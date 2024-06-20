import numpy as np
import torch
from .lora import Linear, Linear8bitLt

pruning_groups = {'self_attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                  'mlp': ['up_proj', 'gate_proj'],
                  'block': ['o_proj', 'down_proj']}

DIM = 128

def _is_target_larer(module):
    return (isinstance(module, Linear) or isinstance(module, Linear8bitLt)) and module.is_prune

def unfreeze(model):
    for name, module in model.named_modules():
        if _is_target_larer(module):
            module.weight.requires_grad = True

def freeze(model):
    layers = len(model.model.model.layers)
    freeze_layer = int(layers * 0.1)
    for name, module in model.named_modules():
        if _is_target_larer(module):
            layer = int(name.split('.')[4])
            if layer < freeze_layer or layer == layers-1:
                module.is_prune = False

def init_sensitivity_dict(model):
    sensitivity_record = {}
    for name, module in model.named_modules():
        if _is_target_larer(module):
            if name.split('.')[-1] in pruning_groups['self_attn']:
                groups = module.out_features // DIM
            else:
                groups = module.out_features
            name = ".".join(name.split('.')[:-1])
            if name in sensitivity_record:
                continue
            sensitivity_record[name] = module.lora_A.weight.data.new_zeros(groups)
    return sensitivity_record

def update_sensitivity_dict(model, s_dict, pruning_type):
    s_all = init_sensitivity_dict(model)
    for name, module in model.named_modules():
        if _is_target_larer(module):
            is_attn = name.split('.')[-1] in pruning_groups['self_attn']
            fan_in = name.split('.')[-1] in pruning_groups['block']
            s = compute_sensitivity(module, is_attn, pruning_type, fan_in)
            name = ".".join(name.split('.')[:-1])
            s_all[name] += s
            #s_dict[i] += s
    for name, imp in s_all.items():
        if torch.isnan(imp.sum()):
            return s_dict
    for name, imp in s_dict.items():
        s_dict[name] = imp * 0.9 + s_all[name] * 0.1
    return s_dict

def compute_sensitivity(layer, is_attn, prune_metric='lora', transpose=False, norm=True):
    a = layer.lora_A.weight.data
    b = layer.lora_B.weight.data
    if prune_metric == 'lora':
        grad_a = layer.lora_A.weight.grad
        grad_b = layer.lora_B.weight.grad
        grad = (grad_b @ a + b @ grad_a - grad_b @ grad_a)
    elif prune_metric == 'magnitude':
        grad = 1
    elif prune_metric == 'grad':
        grad = layer.weight.grad
    else:
        raise NotImplementedError
    if hasattr(layer, 'state'):
        weight = (layer.weight.data * layer.state.SCB.reshape(-1, 1)) / 127
    else:
        weight = layer.weight.data
    s = (grad * (b @ a * layer.scaling + weight)).abs()
    if transpose:
        s = s.t()
    if is_attn:
        s = s.reshape(s.shape[0] // DIM, -1)
    s = s.sum(1)
    if norm:
        s = s / (torch.linalg.norm(s) + 1e-8)
    return s

def prune_fp16_module(module, mask, transpose):
    mask = mask.bool()
    module.train()
    if not transpose:
        module.weight.data = module.weight.data[mask]
        module.out_features = int(mask.sum())
        if module.bias:
            module.bias.data = module.bias.data[mask]
        module.lora_B.weight.data = module.lora_B.weight.data[mask]
        module.lora_B.out_features = int(mask.sum())
    else:
        module.weight.data = module.weight.data[:, mask]
        module.in_features = int(mask.sum())
        module.lora_A.weight.data = module.lora_A.weight.data[:, mask]
        module.lora_A.in_features = int(mask.sum())
    module.merge_weights = True
    module.train(False)

def prune_one_layer(layer):
    ## self_attn
    prune_fp16_module(layer.self_attn.q_proj, layer.self_attn.q_proj.lora_mask, False)
    prune_fp16_module(layer.self_attn.k_proj, layer.self_attn.k_proj.lora_mask, False)
    prune_fp16_module(layer.self_attn.v_proj, layer.self_attn.v_proj.lora_mask, False)
    prune_fp16_module(layer.self_attn.o_proj, layer.self_attn.q_proj.lora_mask, True)
    layer.self_attn.num_heads = int(layer.self_attn.q_proj.lora_mask.sum()) // DIM
    layer.self_attn.hidden_size = int(layer.self_attn.q_proj.lora_mask.sum())

    ## mlp
    prune_fp16_module(layer.mlp.gate_proj, layer.mlp.gate_proj.lora_mask, False)
    prune_fp16_module(layer.mlp.up_proj, layer.mlp.up_proj.lora_mask, False)
    prune_fp16_module(layer.mlp.down_proj, layer.mlp.gate_proj.lora_mask, True)

    ## reset mask
    del(layer.self_attn.q_proj.lora_mask)
    del(layer.self_attn.k_proj.lora_mask)
    del(layer.self_attn.v_proj.lora_mask)
    del(layer.mlp.gate_proj.lora_mask)
    del(layer.mlp.up_proj.lora_mask)

def prune(model):
    for layer_id, layer in enumerate(model.model.model.layers):
        print("pruning layer {}".format(layer_id))
        prune_one_layer(layer)

def local_prune(model, s_dict, ratio, target_ratio):
    original_param_num = 0
    pruned_param_num = 0
    for name, module in model.named_modules():
        if _is_target_larer(module):
            original_param_num += np.prod(module.weight.shape)
            pruned_param_num += np.prod(module.weight.shape) * ratio
            is_attn = name.split('.')[-1] in pruning_groups['self_attn']
            if name.split('.')[-1] in pruning_groups['block']:
                continue
            name = ".".join(name.split('.')[:-1])
            if not hasattr(module, 'lora_mask'):
                continue
            if (1-module.lora_mask.mean()).item() >= target_ratio:
                continue
            total_num = module.lora_mask.numel()
            c_mask = module.lora_mask.data
            mask = torch.ones_like(c_mask)

            if is_attn:
                mask = mask.reshape(-1, DIM)[:, 0]
                c_mask = c_mask.reshape(-1, DIM)[:, 0]
                total_num /= DIM
            need_prune_num = int(total_num * ratio)
            importance = s_dict[name] * c_mask
            can_prune = torch.argsort(importance)[:need_prune_num]
            mask[can_prune] = 0
            if is_attn:
                mask = (mask.new_ones(module.lora_mask.shape).reshape(-1, DIM) * mask.unsqueeze(1)).reshape(-1)
            module.lora_mask.data = mask
        else:
            if hasattr(module, 'weight'):
                original_param_num += np.prod(module.weight.shape)
    print("pruned/original parameters number:{:3f}/{:3f}  ratio:{:3f}".format(pruned_param_num*1e-9,
                                                                               original_param_num*1e-9,
                                                                               pruned_param_num/original_param_num))

def schedule_sparsity_ratio(
    step,
    total_step,
    initial_warmup,
    final_warmup,
    initial_sparsity,
    final_sparsity,
):
    if step <= initial_warmup * total_step:
        sparsity = initial_sparsity
    elif step > (total_step - final_warmup * total_step):
        sparsity = final_sparsity
    else:
        spars_warmup_steps = initial_warmup * total_step
        spars_schedu_steps = (final_warmup + initial_warmup) * total_step
        mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
        sparsity = final_sparsity + (initial_sparsity - final_sparsity) * (mul_coeff ** 3)
    return sparsity

def prune_from_checkpoint(model):
    prune(model)

def print_trainable_parameters(model):
    total_params = 0
    trainable_params = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        total_params += p.numel()
    print("total params:{}   trainable params:{}    ratio:{}".format(total_params * 1e-6, trainable_params * 1e-6, trainable_params / total_params))