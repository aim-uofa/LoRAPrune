CUDA_VISIBLE_DEVICES=0 python inference.py \
    --lora_target_modules '[q_proj, k_proj, v_proj, o_proj, gate_proj,up_proj, down_proj]' \
    --base_model 'llama-7b-hf' \
    --lora_weights 'output_dir' \
    --cutoff_len 2048 \