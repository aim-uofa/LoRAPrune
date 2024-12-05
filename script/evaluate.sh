CUDA_VISIBLE_DEVICES=0 python inference.py \
    --base_model "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset 'MBZUAI/LaMini-instruction' \
    --lora_weights 'outputs_dir' \
    --cutoff_len 512 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"