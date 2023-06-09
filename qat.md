qat 训练入口
```shell
torchrun --nproc_per_node=4 --master_port=20095 train_quant.py \
    --model_name_or_path /nvme/share_data/llama_ckpts/huggingface/7B \
    --data_path ./alpaca_data_min.json \
    --output_dir ./output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --model_max_length=256 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --fsdp "full_shard auto_wrap offload"  \
     --tf32 True
```

蒸馏入口
```shell
torchrun --nproc_per_node=4 --master_port=20095 train_distill.py \
    --model_name_or_path /nvme/share_data/llama_ckpts/huggingface/7B \
    --data_path ./alpaca_data_min.json \
    --output_dir ./output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --model_max_length=256 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --fsdp "full_shard auto_wrap offload"  \
     --tf32 True
```