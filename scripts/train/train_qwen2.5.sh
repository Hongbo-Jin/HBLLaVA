
export CUDA_VISIBLE_DEVICES=0,1,2

NUM_frame=12
model_name='Qwen2.5-VL-3B-Instruct'
output_dir="/mnt/cloud_disk/jhb/binjiang/HBLLaVA/output/ckpts/${model_name}_scannet_${NUM_frame}"
gt_file="/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/scanqa/ScanQA_v1.0_train_forqwen_all.json"
gt_file_processed="/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/scanqa/tmp_processed.json"

# Training hyperparameters
lr=2e-7
batch_size=1
grad_accum_steps=4

# Output configuration
run_name="qwen2.5vl-baseline"

python hbllava/train/preprocess_qwen.py \
    --gt-file ${gt_file} \
    --num-frame ${NUM_frame} \
    --gt-file-processed ${gt_file_processed}


python hbllava/train/train_qwen.py \
    --model_name_or_path /mnt/cloud_disk/public_data/Qwen2.5-VL-3B-Instruct \
    --data_path   ${gt_file_processed}\
    --data_folder /mnt/cloud_disk/public_data/ScanNet_for_ScanQA_SQA3D/downsample_32_w_3d_features/posed_images/ \
    --cache_dir /mnt/cloud_disk/jhb/binjiang/HBLLaVA/cache \
    --gradient_checkpointing True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --sampling_rate 1 \
    --num-frame ${NUM_frame} \
    --output_dir ${output_dir} \
    --model_type "qwen2.5vl" \
    --downsample True \
    --downsample_rate 3 \
    --flash_attn False \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --num_train_epochs 1.0 \
    --learning_rate ${lr} \
    --model_max_length 8192 \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --run_name ${run_name} \
    --fp16 True \
      # --bf16 True \#A100请选用bf16，禁用fp16