
NUM_frame=2
model_name='Qwen2.5-VL-3B-Instruct'
output_dir="/mnt/cloud_disk/jhb/binjiang/HBLLaVA/output/ckpts"

python hbllava/train/train_qwen.py \
    --model_name_or_path /mnt/cloud_disk/public_ckpts/Qwen2.5-VL-3B-Instruct \
    --data_path /mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/scanqa/ScanQA_v1.0_train_forqwen.json  \
    --data_folder /mnt/cloud_disk/public_data/ScanNet_for_ScanQA_SQA3D/downsample_32_w_3d_features/posed_images/ \
    --bf16 True \
    --cache_dir /mnt/cloud_disk/jhb/binjiang/HBLLaVA/cache \
    --gradient_checkpointing True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm False \
    --sampling_rate 2 \
    --num-frame ${NUM_frame} \
    --output_dir ${output_dir} \
    --model_type "qwen2.5vl" \
    --downsample True \
    --downsample_rate 2 \