
NUM_frame=20
model_name='Qwen2.5-VL-3B-Instruct'

python hbllava/train/train_qwen.py \
    --model_name_or_path /mnt/cloud_disk/public_ckpts/Qwen2.5-VL-3B-Instruct \
    --data_path /mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/scanqa/ScanQA_v1.0_train.json  \
    --data_folder /mnt/cloud_disk/public_data/ScanNet_for_ScanQA_SQA3D/downsample_32_w_3d_features/posed_images/ \
    --bf16 True \
    --cache_dir /mnt/cloud_disk/jhb/binjiang/HBLLaVA/cache \
    --gradient_checkpointing True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm False \
    # --num-frame ${NUM_frame} \