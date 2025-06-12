
export CUDA_VISIBLE_DEVICES=0,1,2,3

NUM_frame=5
model_name='Qwen2.5-VL-3B-Instruct'
output_dir="/mnt/cloud_disk/jhb/binjiang/HBLLaVA/output/ckpts/${model_name}_scannet"
gt_file="/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/scanqa/ScanQA_v1.0_train_forqwen_0-10.json"
gt_file_processed="/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/scanqa/ScanQA_v1.0_train_forqwen_0-10_processed.json"

python hbllava/train/preprocess_qwen.py \
    --gt-file ${gt_file} \
    --num-frame ${NUM_frame} \
    --gt-file-processed ${gt_file_processed}


python hbllava/train/train_qwen.py \
    --model_name_or_path /mnt/cloud_disk/public_data/Qwen2.5-VL-3B-Instruct \
    --data_path   ${gt_file_processed}\
    --data_folder /mnt/cloud_disk/public_data/ScanNet_for_ScanQA_SQA3D/downsample_32_w_3d_features/posed_images/ \
    --bf16 True \
    --cache_dir /mnt/cloud_disk/jhb/binjiang/HBLLaVA/cache \
    --gradient_checkpointing True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --sampling_rate 2 \
    --num-frame ${NUM_frame} \
    --output_dir ${output_dir} \
    --model_type "qwen2.5vl" \
    --downsample True \
    --downsample_rate 5 \
    --flash_attn False \