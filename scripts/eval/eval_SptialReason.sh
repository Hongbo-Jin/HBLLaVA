
NUM_frame=8
model_name='Qwen2.5-VL-3B-Instruct'
model_path="/mnt/cloud_disk/public_ckpts/Qwen2.5-VL-3B-Instruct"

python hbllava/eval/eval_SpatialReason.py \
    --model-path ${model_path} \
    --num-frame ${NUM_frame} \
    --gt-file /mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/SpatialReason/local_spatial.json  \
    --data-folder /mnt/cloud_disk/public_data/ScanNet_for_ScanQA_SQA3D/downsample_32_w_3d_features/posed_images/ \
    --answer-file /mnt/cloud_disk/jhb/binjiang/HBLLaVA/output/eval_results/SpatialReason/${NUM_frame}_${model_name}_local.json \
    --downsample-factor 2 \