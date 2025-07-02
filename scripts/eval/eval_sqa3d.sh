
NUM_frame=8
model_name='Qwen2.5-VL-3B-Instruct-scanqa-sqa3d'
model_path="/mnt/cloud_disk/jhb/binjiang/ckpts/Qwen2.5-VL-3B-Instruct-scannet-sqa3d_12"

python hbllava/eval/eval_sqa3d.py \
    --model-path ${model_path} \
    --num-frame ${NUM_frame} \
    --gt-file /mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/sqa3d/gt_test_processed.json  \
    --data-folder /mnt/cloud_disk/public_data/ScanNet_for_ScanQA_SQA3D/downsample_32_w_3d_features/posed_images/ \
    --answer-file /mnt/cloud_disk/jhb/binjiang/HBLLaVA/output/eval_results/sqa3d/${NUM_frame}_${model_name}.json \
    --downsample-factor 2