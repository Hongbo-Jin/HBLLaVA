
NUM_frame=8

python hbllava/eval/eval_sqa3d.py \
    --model-path /mnt/cloud_disk/public_ckpts/Qwen2.5-VL-7B-Instruct \
    --num-frame ${NUM_frame} \
    --gt-file /mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/sqa3d/gt_test_processed.json  \
    --data-folder /mnt/cloud_disk/public_data/ScanNet_for_ScanQA_SQA3D/downsample_32_w_3d_features/posed_images/ \
    --answer-file /mnt/cloud_disk/jhb/binjiang/HBLLaVA/output/${NUM_frame}_qwen2.5vl_7B_sqa3d.json