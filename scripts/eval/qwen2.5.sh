
python hbllava/eval/qwen2.5.py \
    --model-path /mnt/cloud_disk/public_ckpts/Qwen2.5-VL-7B-Instruct \
    --num-frame 8 \
    --gt-file /mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/scannet/scanqa_temp_gt.json  \
    --data-folder /mnt/cloud_disk/public_data/ScanNet_for_ScanQA_SQA3D/downsample_32_w_3d_features/posed_images/ 