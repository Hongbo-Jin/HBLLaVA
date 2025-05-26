python hbllava/eval/eval_scanqa.py \
        --model-path /mnt/cloud_disk/jhb/binjiang/ckpts/HBLLaVA-Coldstart-16 \
        --question-file data/gt_files/scannet/scanqa_temp_gt.json \
        --answers-file /mnt/cloud_disk/jhb/binjiang/HBLLaVA/output/hbllava-0.5b-scanqa_answer_val.json \
        --data-folder /mnt/cloud_disk/public_data/ScanNet_for_ScanQA_SQA3D/downsample_32_w_3d_features/posed_images \
        --conv-mode "qwen2_base" \
        --num-frames 16 \
        --max-new-tokens 1024 

# python llava/eval/scanqa_evaluator.py