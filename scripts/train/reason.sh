#!/bin/bash
SCENE_DATA_PATH="/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/scannet/tmp_gt_reason.json"
SCENE_PATH="/mnt/cloud_disk/public_data/ScanNet_for_ScanQA_SQA3D/downsample_32_w_3d_features/posed_images"

VIDEO_DATA_PATH="/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/Nextqa/nextqa-coldstart-16-p.json"
VIDEO_PATH="/mnt/cloud_disk/public_data/ScanNet_for_ScanQA_SQA3D/downsample_32_w_3d_features/posed_images"

MODEL_PATH=/mnt/cloud_disk/jhb/binjiang/ckpts/HBLLaVA-Coldstart-16
LLM_VERSION=/mnt/cloud_disk/public_ckpts/Qwen2.5-0.5B # llm path
VT_VERSION=/mnt/cloud_disk/public_ckpts/clip-vit-large-patch14-336 #vision tower path
CN_VERSION=groupresampler #connector type
VT_VARIANT="${VT_VERSION##*/}"
LLM_VARIANT="${LLM_VERSION##*/}"
VERSION=base
TRAIN_RECIPE=common
CONV_VERSION=qwen2_base
MODEL_MAX_LENGTH=3072
NUM_FRAME=16
NUM_QUERY=512

deepspeed --include localhost:0,1,2,3 --master_port 29501 hbllava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --video_data_path  $SCENE_DATA_PATH \
    --video_folder $SCENE_PATH \
    --is_multimodal True \
    --conv_version $CONV_VERSION \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VT_VERSION \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type $CN_VERSION \
    --num_frames $NUM_FRAME \
    --num_queries $NUM_QUERY \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --bf16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm full \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --group_by_modality_length False \
    --pretrained_model_path $MODEL_PATH \
    --output_dir /mnt/cloud_disk/jhb/binjiang/ckpts/hbllava_reason \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name hbllava_reason
