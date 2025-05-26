import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from PIL import Image
import math
from hbllava.model import *
from hbllava.data import *
from hbllava.utils import *

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model2(args):
    # Model
    # disable_torch_init()

    model = HBLlavaForConditionalGeneration.from_pretrained(args.model_path).to('cuda')
    tokenizer=model.tokenizer
    image_processor = model.vision_tower.image_processor
    context_len = getattr(model.config, 'max_sequence_length', 2048)
    
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_preprocess = ImagePreprocess(image_processor, data_args)
    scene_preprocess = ScenePreprocess(image_processor)
    
    with open(args.question_file, 'r') as file:
        questions = json.load(file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for line in tqdm(questions):
    
        idx = line["question_id"]
        scene_file = line["scene_id"]
        data_path = os.path.join(args.data_folder, scene_file)
        qs = line["question"]
        answer=line['answer']        
    
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        msg = Message()
        msg.add_message(qs)
        # print(f'----{msg.messages}')
        result = text_processor(msg.messages, mode='eval')
        # print("result:",result)
     
        input_ids = result['input_ids']
        prompt = result['prompt']
        input_ids = input_ids.unsqueeze(0).cuda()
    
        if args.data_folder is not None:
            scene_image_paths=get_jpg_files_os(data_path)[:args.num_frames]
        
        images=[]
        for scene_image_path in scene_image_paths:
            img=Image.open(scene_image_path)
            img_=scene_preprocess(img)
            images.append(img_)
        images=torch.stack(images)       

        # images_feature=model.vision_tower(images)
        # last_dim = images_feature.shape[-1]
        # images_feature = images_feature.reshape(1, -1, last_dim) #torch.Size([1, frame*576, 1024])
        # video_tensor = model.connector(images_feature) #torch.Size([1, 512, 896])

        stop_str = text_processor.template.separator.apply()[1]
        # print(f'stop str {stop_str}')
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            # print("tokenizer.pad_token_id:",tokenizer.pad_token_id)
            output_ids = model.generate(
                input_ids,
                images=None,
                video=[images],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        
        outputs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        # print(outputs)
        
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": prompt,
                                   "text": outputs,
                                   "model_id": 'hbllava-0.5B-coldstart',
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="ckpts/HBLLaVA-Coldstart-16")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-folder", type=str, default="playground/ScanNet_for_ScanQA_SQA3D/downsample_32_w_3d_features/posed_images")
    parser.add_argument("--question-file", type=str, default="data/gt_files/scannet/tmp_gt_reason.json")
    parser.add_argument("--answers-file", type=str, default="output/hbllava_scanqa_val_answer_pred.json")
    parser.add_argument("--conv-mode", type=str, help=('qwen2_base','pretrain'))
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=648)
    args = parser.parse_args()

    eval_model2(args)
