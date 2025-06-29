from hbllava.model import load_pretrained_model
from qwen_vl_utils import process_vision_info
import torch
import json
import shortuuid
from tqdm import tqdm
import argparse
from hbllava.utils import get_jpg_files_os

def downsample_frames(frames: list,factor=2.0):
    
    results=[]
    for frame in frames[0]:
        width, height = frame.size
        new_width = int(width // factor)
        new_height = int(height // factor)
        resized_img = frame.resize((new_width, new_height))
        results.append(resized_img)
        
    return [results]

def eval(args):
    
    fewshot_template="<Question>: The kitchen counter is located to the left of what? <Answer>: refrigerator \n<Question>: On what side of the red door is the refrigerator located? <Answer>: left \n<Question>: What color is the chair? <Answer>: brown"
    qa_data=[]
    with open(args.gt_file,'r') as file:
        qa_data=json.load(file)
    
    ans_file = open(args.answer_file, "w")

    model, processor = load_pretrained_model(
        model_name_or_path=args.model_path,
        device_map="auto",
        device="cuda",
        torch_dtype="auto",
        load_8bit=False,
        load_4bit=False
    )
    
    for qa_sample in tqdm(qa_data):
        question=qa_sample['question']+" Answer the question using one word or one phrase."
        
        scene_path=args.data_folder+qa_sample['scene_id']
        scene_images_path=get_jpg_files_os(scene_path)[0:args.num_frame]
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": scene_images_path,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # print(text)
        # exit(0)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        video_inputs=downsample_frames(frames=video_inputs,factor=args.downsample_factor)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
                                   "question_id":qa_sample['question_id'],
                                   "question": question,
                                   "answer":qa_sample['answer'],
                                   "pred_answer": output_text,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--num-frame", type=int)
    parser.add_argument("--gt-file", type=str)
    parser.add_argument("--data-folder", type=str)
    parser.add_argument("--answer-file", type=str)
    parser.add_argument("--downsample-factor", type=int)

    args = parser.parse_args()
    eval(args)
    