from hbllava.model import load_pretrained_model
from qwen_vl_utils import process_vision_info
import torch
import json
import shortuuid
from tqdm import tqdm
import argparse
from hbllava.utils import get_jpg_files_os

def eval(args):
    
    fewshot_template="<Question>: The kitchen counter is located to the left of what? <Answer>: refrigerator \n<Question>: On what side of the red door is the refrigerator located? <Answer>: left \n<Question>: What color is the chair? <Answer>: brown"
    qa_data=[]
    with open(args.gt_file,'r') as file:
        qa_data=json.load(file)
    
    ans_file = open(f"./output/{args.num_frame}_qwen2.5vl_7B_scanqa_pred.json", "w")

    model, processor = load_pretrained_model(
        model_name_or_path=args.model_path,
        device_map="auto",
        device="cuda",
        torch_dtype="auto"
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
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

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

    args = parser.parse_args()
    eval(args)
    