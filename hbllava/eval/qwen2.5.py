from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import shortuuid
from tqdm import tqdm

def eval():
    
    ans_file = open("/mnt/cloud_disk/jhb/binjiang/SpatialReason/pred/5_qwen2.5vl_7B_scene0_ans.json", "w")

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/mnt/cloud_disk/public_ckpts/Qwen2.5-VL-7B-Instruct", 
        torch_dtype="auto",
        device_map="auto",
    )
    # default processer
    processor = AutoProcessor.from_pretrained("/mnt/cloud_disk/public_ckpts/Qwen2.5-VL-7B-Instruct")
    
    qa_data=[]
    with open('/mnt/cloud_disk/jhb/binjiang/SpatialReason/output0512/scene0000_00_qa.json','r') as file:
        qa_data=json.load(file)
    
    for qa_sample in tqdm(qa_data):
        question=qa_sample['question']+"\nOnly select the best answer:"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": [
                            "/mnt/cloud_disk/jhb/binjiang/SpatialReason/images/downsample_32_w_3d_features/posed_images/scene0000_00/0.jpg",
                            "/mnt/cloud_disk/jhb/binjiang/SpatialReason/images/downsample_32_w_3d_features/posed_images/scene0000_00/174.jpg",
                            "/mnt/cloud_disk/jhb/binjiang/SpatialReason/images/downsample_32_w_3d_features/posed_images/scene0000_00/348.jpg",
                            "/mnt/cloud_disk/jhb/binjiang/SpatialReason/images/downsample_32_w_3d_features/posed_images/scene0000_00/522.jpg",
                            "/mnt/cloud_disk/jhb/binjiang/SpatialReason/images/downsample_32_w_3d_features/posed_images/scene0000_00/696.jpg"
                        ],
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

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
        ans_file.write(json.dumps({"question": question,
                                   "answer":qa_sample['correct_letter'],
                                   "pred_answer": output_text,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__=="__main__":
    eval()
    