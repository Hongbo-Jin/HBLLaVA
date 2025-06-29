import json

gt_file_path="/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/SpatialReason/merged_qa_dataset.json"
output_path="/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/SpatialReason/SR_for_qwen.json"

data_infos=json.load(open(gt_file_path))

print(len(data_infos))
print(data_infos[0])

answer_data=[]

local_data=[]
whole_data=[]

for idx,sample in enumerate(data_infos):

    answer=sample['answers'][0]
    answer_id=sample['answers'][1].replace('Answer: ','').replace('.','')
    conversation=[
        {"from":"human","value":"<image>\n"+sample["question"]+"\nOnly select the best answer."},
        {"from":"gpt","value":answer_id}
    ]
    if sample['question_type']=="local_spatial":
        local_data.append({
            "scene_id":sample['scene_id'],
            "question":sample['question'],
            "question_id":str(idx),
            "options":sample['options'],
            "answer":answer,
            "answer_id":answer_id,
            "question_type":"local_spatial"
        })
    else:
        whole_data.append(
            {
                "scene_id":sample['scene_id'],
                "question":sample['question'],
                "question_id":str(idx),
                "options":sample['options'],
                "answer":answer,
                "answer_id":answer_id,
                "question_type":"holistic_spatial"
            })
    

with open("/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/SpatialReason/local_spatial.json",'w') as file:
    json.dump(local_data,file,indent=4)
    
with open("/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/SpatialReason/holistic_spatial.json",'w') as file:
    json.dump(whole_data,file,indent=4)