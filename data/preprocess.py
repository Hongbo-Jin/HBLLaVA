import json

with open('/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/scannet/scanqa_temp_gt.json','r') as file:
    data=json.load(file)
    
result=[]
for sample in data:
    result.append(
        {
            "id":sample['question_id'],
            "conversations":sample['conversations'],
            "scene":sample['scene']
        }
    )
    
with open('/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/scannet/tmp_gt.json','w') as f :
    json.dump(result,f,indent=4)
    
    