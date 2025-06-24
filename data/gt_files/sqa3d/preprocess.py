import json
import os

def get_jpg_files_os(directory):
    jpg_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))
    return jpg_files
    
    
data_info=json.load(open('/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/sqa3d/gt_train_processed.json','r'))

print(data_info[0])
result=[]

for idx,sample in enumerate(data_info):
    # if idx>7000 :
    #     break
    
    sample['conversations']=[
        {
            'from':"human",
            "value":sample['question']+'\nAnswer the question using a single word or phrase.'
        },
        {
            "from":"gpt",
            "value":sample['answer']
        }
    ]
    
    result.append(sample)

print(result[0])
print(len(result))
with open('/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/sqa3d/sqa3d_train_forqwen_all.json','w') as file:
    json.dump(result,file,indent=4)
