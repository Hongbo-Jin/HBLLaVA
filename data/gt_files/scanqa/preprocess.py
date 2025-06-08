import json

data_info=json.load(open('/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/scanqa/ScanQA_v1.0_train.json','r'))

print(data_info[0])
result=[]

for sample in data_info:
    sample['conversations']=[
        {
            'from':"human",
            "value":"<image>\n"+sample['question']+'\nAnswer the question using a single word or phrase.'
        },
        {
            "from":"gpt",
            "value":sample['answers'][0]
        }
    ]
    result.append(sample)
    
with open('/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/scanqa/ScanQA_v1.0_train_forqwen.json','w') as file:
    json.dump(result,file,indent=4)
