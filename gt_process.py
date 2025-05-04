import json

with open('/mnt/cloud_disk/public_data/ScanNet_for_ScanQA_SQA3D/ScanQA-v1.0/ScanQA_v1.0_train.json','r') as file:
    data_infos=json.load(file)
    
print(len(data_infos))

demo_train_data=data_infos[:500]
    
with open("demo_train.jsonl", "w", encoding="utf-8") as f:
    for item in demo_train_data:
        item['problem']=item['question']
        item.pop('question')
        json_line = json.dumps(item, ensure_ascii=False)  # 处理非ASCII字符
        f.write(json_line + "\n")  # 注意添加换行符
    