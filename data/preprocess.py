import json

with open('./data/gt_files/Nextqa/nextqa-coldstart-16.json','r') as file:
    data=json.load(file)
    

for sample in data:
    sample['video']=sample['video'].replace('/NextQA/NExTVideo','')

with open('./data/gt_files/Nextqa/nextqa-coldstart-16-p.json','w') as f:
    json.dump(data,f,indent=4)

