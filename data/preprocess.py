import json
import re

with open('/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/scannet/tmp_gt_reason.json','r') as file:
    data=json.load(file)
    
result=[]
for idx,sample in enumerate(data):
    
    sample['filename']=sample['filename'].split('/')[-1]

with open('/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/scannet/tmp_gt_reason.json','w') as f :
    json.dump(data,f,indent=4)
    
    # question=sample['conversations'][0]['value'].split('Output the thinking process')[0].split('<image>\n')[-1]
    # solution= re.findall(r'<answer>(.*?)</answer>', sample['conversations'][1]['value'])
    # solution=f'<answer>{solution[0]}</answer>'
 
    # result.append(
    #     {
    #         "problem_id":idx,
    #         "problem":question,
    #         "data_type":'scene',
    #         'problem_type': ["reasoning"],
    #         "solution":solution,
    #         "filename":sample['scene']
    #     }
    # )
    
# with open('/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/scannet/tmp_gt_reason.json','w') as f :
#     json.dump(result,f,indent=4)
    
    