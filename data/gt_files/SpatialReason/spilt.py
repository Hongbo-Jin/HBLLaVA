import json

file1="/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/SpatialReason/SR_for_qwen_1.json"
file2="/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/SpatialReason/SR_for_qwen_2.json"
file3="/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/SpatialReason/SR_for_qwen_3.json"

with open(file1,'r') as file:
    data1=json.load(file)
    
with open(file2,'r') as file:
    data2=json.load(file)
    
with open(file3,'r') as file:
    data3=json.load(file)
    
print(data1[0])
print(len(data1))
print(len(data2))
print(len(data3))

data=data1+data2+data3
print(len(data))

with open("/mnt/cloud_disk/jhb/binjiang/HBLLaVA/data/gt_files/SpatialReason/SR_for_qwen_all.json","w") as file:
    json.dump(data,file,indent=4)
    