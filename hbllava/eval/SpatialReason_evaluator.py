import json

pred_file="/mnt/cloud_disk/jhb/binjiang/HBLLaVA/output/eval_results/SpatialReason/8_Qwen2.5-VL-3B-Instruct_.json"

preds = [json.loads(q) for q in open(pred_file, "r")]
    
print(preds[0])

acc=0

for sample in preds:
    pred=sample['pred_answer'][0]
    answer=sample['answer']
    options=sample['options']
    answer_id=sample['answer_id']
    if answer in pred:
        acc+=1

print(acc)
print(f'total accuracy:{acc/len(preds)}')
    