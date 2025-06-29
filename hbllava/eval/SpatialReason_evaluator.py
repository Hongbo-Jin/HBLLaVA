import json

pred_file="/mnt/cloud_disk/jhb/binjiang/HBLLaVA/output/eval_results/SpatialReason/8_Qwen2.5-VL-3B-Instruct_.json"

preds = [json.loads(q) for q in open(pred_file, "r")]
    
print(preds[0])

acc1=0
acc2=0
acc3=0

for sample in preds:
    pred=sample['pred_answer'][0]
    answer=sample['answer']
    options=sample['options']
    answer_id=sample['answer_id']
    if answer in pred:
        acc1+=1
    elif answer_id == pred[0]:
        acc2+=1
    else:
        tmp_id=f"({answer_id})"
        if tmp_id in pred:
            acc3+=1

print(acc1)
print(acc2)
print(acc3)
print(f'total samples:{len(preds)}')
print(f'total accuracy:{(acc1+acc2+acc3)/len(preds)}')