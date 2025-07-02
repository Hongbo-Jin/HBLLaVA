import json

pred_file="/mnt/cloud_disk/jhb/binjiang/HBLLaVA/output/eval_results/SpatialReason/8_Qwen2.5-VL-3B-Instruct.json"

preds = [json.loads(q) for q in open(pred_file, "r")]

print(preds[0])
question_types=["local_spatial","holistic_spatial"]

preds_list=[[],[]]

for sample in preds:
    if sample['question_type']=="local_spatial":
        preds_list[0].append(sample)
    else:
        preds_list[1].append(sample)
        
print(f'total samples of question type1:{len(preds_list[0])}')
print(f'total samples of question type2:{len(preds_list[1])}')
    
acc1=0
acc2=0
acc3=0

for sample in preds_list[0]:
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
total_acc1=acc1+acc2+acc3
print(f'type1 total samples:{len(preds_list[0])}')
print(f'type1 total accuracy:{total_acc1/len(preds_list[0])}')

for sample in preds_list[1]:
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
print(f'type2 total samples:{len(preds_list[1])}')
print(f'type2 total accuracy:{(acc1+acc2+acc3)/len(preds_list[1])}')

total_acc2=acc1+acc2+acc3

print(f'total accuracy:{(total_acc1+total_acc2)/(len(preds_list[0])+len(preds_list[1]))}')
