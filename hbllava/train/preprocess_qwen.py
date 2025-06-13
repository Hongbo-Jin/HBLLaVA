import json
import argparse


def process(args):
    
    data=json.load(open(args.gt_file,'r'))
    
    results=[]
    for sample in data:
        tmp_conv=sample['conversations'][0]['value']
        tmp_conv="<image>\n"*args.num_frame+tmp_conv
        sample['conversations'][0]['value']=tmp_conv
        results.append(sample)

    with open(args.gt_file_processed,'w') as f:
        json.dump(results,f,indent=4)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", type=str)
    parser.add_argument("--num-frame", type=int)
    parser.add_argument("--gt-file-processed", type=str)

    args = parser.parse_args()
    process(args)
    