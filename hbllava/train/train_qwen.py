import argparse
from hbllava.model import load_pretrained_model
import torch
import transformers
from hbllava.utils import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model, processor = load_pretrained_model(
        model_name_or_path=model_args.model_name_or_path,
        device_map="auto",
        device="cuda",
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        load_8bit=False,
        load_4bit=False,
        attn_implementation=attn_implementation,
        cache_dir=model_args.cache_dir
    )
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
    image_processor=processor.image_processor
    tokenizer=processor.tokenizer   
   
   
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    set_model(model_args, model)
    
        
    print('train finished')
    return None


if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str)
    # parser.add_argument("--num-frame", type=int)
    # parser.add_argument("--gt-file", type=str)
    # parser.add_argument("--data-folder", type=str)
    # parser.add_argument("--bf16", type=bool, default=True)

    # args = parser.parse_args()
    train(attn_implementation="flash_attention_2")