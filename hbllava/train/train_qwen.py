import argparse
from hbllava.model import load_pretrained_model
import torch
import transformers
from hbllava.utils import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from hbllava.data.data_qwen import make_supervised_data_module
from transformers import Trainer
import logging
import pathlib

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


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    
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
        attn_implementation="flash_attention_2" if training_args.flash_attn else None,
        cache_dir=model_args.cache_dir
    )
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
    data_args.image_processor=processor.image_processor
    data_args.model_type = "qwen2.5vl"
    tokenizer=processor.tokenizer   
   
    set_model(model_args, model)
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    training_args.report_to = "none"
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    print('train finished')
    return None


if __name__=="__main__":
    train()