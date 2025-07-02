import os
import torch
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, \
                            Qwen2_5_VLForConditionalGeneration, AutoProcessor
                            # Qwen2_5OmniForConditionalGeneration,Qwen2_5OmniProcessor

from .modeling_hbllava import HBLlavaForConditionalGeneration
from .configuration_hbllava import HBLlavaConfig
 
def load_base_ckp_for_lora(ckp_path):
    ckp = torch.load(ckp_path, map_location=torch.device('cpu'))
    new_ckp = OrderedDict()
    for k, v in ckp.items():
        new_k = k.replace('.base_layer', '')
        new_ckp[new_k] = v
    return new_ckp
    

def load_pretrained_model(model_name_or_path, 
                          device_map="auto",
                          device="cuda", 
                          torch_dtype="auto",
                          load_8bit=False,
                          load_4bit=False,
                          attn_implementation=None,
                          **kwargs):
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        # kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    # else:
    #     kwargs['torch_dtype'] = torch.float16
        
    if "qwen2.5" in model_name_or_path.lower():
        if "vl" in model_name_or_path.lower():
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs)
            print('loading qwen2.5-vl')
            processor= AutoProcessor.from_pretrained(model_name_or_path)
            return model, processor
        elif "omni" in model_name_or_path.lower():
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
            # attn_implementation="flash_attention_2",
            )
            processor = Qwen2_5OmniProcessor.from_pretrained(model_name_or_path)
            
            return model, processor
            
            
    # elif "llava" in model_name_or_path.lower():
    #     print(f"Loaded LLaVA model: {model_name_or_path}")
    #     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    #     if "qwen" in model_name_or_path.lower():
    #         model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, **kwargs)
    #         print(f'llava-qwen model loaded')

    #     return model, processor
    
    context_len = getattr(model.config, 'max_sequence_length', 2048)
    tokenizer = model.tokenizer
    
    return model, tokenizer, image_processor, context_len

def load_pretrained_model_t(model_name_or_path, load_type='hf', load_8bit=False, load_4bit=False, device_map="auto",
                          device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    if model_name_or_path is not None and 'lora' not in model_name_or_path:
        model = HBLlavaForConditionalGeneration.from_pretrained(model_name_or_path, low_cpu_mem_usage=True)
        
    elif model_name_or_path is not None and 'lora' in model_name_or_path:
        if os.path.exists(os.path.join(model_name_or_path, 'adapter_config.json')):
            model_config = TinyLlavaConfig.from_pretrained(model_name_or_path)
            model = TinyLlavaForConditionalGeneration(model_config)

            language_model_ckp_path = os.path.join(model_name_or_path, 'language_model/pytorch_model.bin')
            language_model_ckp = load_base_ckp_for_lora(language_model_ckp_path)
            model.language_model.load_state_dict(language_model_ckp)

            vision_tower_ckp_path = os.path.join(model_name_or_path, 'vision_tower/pytorch_model.bin')
            vision_tower_ckp = load_base_ckp_for_lora(vision_tower_ckp_path)
            model.vision_tower._vision_tower.load_state_dict(vision_tower_ckp)

            connector_ckp_path = os.path.join(model_name_or_path, 'connector/pytorch_model.bin')
            connector_ckp = load_base_ckp_for_lora(connector_ckp_path)
            model.connector.load_state_dict(connector_ckp, strict=False)

            model.to(torch.float16)
            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_name_or_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
    
    image_processor = model.vision_tower._image_processor
    context_len = getattr(model.config, 'max_sequence_length', 2048)
    # tokenizer = AutoTokenizer.from_pretrained(model.config.llm_model_name_or_path, use_fast=False, padding_side="right")
    tokenizer = model.tokenizer
    #tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, image_processor, context_len
