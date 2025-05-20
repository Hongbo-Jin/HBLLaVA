
from transformers import PreTrainedModel, AutoModel, AutoModelForCausalLM,Qwen2ForCausalLM,AutoTokenizer
import torch.nn as nn
from .configuration_hbllava import HBLlavaConfig
from hbllava.model.connector.base import *
from . import ConnectorFactory,LLMFactory

class HBLlavaBase(PreTrainedModel):
    config_class = HBLlavaConfig  # 绑定配置类
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def __init__(self, config):
        super().__init__(config)
        
        self.language_model=AutoModelForCausalLM.from_pretrained(config.llm_model_name_or_path)
        self.vision_tower = AutoModel.from_pretrained(config.vision_model_name_or_path)
        
        self.connector=SimpleResBlock(channels=2048)
        
        self.tokenizer=AutoTokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            cache_dir = config.cache_dir,
            model_max_length = config.tokenizer_model_max_length,
            padding_side = config.tokenizer_padding_side,
            use_fast = config.tokenizer_use_fast,
        )       
    
        self.post_init()
        