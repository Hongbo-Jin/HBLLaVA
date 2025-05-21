
from transformers import PreTrainedModel, AutoModel, AutoModelForCausalLM,Qwen2ForCausalLM,AutoTokenizer
import torch.nn as nn
from .configuration_hbllava import HBLlavaConfig
from hbllava.model.connector.base import *
from . import ConnectorFactory,LLMFactory,VisionTowerFactory


class HBLlavaPreTrainedModel(PreTrainedModel):
    
    config_class = HBLlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    
    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        return self.language_model._supports_sdpa


class HBLlavaBase(HBLlavaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        
        self.language_model=AutoModelForCausalLM.from_pretrained(config.llm_model_name_or_path)
        self.vision_tower = VisionTowerFactory(config.vision_model_name_or_path)(config.vision_config)
        self.connector=SimpleResBlock(channels=2048)
        
        self.tokenizer=AutoTokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            cache_dir = config.cache_dir,
            model_max_length = config.tokenizer_model_max_length,
            padding_side = config.tokenizer_padding_side,
            use_fast = config.tokenizer_use_fast,
        )       
    
        self.post_init()
        