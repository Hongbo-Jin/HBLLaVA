from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import ast

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import Qwen2ForCausalLM, SiglipImageProcessor,AutoModel,AutoTokenizer

from . import LLMFactory, ConnectorFactory, VisionTowerFactory
from .configuration_hbllava import HBLlavaConfig
from ..utils.constants import *
from .vision_tower.clip_encoder import CLIPVisionTower
# from .vision_tower.siglip import SIGLIPVisionTower
from .connector.base import SimpleResBlock
from .connector.custom_connector import Custom_Connector

def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None

class HBLlavaPreTrainedModel(PreTrainedModel):
    
    config_class = HBLlavaConfig
    base_model_prefix = "HBLLaVA_Base"
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


class HBLlavaForConditionalGeneration(HBLlavaPreTrainedModel):
    def __init__(self, config: HBLlavaConfig):
        
        super().__init__(config)
        
        self.language_model = Qwen2ForCausalLM(config.text_config)
        self.connector=Custom_Connector(input_channels=1024,output_channels=config.embedding_dim)
       
        self.post_init()
        
        self.language_model=Qwen2ForCausalLM.from_pretrained(config.llm_model_name_or_path)
        # self.vision_tower=AutoModel.from_pretrained(config.vision_model_name_or_path)
        # self.vision_tower._image_processor = SiglipImageProcessor.from_pretrained(config.vision_model_name_or_path)
        self.vision_tower=CLIPVisionTower(config.vision_model_name_or_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            cache_dir = config.cache_dir,
            model_max_length = config.tokenizer_model_max_length,
            padding_side = config.tokenizer_padding_side,
            use_fast = config.tokenizer_use_fast,
        )
        self.tokenizer.unk_token = self.tokenizer.pad_token
        
        
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        video: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                video,
                image_sizes
            )
        
        return self.language_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        #ano: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
            
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None or video is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                video,
                #ano,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(inputs)

        return self.language_model.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        
    def encode_scene(self, images):
        
        print('encoding---------------------------')
        exit(0)
        return None
    
    def encode_videos(self, videos):
        
        image_features = []
        for video in videos:
            video = video.to(device=self.device, dtype=self.dtype) #torch.Size([frame, 3, 384, 384]) 
            image_feature = self.vision_tower(video) #torch.Size([frame, 576, 1024])
            last_dim = image_feature.shape[-1]
            image_feature = image_feature.reshape(1, -1, last_dim) #torch.Size([1, frame*576, 1024])
            image_feature = self.connector(image_feature) #torch.Size([1, 512, 2048])

            image_features.append(image_feature)
        image_features = torch.cat(image_features, dim=0) #torch.Size([bs, 512, 2048])

        return image_features
    
    
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
        
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, videos, image_sizes=None):
        
        vision_tower = self.vision_tower
        if vision_tower is None or (images is None and videos is None) or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if videos is not None:
            image_features = self.encode_videos(videos)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum() #1
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.language_model.get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]] #[-1, 0, len]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim] #[0, len-1]
            cur_input_embeds = self.language_model.get_input_embeddings()(torch.cat(cur_input_ids_noim)) #torch.Size([len-1, 2560])
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx] #torch.Size([1024, 2560])
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds) #torch.Size([1024+len-1, 2560])
            cur_new_labels = torch.cat(cur_new_labels) #torch.Size([1024+len-1])

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds) #1353,1429...
        batch_size = len(new_input_embeds) #bs=16

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device) #torch.Size([bs, max_len])
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device) #torch.Size([bs, max_len])
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device) #torch.Size([bs, max_len])

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0) #torch.Size([bs, max_len, 2560])

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    
    def load_llm(self, **kwargs):
        language_model_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        pretrained_llm_path = get_value_from_kwargs(kwargs, 'pretrained_llm_path')
        if pretrained_llm_path is not None:
            language_model_name = pretrained_llm_path
        if language_model_name is not None:
            self.language_model = self.language_model.from_pretrained(
                language_model_name, **kwargs
            )
        print('loading language model from ', language_model_name)
        self.language_model.requires_grad_(False)
        
        self.config.text_config.torch_dtype = kwargs.get('torch_dtype', None)
        self.config.pad_token = getattr(self.tokenizer, 'pad_token', None)
        self.config.pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        #self.config.tokenizer_padding_side = getattr(self.tokenizer, 'padding_side', None)
        #self.config.tokenizer_model_max_length =  getattr(self.tokenizer, 'model_max_length', None)
        
    def load_vision_tower(self, **kwargs):
        vision_tower_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        self.vision_tower.load_model(vision_tower_name, **kwargs)
        
    def load_connector(self, **kwargs):
        self.connector.load_model(**kwargs)


            

        
        
