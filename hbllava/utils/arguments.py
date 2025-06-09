from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, TYPE_CHECKING
import transformers


if TYPE_CHECKING:
    import transformers

@dataclass
class ModelArguments:
    cache_dir: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5")
    tune_mm_vision: Optional[bool] = field(default=False)
    tune_mm_mlp: Optional[bool] = field(default=True)
    tune_mm_llm: Optional[bool] = field(default=False)
    
    tokenizer_name_or_path: Optional[str] = field(default=None)
    attn_implementation: Optional[str] = field(default=None)
    vision_tower: Optional[str] = field(default='')
    connector_type: str = field(default='linear')
    
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    resampler_hidden_size: Optional[int] = field(default=768)
    num_queries: Optional[int] = field(default=512)
    num_resampler_layers: Optional[int] = field(default=3)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tokenizer_use_fast: bool = field(default=False)
    tokenizer_padding_side: str = field(default='right')
    # hidden_size: Optional[int] = field(default=896)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_folder: Optional[str] = field(default=None)
    sampling_rate: Optional[int]=field(default=1)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    num_frame: int = field(default=16)
    model_type: str = field(default="qwen2.5")
    downsample: bool = False
    downsample_rate: int =field(default=1.0)
    
    
    image_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    image_folder: Optional[str] = field(default=None)
    video_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    video_folder: Optional[str] = field(default=None)
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_aspect_ratio: str = 'square'
    conv_version: str = 'pretrain'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    
    bf16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    output_dir: str = field(default='./output')
    training_recipe: str = field(default='common')
    flash_attn: bool = field(default=False)
    
    # tune_type_llm: str = field(default="frozen") # support only: frozen, full, lora, qlora_int4, qlora_int8
    # tune_type_vision_tower: str = field(default="frozen") # support only: frozen, full, partially-tune
    # tune_vision_tower_from_layer: Optional[int] = field(default=10)
    # tune_type_connector: str = field(default="full") # support only: frozen, full
    # tune_embed_tokens: Optional[int] = field(default=False)
    
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    vision_tower_lr: Optional[float] = None
    pretrained_model_path: Optional[str] = field(default=None)
    
    
    
   
