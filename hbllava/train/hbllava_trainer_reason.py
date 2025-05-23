import os
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)

from trl.data_utils import is_conversational
from trl.models import prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig

from hbllava.utils import *
from hbllava.model import *

from pytorchvideo.data.encoded_video import EncodedVideo
import numpy as np
from hbllava.utils import get_jpg_files_os
from PIL import Image

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class HBLLaVATrainer_Reason(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        text_processor: Union[str, PreTrainedModel],
        # video_preprocess: Union[str, PreTrainedModel],
        scene_preprocess: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        attn_implementation: str = "flash_attention_2",
        data_path: str = None,
    ):
        self.data_path = data_path
        model_init_kwargs = {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        
        self.ref_model = HBLlavaForConditionalGeneration.from_pretrained(args.pretrained_model_path).to('cuda')
        self.processing_class = processing_class
        self.text_processor = text_processor
        # self.video_preprocess = video_preprocess
        self.scene_preprocess = scene_preprocess
        self.reward_funcs = reward_funcs
        

        def data_collator(features):  # No data collation is needed in GRPO
            return features

        self.beta = 0.01 #args.beta
        self.num_generations = 8 # = G in the GRPO paper
        self.num_frame = 16

        model.warnings_issued["estimate_tokens"] = True
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        
        self.model_accepts_loss_kwargs = False
        self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)

    def _get_per_token_logps(self, model, input_ids, conpletion_ids_length, **kwargs):
        logits = model.forward(input_ids, video=kwargs["video"]).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        
        logits = logits[:, -conpletion_ids_length:, :]  #last conpletion_ids_length logits
        input_ids = input_ids[:, -conpletion_ids_length:]  #last conpletion_ids_length inputs_id
        
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, num_items_in_batch=None):
        
        prompt_inputs = {} 
        prompts = [x["prompt"] for x in inputs]
        prompts_text = []
        for example in inputs:
            query = DEFAULT_IMAGE_TOKEN + example["prompt"][0]["content"][1]["text"]
            msg = Message()
            msg.add_message(query) 
            
            result = self.text_processor(msg.messages, mode='eval')
            prompts_text.append(result['prompt'])
            prompt_inputs["input_ids"] = result['input_ids'].unsqueeze(0).cuda()
        
        for (cur_idx, cur_input) in enumerate(inputs):
            
            scene_file=inputs[cur_idx]['filename']
            scene_folder = os.path.join(self.data_path, scene_file)
            files=get_jpg_files_os(scene_folder)[:self.num_frame]

            scene_images=[]
            for file in files:
                img=Image.open(file)
                image_processed= self.scene_preprocess(image=img)
                scene_images.append(image_processed)
            scene_images=torch.stack(scene_images)
        
            total_frames = scene_images.shape[0]
        
            video_tensor = scene_images.unsqueeze(dim=0)
            prompt_inputs["video"] = video_tensor

        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            all_completions = []
            for i in range(self.num_generations):
                completion = unwrapped_model.generate(
                    inputs=prompt_inputs["input_ids"],
                    video=prompt_inputs["video"],
                    do_sample=True,
                    num_beams=1,
                    pad_token_id=self.processing_class.pad_token_id,
                    max_new_tokens=512,
                    use_cache=True,)
                all_completions.append(completion)
                outputs = self.processing_class.batch_decode(
                    completion, skip_special_tokens=True
                )[0]
                outputs = outputs.strip()

            # Stack all completions and pad if needed
            max_length = max(completion.size(1) for completion in all_completions)
            
            padded_completions = []
            for completion in all_completions:
                if completion.size(1) < max_length:
                    padding = torch.full(
                        (completion.size(0), max_length - completion.size(1)),
                        self.processing_class.pad_token_id,
                        dtype=completion.dtype,
                        device=completion.device,
                    )
                    padded_completion = torch.cat([completion, padding], dim=1)
                else:
                    padded_completion = completion
                padded_completions.append(padded_completion)

            # Stack all padded completions
            prompt_completion_ids = torch.cat(padded_completions, dim=0)

        completion_ids = prompt_completion_ids
        conpletion_ids_length = completion_ids.shape[1]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        prompt_inputs["video"] = prompt_inputs["video"].repeat(len(prompt_completion_ids), 1, 1, 1, 1)
        
        expanded_prompt_inputs = prompt_inputs["input_ids"].expand(prompt_completion_ids.size(0), -1)
        combined_ids = torch.cat((expanded_prompt_inputs, prompt_completion_ids), dim=1)
        
        prompt_inputs.pop("input_ids")
        
        per_token_logps = self._get_per_token_logps(model, combined_ids, conpletion_ids_length, **prompt_inputs)
        with torch.inference_mode():
            ref_per_token_logps = self._get_per_token_logps(self.ref_model, combined_ids, conpletion_ids_length, **prompt_inputs)
        
        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        per_token_kl  = torch.clamp(per_token_kl, min=-100, max=100)
    
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = [[{"role": "assistant", "content": completion.strip()}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device) #torch.Size([num_generations, 2])
        for i, reward_func in enumerate(self.reward_funcs):
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs) #torch.Size([num_generations])
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device) #torch.Size([num_generations])
        
        acc_reward = rewards_per_func[:, 0]
        format_reward = rewards_per_func[:, 1]
        rewards = acc_reward + (2 * acc_reward - 1) * format_reward
        rewards = torch.where(rewards == 0, torch.tensor(-2.0, device=rewards.device, dtype=rewards.dtype), rewards)
        
        """
        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1) #torch.Size([num_generations])
        print("rewards:",rewards)
        """

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        #advantages = (rewards - mean_grouped_rewards)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4) #torch.Size([num_generations])

        noise = torch.randn_like(advantages) * 0.02
        advantages = advantages + noise

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        #loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.shape[1]).mean()
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        
        return loss
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()
