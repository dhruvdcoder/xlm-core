
from dataclasses import dataclass
import random
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple
import torch
from torch import Tensor

def compute_position_id_with_mask(attention_mask_1d: Tensor) -> Tensor:
    """DreamOn-compatible position_ids from a 1D attention mask."""
    if attention_mask_1d.dim() != 1:
        raise ValueError(
            f"attention_mask_1d must be 1D, got shape={tuple(attention_mask_1d.shape)}"
        )
    pos = torch.cumsum(attention_mask_1d.to(torch.long), dim=0) - 1
    pos = pos.clamp_min(0)
    return pos * attention_mask_1d.to(torch.long)


def masking_merge_for_response(input_tokens, tokenizer, merge_prob=0.5, merge_schedule="dynamic_inverse", use_uniform_merge_prob=0.5):
        """
        The process is:
        1. Independently mask each token with probability sampling_ratio.
            If a token is masked it is replaced by "<mask>", otherwise it remains unchanged.
        2. Scan the masked sequence for adjacent "<mask>" tokens. Whenever found, with probability merge_prob:
                - Mark the first token's label as "<expand>" (indicating the head of a merged pair).
                - Modify the attention_mask so that the second token is not attended to (i.e. set its attention_mask to 0).
            Tokens that are not part of a merge or are not masked are labeled as "<nonexpand>".
        3. Compute position_ids such that effective tokens (attention_mask==1) receive sequential indices, 
            while merged-out tokens (attention_mask==0) receive a default position of 0.
        
        Parameters:
        input_tokens (torch.Tensor): The original sequence of tokens as a tensor.
        sampling_ratio (float): The independent probability a token is replaced with "<mask>".
        merge_prob (float): The probability that a pair of adjacent "<mask>" tokens are merged.
        
        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - final_tokens: The token tensor after independent masking (tokens remain as masked or original).
            - labels: A tensor (same length as tokens) with each token labeled as 1 ("<expand>") or 0 ("<nonexpand>").
            - attention_mask: A tensor of binary values (1 for effective tokens, 0 for merged tokens).
        """
        sampling_ratio = torch.rand(1)
        # Step 1: Sampling masks
        mask = torch.rand_like(input_tokens, dtype=torch.float) < sampling_ratio
        final_tokens = input_tokens.clone()
        final_tokens[mask] = tokenizer.mask_token_id

        eos_mask = input_tokens == tokenizer.eos_token_id
        final_tokens[eos_mask] = tokenizer.mask_token_id
        
        # Initialize labels and attention_mask
        labels = input_tokens.clone()
        attention_mask = torch.ones_like(input_tokens, dtype=torch.long)

        ## Step 2: Merge
        num_masked = mask.sum().item()

        if torch.rand(1).item() < use_uniform_merge_prob:
            merge_schedule = "static"
        if merge_schedule == "dynamic_inverse":
            dynamic_merge_prob = merge_prob * (1 - (num_masked / input_tokens.size(0))) 
        elif merge_schedule == "dynamic_proportional":
            dynamic_merge_prob = merge_prob * (num_masked / input_tokens.size(0))
        elif merge_schedule == "static":
            dynamic_merge_prob = merge_prob
        elif merge_schedule == "random":
            dynamic_merge_prob = torch.rand(1).item() * merge_prob
        elif merge_schedule == "full_random":
            # So we need to vary merge_prob to [0,1] to make the model more robust
            dynamic_merge_prob = torch.rand(1).clamp(0.0, 0.95)
        else:
            raise ValueError(f"Unknown merge schedule: {merge_schedule}")

        rand_values = torch.rand(len(final_tokens)-1)
        
        for i in range(len(final_tokens)-1):
            if input_tokens[i] == tokenizer.eos_token_id:
                break
            if (final_tokens[i] == tokenizer.mask_token_id and 
                final_tokens[i+1] == tokenizer.mask_token_id and ## adjacement MASK
                rand_values[i] < dynamic_merge_prob): ## merge
                labels[i] = tokenizer.expand_token_id
                attention_mask[i+1] = 0


        return final_tokens, labels, attention_mask, sampling_ratio

def get_split_ids(example,tokenizer,middle_strategy,middle_line_num):
    if middle_strategy == 'random':
        response_ids = example['input_ids']
        # Split response into prefix, middle, suffix
        total_length = len(response_ids)
        mid_start = random.randint(1, total_length - 2)
        mid_end = random.randint(mid_start + 1, total_length - 1)
        prefix_ids = response_ids[:mid_start]
        middle_ids = response_ids[mid_start:mid_end]
        suffix_ids = response_ids[mid_end:]
    elif middle_strategy == 'line':
        # Split response into prefix, middle, suffix — now using line-based selection
        prefix_str, code_block, suffix_str = example['prefix'], example['middle'], example['suffix']
        code_lines = code_block.split('\n')
        # Choose start and end indices for middle (consecutive lines)
        max_attempts = 5
        for _ in range(max_attempts):
            # Ensure valid random indices
            if not middle_line_num:
                try:
                    middle_start = random.randint(1, len(code_lines) - 2)
                    middle_end = random.randint(middle_start + 1, len(code_lines) - 1)
                except:
                    middle_start = 1
                    middle_end = len(code_lines)
            else:
                try:
                    middle_start = random.randint(1, len(code_lines) - middle_line_num - 1)
                    middle_end = middle_start + middle_line_num
                except:
                    middle_start = 1
                    middle_end = len(code_lines)

            # Extract the slice
            middle_lines = code_lines[middle_start:middle_end]
            middle_str = "\n".join(middle_lines) + '\n'

            # Check your required conditions
            if len(middle_str.split()) >= 3 and 'def' not in middle_str and len(middle_str) > 3:
                break  # Exit loop early if valid one is found


        prefix_lines = code_lines[:middle_start]
        suffix_lines = code_lines[middle_end:]

        # Join the lines back into strings
        prefix_str = prefix_str + "\n".join(prefix_lines) + '\n'
        suffix_str = "\n".join(suffix_lines) + suffix_str
        
        prefix_ids = tokenizer.encode(prefix_str)   
        middle_ids = tokenizer.encode(middle_str)
        suffix_ids = tokenizer.encode(suffix_str)

        return torch.tensor(prefix_ids), torch.tensor(middle_ids), torch.tensor(suffix_ids)


class DreamOnInfillTrainCollator:

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 1024,
        truncation: Literal["error", "left", "right"] = "error",
        middle_strategy: Literal["line", "random"] = "line",
        middle_line_num: Optional[int] = None,
        merge_prob: float = 0.5,
        max_delete: int = 64,
        merge_schedule: str = "dynamic_inverse",
        use_uniform_merge_prob: float = 0.5,
        expand_token_id: int = 151667,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.middle_strategy = middle_strategy
        self.middle_line_num = middle_line_num
        self.merge_prob = merge_prob
        self.max_delete = max_delete
        self.merge_schedule = merge_schedule
        self.use_uniform_merge_prob = use_uniform_merge_prob
        self.expand_token_id = expand_token_id

    def __call__(
        self, examples: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        if self.truncation not in ("error", "left", "right"):
            raise ValueError(f"Invalid truncation={self.truncation}")
        
        input_ids_batch = []
        labels_batch = []
        attention_mask_batch = []
        position_ids_batch = []
        loss_mask_batch = []
        t_batch = []
        for e in examples:
            prefix_ids, middle_ids, suffix_ids = get_split_ids(e, self.tokenizer, self.middle_strategy, self.middle_line_num)
            prompt_ids = torch.tensor(e['prompt_ids'])
            prompt_length = prompt_ids.shape[0]
            response_length = prefix_ids.shape[0] + middle_ids.shape[0] + suffix_ids.shape[0]

            # EOS token handling
            if self.max_length - prompt_length - response_length > 0 and self.max_delete > 0:
                eos_count = torch.randint(
                    low=0,
                    high=min(self.max_delete, self.max_length - prompt_length - response_length),
                    size=(1,),
                ).item()
                eos_tensor = torch.tensor([self.tokenizer.eos_token_id] * eos_count, dtype=middle_ids.dtype)
            else:
                eos_count = 0
                eos_tensor = torch.tensor([], dtype=middle_ids.dtype)
            
            middle_ids = torch.cat([middle_ids, eos_tensor])
        
            masked_middle_ids, labels, middle_attention_mask, t = masking_merge_for_response(
                middle_ids, 
                self.tokenizer, 
                merge_prob=self.merge_prob, 
                merge_schedule=self.merge_schedule,
                use_uniform_merge_prob=self.use_uniform_merge_prob
            )
            # Concat all parts
            input_ids = torch.cat([
                prompt_ids,
                prefix_ids,
                masked_middle_ids,
                suffix_ids
            ], dim=-1)

            attention_mask = torch.cat([
                torch.ones_like(prompt_ids),
                torch.ones_like(prefix_ids),
                middle_attention_mask,
                torch.ones_like(suffix_ids)
            ], dim=-1)
            
            labels = torch.cat([
                prompt_ids,
                prefix_ids,
                labels,
                suffix_ids
            ], dim = -1)

            # Padding or Truncation
            sequence_length = input_ids.shape[0]
            if sequence_length < self.max_length:
                pad_len = self.max_length - sequence_length
                input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)])
                labels = torch.cat([labels, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype = labels.dtype)])
                attention_mask = torch.cat([attention_mask, torch.ones(pad_len, dtype=attention_mask.dtype)])
            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    input_ids = input_ids[-self.max_length:]
                    labels = labels[-self.max_length:]
                    attention_mask = attention_mask[-self.max_length:]
                elif self.truncation == "right":
                    input_ids = input_ids[:self.max_length]
                    labels = labels[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                else:
                    raise ValueError(f"Unknown truncation strategy: {self.truncation}")

            #position_ids = compute_position_id_with_mask(input_ids != self.tokenizer.pad_token_id)
            position_ids = compute_position_id_with_mask(attention_mask)

            # Loss mask (only for merged part)
            loss_mask = (input_ids == self.tokenizer.mask_token_id) & (attention_mask == 1)

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)
            attention_mask_batch.append(attention_mask)
            position_ids_batch.append(position_ids)
            loss_mask_batch.append(loss_mask)
            t_batch.append(t)

        return {
            "input_ids": torch.stack(input_ids_batch, dim=0),
            "labels": torch.stack(labels_batch, dim=0),
            "attention_mask": torch.stack(attention_mask_batch, dim=0),
            "position_ids": torch.stack(position_ids_batch, dim=0),
            "loss_mask": torch.stack(loss_mask_batch, dim=0),
            "t": torch.stack(t_batch, dim=0)
        }
