# coding=utf-8
import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

logger = logging.get_logger(__name__)


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


@dataclass
class DreamOnModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamOnGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.alg: str = kwargs.pop("alg", 'entropy')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None) # for maskgit_plus
        self.number_transfer_tokens = kwargs.pop("number_transfer_tokens", 1)

        # Parameters that control expand and deletion process
        self.expand_budget = kwargs.pop("expand_budget", None)
        self.pad_delete_to_right = kwargs.pop("pad_delete_to_right", False)
        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.expand_token_id = kwargs.pop("expand_token_id", None)
        self.delete_token_id = kwargs.pop("eos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamOnGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        ## INFILLING:We change the condition from >= to = to enable infilling without any expansion.
        if input_ids_length > generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
        num_mask_tokens
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length - num_mask_tokens

        elif has_default_max_length:  # by default let's always generate 32 new tokens at maximum
            if generation_config.max_length == DreamOnGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamOnGenerationConfig], **kwargs: Dict
    ) -> DreamOnGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamOnGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id
                if generation_config.expand_token_id is None:
                    generation_config.expand_token_id = self.generation_config.expand_token_id
        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamOnGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)
        expand_token_tensor = _tensor_or_none(generation_config.expand_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor
        generation_config._expand_token_tensor = expand_token_tensor


    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,  # prefix, mask, suffix
        generation_config: Optional[DreamOnGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamOnModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        generation_config = self._prepare_generation_config(generation_config, **kwargs)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        assert attention_mask is None, 'We currently do not support attention_mask for DreamOn since we recompute attention mask after each denoising step.'
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        ## get number of mask tokens as start_gen_len
        mask_token_id = generation_config.mask_token_id
        mask_token_indices = torch.where(input_ids == mask_token_id)[1]
        
        if mask_token_indices.numel() == 0:
            raise ValueError("No mask tokens found in the input_ids.")
        
        num_mask_tokens = mask_token_indices.numel()
        ## get the first index of mask 
        mask_token_index = mask_token_indices[0]
        ## assign it as prefix_len
        prefix_len = mask_token_index
        start_gen_len = num_mask_tokens  # Ensure start_gen_len is defined

        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
            num_mask_tokens=num_mask_tokens,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        result = self._sample(
            input_ids,
            prefix_len,
            start_gen_len,
            generation_config=generation_config,
        )
        return result

    def _sample(
            self,
            input_ids: torch.LongTensor,
            prefix_len: int,
            start_gen_len: int,
            generation_config: DreamOnGenerationConfig,
        ) -> Union[DreamOnModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate

        pad_delete_to_right = generation_config.pad_delete_to_right
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        device = input_ids.device  

        num_generation_tokens = start_gen_len
        max_length = generation_config.max_length
        max_gen_len = max_length - prefix_len
        expand_budget = generation_config.expand_budget
        if expand_budget is None:
            expand_budget = max_gen_len * 2

        number_transfer_tokens = generation_config.number_transfer_tokens
        eos_token_id = generation_config.eos_token_id 
        delete_token_id = generation_config.delete_token_id  
        expand_token_id = generation_config.expand_token_id
        mask_token_id = generation_config.mask_token_id

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=eos_token_id)

        for i in range(2 * max_gen_len + 2 * expand_budget):
            #### 1. --- Prepare Input ---
            current_window_length = input_ids.shape[1] - start_gen_len + num_generation_tokens
            attention_mask = torch.ones([input_ids.shape[0], current_window_length], dtype=torch.int16).to(device)
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=0)

            mask_index = (x == mask_token_id) & (attention_mask == 1)
            if torch.all(~mask_index[:, :current_window_length]):
                break  # exit if all mask tokens are denoised

            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)

            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )

            output = self(x, attention_mask, tok_idx)
            logits = output.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            logits = logits[mask_index]

            # block the logit for expansion when token budget is all used
            if current_window_length == max_length or expand_budget == 0:
                logits[:, expand_token_id] -= 1e9

            ### 2. ----sample tokens
            if alg == 'maskgit_plus':
                confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
            elif alg == 'topk_margin':
                confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
            elif alg == 'entropy':
                confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
            else:
                raise RuntimeError(f"Unknown alg: {alg}")

            if alg_temp is None or alg_temp == 0:
                _, transfer_index = torch.topk(confidence, number_transfer_tokens)
            else:
                confidence = confidence / alg_temp
                confidence = F.softmax(confidence, dim=-1)
                transfer_index = torch.multinomial(confidence, num_samples=number_transfer_tokens)
            x0_ = torch.zeros_like(x0, device=device, dtype=torch.long) + mask_token_id
            x0_[transfer_index] = x0[transfer_index].clone()
            x[mask_index] = x0_

            if histories is not None:
                histories.append(x[0,:current_window_length].clone())

            ### 3. ---- delete -------
            # pad delete to right if needed
            if pad_delete_to_right:
                x_seq = x[0]  # Flatten to 1D: shape [seq_len]

                # Find indices where EOS occurs
                delete_indices = (x_seq == delete_token_id).nonzero(as_tuple=True)

                if len(delete_indices[0]) > 0:
                    # Get the first occurrence of delete
                    first_delete_idx = delete_indices[0][0].item()
                    position_mask = torch.arange(x_seq.size(0), device=device) >= first_delete_idx
                    replace_mask = position_mask & mask_index[0]
                    # Set all tokens after EOS to eos_id
                    x_seq.masked_fill_(replace_mask, delete_token_id)
                    x = x_seq.unsqueeze(0)

            # delete
            delete_indices = ((x[0] == delete_token_id) & (mask_index[0] == 1)).nonzero(as_tuple=False).squeeze(1)
            if delete_indices.numel() > 0:
                for idx in sorted(delete_indices.tolist(), reverse=True):
                    x = torch.cat((
                        x[:, :idx],
                        x[:, idx + 1:],
                        torch.tensor([[mask_token_id]], device=device)
                    ), dim=1)
                    num_generation_tokens -= 1
                if histories is not None:
                    current_window_length = input_ids.shape[1] - start_gen_len + num_generation_tokens
                    histories.append(x[0,:current_window_length].clone())
            ### 4. ---- expand --------
            expand_indices = (x[0] == expand_token_id).nonzero(as_tuple=False).squeeze(1)
            if expand_indices.numel() > 0:
                # Process from right to left to prevent shifting issues
                for idx in sorted(expand_indices.tolist(), reverse=True):
                    x = torch.cat((
                        x[:, :idx],
                        torch.tensor([[mask_token_id, mask_token_id]], device=device),
                        x[:, idx + 1:]
                    ), dim=1)
                    num_generation_tokens += 1
                    expand_budget -= 1
                    # Truncate back to max_tokens if needed
                    if x.shape[1] > max_length:
                        x = x[:, :max_length]
                
                if histories is not None:
                    current_window_length = input_ids.shape[1] - start_gen_len + num_generation_tokens
                    histories.append(x[0,:current_window_length].clone())

        if return_dict_in_generate:
            return DreamOnModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x