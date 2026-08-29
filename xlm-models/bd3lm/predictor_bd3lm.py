"""Predictor implementation for Bd3lm model.

This file implements the inference/generation logic.
"""

from typing import List, Dict, Any, Optional, Literal, Tuple
import torch
from jaxtyping import Integer, Bool
from torch import Tensor as TT
from xlm.harness import Predictor
from xlm.datamodule import Tokenizer
from xlm.noise import NoiseSchedule
from .types_bd3lm import Bd3lmBatch, Bd3lmPredictionDict, Bd3lmModel
import numpy as np
from tqdm import tqdm

class Bd3lmPredictor(Predictor[Bd3lmBatch, Bd3lmPredictionDict]):
    """Seq2seq Predictor for Bd3lm model.
    """

    def __init__(
        self,
        model: Bd3lmModel = None,
        tokenizer: Tokenizer = None,
        noise_schedule: NoiseSchedule = None,
        max_steps: int = 100,
        max_length: int = 512,
        sampling_method: str = "sample_top_p",
        p: float = 0.9,
        top_k: int = 50,
        temperature: float = 1.0,
        **kwargs
    ):
        """Initialize the predictor.

        Args:
            model: The model instance
            tokenizer: Tokenizer for text processing
            noise_schedule: Noise scheduler
            max_steps: Maximum number of generation steps
            max_length: Maximum sequence length
            sampling_method: Sampling strategy (sample_top_p, sample_top_k, greedy)
            p: Top-p value for nucleus sampling
            top_k: Top-k value for top-k sampling
            temperature: Temperature for sampling
            **kwargs: Additional generation parameters
        """
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.max_steps = max_steps
        self.max_length = max_length
        self.sampling_method = sampling_method
        self.p = p
        self.top_k = top_k
        self.temperature = temperature
        self.mask_index = self.tokenizer.mask_token_id
        self.config = self.model.config if self.model is not None else None        
        self.block_size = self.config.block_size if self.config is not None else None
        self.neg_infinity = -1000000.0
    def _compute_entropy(self, x):
        _, counts = torch.unique(x, return_counts=True, sorted=False)
        entropy = torch.special.entr(counts.float() / counts.sum()).sum()
        return entropy
    def _check_stop_conds(self, x):
        """Check if sampling should stop based on 1) eos, 2) entropy, or 3) likelihood.
        Entropy/likelihood evaluated on last 256 token-block.
    
        Args:
        x: torch.Tensor, current sample.
        Returns:
        stop: bool, whether to stop sampling.
        x: torch.Tensor, sample (potentially truncated for variable-length sampling).
        """
        stop = False # stop sampling?
        truncate_idx = None # truncate sample? (variable-length sampling only)

        # CRITERION 2: always stop sampling if entropy is low
        entropy = self._compute_entropy(x[:, -self.config.sampling.entropy_window:])
        if self.config.sampling.entropy_stop and entropy < self.config.sampling.entropy_threshold:
            stop = True


        # for variable length sampling, check if we should stop
        # sampling, and where to truncate the sample
        if self.config.sampling.var_length:
            # CRITERION 1: stop at sampled EOS token
            if len(torch.where(x == self.tokenizer.eos_token_id)[0]) > 0:
                stop = True
                eos_idx = torch.where(x == self.tokenizer.eos_token_id)
                if len(eos_idx[0]) > 0:
                    is_eos = x == self.tokenizer.eos_token_id
                    first_eos = is_eos.float().argmax(dim=1)  # first eos per row
                    has_eos = is_eos.any(dim=1)
                    truncate_idx = min(
                        int(first_eos[has_eos].max().item()) + 1, x.shape[1])
                    

            # CRITERION 2: stop if entropy/likelihood is low
            if self.config.sampling.entropy_stop and entropy < self.config.sampling.entropy_threshold:
                stop = True
                truncate_idx = x.shape[1] - self.config.sampling.entropy_window

        # truncate sample (variable-length sampling only)
        if truncate_idx is not None:
            x = x[:, :truncate_idx]
            if x.ndim == 1:
                x = x.unsqueeze(0)

        return stop, x


    def _sample_prior(self, *batch_dims,device):
        return self.mask_index * torch.ones(
            * batch_dims, dtype=torch.int64, device=device)

    def _process_sigma(self, sigma):
        # cause of overfitting for block size 1?
        self.config = self.model.config
        if self.config.algo.parameterization == 'ar':
            return None
        assert sigma.ndim == 2
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.config.algo.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma
    def _subs_parameterization(self, logits, xt):
        # log prob at the mask index = - infinity
        logits[:, :, self.mask_index] += self.neg_infinity
    
        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    
        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def bd3lm_forward(self, x, attention_mask, positions, sigma, sample_mode=False, store_kv=False):
        """Returns log score."""
        self.config = self.model.config
        sigma = self._process_sigma(sigma)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            if self.config.algo.name == 'bd3lm':
                logits = self.model(x, sigma,
                              store_kv=store_kv,attention_mask=attention_mask,positions = positions,
                              sample_mode=sample_mode)
            elif self.config.algo.name == 'ar':
                if self.config.algo.backbone == 'hf_dit':
                    logits = self.model(x, None)     
                else:
                    logits = self.model(x, sigma, sample_mode=sample_mode, store_kv=store_kv)
                logits[:, :, self.mask_index] = self.neg_infinity
                logits = logits.log_softmax(-1)
            else:
                logits = self.model(x, sigma)

        if self.config.algo.cross_attn:
            x = x[:, :self.config.model.length]
        if self.config.algo.parameterization == 'subs':
            return self._subs_parameterization(logits=logits,
                                      xt=x)
        elif self.config.algo.parameterization == 'sedd':
            return self._sedd_parameterization(logits=logits,
                                        xt=x,
                                        sigma=sigma)
        return logits
    def _nucleus_sample(self, p_x0):
        self.block_size = self.config.block_size
        self.config = self.model.config
        p = self.config.sampling.nucleus_p
        if p == 1.0:
            return p_x0
        p_x0_ = p_x0[:, -self.block_size:].clone()
        sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
        cum_probs = sorted_probs.cumsum(dim=-1)
        nucleus_mask = cum_probs <= p
        nucleus_mask[..., 0] = 1
        sorted_probs = sorted_probs * nucleus_mask
        p_x0_.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
        p_x0_ /= p_x0_.sum(-1, keepdim=True)
        p_x0[:, -self.block_size:] = p_x0_
        return p_x0
    def _sigma_from_p(self, p):
        return torch.min(- torch.log(1 - p), self.noise_schedule.noise.sigma_max)
    def _sample_categorical(self, categorical_probs):
        gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
        samples = (categorical_probs / gumbel_norm).argmax(dim=-1)
        return samples

    def _compute_confidence(
        self,
        logits: TT,
        masked: Bool[TT, " batch seq_len"],
    ) -> TT:
        """Per-position confidence score (higher = more confident). -inf at non-mask.

        Taken from xlm-core's MLM predictor:
        https://github.com/dhruvdcoder/xlm-core/blob/a763b6ea632564e85531436e6e658e6813a93a57/xlm-models/mlm/predictor_mlm.py#L130
        """
        self.config = self.model.config
        probs = logits.softmax(dim=-1)
        if self.config.sampling.confidence == "top_prob":
            score = probs.max(dim=-1)[0]
        elif self.config.sampling.confidence == "prob_diff":
            top2, _ = torch.topk(probs, k=2, dim=-1)
            score = top2[..., 0] - top2[..., 1]
        elif self.config.sampling.confidence == "entropy":
            score = torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        else:
            raise ValueError(f"Unknown confidence: {self.config.sampling.confidence}")
        score = score.clone()
        score[~masked] = float("-inf")
        return score
    @torch.no_grad()
    def _ddpm_caching_update(self, x, attention_mask_fwd, positions_fwd, t, dt, p_x0=None):
        self.config = self.model.config
        self.block_size = self.config.block_size
        _, move_chance_t = self.noise_schedule(t)
        _, move_chance_s = self.noise_schedule(t - dt)
        sigma_t = self._sigma_from_p(move_chance_t)
        move_chance_t = move_chance_t[:, None]
        move_chance_s = move_chance_s[:, None]
        mask_prob = move_chance_s / move_chance_t

        if p_x0 is None:
            if self.config.sampling.kv_cache:
                p_x0 = self.bd3lm_forward(x[:, -self.block_size:],
                        attention_mask_fwd,
                        positions_fwd,
                        sigma_t,
                        sample_mode=True).to(torch.float64)
            else:   
                p_x0 = self.bd3lm_forward(x,
                          attention_mask_fwd,
                          positions_fwd,
                          sigma_t,
                          sample_mode=True).to(torch.float64)
                p_x0 = p_x0[:, -self.block_size:]
            

        if self.config.sampling.first_hitting:
            
            if self.config.sampling.confidence_decoding:
                
                log_p_x0 = p_x0[:, -self.block_size:]
                masked = (
                    x[:, -self.block_size:] == self.mask_index
                )
                confidence_scores = self._compute_confidence(
                    log_p_x0,
                    masked,)

                ind = confidence_scores.argmax(dim=-1)
                
                p_x0 = p_x0.exp()
                p_x0 = self._nucleus_sample(p_x0)
                x_block = self._sample_categorical(p_x0)
                
                selection_mask = (
                    torch.arange(self.block_size, device=x.device)
                    == ind[:, None]).to(x_block.dtype)
                
                x_block = (x_block * selection_mask
                    + x[:, -self.block_size:] * (1 - selection_mask))
                
                
            else:
                p_x0 = p_x0.exp()
                p_x0 = self._nucleus_sample(p_x0)
                x_block = self._sample_categorical(p_x0)
                is_masked = (x[:, -self.block_size:] == self.mask_index)
                probs = is_masked.float()
                probs[probs.sum(-1) == 0] = 1.0
                ind = torch.multinomial(probs, num_samples=1).squeeze(-1)
                mask = (torch.arange(self.block_size, device=x.device) == ind[:, None]).to(x_block.dtype)
                x_block = x_block * mask + x[:, -self.block_size:] * (1 - mask)
                
        else:
            
            q_xs = p_x0 * (1 - mask_prob)
            q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
            x_block = self._sample_categorical(q_xs)
        copy_flag = (x[:, -self.block_size:] != self.mask_index).to(x.dtype)
        x_block =  copy_flag * x[:, -self.block_size:] + (1 - copy_flag) * x_block
        x_new = torch.cat((x[:, :-self.block_size], x_block), dim=-1)
        
        # compute kv cache if all tokens in a block are sampled
        if self.config.sampling.kv_cache and self.mask_index not in x_block:
            _ = self.bd3lm_forward(x_block, attention_mask_fwd, positions_fwd,
                                   sigma_t, sample_mode=True, store_kv=True)
            
        if not torch.allclose(x_new, x):
            return None, x_new
        else:
            return p_x0, x_new


    @torch.no_grad
    def _semi_ar_sampler(
        self, prompt, attention_mask,positions, n_samples, num_steps, num_strides, seqlen, context_size=1024):
        config = self.model.config
        block_size = config.block_size
        device = next(self.model.parameters()).device
        dtype = torch.float32
        
        if seqlen is None:
            seqlen = config.model.length
        sampling_steps = 0
          
        mdlm_semi_ar = config.algo.name == 'mdlm' and config.model.length > block_size
        if mdlm_semi_ar:
            # sliding window of length 512 for mdlm semi-ar decoding
            num_strides = config.model.length // 512
            num_strides -= 1

        ones = torch.ones((n_samples,1), dtype=dtype,
                      device=device)
        
        # reset kv-cache 
        if config.sampling.kv_cache:
            config.loader.eval_batch_size = n_samples
            self.model.reset_kv_cache()
        prompt_len = prompt.shape[1] if prompt is not None else 0
        if prompt is not None and attention_mask is not None:
            target_attention_mask = torch.ones(
                (n_samples, seqlen),
                dtype=attention_mask.dtype,
                device=device,
            )

            full_attention_mask = torch.cat(
                [attention_mask.to(device), target_attention_mask],
                dim=1,
            )
        else:
            full_attention_mask = None
        if positions is not None:
            positions = None
            positions = full_attention_mask.cumsum(dim=1) - 1
            positions *= full_attention_mask
        ### prompt filling for kv_cache..
        if config.sampling.kv_cache and prompt is not None:
            if prompt_len % block_size != 0:
                raise ValueError(
                    f"kv_cache needs prompt_len ({prompt_len}) to be a multiple "
                    f"of block_size ({block_size})")
            # zeros, matching what _process_sigma produces for every other call
            seed_sigma = torch.zeros((n_samples, 1), dtype=dtype, device=device)
            for seed_start in range(0, prompt_len, block_size):
                seed_end = seed_start + block_size
                _ = self.bd3lm_forward(
                    prompt[:, seed_start:seed_end].to(device),
                    full_attention_mask[:, :seed_end]
                    if full_attention_mask is not None else None,
                    positions[:, :seed_end] if positions is not None else None,
                    seed_sigma,
                    sample_mode=True,
                    store_kv=True)

        for stride_num in tqdm(range(num_strides)):
            # sample next block
            if stride_num == 0:
                if prompt is not None:
                    x_accum = prompt.to(device)
                    x = self._sample_prior(n_samples, block_size, device=device).to(device)
                    x_accum = torch.cat((x_accum, x), dim=1)
                else:
                    x_accum = self._sample_prior(n_samples, block_size, device=device).to(device)
                    x_accum[:, 0] = self.tokenizer.bos_token_id
            else:
                if mdlm_semi_ar:
                    x = self._sample_prior(n_samples, 512,device=device).to(device)
                else:
                    x = self._sample_prior(n_samples, block_size,device=device).to(device)
                x_accum = torch.cat((x_accum, x), dim=1)

            # compute logits in a sliding window (context passed to model can't exceed context_size)
            
            end_idx = prompt_len + (stride_num + 1) * block_size
            start_idx = max(end_idx - context_size, 0)
            fwd_idx = torch.arange(start_idx, end_idx, device=device)

            if mdlm_semi_ar and stride_num > 0: # MDLM
                fwd_idx = torch.arange(512*(stride_num), (512*(stride_num))+block_size)

            dt = 1 / num_steps
            p_x0_cache = None
            timesteps = torch.linspace(1, 0, num_steps, device=device)
            t = 1
            for i in range(num_steps):
                if self.mask_index not in x_accum:
                    break

                
                if config.sampling.first_hitting:
                    u = torch.rand(x_accum.shape[0], 1, device=device)
                    num_masked = (x_accum[:, fwd_idx] == self.mask_index).sum(
                        -1, keepdim=True).clamp(min=1)
                    t = t * u ** (1.0 / num_masked)
              
                elif not config.sampling.first_hitting:
                    t = timesteps[i]
                attention_mask_fwd = (full_attention_mask[:, fwd_idx] if full_attention_mask is not None else None
                    )
                positions_fwd =  (positions[:, fwd_idx] if positions is not None else None
                    )
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x=x_accum[:, fwd_idx],
                    attention_mask_fwd = attention_mask_fwd,
                    positions_fwd = positions_fwd,
                    t=t * ones,
                    dt=dt,
                    p_x0=p_x0_cache,)
                if p_x0_cache is None:
                    sampling_steps += 1
       
                x_accum[:, fwd_idx] = x_next
            
        
        # truncate the sample if it exceeds 30 tokens and check stopping conditions it will be done after generating all the block for each sequence in the batch.
        if x_accum.shape[1] > 30:
            stop, x_accum = self._check_stop_conds(x_accum)
            if (stop and not config.sampling.var_length) \
                or (stop and x.shape[-1] == 1):
                return None, None

        return x_accum, sampling_steps

    @torch.no_grad()
    def _sample(
        self, prompt=None, attention_mask = None, positions = None, seqlen=None, num_steps=None, eps=1e-5, batch_size_per_gpu=None):
        """Generate samples from the model."""
        if seqlen is None:
            seqlen = self.model.config.model.length
        if batch_size_per_gpu is None:
            batch_size_per_gpu = self.model.config.loader.eval_batch_size
        samples = []
        if self.model.config.algo.sampler == 'semi_ar':
            for _ in range(1):
                sample_i, num_tries = None, 0
                while sample_i is None:
                    num_tries += 1
                    sample_i, nfes = self._semi_ar_sampler(prompt=prompt,
                        attention_mask = attention_mask,
                        positions = positions,
                        n_samples=batch_size_per_gpu,
                        num_strides=(seqlen // self.model.config.block_size), 
                        num_steps=num_steps,
                        seqlen=seqlen)
                    if num_tries > 10:
                        raise ValueError('Sampling failed.')
                samples.append(sample_i)
        samples = torch.cat(samples, dim=0) 
        return samples
    
    @torch._dynamo.disable()
    def predict(
        self,
        batch: Dict[str, Any],  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
        max_len: int = 0,
    ) -> Bd3lmPredictionDict:
        """Generate predictions from the model.

        Args:
            batch: Input batch
            batch_idx: Batch index
            dataloader_idx: Dataloader index  
            dataloader_name: Dataloader name
            max_len: Maximum length override

        Returns:
            Dictionary containing generated text and token IDs
        """
        # Record start time
        import time
        start_time = time.time()
        
        # Get batch information
        batch_size = batch["input_ids"].size(0)
        input_length = batch["input_ids"].size(1)
        device = batch["input_ids"].device
        
        # Initialize generation state
        input_ids = batch["input_ids"].clone()
        target_ids = batch["target_ids"].clone()
        current_attention_mask = batch["attention_mask"].clone()
        
        # Track positions
        positions = current_attention_mask.cumsum(dim=1) - 1
        positions *= current_attention_mask
        
        ## BD3lm code...
        samples = self._sample(prompt = input_ids,
            attention_mask = current_attention_mask,
            positions = positions,
            seqlen=self.model.config.model.target_length,
            batch_size_per_gpu= batch_size,
            num_steps=self.model.config.algo.T,
            eps=1e-5)
        current_ids = samples
        # Decode to text
        generated_text = []
        generated_text_with_spl = []
        for i in range(batch_size):
            tokens = current_ids[i].tolist()
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            text_with_spl = self.tokenizer.decode(tokens, skip_special_tokens=False)
            generated_text.append(text)
            generated_text_with_spl.append(text_with_spl)
        
        # Record end time
        end_time = time.time()
        time_taken = [end_time - start_time] * batch_size
        
        return {
            "text": generated_text,
            "text_with_spl_tokens": generated_text_with_spl,
            "ids": current_ids,
            "attention_mask": current_attention_mask,
            "positions": positions,
            "time_taken": time_taken,
            "output_start_idx": input_length,
        }

    def _sample_top_k(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k sampling implementation."""
        return torch.argmax(logits, dim=-1, keepdim=True)

    def _sample_top_p(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Top-p (nucleus) sampling implementation."""
        return torch.argmax(logits, dim=-1, keepdim=True)

    def to_dict(self, batch: Bd3lmBatch, preds: Bd3lmPredictionDict, **kwargs) -> List[Dict[str, Any]]:
        """Convert predictions to dictionary format for logging.

        Args:
            batch: Original input batch
            preds: Model predictions
            **kwargs: Additional arguments

        Returns:
            List of dictionaries containing prediction results

        Note the generated sequence must be under "text" and "ids", the names every
        other xlm model uses. LogPredictions only keeps fields starting with one of
        text/length/perplexity/entropy/nll, and the post-hoc evaluators read
        pred["text"] directly - under any other name the rows come out empty.
        """
        results = []
        for i in range(len(preds["text"])):
            result = {
                "text": preds["text"][i],
                "ids": preds["ids"][i].tolist(),
                "text_with_spl_tokens": preds["text_with_spl_tokens"][i],
                "input_text": self.tokenizer.decode(batch["input_ids"][i], skip_special_tokens=False),
            }
            results.append(result)

        return results


class Bd3lmUnconditionalPredictor(Bd3lmPredictor):
    """Predictor for unconditional generation.

    Everything is inherited from Bd3lmPredictor except predict(), which
    calls _sample with prompt=None. 
    """

    @torch._dynamo.disable()
    def predict(
        self,
        batch: Dict[str, Any],  # type: ignore
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
        max_len: int = 0,
    ) -> Bd3lmPredictionDict:
        """Generate unconditionally.

        The batch comes from Bd3lmUnconditionalPredCollator and carries no
        prompt - it is read only for the batch size and device.
        """
        import time
        start_time = time.time()

        batch_size = batch["input_ids"].size(0)

       
        seqlen = self.model.config.model.length

        samples = self._sample(
            prompt=None,
            attention_mask=None,
            positions=None,
            seqlen=seqlen,
            batch_size_per_gpu=batch_size,
            num_steps=self.model.config.algo.T,
            eps=1e-5)
        current_ids = samples

        generated_text = []
        generated_text_with_spl = []
        for i in range(batch_size):
            tokens = current_ids[i].tolist()
            generated_text.append(
                self.tokenizer.decode(tokens, skip_special_tokens=True))
            generated_text_with_spl.append(
                self.tokenizer.decode(tokens, skip_special_tokens=False))

        end_time = time.time()
        time_taken = [end_time - start_time] * batch_size

        return {
            "text": generated_text,
            "text_with_spl_tokens": generated_text_with_spl,
            "ids": current_ids,
            "attention_mask": batch["attention_mask"],
            "positions": None,
            "time_taken": time_taken,
            
            "output_start_idx": 0,
        }
