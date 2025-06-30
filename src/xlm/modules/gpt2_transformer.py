"""
Adapted from MINGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(
            input, self.weight.shape, self.weight, self.bias, 1e-5
        )


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias
        )
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )
        if not self.flash:
            raise ValueError(
                "Flash Attention not available. Please install PyTorch >= 2.0"
            )

    def forward(self, inp, attention_mask):
        x = inp
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # Make the attention mask broadcastable to (bsz, query_seq_len(1), key_seq_len(seq_len))
        # Note we want to broadcast (copy) along the query_seq_len dimension
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:  # (bsz, seq_len)
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(
                    1
                )  # shape (bsz, 1, 1, seq_len)
            elif (
                attention_mask.ndim == 3
            ):  # (bsz, seq_len (query), seq_len (key-value))
                attn_mask = attention_mask.unsqueeze(
                    1
                )  # shape (bsz, 1, seq_len (query), seq_len (key-value))
            else:
                raise ValueError(
                    f"Attention mask must be of shape (bsz, seq_len) or (bsz, seq_len (query), seq_len (key-value)). Got {attention_mask.shape}"
                )
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        dim_ff = config.dim_feedforward
        self.c_fc = nn.Linear(config.n_embd, dim_ff, bias=config.bias)
        if config.activation == "gelu":
            self.gelu = nn.GELU()
        elif config.activation == "relu":
            self.gelu = nn.ReLU()
        else:
            raise ValueError(f"Activation {config.activation} not supported")
        self.c_proj = nn.Linear(dim_ff, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # max_seq_len
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency  # num_embeddings
    )
    n_layer: int = 12  # num_layers
    n_head: int = 12  # nhead
    n_embd: int = 768  # d_model
    dim_feedforward: int = 3072  # 768*4
    activation: str = "gelu"
    dropout: float = 0.0  # dropout
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster  # we don't control this.
    )
    padding_idx: Optional[int] = None
    final_layer_without_normalization: bool = False


class GPT(nn.Module):
    config: GPTConfig
    transformer: nn.ModuleDict

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                **dict(
                    wte=nn.Embedding(
                        config.vocab_size,
                        config.n_embd,
                        padding_idx=config.padding_idx,
                    ),
                    wpe=nn.Embedding(config.block_size, config.n_embd),
                    drop=nn.Dropout(config.dropout),
                    h=nn.ModuleList(
                        [Block(config) for _ in range(config.n_layer)]
                    ),
                ),
                **(
                    dict(
                        ln_f=LayerNorm(config.n_embd, bias=config.bias),
                    )
                    if not config.final_layer_without_normalization
                    else dict()
                ),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = (
        #    self.lm_head.weight
        # )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, pos=None):
        device = idx.device
        b, t = idx.size()  # (batch_size, seq_len)
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        if pos is None:
            pos = torch.arange(
                0, t, dtype=torch.long, device=device
            )  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(
            idx
        )  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        if self.config.final_layer_without_normalization:
            x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return x, logits

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[
                    :, :, :block_size, :block_size
                ]

    def configure_optimizers(
        self, weight_decay, learning_rate, betas, device_type
    ):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = (
            cfg.n_layer,
            cfg.n_head,
            cfg.n_embd // cfg.n_head,
            cfg.block_size,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
