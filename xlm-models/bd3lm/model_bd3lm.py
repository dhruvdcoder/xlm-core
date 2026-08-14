import math
import typing

import einops
from einops import rearrange
from functools import partial
try:
  import flash_attn
  import flash_attn.layers.rotary
except:
  pass
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
  from torch.nn.attention.flex_attention import flex_attention, create_block_mask
  FLEX_ATTN_AVAILABLE = True
except:
  FLEX_ATTN_AVAILABLE = False

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
  """
  Constructs the specialized block diffusion attention mask for training
  composed of three masks:
  - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
  - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
  - **Block Causal Mask (M_BC)**: Attention to update x0

  Args:
      b, h: Batch and head indices (ignored for mask logic).
      q_idx, kv_idx: Query and Key indices.
      seq_len: Total sequence length.
      block_size: Defines the block structure.

  Returns:
      A boolean attention mask.
  """

  # Indicate whether token belongs to xt or x0
  x0_flag_q = (q_idx >= n)
  x0_flag_kv = (kv_idx >= n)

  # Compute block indices
  block_q = torch.where(x0_flag_q == 1,
                        (q_idx - n) // block_size,
                        q_idx // block_size)
  block_kv = torch.where(x0_flag_kv == 1,
                        (kv_idx - n) // block_size,
                        kv_idx // block_size)

  # **1. Block Diagonal Mask (M_BD) **
  block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

  # **2. Offset Block-Causal Mask (M_OBC) **
  offset_block_causal = (
    (block_q > block_kv)
    & (x0_flag_kv == 1)
    & (x0_flag_q == 0)
  )

  # **3. Block-Causal Mask (M_BC) **
  block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

  # **4. Combine Masks **
  return block_diagonal | offset_block_causal | block_causal

@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def fused_flex_attention(q, k, v, mask=None):
    return flex_attention(q, k, v, block_mask=mask)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift

@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)

@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)

@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x,positions=None, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if positions is not None:
      # Building the position embedding for the batch having different sequence lengths.
      t = positions.to(device=x.device, dtype=self.inv_freq.dtype)  # [B, S]
      freqs = torch.einsum("bi,j->bij", t, self.inv_freq)
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      cos = emb.cos()[:, :, None, None, :].repeat(1, 1, 3, 1, 1)
      sin = emb.sin()[:, :, None, None, :].repeat(1, 1, 3, 1, 1)
      cos[:, :, 2, :, :].fill_(1.)
      sin[:, :, 2, :, :].fill_(0.)

      return cos, sin


    else:
      if (self.seq_len_cached != seq_len
          or self.cos_cached is None
          or self.cos_cached.device != x.device
          or self.cos_cached.dtype != x.dtype):
        self.seq_len_cached = seq_len

        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

        # dims are: batch, seq_len, qkv, head, dim
        cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
        sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
        cos_cached[:,:,2,:,:].fill_(1.)
        sin_cached[:,:,2,:,:].fill_(0.)
        self.cos_cached = cos_cached.to(x.dtype)
        self.sin_cached = sin_cached.to(x.dtype)

      return self.cos_cached, self.sin_cached



def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    cos = cos[0,:,0,0,:cos.shape[-1]//2]
    sin = sin[0,:,0,0,:sin.shape[-1]//2]
    q, k, v = qkv.chunk(3, dim=2)
    q = flash_attn.layers.rotary.apply_rotary_emb_torch(
      q.squeeze(dim=2), cos, sin)
    k = flash_attn.layers.rotary.apply_rotary_emb_torch(
      k.squeeze(dim=2), cos, sin)
    v = v.squeeze(dim=2)
  return q, k, v

def apply_rotary_pos_emb_torchscript(qkv, cos, sin):
    return (qkv * cos) + (rotate_half(qkv) * sin)

def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


def regular_attention_multi_headed(q, k, v):
  # Assuming qkv is a tensor with shape [batch, seq_len, 3, num_heads, head_dim]
  # where the 3 represents Q, K, V packed in that order
  attention_output = F.scaled_dot_product_attention(
    query=q.transpose(1, 2),
    key=k.transpose(1, 2),
    value=v.transpose(1, 2),
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False)
  # [batch_size, seq_len, num_heads, head_dim]
  attention_output = attention_output.transpose(1, 2)
  return einops.rearrange(attention_output, 'b s h d -> b s (h d)')


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.amp.autocast('cuda', enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half).to(t.dtype).to(t.device)
      / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.
  
  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes
  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################

class DDiTBlockCausal(nn.Module):
  def __init__(self, n, dim, n_heads, mlp_ratio=4, dropout=0.1, max_batch_size=64, max_seqlen=1024, adaLN=False, cond_dim=None, attn_backend='flash_attn'):
    super().__init__()
    self.n_heads = n_heads
    self.max_seqlen = max_seqlen
    self.n = n

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout
    self.adaLN = adaLN
    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    self.attn_backend = attn_backend
    self.kv_cache = None

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def get_qkv(self, x, rotary_cos_sin, store_kv=False):
    # compute qkv (potentially use cache)
    if self.kv_cache is not None:
      new_qkv = self.attn_qkv(x[:, -1:])
      qkv = torch.cat((self.kv_cache, new_qkv), dim=1)
    else:
      qkv = self.attn_qkv(x)
    # store kv cache in a sliding window (can't exceed context len)
    if store_kv:
      self.kv_cache = qkv[:, -(self.max_seqlen-1):].clone()
      
    qkv = einops.rearrange(
      qkv,
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      if self.attn_backend == 'flash_attn':
        qkv = apply_rotary_pos_emb(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
      else:
        qkv = apply_rotary_pos_emb_torchscript(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
          
    return qkv

  def cross_attn(self, qkv, mask=None):
    scale = qkv.shape[-1]
    qkv = qkv.transpose(1, 3)
    mask = mask.bool() if mask is not None else None
    x = F.scaled_dot_product_attention(
      query=qkv[:, :, 0],
      key=qkv[:, :, 1],
      value=qkv[:, :, 2],
      attn_mask=mask,
      is_causal=True,
      scale=1 / math.sqrt(scale))
    x = x.transpose(1, 2)
    x = rearrange(x, 'b s h d -> b s (h d)')
    return x

  def forward(self,
              x,
              rotary_cos_sin,
              c=None,
              causal=True,
              mask=None,
              store_kv=False,
              **kwargs):
    del kwargs
    batch_size, seq_len = x.shape[0], x.shape[1]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = None, None, None, None, None, None
    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    if c is not None and c.shape[0] == batch_size:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
      scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
    elif c is not None:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
      scale_mlp, gate_mlp) = rearrange(
        self.adaLN_modulation(c), '(b h) d -> b h d', b=batch_size
        ).chunk(6, dim=-1)

    # attention operation
    x_skip = x
    if c is not None:
      x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
    else:
      x = self.norm1(x)
    
    qkv = self.get_qkv(x, rotary_cos_sin, store_kv=store_kv)
    if self.attn_backend == 'flash_attn':
      qkv = einops.rearrange(qkv, 'b s ... -> (b s) ...')
      cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len,
        step=seq_len, dtype=torch.int32, device=qkv.device)
      x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, seq_len, 0.0, causal=True)
      x = einops.rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
    else:
      x = self.cross_attn(qkv, c)
      
    if c is not None:
      x = bias_dropout_scale_fn(self.attn_out(x),
        None,
        gate_msa,
        x_skip,
        self.dropout)
      # mlp operation
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(
          self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x


class DDiTBlock(nn.Module):
  def __init__(self, n, dim, n_heads, adaLN,
               latent_dim=None, cond_dim=None,
               latent_conditioning=-1, mlp_ratio=4,
               dropout=0.1, block_size=1,
               max_batch_size=64, max_seqlen=1024, attn_backend='flash_attn'):
    super().__init__()
    self.max_seqlen = max_seqlen
    self.n = n
    self.n_heads = n_heads
    self.adaLN = adaLN
    self.latent_conditioning = latent_conditioning
    self.block_size = block_size

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout
    self.kv_cache = None
    self.cache_idx = 0

    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    self.attn_backend = attn_backend

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def get_qkv(self, x, rotary_cos_sin, store_kv=False):
    # compute qkv (potentially use cache)
    if self.kv_cache is not None:
      new_qkv = self.attn_qkv(x)
      self.kv_cache[:, self.cache_idx:self.cache_idx+self.block_size] = new_qkv
      qkv = self.kv_cache[:, :self.cache_idx+self.block_size].clone()
    else:
      qkv = self.attn_qkv(x)
    # store kv cache in a sliding window (can't exceed context len)
    if store_kv:
      self.cache_idx += self.block_size
      if self.cache_idx >= self.max_seqlen:
        # left-shift the cache
        self.cache_idx = self.max_seqlen - self.block_size
        self.kv_cache[:, :-self.block_size] = self.kv_cache[:, self.block_size:].clone()

    qkv = einops.rearrange(
      qkv,
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      if self.attn_backend == 'flash_attn':
        qkv = apply_rotary_pos_emb(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
      else:
        qkv = apply_rotary_pos_emb_torchscript(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    return qkv
  
  def attn_mlp(self, x, c, gate_msa, gate_mlp, shift_mlp, scale_mlp, x_skip):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    if c is not None:
      x = bias_dropout_scale_fn(self.attn_out(x),
        None,
        gate_msa,
        x_skip,
        self.dropout)
      # mlp operation
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(
          self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x

  def cross_attn(self, qkv, mask=None):
    scale = qkv.shape[-1]
    qkv = qkv.transpose(1, 3)
    mask = mask.bool() if mask is not None else None
    x = F.scaled_dot_product_attention(
      query=qkv[:, :, 0],
      key=qkv[:, :, 1],
      value=qkv[:, :, 2],
      attn_mask=mask,
      is_causal=False,
      scale=1 / math.sqrt(scale))
    x = x.transpose(1, 2)
    x = rearrange(x, 'b s h d -> b s (h d)')
    return x

  def cross_attn_flex(self, qkv, mask=None):
    qkv = rearrange(qkv, 'b s three h d -> b h three s d', h=self.n_heads)
    x = fused_flex_attention(
      qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], mask=mask)
    x = rearrange(x, 'b h s d -> b s (h d)')
    return x

  def forward(self,
              x,
              rotary_cos_sin,
              c,
              causal=False,
              mask=None,
              sample_mode=False,
              store_kv=False):
    batch_size, seq_len = x.shape[0], x.shape[1]

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = None, None, None, None, None, None
    if c is not None and c.shape[0] == batch_size:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
      scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
    elif c is not None:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
      scale_mlp, gate_mlp) = rearrange(
        self.adaLN_modulation(c), '(b h) d -> b h d', b=batch_size
        ).chunk(6, dim=-1)

    x_skip = x
    if c is not None:
      x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
    else:
      x = self.norm1(x)

    # get qkvs
    if mask is not None and not sample_mode:
      qkv_x = self.get_qkv(x[:,:self.n], rotary_cos_sin)
      qkv_x0 = self.get_qkv(x[:,self.n:], rotary_cos_sin)
      qkv = torch.cat((qkv_x, qkv_x0), dim=1)
    else:
      qkv = self.get_qkv(x, rotary_cos_sin, store_kv=store_kv)
      
    # attention
    if self.attn_backend == 'flash_attn' and mask is None:
      qkv = einops.rearrange(qkv, 'b s ... -> (b s) ...')
      cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, step=seq_len,
        dtype=torch.int32, device=qkv.device)
      x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, seq_len, 0., causal=causal)
      x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)     
    elif self.attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
      x = self.cross_attn_flex(qkv, mask=mask)
    elif self.attn_backend == 'sdpa':
      x = self.cross_attn(qkv, mask=mask)
    else:
      raise ValueError('Unknown attention backend')
    if self.kv_cache is not None:
      x = x[:, -self.block_size:]
    x = self.attn_mlp(x, c, gate_msa, gate_mlp, shift_mlp, scale_mlp, x_skip)
    return x
   
class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    return self.embedding[x]


class DDiTFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim, 
               adaLN, tie_word_embeddings=False):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()
    self.adaLN = adaLN
    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim,
                                        2 * hidden_size,
                                        bias=True)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    self.tie_word_embeddings = tie_word_embeddings

  def forward(self, x, c):
    x = self.norm_final(x)
    if c is not None:
      if c.shape[0] == x.shape[0]:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
      else:
        shift, scale = rearrange(
          self.adaLN_modulation(c), '(b h) d -> b h d', b=x.shape[0]).chunk(2, dim=-1)
      x = modulate_fused(x, shift, scale)
    x = self.linear(x)
    return x



class DDiT(nn.Module):
  """The transformer. Held by Bd3lmModel as `self.backbone` so that the state dict
  matches the released checkpoints, whose tensors are named backbone.blocks.0...
  """

  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)
    self.causal = getattr(config.model, 'causal_attention', config.algo.parameterization == 'ar')
    self.n = config.model.length
    self.adaLN = not self.causal or getattr(config.model, 'adaln', False)
    self.config = config
    self.vocab_size = vocab_size
    self.block_size = config.block_size
    dim = config.model.hidden_size
    cond_dim = config.model.cond_dim
    self.n_heads = config.model.n_heads
    self.vocab_embed = EmbeddingLayer(dim, vocab_size)
    if self.adaLN == True:
      self.sigma_map = TimestepEmbedder(cond_dim)
    if not self.causal:
      self.sigma_map = TimestepEmbedder(cond_dim)
    self.rotary_emb = Rotary(dim // config.model.n_heads)
    self.attn_backend = getattr(config.model, 'attn_backend', 'flash_attn')
    self.max_seqlen = config.model.length

    blocks = []
    for _ in range(config.model.n_blocks):
      if self.causal:
        
        block = DDiTBlockCausal(
          n=config.model.length,
          dim=dim,
          n_heads=config.model.n_heads,
          dropout=config.model.dropout,
          max_batch_size=config.loader.eval_batch_size,
          adaLN=self.adaLN,
          cond_dim=cond_dim,
          attn_backend=self.attn_backend)
      else:
        
        block = DDiTBlock(
          n=config.model.length,
          dim=dim,
          n_heads=config.model.n_heads,
          cond_dim=cond_dim,
          adaLN=self.adaLN,
          dropout=config.model.dropout,
          block_size=self.block_size,
          attn_backend=self.attn_backend,
          max_seqlen=self.max_seqlen)
      blocks.append(block)
    
    self.blocks = nn.ModuleList(blocks)
    self.output_layer = DDiTFinalLayer(
      hidden_size=dim,
      out_channels=vocab_size,
      cond_dim=cond_dim,
      adaLN=self.adaLN,
      tie_word_embeddings=config.model.tie_word_embeddings)
    if config.algo.cross_attn:
      self.gen_mask(config.model.length, self.block_size, self.attn_backend)

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference
    
  def gen_mask(self, seqlen, block_size, attn_backend='sdpa'):
    """Genererates attention mask"""
    if attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
      self.block_diff_mask = create_block_mask(
        partial(block_diff_mask, block_size=block_size, n=seqlen),
        B=None, H=None, Q_LEN=seqlen*2, KV_LEN=seqlen*2)
    elif attn_backend == 'sdpa':
      self.block_diff_mask = block_diff_mask(
        b=None, h=None, q_idx=torch.arange(seqlen*2)[:, None], 
        kv_idx=torch.arange(seqlen*2)[None, :], block_size=block_size, n=seqlen)
    else:
      raise ValueError('Unknown attention backend')
    
  def reset_kv_cache(self):
    for block in self.blocks:
      block.kv_cache = torch.zeros(
        self.config.loader.eval_batch_size,
        self.max_seqlen,
        self.config.model.hidden_size * 3,
        device=next(self.parameters()).device,
        dtype=next(self.parameters()).dtype)
      block.cache_idx = 0

  def forward(self, indices, sigma, attention_mask = None,positions=None, sample_mode=False, store_kv=False):
    x = self.vocab_embed(indices)
    if sigma is None:
      t_cond = None
    else:
      t_cond = F.silu(self.sigma_map(sigma))

    cross_attn = hasattr(self, 'block_diff_mask')
    if cross_attn:
      mask = self.block_diff_mask
      # special cases for sampling
      if sample_mode:
        if self.config.sampling.kv_cache:
          # full cross-attention to kv cache
          mask = None
          accum_length = self.blocks[0].cache_idx + self.block_size
          # positional encodings for cache
          x_full = torch.zeros((
            x.shape[0], accum_length, x.shape[2]), device=x.device)
          rotary_cos_sin = self.rotary_emb(x_full,positions)
        else:
          # index block-causal mask only during sampling
          mask = mask[
            self.n:self.n+x.shape[1], self.n:self.n+x.shape[1]]
          rotary_cos_sin = self.rotary_emb(x,positions)

      else:
        rotary_cos_sin = self.rotary_emb(x[:, :self.n],positions)

    else:
      rotary_cos_sin = self.rotary_emb(x,positions)
      mask = None
    ## mask : block-causal mask for cross-attention
    ## attention_mask : padding mask for real and pad tokens
    if attention_mask is not None:
        if not sample_mode:
            attention_mask = torch.cat([attention_mask, attention_mask], dim=-1)
        pad_key_mask = attention_mask[:, None, None, :].bool()
        if mask is None:
            mask = pad_key_mask
        else:
            mask = mask.to(x.device)
            mask = mask[None, None, :, :] & pad_key_mask
    if not torch.isfinite(x).all():
      print("NaN before blocks")
      print(x)
      raise RuntimeError("NaN before blocks")

    for i in range(len(self.blocks)):
      x = self.blocks[i](
        x,
        rotary_cos_sin,
        c=t_cond,
        causal=self.causal,
        sample_mode=sample_mode,
        mask=mask,
        store_kv=store_kv)
      if not torch.isfinite(x).all():
        print("NaN after block", i)
        print("x min/max:", torch.nan_to_num(x).min().item(), torch.nan_to_num(x).max().item())
        raise RuntimeError("NaN inside transformer block")
    x = self.output_layer(x, t_cond)
    if cross_attn and not sample_mode:
      x = x[:, :self.n]
    return x


class Bd3lmModel(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  """Holds the transformer as `self.backbone`.

  The released BD3-LM checkpoints are a module of the same shape, so their tensors are
  already named backbone.blocks.0... and xlm's shared loader reads them directly:

      xlm job_type=eval ... hub.repo_id=kuleshov-group/bd3lm-owt-block_size4
  """

  # The released checkpoints carry these two, we do not. They are the
  # clipped-noise-schedule range the paper searched per block size (4: 0.45-0.95,
  # 8: 0.35-0.85, 16: 0.30-0.80); here the noise schedule config owns that range.
  UNUSED_BUFFERS = ('sampling_eps_min', 'sampling_eps_max')

  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)
    self.config = config
    self.vocab_size = vocab_size
    self.backbone = DDiT(config, vocab_size)
    self._register_load_state_dict_pre_hook(self._adapt_state_dict)

  def _adapt_state_dict(self, state_dict, prefix, *args):
    """Drop what a released checkpoint carries that this model cannot take.

    The eval path loads with strict=True, so both of these have to be handled here
    rather than by strict_model_only_load.
    """
    import logging

    dropped = {name: state_dict.pop(prefix + name).tolist()
               for name in self.UNUSED_BUFFERS if prefix + name in state_dict}
    if dropped:
      logging.getLogger(__name__).info(
        "[bd3lm] checkpoint was trained with %s; this run uses the range in the "
        "noise_schedule config", dropped)

    
    own = self.state_dict()
    mismatched = []
    for key in [k for k in state_dict if k.startswith(prefix)]:
      tail = key[len(prefix):]
      if tail in own and own[tail].shape != state_dict[key].shape:
        mismatched.append((tail, tuple(own[tail].shape), tuple(state_dict.pop(key).shape)))
    if mismatched:
      log = logging.getLogger(__name__).warning
      log("[bd3lm] %d tensor(s) skipped on shape and will train from scratch - "
          "usually a vocabulary difference:", len(mismatched))
      for name, ours, theirs in mismatched:
        log("[bd3lm]     %s: model %s vs checkpoint %s", name, ours, theirs)

  def reset_kv_cache(self):
    self.backbone.reset_kv_cache()

  def forward(self, *args, **kwargs):
    return self.backbone(*args, **kwargs)