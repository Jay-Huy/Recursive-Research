import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for RoPE (Rotary Position Embedding).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding to queries and keys.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SiLU_MLP(nn.Module):
    """
    Standard Gated SiLU MLP as used in modern Transformers (Llama, Huginn, vv.)
    """
    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention with RoPE.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Transpose for attention computation -> (bs, n_heads, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Causal mask (upper triangular)
        mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv) 

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class TransformerPreNormBlock(nn.Module):
    """
    Standard Base Transformer Block using Pre-LayerNorm.
    x = x + Attn(Norm(x))
    x = x + MLP(Norm(x))
    """
    def __init__(self, d_model: int, n_heads: int, intermediate_size: int, norm_eps: float = 1e-6):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        self.mlp = SiLU_MLP(d_model, intermediate_size)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cis)
        x = x + self.mlp(self.norm2(x))
        return x

class SandwichBlock(nn.Module):
    """
    [Huginn-0125 Enforced Condition]
    Transformer Block using Sandwich Norm.
    x = n2(x + Attn(n1(x)))
    x = n4(x + MLP(n3(x)))
    """
    def __init__(self, d_model: int, n_heads: int, intermediate_size: int, norm_eps: float = 1e-6):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        
        self.norm3 = RMSNorm(d_model, eps=norm_eps)
        self.mlp = SiLU_MLP(d_model, intermediate_size)
        self.norm4 = RMSNorm(d_model, eps=norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Sandwich format: x = n2(x + Attn(n1(x)))
        x = self.norm2(x + self.attn(self.norm1(x), freqs_cis))
        # Sandwich format: x = n4(x + MLP(n3(x)))
        x = self.norm4(x + self.mlp(self.norm3(x)))
        return x
