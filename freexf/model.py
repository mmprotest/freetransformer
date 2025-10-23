"""Core model components for the Free Transformer.

This module exposes three building blocks used throughout the project:

``BinaryMapper``
    Implements the stochastic binary-to-categorical mapper described in the
    paper.  The mapper samples a categorical latent variable whose categories
    are defined by the binary code book of length ``H``.  It returns a
    straight-through one-hot tensor that allows gradients to propagate through
    the Bernoulli probabilities as specified in the paper.

``EncoderBlock``
    A lightweight non-causal Transformer encoder block that consumes the
    mid-layer representation of the decoder and produces logits over the
    ``H`` latent bits for every position.

``FreeTransformer``
    A decoder-only Transformer that shares its first half with the encoder
    pathway, samples the latent variable using ``BinaryMapper`` and injects the
    latent representation into the second half of the decoder.

The implementation deliberately mirrors the interface of HuggingFace causal
language models in order to make the module reusable with existing tooling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ModeType = Literal["train", "prefill", "generate"]


def _build_codebook(num_bits: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a ``(2**num_bits, num_bits)`` tensor with binary codes.

    The result is cached on the BinaryMapper instance to avoid recomputing for
    repeated calls.  Values are ``0`` or ``1`` stored as floating point numbers
    for ease of use in log-probability computations.
    """

    num_categories = 1 << num_bits
    codes = torch.arange(num_categories, device=device, dtype=torch.long)
    bits = ((codes.unsqueeze(-1) >> torch.arange(num_bits, device=device)) & 1).to(dtype)
    return bits.flip(-1)


class BinaryMapper(nn.Module):
    """Map logits over bits to a one-hot categorical sample with pass-through gradients.

    Parameters
    ----------
    num_bits:
        Number of independent Bernoulli variables.  The categorical variable
        has ``2 ** num_bits`` categories.
    deterministic:
        If set, ``forward`` will return the most likely category instead of a
        sample.  This is primarily used during evaluation.

    Notes
    -----
    The implementation keeps the categorical probabilities in log-space for
    numerical stability.  Although enumerating all categories has a cost of
    ``O(2**H)``, the default ``H=16`` keeps the tensors at a manageable size
    for the small and medium-sized models we target.
    """

    def __init__(self, num_bits: int, deterministic: bool = False) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.num_categories = 1 << num_bits
        self.deterministic = deterministic
        self.register_buffer("_codebook", torch.empty(0), persistent=False)

    def _get_codebook(self, reference: torch.Tensor) -> torch.Tensor:
        if self._codebook.numel() == 0 or self._codebook.shape[0] != self.num_categories:
            codebook = _build_codebook(self.num_bits, reference.device, reference.dtype)
            self.register_buffer("_codebook", codebook, persistent=False)
        return self._codebook

    def forward(
        self,
        logits_bits: torch.Tensor,
        *,
        sample: bool = True,
        forced_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample binary codes and return straight-through one-hot tensors.

        Parameters
        ----------
        logits_bits:
            Tensor of shape ``(batch, seq, H)`` containing the Bernoulli logits.
        sample:
            Whether to draw a sample (True) or to use the categorical mode.  The
            latter is primarily used for evaluation.
        forced_index:
            Optional tensor of integers ``(batch, seq)`` that overrides the
            sampled category.  Useful when external code supplies the latent
            indices (e.g. for teacher forcing during unit tests).

        Returns
        -------
        indices:
            Tensor with the sampled integer code per token.
        probs:
            Tensor ``(batch, seq, C)`` with the categorical probabilities.  The
            probabilities are differentiable with respect to the input logits.
        one_hot_st:
            Straight-through one-hot representation ``Y + G - detach(G)`` where
            ``G`` are the probabilities.
        """

        if logits_bits.dim() != 3:
            raise ValueError("BinaryMapper expects logits of shape (batch, seq, H)")

        codebook = self._get_codebook(logits_bits)

        logits_bits_fp32 = logits_bits.to(dtype=torch.float32)
        # Compute log-probabilities of every category.
        logits = logits_bits_fp32.unsqueeze(-2)  # (B, S, 1, H)
        codebook = codebook.to(logits.device, logits.dtype)
        # log p(bit = 1) and log p(bit = 0)
        log_prob_one = F.logsigmoid(logits)
        log_prob_zero = F.logsigmoid(-logits)
        log_probs = (
            codebook.view(1, 1, self.num_categories, self.num_bits) * log_prob_one
            + (1.0 - codebook.view(1, 1, self.num_categories, self.num_bits)) * log_prob_zero
        ).sum(-1)

        log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
        probs = torch.exp(log_probs).to(dtype=logits_bits.dtype)
        log_probs = log_probs.to(dtype=logits_bits.dtype)

        if forced_index is not None:
            indices = forced_index.to(dtype=torch.long)
        else:
            if self.deterministic or not sample:
                indices = probs.argmax(-1)
            else:
                flat_probs = probs.view(-1, self.num_categories)
                indices = torch.multinomial(flat_probs, num_samples=1).view(probs.shape[:-1])

        one_hot = F.one_hot(indices, num_classes=self.num_categories).to(probs.dtype)
        st = one_hot + probs - probs.detach()
        return indices, probs, st


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return (self.scale * x).to(dtype=x.dtype)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bsz, num_heads, seq_len, head_dim = x.shape
    x = x.unsqueeze(2).expand(bsz, num_heads, n_rep, seq_len, head_dim)
    return x.reshape(bsz, num_heads * n_rep, seq_len, head_dim)


class CausalSelfAttention(nn.Module):
    """Standard multi-head attention with causal masking and KV caching."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        rope: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope = rope

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.rope:
            return q, k
        device = q.device
        dtype = q.dtype
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        freq_seq = torch.arange(0, self.head_dim, 2, device=device, dtype=dtype)
        inv_freq = 1.0 / (10000 ** (freq_seq / self.head_dim))
        sinusoid = torch.einsum("s,d->sd", positions, inv_freq)
        sin = sinusoid.sin()[None, None, :, :]
        cos = sinusoid.cos()[None, None, :, :]

        def rope_apply(x: torch.Tensor) -> torch.Tensor:
            x = x.view(*x.shape[:-1], self.head_dim // 2, 2)
            x1, x2 = x.unbind(-1)
            rot_x1 = x1 * cos - x2 * sin
            rot_x2 = x1 * sin + x2 * cos
            return torch.stack((rot_x1, rot_x2), dim=-1).flatten(-2)

        return rope_apply(q), rope_apply(k)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        key_value_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = x.size()
        q = self.q_proj(x)
        if key_value_override is None:
            kv_input = x
        else:
            kv_input = key_value_override
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            pk, pv = past_key_value
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        q, k = self._apply_rope(q, k, k.size(2))

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        attn_output = self.o_proj(attn_output)

        present = (k, v) if use_cache else None
        return attn_output, present


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, activation: str = "gelu") -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        *,
        rope: bool = False,
    ) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, rope=rope)
        self.attn_norm = RMSNorm(dim)
        self.mlp = FeedForward(dim, mlp_dim)
        self.mlp_norm = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        key_value_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        x = self.attn_norm(x)
        attn_out, present = self.attn(
            x,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            key_value_override=key_value_override,
        )
        x = residual + attn_out

        residual = x
        x = self.mlp_norm(x)
        x = residual + self.mlp(x)
        return x, present


class EncoderBlock(nn.Module):
    """Single non-causal Transformer block for the latent encoder."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, num_bits: int) -> None:
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(dim))
        self.attn = CausalSelfAttention(dim, num_heads, rope=False)
        self.attn_norm = RMSNorm(dim)
        self.mlp = FeedForward(dim, mlp_dim)
        self.mlp_norm = RMSNorm(dim)
        self.readout = nn.Linear(dim, num_bits)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = hidden_states.shape
        queries = self.query_token.expand(bsz, seq_len, dim)
        x = self.attn_norm(queries)
        # Non-causal attention: override mask with zeros and use the decoder
        # hidden states as keys/values.
        attn_out, _ = self.attn(x, attention_mask=None, key_value_override=hidden_states)
        x = queries + attn_out
        residual = x
        x = self.mlp_norm(x)
        x = residual + self.mlp(x)
        logits = self.readout(x)
        return logits


@dataclass
class FreeTransformerOutput:
    logits: torch.Tensor
    loss_ce: Optional[torch.Tensor]
    loss_kl: Optional[torch.Tensor]
    loss_total: Optional[torch.Tensor]
    z_indices: Optional[torch.Tensor] = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None


class FreeTransformer(nn.Module):
    """Implementation of the Free Transformer."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_mlp: int,
        num_bits: int = 16,
        tokenizer: Optional[Any] = None,
        rope: bool = False,
        tie_weights: bool = True,
        beta_kl: float = 1.0,
    ) -> None:
        super().__init__()
        if n_layers % 2 != 0:
            raise ValueError("Number of layers must be even for mid-layer injection")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.num_bits = num_bits
        self.tokenizer = tokenizer
        self.beta_kl = beta_kl

        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(4096, d_model)
        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_mlp, rope=rope) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

        self.encoder_block = EncoderBlock(d_model, n_heads, d_mlp, num_bits)
        self.binary_mapper = BinaryMapper(num_bits)
        self.post_sampler = nn.Linear(1 << num_bits, d_model, bias=False)

    def _get_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            causal_mask = mask
            expanded = attention_mask[:, None, None, :]
            causal_mask = causal_mask + (1.0 - expanded) * -1e4
            return causal_mask
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mode: ModeType = "train",
        kappa: float = 0.0,
        return_loss: bool = True,
        z_force: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> FreeTransformerOutput:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        pos = torch.arange(seq_len, device=device)
        hidden = self.embed_tokens(input_ids) + self.pos_embed(pos)[None, :, :]
        hidden = self.dropout(hidden)

        attention_mask_full = self._get_attention_mask(attention_mask, seq_len, device, hidden.dtype)

        mid = self.n_layers // 2
        new_past: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for idx, block in enumerate(self.blocks):
            block_attention_mask = attention_mask_full
            past = None if past_key_values is None else past_key_values[idx]
            key_value_override = None
            if idx == mid and mode in {"train", "prefill"}:
                # We will run encoder after processing first half, so break loop.
                break
            hidden, present = block(
                hidden,
                attention_mask=block_attention_mask,
                past_key_value=past,
                use_cache=use_cache,
                key_value_override=key_value_override,
            )
            if use_cache:
                new_past.append(present)

        # Encoder path (only during training/prefill).
        z_indices = None
        kl_loss = None
        mapper_probs = None
        if mode in {"train", "prefill"}:
            encoder_logits = self.encoder_block(hidden)
            if z_force is not None:
                forced = z_force
            else:
                forced = None
            z_indices, mapper_probs, z_onehot = self.binary_mapper(
                encoder_logits, sample=(mode == "train"), forced_index=forced
            )
            post_injection = F.linear(z_onehot, self.post_sampler.weight)
            hidden_inject = hidden + post_injection

            # Compute KL with uniform prior.
            if return_loss:
                log_probs = torch.log(mapper_probs + 1e-12)
                kl = torch.sum(
                    mapper_probs * (log_probs + math.log(self.binary_mapper.num_categories)),
                    dim=-1,
                )
                kl = torch.clamp(kl - kappa, min=0.0)
                kl_loss = kl.mean() * self.beta_kl
        else:
            # Sample uniform Z for each token.
            num_categories = 1 << self.num_bits
            z_indices = torch.randint(num_categories, (bsz, seq_len), device=device)
            z_onehot = F.one_hot(z_indices, num_classes=num_categories).to(hidden.dtype)
            hidden_inject = hidden + F.linear(z_onehot, self.post_sampler.weight)

        # Continue with second half blocks, injecting latents as KV override for the
        # immediate next block.
        hidden = hidden_inject
        for idx in range(mid, self.n_layers):
            block = self.blocks[idx]
            past = None if past_key_values is None else past_key_values[idx]
            key_value_override = hidden_inject if idx == mid else None
            hidden, present = block(
                hidden,
                attention_mask=attention_mask_full,
                past_key_value=past,
                use_cache=use_cache,
                key_value_override=key_value_override,
            )
            if use_cache:
                new_past.append(present)

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        loss_ce = None
        if labels is not None and return_loss:
            loss_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        total_loss = None
        if return_loss and loss_ce is not None:
            total_loss = loss_ce
            if kl_loss is not None:
                total_loss = total_loss + kl_loss

        cache = new_past if use_cache else None

        return FreeTransformerOutput(
            logits=logits,
            loss_ce=loss_ce,
            loss_kl=kl_loss,
            loss_total=total_loss,
            z_indices=z_indices,
            past_key_values=cache,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        for _ in range(max_new_tokens):
            outputs = self(
                generated,
                mode="generate",
                return_loss=False,
            )
            logits = outputs.logits[:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            if top_k > 0:
                top_probs, top_idx = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs).scatter_(1, top_idx, top_probs)
                probs = probs / probs.sum(-1, keepdim=True)
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)
                probs = probs / probs.sum(-1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
        return generated

