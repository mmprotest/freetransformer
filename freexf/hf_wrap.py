"""Utilities for interfacing the Free Transformer with HuggingFace."""

from __future__ import annotations

from typing import Optional

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from .model import FreeTransformer


def load_tokenizer(tokenizer_name: str):
    """Load a HuggingFace tokenizer and ensure a padding token exists."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _copy_linear(dst: torch.nn.Linear, src_weight: torch.Tensor, src_bias: Optional[torch.Tensor]) -> None:
    if dst.weight.shape != src_weight.shape:
        raise ValueError("Shape mismatch for linear weight copy")
    dst.weight.data.copy_(src_weight)
    if dst.bias is not None and src_bias is not None:
        if dst.bias.shape != src_bias.shape:
            raise ValueError("Shape mismatch for linear bias copy")
        dst.bias.data.copy_(src_bias)


def init_from_hf(model: FreeTransformer, repo_id: str) -> None:
    """Initialise a ``FreeTransformer`` instance from a HuggingFace checkpoint.

    The procedure copies embeddings, attention and MLP projections whenever the
    shapes match a GPT-2 style architecture.  Layers that do not match are left
    at their randomly initialised values.
    """

    hf_model = AutoModelForCausalLM.from_pretrained(repo_id)
    hf_state = hf_model.state_dict()

    if hasattr(hf_model, "get_input_embeddings"):
        embed = hf_model.get_input_embeddings().weight.data
        if embed.shape == model.embed_tokens.weight.shape:
            model.embed_tokens.weight.data.copy_(embed)

    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "wpe"):
        pos_embed = hf_model.transformer.wpe.weight.data
        if pos_embed.shape == model.pos_embed.weight.shape:
            model.pos_embed.weight.data.copy_(pos_embed)

    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        hf_blocks = hf_model.transformer.h
        num_shared = min(len(hf_blocks), model.n_layers)
        for idx in range(num_shared):
            src_block = hf_blocks[idx]
            dst_block = model.blocks[idx]

            if hasattr(src_block, "attn") and hasattr(src_block.attn, "c_attn"):
                qkv_weight = src_block.attn.c_attn.weight.data
                qkv_bias = src_block.attn.c_attn.bias.data
                q_w, k_w, v_w = torch.chunk(qkv_weight, 3, dim=0)
                q_b, k_b, v_b = torch.chunk(qkv_bias, 3, dim=0)
                if q_w.shape == dst_block.attn.q_proj.weight.shape:
                    dst_block.attn.q_proj.weight.data.copy_(q_w)
                    dst_block.attn.k_proj.weight.data.copy_(k_w)
                    dst_block.attn.v_proj.weight.data.copy_(v_w)
                    if dst_block.attn.q_proj.bias is not None:
                        dst_block.attn.q_proj.bias.data.copy_(q_b)
                        dst_block.attn.k_proj.bias.data.copy_(k_b)
                        dst_block.attn.v_proj.bias.data.copy_(v_b)

            if hasattr(src_block, "attn") and hasattr(src_block.attn, "c_proj"):
                proj_w = src_block.attn.c_proj.weight.data
                proj_b = src_block.attn.c_proj.bias.data
                if proj_w.shape == dst_block.attn.o_proj.weight.shape:
                    dst_block.attn.o_proj.weight.data.copy_(proj_w)
                    if dst_block.attn.o_proj.bias is not None:
                        dst_block.attn.o_proj.bias.data.copy_(proj_b)

            if hasattr(src_block, "mlp") and hasattr(src_block.mlp, "c_fc"):
                fc_w = src_block.mlp.c_fc.weight.data
                fc_b = src_block.mlp.c_fc.bias.data
                proj_w = src_block.mlp.c_proj.weight.data
                proj_b = src_block.mlp.c_proj.bias.data
                if fc_w.shape == dst_block.mlp.fc1.weight.shape:
                    dst_block.mlp.fc1.weight.data.copy_(fc_w)
                    dst_block.mlp.fc1.bias.data.copy_(fc_b)
                if proj_w.shape == dst_block.mlp.fc2.weight.shape:
                    dst_block.mlp.fc2.weight.data.copy_(proj_w)
                    dst_block.mlp.fc2.bias.data.copy_(proj_b)

    for tensor in hf_state.values():
        if tensor.shape == model.lm_head.weight.shape:
            model.lm_head.weight.data.copy_(tensor)
            break

    model.tokenizer = getattr(hf_model, "tokenizer", model.tokenizer)

