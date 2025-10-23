"""Text generation entry point for trained Free Transformer checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from freexf.hf_wrap import load_tokenizer
from freexf.model import FreeTransformer


def load_model(checkpoint: Path, device: str) -> FreeTransformer:
    state = torch.load(checkpoint, map_location=device)
    cfg = state.get("config", {})
    model = FreeTransformer(
        vocab_size=cfg.get("vocab_size", state["model"]["embed_tokens.weight"].shape[0]),
        d_model=cfg.get("d_model", state["model"]["embed_tokens.weight"].shape[1]),
        n_layers=cfg.get("n_layers", 6),
        n_heads=cfg.get("n_heads", 8),
        d_mlp=cfg.get("d_mlp", 4 * state["model"]["embed_tokens.weight"].shape[1]),
        num_bits=cfg.get("num_bits", 16),
        beta_kl=cfg.get("beta_kl", 1.0),
    )
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with the Free Transformer")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(Path(args.checkpoint), args.device)

    if not args.prompt:
        prompt = input("Prompt: ")
    else:
        prompt = args.prompt

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(args.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()

