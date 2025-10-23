"""Training script for the Free Transformer."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm

from datasets import load_dataset

from freexf.hf_wrap import init_from_hf, load_tokenizer
from freexf.model import FreeTransformer


@dataclass
class TrainConfig:
    dataset: str
    tokenizer: str
    output_dir: str
    seq_len: int = 512
    batch_size: int = 4
    eval_batch_size: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    total_steps: int = 10000
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    grad_clip: float = 1.0
    grad_accum: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = False
    bf16: bool = False
    compile: bool = False
    init_from_hf: Optional[str] = None
    resume: Optional[str] = None
    beta_kl: float = 1.0
    kappa: float = 0.0
    pad_side: str = "right"
    num_workers: int = 2
    save_final: bool = True
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_mlp: int = 1024
    num_bits: int = 16
    vocab_size: Optional[int] = None


class TextDataset(Dataset):
    def __init__(self, tokenized: List[List[int]], seq_len: int, pad_id: int, pad_side: str) -> None:
        self.tokenized = tokenized
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.pad_side = pad_side

    def __len__(self) -> int:
        return len(self.tokenized)

    def __getitem__(self, idx: int):
        tokens = self.tokenized[idx]
        tokens = tokens[: self.seq_len + 1]
        if len(tokens) < self.seq_len + 1:
            pad_amount = self.seq_len + 1 - len(tokens)
            pad_tokens = [self.pad_id] * pad_amount
            if self.pad_side == "right":
                tokens = tokens + pad_tokens
            else:
                tokens = pad_tokens + tokens
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def cycle(dl: Iterable):
    while True:
        for batch in dl:
            yield batch


def get_dataloaders(tokenizer_name: str, dataset_name: str, seq_len: int, batch_size: int, eval_batch_size: int, pad_side: str, num_workers: int) -> tuple[DataLoader, DataLoader, int]:
    tokenizer = load_tokenizer(tokenizer_name)
    dataset = load_dataset(dataset_name)
    column = "text"
    if column not in dataset["train"].column_names:
        column = dataset["train"].column_names[0]

    def tokenize(example):
        return tokenizer(example[column])

    train_tokens = [item["input_ids"] for item in map(tokenize, dataset["train"])]
    eval_split = dataset.get("validation") or dataset.get("test") or dataset["train"].select(range(min(1000, len(dataset["train"]))))
    eval_tokens = [item["input_ids"] for item in map(tokenize, eval_split)]

    pad_id = tokenizer.pad_token_id
    train_dataset = TextDataset(train_tokens, seq_len, pad_id, pad_side)
    eval_dataset = TextDataset(eval_tokens, seq_len, pad_id, pad_side)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, eval_loader, tokenizer.vocab_size


def save_checkpoint(path: Path, model: FreeTransformer, optimizer: torch.optim.Optimizer, scaler: Optional[torch.cuda.amp.GradScaler], step: int, config: TrainConfig) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "config": asdict(config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, model: FreeTransformer, optimizer: torch.optim.Optimizer, scaler: Optional[torch.cuda.amp.GradScaler]) -> int:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return state.get("step", 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Free Transformer")
    parser.add_argument("--config", type=str, help="Path to JSON config", required=False)
    parser.add_argument("--dataset", type=str, default="wikitext", help="HF dataset name")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--total_steps", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--init_from_hf", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--beta_kl", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--pad_side", type=str, default="right", choices=["left", "right"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_final", action="store_true")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_mlp", type=int, default=1024)
    parser.add_argument("--num_bits", type=int, default=16)

    args = parser.parse_args()

    config = TrainConfig(
        dataset=args.dataset,
        tokenizer=args.tokenizer,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        grad_clip=args.grad_clip,
        grad_accum=args.grad_accum,
        device=args.device,
        fp16=args.fp16,
        bf16=args.bf16,
        compile=args.compile,
        init_from_hf=args.init_from_hf,
        resume=args.resume,
        beta_kl=args.beta_kl,
        kappa=args.kappa,
        pad_side=args.pad_side,
        num_workers=args.num_workers,
        save_final=args.save_final,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        num_bits=args.num_bits,
    )

    train_loader, eval_loader, vocab_size = get_dataloaders(
        config.tokenizer,
        config.dataset,
        config.seq_len,
        config.batch_size,
        config.eval_batch_size,
        config.pad_side,
        config.num_workers,
    )

    config.vocab_size = vocab_size

    model = FreeTransformer(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_mlp=config.d_mlp,
        num_bits=config.num_bits,
        beta_kl=config.beta_kl,
    )

    if config.init_from_hf:
        init_from_hf(model, config.init_from_hf)

    model.to(config.device)
    if config.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=config.weight_decay)

    scaler: Optional[torch.cuda.amp.GradScaler]
    if config.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return max(1e-3, (step + 1) / max(1, config.warmup_steps))
        progress = (step - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    start_step = 0
    if config.resume:
        start_step = load_checkpoint(Path(config.resume), model, optimizer, scaler)
        global_step = start_step

    model.train()
    data_iter = cycle(train_loader)
    autocast_dtype = torch.float16 if config.fp16 else torch.bfloat16 if config.bf16 else None

    for step in tqdm(range(start_step, config.total_steps), initial=start_step, total=config.total_steps):
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(config.grad_accum):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(config.device)
            labels = batch["labels"].to(config.device)
            with torch.cuda.amp.autocast(enabled=autocast_dtype is not None, dtype=autocast_dtype):
                outputs = model(input_ids, labels=labels, mode="train", kappa=config.kappa)
                loss = outputs.loss_total if outputs.loss_total is not None else outputs.loss_ce
            if loss is None:
                continue
            loss = loss / config.grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum += loss.item()

        if config.grad_clip > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        global_step += 1

        if global_step % config.log_interval == 0:
            print(f"Step {global_step}: loss={loss_accum:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        if global_step % config.eval_interval == 0:
            evaluate(model, eval_loader, config)

        if global_step % config.save_interval == 0:
            save_path = Path(config.output_dir) / f"ckpt_{global_step}.pt"
            save_checkpoint(save_path, model, optimizer, scaler, global_step, config)

    if config.save_final:
        save_path = Path(config.output_dir) / "final.pt"
        save_checkpoint(save_path, model, optimizer, scaler, global_step, config)


def evaluate(model: FreeTransformer, eval_loader: DataLoader, config: TrainConfig) -> None:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(config.device)
            labels = batch["labels"].to(config.device)
            outputs = model(input_ids, labels=labels, mode="prefill", kappa=config.kappa)
            loss = outputs.loss_total if outputs.loss_total is not None else outputs.loss_ce
            if loss is not None:
                losses.append(loss.item())
    if losses:
        print(f"Eval loss: {sum(losses)/len(losses):.4f}")
    model.train()


if __name__ == "__main__":
    main()

