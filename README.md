# Free Transformer

Implementation of the *Free Transformer* (Fleuret, 2025), a decoder-only model that augments its autoregressive pathway with discrete latent variables sampled through a conditional variational auto-encoder. The implementation targets PyTorch 2.2+ and is designed to interoperate with HuggingFace tokenizers and weights.

## Features

* `freexf.model.FreeTransformer` – decoder-only Transformer with latent sampling, encoder bridge and KL free-bits objective.
* `BinaryMapper` – Bernoulli-to-categorical mapper with straight-through gradients and numerically stable probability computation.
* HuggingFace integration utilities (`freexf.hf_wrap`) for tokenizer loading and weight initialisation from GPT-style checkpoints.
* Training script (`train.py`) supporting mixed precision, gradient accumulation/clipping, cosine scheduling, checkpointing and optional HuggingFace dataset ingestion.
* `generate.py` for text generation using nucleus / top-k sampling with uniform latent sampling.
* Unit tests validating the mapper, loss shaping and model tensor shapes.

## Latent injection overview

Tokens are embedded and processed by the first half of the decoder blocks. A lightweight non-causal encoder block queries the intermediate activations and outputs logits over independent latent bits. The `BinaryMapper` interprets the bit logits as a categorical distribution over `2**H` latent codes, samples a code per token and projects the corresponding one-hot vector through the `post_sampler` linear layer. The projection is added to the intermediate activations and fed to the second half of the decoder, effectively conditioning future computations on the latent sample.

During training the KL divergence between the encoder distribution and the uniform prior is computed token-wise. A configurable free-bits threshold `κ` ensures that only the KL surplus contributes to the objective. The total loss is the sum of next-token cross-entropy and the scaled KL penalty.

## Usage

### Training

```bash
python train.py \
  --dataset wikitext-2-raw-v1 \
  --tokenizer gpt2 \
  --seq_len 256 \
  --batch_size 8 \
  --total_steps 2000 \
  --d_model 256 --n_layers 6 --n_heads 8 --d_mlp 1024
```

To start from a HuggingFace causal LM (e.g. GPT-2 small), pass `--init_from_hf gpt2`. The script will copy matching weights where possible and leave the remaining parameters randomly initialised.

Checkpoints are stored under `--output_dir` and can be resumed with `--resume`.

### Generation

```bash
python generate.py --checkpoint checkpoints/final.pt --tokenizer gpt2 \
  --prompt "A free transformer walks into a bar" --max_new_tokens 64
```

Latent variables are sampled uniformly during generation; to make use of prefill/postfill latents supply an encoded prefix to the model (see `FreeTransformer.generate`).

## Tests

Run the unit test suite with

```bash
pytest -q
```

## Citation

```text
@article{fleuret2025freetransformer,
  title   = {The Free Transformer},
  author  = {François Fleuret},
  journal = {arXiv preprint arXiv:2510.17558},
  year    = {2025}
}
```
