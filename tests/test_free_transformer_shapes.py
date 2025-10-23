import torch

from freexf.model import FreeTransformer


def test_free_transformer_forward_shapes():
    model = FreeTransformer(
        vocab_size=32,
        d_model=32,
        n_layers=4,
        n_heads=4,
        d_mlp=64,
        num_bits=3,
    )
    input_ids = torch.randint(0, 32, (2, 5))
    labels = torch.randint(0, 32, (2, 5))
    outputs = model(input_ids, labels=labels, mode="train", kappa=0.0)

    assert outputs.logits.shape == (2, 5, 32)
    assert outputs.loss_ce is not None
    assert outputs.loss_total is not None
    assert outputs.z_indices is not None


def test_free_transformer_prefill_mode():
    model = FreeTransformer(
        vocab_size=16,
        d_model=16,
        n_layers=4,
        n_heads=4,
        d_mlp=32,
        num_bits=2,
    )
    input_ids = torch.randint(0, 16, (1, 6))
    outputs = model(input_ids, mode="prefill", kappa=0.0)
    assert outputs.logits.shape == (1, 6, 16)
    assert outputs.z_indices is not None
