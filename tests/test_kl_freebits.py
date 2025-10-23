import torch

from freexf.model import FreeTransformer


def test_kl_freebits_clamping():
    torch.manual_seed(0)
    model = FreeTransformer(
        vocab_size=20,
        d_model=16,
        n_layers=4,
        n_heads=4,
        d_mlp=32,
        num_bits=2,
    )

    with torch.no_grad():
        model.encoder_block.readout.weight.zero_()
        model.encoder_block.readout.bias.zero_()

    input_ids = torch.randint(0, 20, (2, 6))
    outputs = model(input_ids, mode="train", kappa=1.0)
    assert outputs.loss_kl is not None
    assert outputs.loss_kl.item() < 1e-5

    with torch.no_grad():
        model.encoder_block.readout.bias.fill_(5.0)

    outputs_no_free_bits = model(input_ids, mode="train", kappa=0.0)
    assert outputs_no_free_bits.loss_kl is not None
    assert outputs_no_free_bits.loss_kl.item() > 0.0
