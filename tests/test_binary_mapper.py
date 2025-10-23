import torch

from freexf.model import BinaryMapper


def test_binary_mapper_shapes_and_grad():
    mapper = BinaryMapper(num_bits=3)
    logits = torch.randn(2, 4, 3, requires_grad=True)
    indices, probs, st = mapper(logits, sample=False)

    assert indices.shape == (2, 4)
    assert probs.shape == (2, 4, 8)
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs[..., 0]))

    loss = st.sum()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
