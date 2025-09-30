# tests/test_models.py
import pytest
import torch
import torch.nn as nn
import math
from Rougeformer import Rougeformer, GQA_RoPE_Attention, to_additive_mask
from RhoBART import RhoBARTClean


def test_mask_conversion():
    bool_mask = torch.tensor([[True, False], [False, True]])
    add_mask = to_additive_mask(bool_mask)
    assert add_mask.shape == bool_mask.shape
    # Valid entries should be 0, masked entries -inf
    assert (add_mask[0, 0] == 0) and (add_mask[0, 1] == float("-inf"))

    int_mask = torch.tensor([[1, 0], [0, 1]])
    add_mask2 = to_additive_mask(int_mask)
    assert torch.allclose(add_mask, add_mask2)


@pytest.mark.parametrize("batch_size, seq_len, vocab_size", [(2, 16, 1000)])
def test_rougeformer_forward(batch_size, seq_len, vocab_size):
    model = Rougeformer(vocab_size=vocab_size, max_seq=seq_len, num_layers=2, hidden_size=64)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, vocab_size)

    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss, logits = model(input_ids, labels=labels)
    assert loss.ndim == 0
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("batch_size, src_len, tgt_len, vocab_size", [(2, 16, 10, 1000)])
def test_rhobart_forward(batch_size, src_len, tgt_len, vocab_size):
    model = RhoBARTClean(vocab_size=vocab_size, num_layers=2, hidden_size=64)
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
    out = model(input_ids=src, labels=tgt)
    assert out["loss"].ndim == 0
    out["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_rope_attention_shapes():
    hidden_size = 64
    num_heads = 4
    model = GQA_RoPE_Attention(hidden_size=hidden_size, num_heads=num_heads, kv_groups=1)
    x = torch.randn(2, 16, hidden_size)
    out = model(x)
    assert out.shape == (2, 16, hidden_size)
    out2, past_kv = model(x, use_cache=True)
    assert past_kv[0].dim() == 4 and past_kv[1].dim() == 4


def test_rhobart_generate_runs():
    vocab_size = 1000
    model = RhoBARTClean(vocab_size=vocab_size, num_layers=2, hidden_size=64)
    src = torch.randint(0, vocab_size, (2, 8))
    out = model.generate(src, max_new_tokens=5, do_sample=False)
    assert out.shape == (2, 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_models_on_cuda():
    device = torch.device("cuda")
    model = Rougeformer(vocab_size=500, max_seq=32, num_layers=2, hidden_size=32).to(device)
    x = torch.randint(0, 500, (2, 16), device=device)
    out = model(x)
    assert out.shape == (2, 16, 500)

def test_gradient_clipping_check():
    model = RhoBARTClean(vocab_size=100, num_layers=2, hidden_size=32)
    src = torch.randint(0, 100, (2, 8))
    tgt = torch.randint(0, 100, (2, 6))
    out = model(input_ids=src, labels=tgt)
    out["loss"].backward()
    grad_norm, needs_clip = model.check_gradients(max_norm=0.1)
    assert not math.isnan(grad_norm)
    assert grad_norm >= 0   # allow 0.0
    assert isinstance(needs_clip, bool)


def test_long_sequence_near_max_seq():
    vocab_size = 500
    max_seq = 64
    model = Rougeformer(vocab_size=vocab_size, max_seq=max_seq, num_layers=2, hidden_size=32)
    # sequence length exactly at max_seq
    x = torch.randint(0, vocab_size, (2, max_seq))
    logits = model(x)
    assert logits.shape == (2, max_seq, vocab_size)

def test_rougeformer_gradient_utils(capsys):
    model = Rougeformer(vocab_size=100, max_seq=16, num_layers=2, hidden_size=32)
    input_ids = torch.randint(0, 100, (2, 8))
    labels = torch.randint(0, 100, (2, 8))

    loss, logits = model(input_ids, labels=labels)
    loss.backward()

    grad_norm, needs_clip = model.check_gradients(max_norm=0.1)
    assert not math.isnan(grad_norm)
    assert grad_norm >= 0
    assert isinstance(needs_clip, bool)

    # Check logging output
    model.log_training_step(loss, step=1)
    captured = capsys.readouterr()
    assert "[Rougeformer]" in captured.out
    assert "loss=" in captured.out
    assert "grad_norm=" in captured.out
