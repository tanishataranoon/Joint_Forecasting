"""
Unit tests for the BiLSTM + Temporal Attention backbone.

Input:
    (B, T, D_MODEL)

Expected:
    pooled representation -> (B, 2 * hidden_dim)
    temporal attention    -> (B, T)

Current project configuration:
    Batch size      = 4
    Sequence length = 30
    D_MODEL         = 64
    Hidden dimension= 128
"""

import sys
from pathlib import Path

import torch


# ----------------------------------------------------------------------
# Allow imports from project root
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.models.temporal_backbones.Bilstm_attention.bilstm_attention import (
    BiLSTMAttentionBackbone,
    TemporalAttentionPool,
)


# ======================================================================
# CONFIGURATION
# ======================================================================

BATCH_SIZE = 4
SEQUENCE_LENGTH = 30
D_MODEL = 64
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.1


# ======================================================================
# HELPERS
# ======================================================================

def check_finite(name, tensor):
    assert torch.isfinite(tensor).all(), f"{name} contains NaN or Inf"


# ======================================================================
# MAIN TEST
# ======================================================================

def main():

    print()
    print("=" * 70)
    print("BiLSTM + TEMPORAL ATTENTION TEST")
    print("=" * 70)

    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Sequence length  : {SEQUENCE_LENGTH}")
    print(f"D_MODEL          : {D_MODEL}")
    print(f"Hidden dimension : {HIDDEN_DIM}")
    print(f"BiLSTM output    : {HIDDEN_DIM * 2}")
    print(f"Number of layers : {NUM_LAYERS}")
    print("=" * 70)

    # --------------------------------------------------------------
    # Device
    # --------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device           : {device}")

    # --------------------------------------------------------------
    # Create model
    # --------------------------------------------------------------

    model = BiLSTMAttentionBackbone(
        d_model=D_MODEL,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    model.train()

    # --------------------------------------------------------------
    # Synthetic fused representation
    # --------------------------------------------------------------

    x = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        D_MODEL,
        device=device,
    )

    # ==============================================================
    # TEST 1
    # ==============================================================

    print("\n[TEST 1] Forward pass...")

    output = model(x)

    assert isinstance(output, dict)

    assert "pooled" in output
    assert "temporal_attn" in output

    pooled = output["pooled"]
    temporal_attn = output["temporal_attn"]

    print(f"  ✓ Pooled representation : {tuple(pooled.shape)}")
    print(f"  ✓ Temporal attention    : {tuple(temporal_attn.shape)}")

    assert pooled.shape == (
        BATCH_SIZE,
        HIDDEN_DIM * 2,
    )

    assert temporal_attn.shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
    )

    # ==============================================================
    # TEST 2
    # ==============================================================

    print("\n[TEST 2] Checking for NaN / Inf...")

    check_finite(
        "Pooled representation",
        pooled,
    )

    check_finite(
        "Temporal attention",
        temporal_attn,
    )

    print("  ✓ Pooled representation is finite")
    print("  ✓ Temporal attention is finite")

    # ==============================================================
    # TEST 3
    # ==============================================================

    print("\n[TEST 3] Checking temporal attention normalization...")

    attention_sums = temporal_attn.sum(dim=1)

    max_error = torch.abs(
        attention_sums - 1.0
    ).max().item()

    print(
        f"  Maximum normalization error: "
        f"{max_error:.10f}"
    )

    assert torch.allclose(
        attention_sums,
        torch.ones_like(attention_sums),
        atol=1e-5,
    )

    print("  ✓ Attention weights sum to 1")

    # ==============================================================
    # TEST 4
    # ==============================================================

    print("\n[TEST 4] Checking attention weight range...")

    min_weight = temporal_attn.min().item()
    max_weight = temporal_attn.max().item()

    print(f"  Minimum attention weight: {min_weight:.8f}")
    print(f"  Maximum attention weight: {max_weight:.8f}")

    assert min_weight >= 0.0
    assert max_weight <= 1.0

    print("  ✓ Attention weights are in [0, 1]")

    # ==============================================================
    # TEST 5
    # ==============================================================

    print("\n[TEST 5] Checking temporal attention adaptation...")

    # Compare attention distributions between samples.
    difference = torch.abs(
        temporal_attn[0] - temporal_attn[1]
    ).mean().item()

    print(
        f"  Mean attention difference "
        f"between sample 0 and 1: {difference:.8f}"
    )

    assert difference > 0.0

    print(
        "  ✓ Temporal attention produces "
        "sample-dependent weights"
    )

    # ==============================================================
    # TEST 6
    # ==============================================================

    print("\n[TEST 6] Checking BiLSTM structure...")

    assert isinstance(model.lstm, torch.nn.LSTM)

    assert model.lstm.bidirectional is True

    assert model.lstm.input_size == D_MODEL

    assert model.lstm.hidden_size == HIDDEN_DIM

    assert model.lstm.num_layers == NUM_LAYERS

    print("  ✓ BiLSTM exists")
    print("  ✓ Bidirectional=True")
    print(f"  ✓ Input dimension={D_MODEL}")
    print(f"  ✓ Hidden dimension={HIDDEN_DIM}")
    print(f"  ✓ Number of layers={NUM_LAYERS}")

    # ==============================================================
    # TEST 7
    # ==============================================================

    print("\n[TEST 7] Checking temporal attention module...")

    assert isinstance(
        model.pool,
        TemporalAttentionPool,
    )

    print("  ✓ TemporalAttentionPool exists")

    assert model.out_dim == HIDDEN_DIM * 2

    print(
        f"  ✓ Backbone output dimension="
        f"{model.out_dim}"
    )

    # ==============================================================
    # TEST 8
    # ==============================================================

    print("\n[TEST 8] Testing backward pass...")

    loss = pooled.mean()

    loss.backward()

    print(f"  Loss: {loss.item():.8f}")
    print("  ✓ Backward pass completed")

    # ==============================================================
    # TEST 9
    # ==============================================================

    print("\n[TEST 9] Checking gradients...")

    lstm_params = 0
    lstm_grad_params = 0

    for param in model.lstm.parameters():

        lstm_params += 1

        if param.grad is not None:
            lstm_grad_params += 1

    print(
        f"  BiLSTM: "
        f"{lstm_grad_params}/{lstm_params} "
        f"parameters received gradients"
    )

    assert lstm_grad_params == lstm_params

    pool_params = 0
    pool_grad_params = 0

    for param in model.pool.parameters():

        pool_params += 1

        if param.grad is not None:
            pool_grad_params += 1

    print(
        f"  Attention pool: "
        f"{pool_grad_params}/{pool_params} "
        f"parameters received gradients"
    )

    assert pool_grad_params == pool_params

    print("  ✓ BiLSTM received gradients")
    print("  ✓ Temporal attention received gradients")

    # ==============================================================
    # TEST 10
    # ==============================================================

    print("\n[TEST 10] Checking parameter count...")

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"  Total trainable parameters: "
        f"{total_params:,}"
    )

    assert total_params > 0

    print("  ✓ Backbone contains trainable parameters")

    # ==============================================================
    # TEST 11
    # ==============================================================

    print("\n[TEST 11] Checking output statistics...")

    pooled_mean = pooled.mean().item()
    pooled_std = pooled.std().item()

    attention_mean = temporal_attn.mean().item()
    attention_std = temporal_attn.std().item()

    print(
        f"  Pooled representation:"
        f"\n    mean = {pooled_mean:.6f}"
        f"\n    std  = {pooled_std:.6f}"
    )

    print(
        f"  Temporal attention:"
        f"\n    mean = {attention_mean:.6f}"
        f"\n    std  = {attention_std:.6f}"
    )

    assert torch.isfinite(
        torch.tensor(pooled_mean)
    )

    assert torch.isfinite(
        torch.tensor(pooled_std)
    )

    print("  ✓ Output statistics are finite")

    # ==============================================================
    # SUCCESS
    # ==============================================================

    print()
    print("=" * 70)
    print("BiLSTM + TEMPORAL ATTENTION TEST PASSED")
    print("=" * 70)

    print("\nVerified:")
    print("  ✓ Correct input dimensions")
    print("  ✓ Correct BiLSTM configuration")
    print("  ✓ Bidirectional temporal modeling")
    print("  ✓ Correct output dimension")
    print("  ✓ Temporal attention pooling")
    print("  ✓ Attention weights sum to 1")
    print("  ✓ Attention weights are non-negative")
    print("  ✓ Temporal attention adapts across samples")
    print("  ✓ No NaN / Inf")
    print("  ✓ Forward pass")
    print("  ✓ Backward pass")
    print("  ✓ Gradient propagation")
    print("  ✓ Trainable parameters")
    print()


if __name__ == "__main__":
    main()