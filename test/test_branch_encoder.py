"""
Unit tests for the BranchEncoder.

Tests:
    1. Correct input/output shapes
    2. Residual connection is present
    3. Forward pass produces finite values
    4. Backward pass produces gradients
    5. Encoder parameters receive gradients

Run from project root:

    python tests/test_branch_encoder.py
"""

import sys
from pathlib import Path

import torch


# ----------------------------------------------------------------------
# Add project root to Python path
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------------------------------------------------
# Import project module
# ----------------------------------------------------------------------

from src.models.branch_encoder import BranchEncoder


# ======================================================================
# Configuration
# ======================================================================

BATCH_SIZE = 4
WINDOW = 30

# Actual branch dimensions from the project
MET_DIM = 7
VEG_DIM = 2
ENG_DIM = 5

# Change this ONLY if your config.yaml uses a different d_model.
D_MODEL = 64

HIDDEN_DIM = D_MODEL

DROPOUT = 0.1


# ======================================================================
# Test
# ======================================================================

def main():

    print()
    print("=" * 70)
    print("BRANCH ENCODER TEST")
    print("=" * 70)

    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Sequence length  : {WINDOW}")
    print(f"Meteorological   : {MET_DIM}")
    print(f"Vegetation       : {VEG_DIM}")
    print(f"Engineered       : {ENG_DIM}")
    print(f"D_MODEL          : {D_MODEL}")
    print("=" * 70)


    # ------------------------------------------------------------------
    # Create deterministic random inputs
    # ------------------------------------------------------------------

    torch.manual_seed(42)

    x_met = torch.randn(
        BATCH_SIZE,
        WINDOW,
        MET_DIM,
    )

    x_veg = torch.randn(
        BATCH_SIZE,
        WINDOW,
        VEG_DIM,
    )

    x_eng = torch.randn(
        BATCH_SIZE,
        WINDOW,
        ENG_DIM,
    )


    # ------------------------------------------------------------------
    # Create three independent branch encoders
    # ------------------------------------------------------------------

    met_encoder = BranchEncoder(
        in_dim=MET_DIM,
        hidden_dim=HIDDEN_DIM,
        out_dim=D_MODEL,
        dropout=DROPOUT,
    )

    veg_encoder = BranchEncoder(
        in_dim=VEG_DIM,
        hidden_dim=HIDDEN_DIM,
        out_dim=D_MODEL,
        dropout=DROPOUT,
    )

    eng_encoder = BranchEncoder(
        in_dim=ENG_DIM,
        hidden_dim=HIDDEN_DIM,
        out_dim=D_MODEL,
        dropout=DROPOUT,
    )


    # ==================================================================
    # TEST 1 — Forward pass
    # ==================================================================

    print("\n[TEST 1] Forward pass...")

    met_out = met_encoder(x_met)
    veg_out = veg_encoder(x_veg)
    eng_out = eng_encoder(x_eng)

    expected_met_shape = (
        BATCH_SIZE,
        WINDOW,
        D_MODEL,
    )

    expected_veg_shape = (
        BATCH_SIZE,
        WINDOW,
        D_MODEL,
    )

    expected_eng_shape = (
        BATCH_SIZE,
        WINDOW,
        D_MODEL,
    )

    assert met_out.shape == expected_met_shape, (
        f"Meteorological output shape incorrect: "
        f"{met_out.shape} != {expected_met_shape}"
    )

    assert veg_out.shape == expected_veg_shape, (
        f"Vegetation output shape incorrect: "
        f"{veg_out.shape} != {expected_veg_shape}"
    )

    assert eng_out.shape == expected_eng_shape, (
        f"Engineered output shape incorrect: "
        f"{eng_out.shape} != {expected_eng_shape}"
    )

    print("  ✓ Meteorological output:", tuple(met_out.shape))
    print("  ✓ Vegetation output    :", tuple(veg_out.shape))
    print("  ✓ Engineered output    :", tuple(eng_out.shape))


    # ==================================================================
    # TEST 2 — Finite output
    # ==================================================================

    print("\n[TEST 2] Checking for NaN / Inf...")

    assert torch.isfinite(met_out).all(), (
        "Meteorological encoder produced NaN or Inf."
    )

    assert torch.isfinite(veg_out).all(), (
        "Vegetation encoder produced NaN or Inf."
    )

    assert torch.isfinite(eng_out).all(), (
        "Engineered encoder produced NaN or Inf."
    )

    print("  ✓ Meteorological output is finite")
    print("  ✓ Vegetation output is finite")
    print("  ✓ Engineered output is finite")


    # ==================================================================
    # TEST 3 — Branch independence
    # ==================================================================

    print("\n[TEST 3] Checking branch independence...")

    met_params = list(met_encoder.parameters())
    veg_params = list(veg_encoder.parameters())
    eng_params = list(eng_encoder.parameters())

    assert met_params, "Meteorological encoder has no parameters."
    assert veg_params, "Vegetation encoder has no parameters."
    assert eng_params, "Engineered encoder has no parameters."

    met_ids = {id(p) for p in met_params}
    veg_ids = {id(p) for p in veg_params}
    eng_ids = {id(p) for p in eng_params}

    assert met_ids.isdisjoint(veg_ids), (
        "Meteorological and vegetation encoders share parameters."
    )

    assert met_ids.isdisjoint(eng_ids), (
        "Meteorological and engineered encoders share parameters."
    )

    assert veg_ids.isdisjoint(eng_ids), (
        "Vegetation and engineered encoders share parameters."
    )

    print("  ✓ Meteorological encoder has independent parameters")
    print("  ✓ Vegetation encoder has independent parameters")
    print("  ✓ Engineered encoder has independent parameters")


    # ==================================================================
    # TEST 4 — Residual path
    # ==================================================================

    print("\n[TEST 4] Checking residual projection...")

    assert hasattr(met_encoder, "shortcut"), (
        "BranchEncoder does not contain the residual shortcut."
    )

    assert isinstance(
        met_encoder.shortcut,
        torch.nn.Linear,
    ), (
        "BranchEncoder shortcut is not a Linear projection."
    )

    print("  ✓ Residual shortcut exists")
    print("  ✓ Shortcut uses Linear projection")


    # ==================================================================
    # TEST 5 — Backward pass
    # ==================================================================

    print("\n[TEST 5] Testing backward pass...")

    loss = (
        met_out.mean()
        + veg_out.mean()
        + eng_out.mean()
    )

    loss.backward()

    print("  ✓ Backward pass completed")


    # ==================================================================
    # TEST 6 — Gradient verification
    # ==================================================================

    print("\n[TEST 6] Checking gradients...")

    encoders = {
        "meteorological": met_encoder,
        "vegetation": veg_encoder,
        "engineered": eng_encoder,
    }

    for name, encoder in encoders.items():

        parameters_with_grad = 0
        parameters_without_grad = []

        for param_name, param in encoder.named_parameters():

            if param.grad is None:

                parameters_without_grad.append(
                    param_name
                )

            else:

                parameters_with_grad += 1

                assert torch.isfinite(param.grad).all(), (
                    f"{name} encoder parameter "
                    f"{param_name} has NaN/Inf gradient."
                )

        assert not parameters_without_grad, (
            f"{name} encoder parameters without gradients: "
            f"{parameters_without_grad}"
        )

        print(
            f"  ✓ {name.capitalize()} encoder: "
            f"{parameters_with_grad} parameters received gradients"
        )


    # ==================================================================
    # TEST 7 — Output statistics
    # ==================================================================

    print("\n[TEST 7] Output statistics...")

    print(
        f"  Meteorological -> "
        f"mean={met_out.detach().mean():.6f}, "
        f"std={met_out.detach().std():.6f}"
    )

    print(
        f"  Vegetation     -> "
        f"mean={veg_out.detach().mean():.6f}, "
        f"std={veg_out.detach().std():.6f}"
    )

    print(
        f"  Engineered     -> "
        f"mean={eng_out.detach().mean():.6f}, "
        f"std={eng_out.detach().std():.6f}"
    )


    # ==================================================================
    # COMPLETE
    # ==================================================================

    print()
    print("=" * 70)
    print("BRANCH ENCODER TEST PASSED")
    print("=" * 70)
    print()
    print("Verified:")
    print("  ✓ Correct sequence dimensions")
    print("  ✓ Correct D_MODEL projection")
    print("  ✓ No NaN / Inf")
    print("  ✓ Independent branch parameters")
    print("  ✓ Residual shortcut")
    print("  ✓ Forward pass")
    print("  ✓ Backward pass")
    print("  ✓ Gradient propagation")
    print("=" * 70)


# ======================================================================
# ENTRYPOINT
# ======================================================================

if __name__ == "__main__":
    main()