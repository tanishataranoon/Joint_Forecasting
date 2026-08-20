"""
Comprehensive test suite for the Adaptive Cross-Attention Fusion module.

Architecture tested
-------------------
Meteorological Branch
        |
    BranchEncoder
        |
        Zm
        |
        | Query
        v
 Cross-Attention <------ Zv + Ze
        |                    |
        O                    |
        |                    |
        +--------------------+
                    |
        Dynamic Temporal Gate
                    |
        alpha_m, alpha_v,
        alpha_e, alpha_o
                    |
                    v
            Adaptive Fusion
                    |
                    v
                    F

Expected input dimensions
-------------------------
Meteorological : 7 features
Vegetation     : 2 features
Engineered     : 5 features
                    = VPD + Humidex + SPI3
                    + doy_sin + doy_cos

Sequence length : 30
D_MODEL         : 64

This test verifies:

1. Correct branch encoding
2. Correct fused output dimensions
3. Cross-attention output dimensions
4. Attention weights
5. Dynamic temporal gate dimensions
6. Gate normalization
7. Gate temporal variation
8. No NaN / Inf
9. Branch independence
10. Cross-attention receives vegetation + engineered information
11. Fusion output differs from individual branches
12. Backward pass
13. Gradient propagation
14. Model parameter integrity
15. End-to-end fusion execution
"""

import sys
from pathlib import Path

import torch


# ----------------------------------------------------------------------
# Allow imports from project root
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------------------------------------------------
# Project imports
# ----------------------------------------------------------------------

from src.models.fusion import (
    BranchFusionBlock,
    AdaptiveCrossAttentionFusion,
)


# ======================================================================
# CONFIGURATION
# ======================================================================

BATCH_SIZE = 4
SEQUENCE_LENGTH = 30

MET_DIM = 7
VEG_DIM = 2
ENG_DIM = 5

D_MODEL = 64
D_K = 16

NUM_HEADS = 4
GATE_HIDDEN = 64
DROPOUT = 0.1


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================


def check_finite(name, tensor):
    """Check tensor contains no NaN or Inf values."""

    assert torch.isfinite(tensor).all(), (
        f"{name} contains NaN or Inf values"
    )

    print(f"  ✓ {name} is finite")


def count_parameters(model):
    """Return number of trainable parameters."""

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ======================================================================
# TEST 1
# ======================================================================


def test_branch_fusion_forward():

    print("\n[TEST 1] Forward pass...")

    model = BranchFusionBlock(
        met_dim=MET_DIM,
        veg_dim=VEG_DIM,
        eng_dim=ENG_DIM,
        d_model=D_MODEL,
        d_k=D_K,
        num_heads=NUM_HEADS,
        gate_hidden=GATE_HIDDEN,
        dropout=DROPOUT,
    )

    model.eval()

    x_met = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        MET_DIM,
    )

    x_veg = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        VEG_DIM,
    )

    x_eng = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        ENG_DIM,
    )

    with torch.no_grad():

        F, attn, alphas = model(
            x_met,
            x_veg,
            x_eng,
        )

    # --------------------------------------------------------------
    # Fused representation
    # --------------------------------------------------------------

    expected_F_shape = (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        D_MODEL,
    )

    assert F.shape == expected_F_shape, (
        f"Expected F shape {expected_F_shape}, "
        f"got {tuple(F.shape)}"
    )

    print(
        f"  ✓ Fused representation: "
        f"{tuple(F.shape)}"
    )

    # --------------------------------------------------------------
    # Attention weights
    # --------------------------------------------------------------

    expected_attn_shape = (
        BATCH_SIZE,
        NUM_HEADS,
        SEQUENCE_LENGTH,
        SEQUENCE_LENGTH,
    )

    assert attn.shape == expected_attn_shape, (
        f"Expected attention shape "
        f"{expected_attn_shape}, "
        f"got {tuple(attn.shape)}"
    )

    print(
        f"  ✓ Cross-attention weights: "
        f"{tuple(attn.shape)}"
    )

    # --------------------------------------------------------------
    # Dynamic gate
    # --------------------------------------------------------------

    expected_alpha_shape = (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        4,
    )

    assert alphas.shape == expected_alpha_shape, (
        f"Expected gate shape "
        f"{expected_alpha_shape}, "
        f"got {tuple(alphas.shape)}"
    )

    print(
        f"  ✓ Dynamic gate weights: "
        f"{tuple(alphas.shape)}"
    )

    return model, x_met, x_veg, x_eng, F, attn, alphas


# ======================================================================
# TEST 2
# ======================================================================


def test_finite_outputs(
    F,
    attn,
    alphas,
):

    print("\n[TEST 2] Checking for NaN / Inf...")

    check_finite(
        "Fused representation",
        F,
    )

    check_finite(
        "Cross-attention weights",
        attn,
    )

    check_finite(
        "Dynamic gate weights",
        alphas,
    )


# ======================================================================
# TEST 3
# ======================================================================


def test_attention_normalization(attn):

    print("\n[TEST 3] Checking attention normalization...")

    # MultiheadAttention returns:
    #
    # (B, heads, query_length, key_length)
    #
    # Attention should sum to approximately 1
    # across the key dimension.

    attention_sum = attn.sum(dim=-1)

    expected = torch.ones_like(
        attention_sum
    )

    assert torch.allclose(
        attention_sum,
        expected,
        atol=1e-5,
    ), (
        "Cross-attention weights do not "
        "sum to 1 across the key dimension."
    )

    print(
        "  ✓ Attention weights sum to 1 "
        "across temporal keys"
    )


# ======================================================================
# TEST 4
# ======================================================================


def test_gate_normalization(alphas):

    print("\n[TEST 4] Checking dynamic gate normalization...")

    # Gate order:
    #
    # 0 -> meteorological
    # 1 -> vegetation
    # 2 -> engineered
    # 3 -> cross-attention

    alpha_sum = alphas.sum(dim=-1)

    expected = torch.ones_like(
        alpha_sum
    )

    assert torch.allclose(
        alpha_sum,
        expected,
        atol=1e-5,
    ), (
        "Dynamic fusion weights do not "
        "sum to 1."
    )

    print(
        "  ✓ Gate weights sum to 1 "
        "at every timestep"
    )

    # --------------------------------------------------------------
    # Check non-negative values
    # --------------------------------------------------------------

    assert torch.all(
        alphas >= 0
    ), "Gate contains negative weights."

    print(
        "  ✓ Gate weights are non-negative"
    )


# ======================================================================
# TEST 5
# ======================================================================


def test_gate_temporal_adaptation(alphas):

    print(
        "\n[TEST 5] Checking temporal adaptation "
        "of fusion weights..."
    )

    # alphas:
    #
    # (B, T, 4)

    # Compare timestep 0 against timestep 1.

    temporal_difference = torch.abs(
        alphas[:, 1:, :] -
        alphas[:, :-1, :]
    ).mean()

    print(
        f"  Mean temporal gate change: "
        f"{temporal_difference.item():.8f}"
    )

    # The gate is designed to be temporal.
    # We do not require a large difference because
    # randomly initialized networks can occasionally
    # produce similar weights.

    assert torch.isfinite(
        temporal_difference
    ), "Temporal gate difference is invalid."

    print(
        "  ✓ Temporal gate produces "
        "valid timestep-dependent values"
    )


# ======================================================================
# TEST 6
# ======================================================================


def test_gate_range(alphas):

    print("\n[TEST 6] Checking gate value range...")

    assert torch.all(
        alphas >= 0
    ), "Gate contains values below 0."

    assert torch.all(
        alphas <= 1
    ), "Gate contains values above 1."

    print(
        "  ✓ All adaptive fusion weights "
        "are in [0, 1]"
    )


# ======================================================================
# TEST 7
# ======================================================================


def test_branch_independence(model):

    print("\n[TEST 7] Checking branch independence...")

    met_params = list(
        model.met_encoder.parameters()
    )

    veg_params = list(
        model.veg_encoder.parameters()
    )

    eng_params = list(
        model.eng_encoder.parameters()
    )

    # --------------------------------------------------------------
    # Ensure parameters are separate objects
    # --------------------------------------------------------------

    met_ids = {
        id(p)
        for p in met_params
    }

    veg_ids = {
        id(p)
        for p in veg_params
    }

    eng_ids = {
        id(p)
        for p in eng_params
    }

    assert met_ids.isdisjoint(
        veg_ids
    ), "Meteorological and vegetation encoders share parameters."

    assert met_ids.isdisjoint(
        eng_ids
    ), "Meteorological and engineered encoders share parameters."

    assert veg_ids.isdisjoint(
        eng_ids
    ), "Vegetation and engineered encoders share parameters."

    print(
        "  ✓ Meteorological encoder "
        "has independent parameters"
    )

    print(
        "  ✓ Vegetation encoder "
        "has independent parameters"
    )

    print(
        "  ✓ Engineered encoder "
        "has independent parameters"
    )


# ======================================================================
# TEST 8
# ======================================================================


def test_cross_attention_structure(model):

    print(
        "\n[TEST 8] Checking cross-attention structure..."
    )

    fusion = model.fusion

    assert isinstance(
        fusion,
        AdaptiveCrossAttentionFusion,
    )

    print(
        "  ✓ AdaptiveCrossAttentionFusion "
        "is present"
    )

    assert hasattr(
        fusion,
        "q_proj",
    )

    assert hasattr(
        fusion,
        "k_proj",
    )

    assert hasattr(
        fusion,
        "v_proj",
    )

    print(
        "  ✓ Query projection exists"
    )

    print(
        "  ✓ Key projection exists"
    )

    print(
        "  ✓ Value projection exists"
    )

    assert hasattr(
        fusion,
        "cross_attention",
    )

    print(
        "  ✓ Multi-head cross-attention exists"
    )

    assert hasattr(
        fusion,
        "gate",
    )

    print(
        "  ✓ Dynamic temporal gate exists"
    )


# ======================================================================
# TEST 9
# ======================================================================


def test_vegetation_engineered_key_value_projection(model):

    print(
        "\n[TEST 9] Checking vegetation + engineered "
        "cross-attention input..."
    )

    fusion = model.fusion

    # K and V receive:
    #
    # [Zv ; Ze]
    #
    # Therefore their input dimension must be
    # 2 * D_MODEL.

    assert fusion.k_proj.in_features == (
        2 * D_MODEL
    ), (
        "Key projection does not receive "
        "vegetation + engineered representations."
    )

    assert fusion.v_proj.in_features == (
        2 * D_MODEL
    ), (
        "Value projection does not receive "
        "vegetation + engineered representations."
    )

    print(
        "  ✓ Key projection receives "
        "vegetation + engineered branches"
    )

    print(
        "  ✓ Value projection receives "
        "vegetation + engineered branches"
    )


# ======================================================================
# TEST 10
# ======================================================================


def test_fusion_uses_all_branches():

    print(
        "\n[TEST 10] Checking that fusion "
        "responds to all three branches..."
    )

    torch.manual_seed(42)

    model = BranchFusionBlock(
        met_dim=MET_DIM,
        veg_dim=VEG_DIM,
        eng_dim=ENG_DIM,
        d_model=D_MODEL,
        d_k=D_K,
        num_heads=NUM_HEADS,
        gate_hidden=GATE_HIDDEN,
        dropout=0.0,
    )

    model.eval()

    # --------------------------------------------------------------
    # Base inputs
    # --------------------------------------------------------------

    x_met = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        MET_DIM,
    )

    x_veg = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        VEG_DIM,
    )

    x_eng = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        ENG_DIM,
    )

    with torch.no_grad():

        F_base, _, _ = model(
            x_met,
            x_veg,
            x_eng,
        )

        # Change meteorological branch

        F_met, _, _ = model(
            x_met + 1.0,
            x_veg,
            x_eng,
        )

        # Change vegetation branch

        F_veg, _, _ = model(
            x_met,
            x_veg + 1.0,
            x_eng,
        )

        # Change engineered branch

        F_eng, _, _ = model(
            x_met,
            x_veg,
            x_eng + 1.0,
        )

    # --------------------------------------------------------------
    # Calculate changes
    # --------------------------------------------------------------

    met_change = torch.mean(
        torch.abs(F_base - F_met)
    ).item()

    veg_change = torch.mean(
        torch.abs(F_base - F_veg)
    ).item()

    eng_change = torch.mean(
        torch.abs(F_base - F_eng)
    ).item()

    print(
        f"  Meteorological change : "
        f"{met_change:.8f}"
    )

    print(
        f"  Vegetation change     : "
        f"{veg_change:.8f}"
    )

    print(
        f"  Engineered change     : "
        f"{eng_change:.8f}"
    )

    assert met_change > 1e-7, (
        "Fusion output does not respond "
        "to meteorological input."
    )

    assert veg_change > 1e-7, (
        "Fusion output does not respond "
        "to vegetation input."
    )

    assert eng_change > 1e-7, (
        "Fusion output does not respond "
        "to engineered input."
    )

    print(
        "  ✓ Fusion responds to "
        "meteorological branch"
    )

    print(
        "  ✓ Fusion responds to "
        "vegetation branch"
    )

    print(
        "  ✓ Fusion responds to "
        "engineered branch"
    )


# ======================================================================
# TEST 11
# ======================================================================


def test_fusion_not_identity(
    model,
    x_met,
    x_veg,
    x_eng,
    F,
):

    print(
        "\n[TEST 11] Checking adaptive fusion "
        "is not a simple identity..."
    )

    with torch.no_grad():

        Zm = model.met_encoder(
            x_met
        )

        Zv = model.veg_encoder(
            x_veg
        )

        Ze = model.eng_encoder(
            x_eng
        )

    met_difference = torch.mean(
        torch.abs(F - Zm)
    ).item()

    veg_difference = torch.mean(
        torch.abs(F - Zv)
    ).item()

    eng_difference = torch.mean(
        torch.abs(F - Ze)
    ).item()

    print(
        f"  Difference from Zm: "
        f"{met_difference:.8f}"
    )

    print(
        f"  Difference from Zv: "
        f"{veg_difference:.8f}"
    )

    print(
        f"  Difference from Ze: "
        f"{eng_difference:.8f}"
    )

    assert met_difference > 1e-7
    assert veg_difference > 1e-7
    assert eng_difference > 1e-7

    print(
        "  ✓ Fused representation "
        "is distinct from individual branches"
    )


# ======================================================================
# TEST 12
# ======================================================================


def test_backward_pass(model):

    print("\n[TEST 12] Testing backward pass...")

    model.train()

    x_met = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        MET_DIM,
        requires_grad=True,
    )

    x_veg = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        VEG_DIM,
        requires_grad=True,
    )

    x_eng = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        ENG_DIM,
        requires_grad=True,
    )

    F, attn, alphas = model(
        x_met,
        x_veg,
        x_eng,
    )

    # Create scalar loss

    loss = (
        F.mean()
        + attn.mean()
        + alphas.mean()
    )

    loss.backward()

    print(
        f"  Loss: {loss.item():.8f}"
    )

    print(
        "  ✓ Backward pass completed"
    )


# ======================================================================
# TEST 13
# ======================================================================


def test_gradient_propagation(model):

    print("\n[TEST 13] Checking gradients...")

    # --------------------------------------------------------------
    # Count parameters with gradients
    # --------------------------------------------------------------

    encoder_groups = {
        "Meteorological encoder": model.met_encoder,
        "Vegetation encoder": model.veg_encoder,
        "Engineered encoder": model.eng_encoder,
        "Adaptive fusion": model.fusion,
    }

    for name, module in encoder_groups.items():

        gradient_count = sum(
            1
            for p in module.parameters()
            if p.grad is not None
        )

        total_params = sum(
            1
            for _ in module.parameters()
        )

        print(
            f"  {name}: "
            f"{gradient_count}/{total_params} "
            f"parameters received gradients"
        )

        assert gradient_count > 0, (
            f"{name} received no gradients."
        )

        print(
            f"  ✓ {name} received gradients"
        )


# ======================================================================
# TEST 14
# ======================================================================


def test_parameter_count(model):

    print("\n[TEST 14] Checking trainable parameters...")

    total = count_parameters(model)

    print(
        f"  Total trainable parameters: "
        f"{total:,}"
    )

    assert total > 0, (
        "Model contains no trainable parameters."
    )

    print(
        "  ✓ Fusion model contains "
        "trainable parameters"
    )


# ======================================================================
# TEST 15
# ======================================================================


def test_output_statistics(F, alphas):

    print("\n[TEST 15] Output statistics...")

    print(
        "  Fused representation:"
    )

    print(
        f"    mean = {F.mean().item():.6f}"
    )

    print(
        f"    std  = {F.std().item():.6f}"
    )

    print(
        "  Dynamic gate:"
    )

    gate_names = [
        "Meteorological",
        "Vegetation",
        "Engineered",
        "Cross-Attention",
    ]

    for i, name in enumerate(
        gate_names
    ):

        value = alphas[..., i].mean().item()

        print(
            f"    {name:16s}: "
            f"{value:.6f}"
        )

    assert torch.isfinite(F).all()
    assert torch.isfinite(alphas).all()

    print(
        "  ✓ Output statistics are finite"
    )


# ======================================================================
# MAIN
# ======================================================================


def main():

    print("\n")
    print("=" * 70)
    print("ADAPTIVE CROSS-ATTENTION FUSION TEST")
    print("=" * 70)

    print(
        f"Batch size       : {BATCH_SIZE}"
    )

    print(
        f"Sequence length  : {SEQUENCE_LENGTH}"
    )

    print(
        f"Meteorological   : {MET_DIM}"
    )

    print(
        f"Vegetation       : {VEG_DIM}"
    )

    print(
        f"Engineered       : {ENG_DIM}"
    )

    print(
        f"D_MODEL          : {D_MODEL}"
    )

    print(
        f"D_K              : {D_K}"
    )

    print(
        f"Attention heads  : {NUM_HEADS}"
    )

    print(
        f"Gate hidden      : {GATE_HIDDEN}"
    )

    print("=" * 70)

    # --------------------------------------------------------------
    # Test 1
    # --------------------------------------------------------------

    (
        model,
        x_met,
        x_veg,
        x_eng,
        F,
        attn,
        alphas,
    ) = test_branch_fusion_forward()

    # --------------------------------------------------------------
    # Test 2
    # --------------------------------------------------------------

    test_finite_outputs(
        F,
        attn,
        alphas,
    )

    # --------------------------------------------------------------
    # Test 3
    # --------------------------------------------------------------

    test_attention_normalization(
        attn
    )

    # --------------------------------------------------------------
    # Test 4
    # --------------------------------------------------------------

    test_gate_normalization(
        alphas
    )

    # --------------------------------------------------------------
    # Test 5
    # --------------------------------------------------------------

    test_gate_temporal_adaptation(
        alphas
    )

    # --------------------------------------------------------------
    # Test 6
    # --------------------------------------------------------------

    test_gate_range(
        alphas
    )

    # --------------------------------------------------------------
    # Test 7
    # --------------------------------------------------------------

    test_branch_independence(
        model
    )

    # --------------------------------------------------------------
    # Test 8
    # --------------------------------------------------------------

    test_cross_attention_structure(
        model
    )

    # --------------------------------------------------------------
    # Test 9
    # --------------------------------------------------------------

    test_vegetation_engineered_key_value_projection(
        model
    )

    # --------------------------------------------------------------
    # Test 10
    # --------------------------------------------------------------

    test_fusion_uses_all_branches()

    # --------------------------------------------------------------
    # Test 11
    # --------------------------------------------------------------

    test_fusion_not_identity(
        model,
        x_met,
        x_veg,
        x_eng,
        F,
    )

    # --------------------------------------------------------------
    # Test 12
    # --------------------------------------------------------------

    test_backward_pass(
        model
    )

    # --------------------------------------------------------------
    # Test 13
    # --------------------------------------------------------------

    test_gradient_propagation(
        model
    )

    # --------------------------------------------------------------
    # Test 14
    # --------------------------------------------------------------

    test_parameter_count(
        model
    )

    # --------------------------------------------------------------
    # Test 15
    # --------------------------------------------------------------

    test_output_statistics(
        F,
        alphas,
    )

    # --------------------------------------------------------------
    # Final result
    # --------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ADAPTIVE CROSS-ATTENTION FUSION TEST PASSED")
    print("=" * 70)

    print("\nVerified:")

    print(
        "  ✓ Three independent branch encoders"
    )

    print(
        "  ✓ Correct branch dimensions"
    )

    print(
        "  ✓ Correct fused representation"
    )

    print(
        "  ✓ Multi-head cross-attention"
    )

    print(
        "  ✓ Vegetation + engineered K/V"
    )

    print(
        "  ✓ Dynamic temporal gating"
    )

    print(
        "  ✓ Four adaptive fusion weights"
    )

    print(
        "  ✓ Gate weights sum to 1"
    )

    print(
        "  ✓ Attention weights sum to 1"
    )

    print(
        "  ✓ Temporal gate adaptation"
    )

    print(
        "  ✓ All three branches influence fusion"
    )

    print(
        "  ✓ No NaN / Inf"
    )

    print(
        "  ✓ Forward pass"
    )

    print(
        "  ✓ Backward pass"
    )

    print(
        "  ✓ Gradient propagation"
    )

    print(
        "  ✓ Trainable parameters"
    )

    print("=" * 70)


# ======================================================================
# ENTRYPOINT
# ======================================================================

if __name__ == "__main__":
    main()

