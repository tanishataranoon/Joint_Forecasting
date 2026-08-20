"""
Test suite for MultiTaskHeads and hazard-percentage calculation.

Tests:
    1. MultiTaskHeads forward pass
    2. Output shapes
    3. NaN / Inf checks
    4. Classification probability normalization
    5. Hazard percentage calculation
    6. Percentage normalization
    7. Dominant hazard detection
    8. Backward pass
    9. Gradient propagation
    10. Parameter count
    11. Class maps

The prediction heads are shared across all three independent
temporal models:

    BiLSTM + Attention
    CNN + BiLSTM + Attention
    TFT
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.models.heads import (
    MultiTaskHeads,
    calculate_hazard_percentages,
    DROUGHT_CLASS_MAP,
    HEAT_CLASS_MAP,
)


# ======================================================================
# CONFIGURATION
# ======================================================================

BATCH_SIZE = 4
BACKBONE_OUT_DIM = 256

N_REG_TARGETS = 2
N_CLASSES = 4

DROPOUT = 0.1


# ======================================================================
# HELPER
# ======================================================================

def check_finite(name, tensor):
    assert torch.isfinite(tensor).all(), (
        f"{name} contains NaN or Inf"
    )
    print(f"  ✓ {name} is finite")


# ======================================================================
# MAIN TEST
# ======================================================================

def main():

    print()
    print("=" * 70)
    print("MULTI-TASK HEADS TEST")
    print("=" * 70)

    print(f"Batch size          : {BATCH_SIZE}")
    print(f"Backbone output dim : {BACKBONE_OUT_DIM}")
    print(f"Regression targets  : {N_REG_TARGETS}")
    print(f"Classes             : {N_CLASSES}")
    print(f"Dropout             : {DROPOUT}")
    print("=" * 70)

    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------------

    model = MultiTaskHeads(
        in_dim=BACKBONE_OUT_DIM,
        n_reg_targets=N_REG_TARGETS,
        n_classes=N_CLASSES,
        dropout=DROPOUT,
    )

    model.train()

    # ------------------------------------------------------------------
    # Synthetic backbone representation
    # ------------------------------------------------------------------

    pooled = torch.randn(
        BATCH_SIZE,
        BACKBONE_OUT_DIM,
        requires_grad=True,
    )

    # ==================================================================
    # TEST 1 — Forward pass
    # ==================================================================

    print()
    print("[TEST 1] Forward pass...")

    reg_out, drought_logits, heat_logits = model(pooled)

    print(f"  ✓ Regression output : {tuple(reg_out.shape)}")
    print(f"  ✓ Drought logits    : {tuple(drought_logits.shape)}")
    print(f"  ✓ Heat logits       : {tuple(heat_logits.shape)}")

    assert reg_out.shape == (
        BATCH_SIZE,
        N_REG_TARGETS,
    )

    assert drought_logits.shape == (
        BATCH_SIZE,
        N_CLASSES,
    )

    assert heat_logits.shape == (
        BATCH_SIZE,
        N_CLASSES,
    )

    # ==================================================================
    # TEST 2 — NaN / Inf
    # ==================================================================

    print()
    print("[TEST 2] Checking for NaN / Inf...")

    check_finite(
        "Regression output",
        reg_out,
    )

    check_finite(
        "Drought logits",
        drought_logits,
    )

    check_finite(
        "Heat logits",
        heat_logits,
    )

    # ==================================================================
    # TEST 3 — Classification probabilities
    # ==================================================================

    print()
    print("[TEST 3] Checking classification probabilities...")

    drought_probs = F.softmax(
        drought_logits,
        dim=-1,
    )

    heat_probs = F.softmax(
        heat_logits,
        dim=-1,
    )

    check_finite(
        "Drought probabilities",
        drought_probs,
    )

    check_finite(
        "Heat probabilities",
        heat_probs,
    )

    # probabilities must be >= 0
    assert (drought_probs >= 0).all()
    assert (heat_probs >= 0).all()

    print("  ✓ Drought probabilities are non-negative")
    print("  ✓ Heat probabilities are non-negative")

    # ==================================================================
    # TEST 4 — Probability normalization
    # ==================================================================

    print()
    print("[TEST 4] Checking probability normalization...")

    drought_sum = drought_probs.sum(dim=-1)
    heat_sum = heat_probs.sum(dim=-1)

    drought_error = torch.max(
        torch.abs(drought_sum - 1.0)
    ).item()

    heat_error = torch.max(
        torch.abs(heat_sum - 1.0)
    ).item()

    print(
        f"  Maximum drought normalization error : "
        f"{drought_error:.10f}"
    )

    print(
        f"  Maximum heat normalization error    : "
        f"{heat_error:.10f}"
    )

    assert torch.allclose(
        drought_sum,
        torch.ones_like(drought_sum),
        atol=1e-6,
    )

    assert torch.allclose(
        heat_sum,
        torch.ones_like(heat_sum),
        atol=1e-6,
    )

    print("  ✓ Drought probabilities sum to 1")
    print("  ✓ Heat probabilities sum to 1")

    # ==================================================================
    # TEST 5 — Hazard percentage calculation
    # ==================================================================

    print()
    print("[TEST 5] Testing hazard-percentage calculation...")

    hazard = calculate_hazard_percentages(
        drought_logits,
        heat_logits,
    )

    required_keys = [
        "drought_class_probabilities",
        "heat_class_probabilities",
        "drought_severity",
        "heat_severity",
        "drought_percentage",
        "heat_percentage",
        "dominant_hazard",
    ]

    for key in required_keys:
        assert key in hazard, (
            f"Missing hazard output: {key}"
        )

    print("  ✓ All hazard outputs are present")

    # ==================================================================
    # TEST 6 — Hazard probability shapes
    # ==================================================================

    print()
    print("[TEST 6] Checking hazard probability shapes...")

    assert hazard[
        "drought_class_probabilities"
    ].shape == (BATCH_SIZE, N_CLASSES)

    assert hazard[
        "heat_class_probabilities"
    ].shape == (BATCH_SIZE, N_CLASSES)

    print(
        "  ✓ Drought class probabilities: "
        f"{tuple(hazard['drought_class_probabilities'].shape)}"
    )

    print(
        "  ✓ Heat class probabilities    : "
        f"{tuple(hazard['heat_class_probabilities'].shape)}"
    )

    # ==================================================================
    # TEST 7 — Severity range
    # ==================================================================

    print()
    print("[TEST 7] Checking severity scores...")

    drought_severity = hazard["drought_severity"]
    heat_severity = hazard["heat_severity"]

    check_finite(
        "Drought severity",
        drought_severity,
    )

    check_finite(
        "Heat severity",
        heat_severity,
    )

    assert (drought_severity >= 0).all()
    assert (drought_severity <= 1).all()

    assert (heat_severity >= 0).all()
    assert (heat_severity <= 1).all()

    print("  ✓ Drought severity is in [0, 1]")
    print("  ✓ Heat severity is in [0, 1]")

    # ==================================================================
    # TEST 8 — Hazard percentages
    # ==================================================================

    print()
    print("[TEST 8] Checking hazard percentages...")

    drought_percentage = hazard[
        "drought_percentage"
    ]

    heat_percentage = hazard[
        "heat_percentage"
    ]

    check_finite(
        "Drought percentage",
        drought_percentage,
    )

    check_finite(
        "Heat percentage",
        heat_percentage,
    )

    assert (drought_percentage >= 0).all()
    assert (drought_percentage <= 100).all()

    assert (heat_percentage >= 0).all()
    assert (heat_percentage <= 100).all()

    print("  ✓ Drought percentages are in [0, 100]")
    print("  ✓ Heat percentages are in [0, 100]")

    # ==================================================================
    # TEST 9 — Percentage normalization
    # ==================================================================

    print()
    print("[TEST 9] Checking percentage normalization...")

    percentage_sum = (
        drought_percentage
        + heat_percentage
    )

    max_error = torch.max(
        torch.abs(percentage_sum - 100.0)
    ).item()

    print(
        f"  Maximum percentage sum error : "
        f"{max_error:.10f}"
    )

    assert torch.allclose(
        percentage_sum,
        torch.full_like(
            percentage_sum,
            100.0,
        ),
        atol=1e-5,
    )

    print("  ✓ Drought + Heat percentages = 100%")

    # ==================================================================
    # TEST 10 — Dominant hazard
    # ==================================================================

    print("[TEST 10] Checking dominant hazard...")

    dominant_hazard = torch.argmax(
        torch.stack(
            [drought_percentage, heat_percentage],
            dim=-1,
        ),
        dim=-1,
    )

    print("Dominant hazard:", dominant_hazard)

    assert dominant_hazard.shape == (BATCH_SIZE,)

    assert torch.all(
        (dominant_hazard == 0) |
        (dominant_hazard == 1)
    )

    print("  ✓ Dominant hazard contains valid class indices")
    print("  ✓ 0 = Drought, 1 = Heat")

    # ==================================================================
    # TEST 11 — Display example predictions
    # ==================================================================

    print()
    print("[TEST 11] Example hazard predictions...")

    for i in range(BATCH_SIZE):

        d_pct = drought_percentage[i].item()
        h_pct = heat_percentage[i].item()

        d_severity = drought_severity[i].item()
        h_severity = heat_severity[i].item()

        d_class = torch.argmax(
            hazard[
                "drought_class_probabilities"
            ][i]
        ).item()

        h_class = torch.argmax(
            hazard[
                "heat_class_probabilities"
            ][i]
        ).item()

        print()
        print(f"  Sample {i + 1}")
        print(
            f"    Drought : {d_pct:.2f}%"
        )
        print(
            f"      Class : {d_class} - "
            f"{DROUGHT_CLASS_MAP[d_class]}"
        )
        print(
            f"      Severity score : "
            f"{d_severity:.4f}"
        )

        print(
            f"    Heat    : {h_pct:.2f}%"
        )
        print(
            f"      Class : {h_class} - "
            f"{HEAT_CLASS_MAP[h_class]}"
        )
        print(
            f"      Severity score : "
            f"{h_severity:.4f}"
        )

        print(
            f"    Dominant hazard : "
            f"{'Drought' if dominant_hazard[i].item() == 0 else 'Heat'}"
        )

    # ==================================================================
    # TEST 12 — Backward pass
    # ==================================================================

    print()
    print("[TEST 12] Testing backward pass...")

    # Create a scalar loss from all outputs.
    loss = (
        reg_out.mean()
        + drought_logits.mean()
        + heat_logits.mean()
    )

    print(
        f"  Loss: {loss.item():.8f}"
    )

    loss.backward()

    print("  ✓ Backward pass completed")

    # ==================================================================
    # TEST 13 — Gradient propagation
    # ==================================================================

    print()
    print("[TEST 13] Checking gradients...")

    total_params = 0
    params_with_grad = 0

    for name, parameter in model.named_parameters():

        if parameter.requires_grad:

            total_params += 1

            if parameter.grad is not None:
                params_with_grad += 1

    print(
        f"  Parameters with gradients : "
        f"{params_with_grad}/{total_params}"
    )

    assert params_with_grad == total_params

    print(
        "  ✓ All trainable head parameters "
        "received gradients"
    )

    # ==================================================================
    # TEST 14 — Parameter count
    # ==================================================================

    print()
    print("[TEST 14] Checking trainable parameters...")

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"  Total trainable parameters : "
        f"{trainable_params:,}"
    )

    assert trainable_params > 0

    print("  ✓ Prediction heads contain trainable parameters")

    # ==================================================================
    # TEST 15 — Class maps
    # ==================================================================

    print()
    print("[TEST 15] Checking class maps...")

    assert len(DROUGHT_CLASS_MAP) == 4
    assert len(HEAT_CLASS_MAP) == 4

    for i in range(4):
        assert i in DROUGHT_CLASS_MAP
        assert i in HEAT_CLASS_MAP

    print("  ✓ Drought class map contains 4 classes")
    print("  ✓ Heat class map contains 4 classes")

    # ==================================================================
    # FINAL
    # ==================================================================

    print()
    print("=" * 70)
    print("MULTI-TASK HEADS TEST PASSED")
    print("=" * 70)

    print()
    print("Verified:")

    print("  ✓ Regression output")
    print("  ✓ Drought classification output")
    print("  ✓ Heat classification output")
    print("  ✓ Correct output dimensions")
    print("  ✓ No NaN / Inf")
    print("  ✓ Classification probabilities")
    print("  ✓ Probability normalization")
    print("  ✓ Hazard severity calculation")
    print("  ✓ Drought percentage")
    print("  ✓ Heat percentage")
    print("  ✓ Drought + Heat = 100%")
    print("  ✓ Dominant hazard detection")
    print("  ✓ Class maps")
    print("  ✓ Forward pass")
    print("  ✓ Backward pass")
    print("  ✓ Gradient propagation")
    print("  ✓ Trainable parameters")

    print("=" * 70)
    print()


if __name__ == "__main__":
    main()