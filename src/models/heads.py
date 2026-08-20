"""
Multi-task prediction heads and complete forecasting model.

The complete architecture is:

    Meteorological Branch
            |
    Vegetation Branch
            |----> Branch Encoders
    Engineered Branch
            |
            v
    Adaptive Cross-Attention Fusion
            |
            v
    Temporal Backbone
            |
            v
    Multi-Task Prediction Heads
            |
            +---- Regression
            |       ├── SPI3
            |       └── Humidex
            |
            +---- Drought Classification
            |       └── 4 classes
            |
            +---- Heat Classification
                    └── 4 classes

The three temporal backbones are independent models:

    1. BiLSTM + Attention
    2. CNN + BiLSTM + Attention
    3. TFT

The prediction heads remain identical across all three models.

Hazard percentages are calculated from the predicted class
probabilities using severity-weighted hazard scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# CLASS MAPS
# ======================================================================

DROUGHT_CLASS_MAP = {
    0: "Normal/Wet",
    1: "Moderate",
    2: "Severe",
    3: "Extreme drought",
}


HEAT_CLASS_MAP = {
    0: "No significant discomfort",
    1: "Some discomfort",
    2: "Great discomfort (avoid exertion)",
    3: "Dangerous (heat stroke possible)",
}


# ======================================================================
# HAZARD PERCENTAGE CALCULATION
# ======================================================================

def calculate_hazard_percentages(
    drought_logits: torch.Tensor,
    heat_logits: torch.Tensor,
):
    """
    Convert drought and heat classification logits into
    relative hazard-severity percentages.

    IMPORTANT
    ---------
    These are NOT independent probabilities.

    The drought and heat classifiers each have their own
    4-class probability distribution.

    We first calculate an expected severity score:

        severity = sum(class_index * P(class)) / 3

    This produces a value between 0 and 1.

    The drought and heat severity scores are then normalized
    against each other to produce:

        Drought : XX%
        Heat    : YY%

    Example:

        Drought : 60%
        Heat    : 40%

    Interpretation:
        Drought contributes approximately 60% of the predicted
        combined hazard severity, while heat contributes 40%.

    Parameters
    ----------
    drought_logits : torch.Tensor
        Shape (B, 4)

    heat_logits : torch.Tensor
        Shape (B, 4)

    Returns
    -------
    dictionary containing:

        drought_class_probabilities
        heat_class_probabilities
        drought_severity
        heat_severity
        drought_percentage
        heat_percentage
        dominant_hazard
    """

    # --------------------------------------------------------------
    # Convert logits to class probabilities
    # --------------------------------------------------------------

    drought_probs = F.softmax(
        drought_logits,
        dim=-1,
    )

    heat_probs = F.softmax(
        heat_logits,
        dim=-1,
    )

    # --------------------------------------------------------------
    # Severity class indices
    #
    # 0 = normal / no significant hazard
    # 1 = moderate
    # 2 = severe
    # 3 = extreme
    # --------------------------------------------------------------

    severity_levels = torch.tensor(
        [0.0, 1.0, 2.0, 3.0],
        device=drought_logits.device,
        dtype=drought_logits.dtype,
    )

    severity_levels = severity_levels.unsqueeze(0)

    # --------------------------------------------------------------
    # Expected severity
    # --------------------------------------------------------------

    drought_severity = (
        drought_probs * severity_levels
    ).sum(dim=-1) / 3.0

    heat_severity = (
        heat_probs * severity_levels
    ).sum(dim=-1) / 3.0

    # --------------------------------------------------------------
    # Relative hazard contribution
    # --------------------------------------------------------------

    total_severity = (
        drought_severity
        + heat_severity
    )

    # Avoid division by zero.
    denominator = torch.where(
        total_severity > 1e-8,
        total_severity,
        torch.ones_like(total_severity),
    )

    drought_percentage = (
        drought_severity
        / denominator
        * 100.0
    )

    heat_percentage = (
        heat_severity
        / denominator
        * 100.0
    )

    # --------------------------------------------------------------
    # If both hazards have zero predicted severity,
    # neither hazard dominates.
    #
    # Return 50 / 50 rather than producing an artificial
    # 0 / 0 result.
    # --------------------------------------------------------------

    no_hazard = total_severity <= 1e-8

    drought_percentage = torch.where(
        no_hazard,
        torch.full_like(drought_percentage, 50.0),
        drought_percentage,
    )

    heat_percentage = torch.where(
        no_hazard,
        torch.full_like(heat_percentage, 50.0),
        heat_percentage,
    )

    # --------------------------------------------------------------
    # Dominant hazard
    # --------------------------------------------------------------

    dominant_hazard = torch.where(
        drought_percentage >= heat_percentage,
        torch.zeros_like(drought_percentage, dtype=torch.long),
        torch.ones_like(heat_percentage, dtype=torch.long),
    )

    return {
        "drought_class_probabilities": drought_probs,
        "heat_class_probabilities": heat_probs,

        "drought_severity": drought_severity,
        "heat_severity": heat_severity,

        "drought_percentage": drought_percentage,
        "heat_percentage": heat_percentage,

        # 0 = drought
        # 1 = heat
        "dominant_hazard": dominant_hazard,
    }


# ======================================================================
# MULTI-TASK HEADS
# ======================================================================

class MultiTaskHeads(nn.Module):
    """
    Shared prediction heads used by every temporal backbone.

    Tasks
    -----

    Regression:
        SPI3
        Humidex

    Classification:
        Drought severity -> 4 classes
        Heat severity    -> 4 classes
    """

    def __init__(
        self,
        in_dim: int,
        n_reg_targets: int = 2,
        n_classes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dim = in_dim // 2

        # ----------------------------------------------------------
        # Regression
        # ----------------------------------------------------------

        self.regression_head = nn.Sequential(
            nn.Linear(
                in_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                n_reg_targets,
            ),
        )

        # ----------------------------------------------------------
        # Drought classification
        # ----------------------------------------------------------

        self.drought_head = nn.Sequential(
            nn.Linear(
                in_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                n_classes,
            ),
        )

        # ----------------------------------------------------------
        # Heat classification
        # ----------------------------------------------------------

        self.heat_head = nn.Sequential(
            nn.Linear(
                in_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                n_classes,
            ),
        )

    def forward(
        self,
        pooled: torch.Tensor,
    ):
        """
        Parameters
        ----------
        pooled : torch.Tensor
            Sample-level temporal representation.

            Shape:
                (B, in_dim)

        Returns
        -------
        reg_out :
            (B, 2)

        drought_logits :
            (B, 4)

        heat_logits :
            (B, 4)
        """

        reg_out = self.regression_head(
            pooled
        )

        drought_logits = self.drought_head(
            pooled
        )

        heat_logits = self.heat_head(
            pooled
        )

        return (
            reg_out,
            drought_logits,
            heat_logits,
        )


# ======================================================================
# COMPLETE FORECASTING MODEL
# ======================================================================

class HazardForecastModel(nn.Module):
    """
    Complete forecasting architecture.

    Pipeline:

        Meteorological input
                  |
        Vegetation input
                  |
        Engineered input
                  |
                  v
        BranchFusionBlock
                  |
                  v
        Adaptive Cross-Attention
                  |
                  v
        Temporal Backbone
                  |
                  v
        MultiTaskHeads
                  |
          +-------+-------+
          |       |       |
        SPI3   Drought   Heat
               class     class

    The temporal backbone is injected so that the same wrapper
    can independently build:

        BiLSTM + Attention
        CNN + BiLSTM + Attention
        TFT
    """

    def __init__(
        self,
        branch_fusion_block: nn.Module,
        temporal_backbone: nn.Module,
        n_reg_targets: int = 2,
        n_classes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ----------------------------------------------------------
        # Multi-branch adaptive fusion
        # ----------------------------------------------------------

        self.branch_fusion = branch_fusion_block

        # ----------------------------------------------------------
        # Temporal modeling
        # ----------------------------------------------------------

        self.temporal_backbone = temporal_backbone

        # ----------------------------------------------------------
        # Prediction heads
        # ----------------------------------------------------------

        self.heads = MultiTaskHeads(
            in_dim=temporal_backbone.out_dim,
            n_reg_targets=n_reg_targets,
            n_classes=n_classes,
            dropout=dropout,
        )

    def forward(
        self,
        x_met: torch.Tensor,
        x_veg: torch.Tensor,
        x_eng: torch.Tensor,
    ):
        """
        Parameters
        ----------
        x_met :
            (B, T, met_dim)

        x_veg :
            (B, T, veg_dim)

        x_eng :
            (B, T, eng_dim)

        Returns
        -------
        Dictionary containing all outputs needed for:

            training
            evaluation
            explainability
            final prediction
        """

        # ==========================================================
        # 1. Adaptive branch fusion
        # ==========================================================

        fused, cross_attn, fusion_alphas = (
            self.branch_fusion(
                x_met,
                x_veg,
                x_eng,
            )
        )

        # ==========================================================
        # 2. Temporal modeling
        # ==========================================================

        temporal_output = self.temporal_backbone(
            fused
        )

        # ==========================================================
        # 3. Extract pooled representation
        # ==========================================================

        if isinstance(
            temporal_output,
            dict,
        ):
            pooled = temporal_output["pooled"]
        else:
            pooled = temporal_output

        # ==========================================================
        # 4. Multi-task prediction heads
        # ==========================================================

        (
            reg_out,
            drought_logits,
            heat_logits,
        ) = self.heads(
            pooled
        )

        # ==========================================================
        # 5. Hazard probability / severity calculation
        # ==========================================================

        hazard_info = calculate_hazard_percentages(
            drought_logits,
            heat_logits,
        )

        # ==========================================================
        # 6. Final predicted classes
        # ==========================================================

        drought_class = torch.argmax(
            drought_logits,
            dim=-1,
        )

        heat_class = torch.argmax(
            heat_logits,
            dim=-1,
        )

        # ==========================================================
        # 7. Final output
        # ==========================================================

        return {
            # ------------------------------------------------------
            # Regression
            # ------------------------------------------------------

            "reg_out": reg_out,

            # ------------------------------------------------------
            # Classification logits
            # ------------------------------------------------------

            "drought_logits": drought_logits,
            "heat_logits": heat_logits,

            # ------------------------------------------------------
            # Classification probabilities
            # ------------------------------------------------------

            "drought_class_probabilities":
                hazard_info[
                    "drought_class_probabilities"
                ],

            "heat_class_probabilities":
                hazard_info[
                    "heat_class_probabilities"
                ],

            # ------------------------------------------------------
            # Predicted classes
            # ------------------------------------------------------

            "drought_class": drought_class,
            "heat_class": heat_class,

            # ------------------------------------------------------
            # Hazard severity
            # ------------------------------------------------------

            "drought_severity":
                hazard_info[
                    "drought_severity"
                ],

            "heat_severity":
                hazard_info[
                    "heat_severity"
                ],

            # ------------------------------------------------------
            # Final relative hazard percentages
            # ------------------------------------------------------

            "drought_percentage":
                hazard_info[
                    "drought_percentage"
                ],

            "heat_percentage":
                hazard_info[
                    "heat_percentage"
                ],

            "dominant_hazard":
                hazard_info[
                    "dominant_hazard"
                ],

            # ------------------------------------------------------
            # Fusion explainability
            # ------------------------------------------------------

            "cross_attention":
                cross_attn,

            "fusion_alphas":
                fusion_alphas,

            # ------------------------------------------------------
            # Temporal explainability
            # ------------------------------------------------------

            "temporal_output":
                temporal_output,

            # ------------------------------------------------------
            # Fused representation
            # ------------------------------------------------------

            "fused_representation":
                fused,
        }