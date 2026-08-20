"""
Adaptive Cross-Attention Dynamic Temporal Fusion Module.

This module implements the thesis's primary technical contribution (TC1).

Four stages:

1. Linear projections:
   Query from Zm, Key/Value from [Zv; Ze]

2. Multi-head scaled dot-product cross-attention:
   Meteorological branch queries the vegetation + engineered branches.

3. Dynamic temporal gating:
   A bidirectional GRU processes [Zm; Zv; Ze; O] and produces
   four time-dependent weights:
       meteorological
       vegetation
       engineered
       cross-attention output

4. Residual adaptive fusion:
   The four representations are combined using the learned gate
   weights and stabilized with LayerNorm.

Attention weights and temporal gate weights are returned for
downstream explainability/analysis.
"""

import torch
import torch.nn as nn

from src.models.branch_encoder import BranchEncoder


class DynamicTemporalGate(nn.Module):
    """
    Produces four time-dependent fusion weights.

    Input:
        Zm, Zv, Ze, O
        each of shape (B, T, D_MODEL)

    Output:
        alphas of shape (B, T, 4)

    Gate order:
        0 -> meteorological
        1 -> vegetation
        2 -> engineered
        3 -> cross-attention output
    """

    def __init__(
        self,
        d_model: int,
        gate_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        gate_in_dim = 4 * d_model

        self.temporal = nn.GRU(
            input_size=gate_in_dim,
            hidden_size=gate_hidden // 2,
            batch_first=True,
            bidirectional=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(gate_hidden, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 4),
        )

    def forward(
        self,
        Zm: torch.Tensor,
        Zv: torch.Tensor,
        Ze: torch.Tensor,
        O: torch.Tensor,
    ) -> torch.Tensor:

        # (B, T, 4 * D_MODEL)
        gate_input = torch.cat(
            [Zm, Zv, Ze, O],
            dim=-1,
        )

        # (B, T, gate_hidden)
        gru_out, _ = self.temporal(gate_input)

        # (B, T, 4)
        raw_scores = self.mlp(gru_out)

        # Four weights sum to 1 at every timestep
        alphas = torch.softmax(
            raw_scores,
            dim=-1,
        )

        return alphas


class AdaptiveCrossAttentionFusion(nn.Module):
    """
    Adaptive cross-attention fusion.

    Meteorological representation Zm provides the query.

    Vegetation and engineered representations are concatenated
    and provide keys and values.

    Returns:
        F     -> fused representation
        attn  -> multi-head attention weights
        alphas -> dynamic temporal gate weights
    """

    def __init__(
        self,
        d_model: int,
        d_k: int,
        num_heads: int = 4,
        gate_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Kept to preserve the thesis configuration.
        # PyTorch MultiheadAttention derives the actual head dimension
        # as d_model // num_heads.
        self.d_k = d_k

        # ---------------------------------------------------------
        # Step 1: Linear projections
        # ---------------------------------------------------------

        self.q_proj = nn.Linear(
            d_model,
            d_model,
        )

        self.k_proj = nn.Linear(
            2 * d_model,
            d_model,
        )

        self.v_proj = nn.Linear(
            2 * d_model,
            d_model,
        )

        # ---------------------------------------------------------
        # Step 2: Multi-head cross-attention
        # ---------------------------------------------------------

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ---------------------------------------------------------
        # Step 3: Dynamic temporal gate
        # ---------------------------------------------------------

        self.gate = DynamicTemporalGate(
            d_model=d_model,
            gate_hidden=gate_hidden,
            dropout=dropout,
        )

        # ---------------------------------------------------------
        # Step 4: Fusion normalization
        # ---------------------------------------------------------

        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        Zm: torch.Tensor,
        Zv: torch.Tensor,
        Ze: torch.Tensor,
    ):
        """
        Args:
            Zm: Meteorological branch
                (B, T, D_MODEL)

            Zv: Vegetation branch
                (B, T, D_MODEL)

            Ze: Engineered branch
                (B, T, D_MODEL)

        Returns:
            F:
                (B, T, D_MODEL)

            attn:
                (B, NUM_HEADS, T, T)

            alphas:
                (B, T, 4)
        """

        # ---------------------------------------------------------
        # Step 1: Projections
        # ---------------------------------------------------------

        Q = self.q_proj(Zm)

        KV = torch.cat(
            [Zv, Ze],
            dim=-1,
        )

        K = self.k_proj(KV)
        V = self.v_proj(KV)

        # ---------------------------------------------------------
        # Step 2: Multi-head cross-attention
        # ---------------------------------------------------------

        O, attn = self.cross_attention(
            query=Q,
            key=K,
            value=V,
            need_weights=True,
            average_attn_weights=False,
        )

        # ---------------------------------------------------------
        # Step 3: Dynamic temporal gating
        # ---------------------------------------------------------

        alphas = self.gate(
            Zm,
            Zv,
            Ze,
            O,
        )

        alpha_m = alphas[..., 0:1]
        alpha_v = alphas[..., 1:2]
        alpha_e = alphas[..., 2:3]
        alpha_o = alphas[..., 3:4]

        # ---------------------------------------------------------
        # Step 4: Fully gated adaptive fusion
        # ---------------------------------------------------------

        F_t = alpha_m * Zm + alpha_v * Zv + alpha_e * Ze + alpha_o * O

        F = self.fusion_norm(F_t)

        return F, attn, alphas


class BranchFusionBlock(nn.Module):
    """
    Complete three-branch encoding + adaptive fusion block.

    Input:
        meteorological -> 7 features
        vegetation     -> 2 features
        engineered     -> 5 features

    Each branch is projected to D_MODEL and then passed through
    adaptive cross-attention fusion.

    Output:
        F -> fused sequence representation
    """

    def __init__(
        self,
        met_dim: int,
        veg_dim: int,
        eng_dim: int,
        d_model: int,
        d_k: int,
        num_heads: int = 4,
        gate_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.met_encoder = BranchEncoder(
            met_dim,
            d_model,
            d_model,
            dropout,
        )

        self.veg_encoder = BranchEncoder(
            veg_dim,
            d_model,
            d_model,
            dropout,
        )

        self.eng_encoder = BranchEncoder(
            eng_dim,
            d_model,
            d_model,
            dropout,
        )

        self.fusion = AdaptiveCrossAttentionFusion(
            d_model=d_model,
            d_k=d_k,
            num_heads=num_heads,
            gate_hidden=gate_hidden,
            dropout=dropout,
        )

    def forward(
        self,
        x_met: torch.Tensor,
        x_veg: torch.Tensor,
        x_eng: torch.Tensor,
    ):
        """
        Args:
            x_met: (B, T, met_dim)
            x_veg: (B, T, veg_dim)
            x_eng: (B, T, eng_dim)

        Returns:
            F:
                (B, T, D_MODEL)

            attn:
                (B, NUM_HEADS, T, T)

            alphas:
                (B, T, 4)
        """

        Zm = self.met_encoder(x_met)
        Zv = self.veg_encoder(x_veg)
        Ze = self.eng_encoder(x_eng)

        F, attn, alphas = self.fusion(
            Zm,
            Zv,
            Ze,
        )

        return F, attn, alphas
