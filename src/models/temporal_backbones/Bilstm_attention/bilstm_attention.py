"""
BiLSTM + Temporal Attention backbone.

Input:
    F: (B, T, D_MODEL)

With the current configuration:
    T = 30
    D_MODEL = 64

The BiLSTM processes the complete historical sequence and the
additive temporal attention layer learns a weighted representation
of the historical timesteps.

Output:
    pooled:
        (B, 2 * hidden_dim)

    temporal_attn:
        (B, T)

The temporal attention weights are retained for later
interpretability/analysis.
"""

import torch
import torch.nn as nn


class TemporalAttentionPool(nn.Module):
    """
    Additive temporal attention pooling.

    Input:
        seq: (B, T, H)

    Output:
        pooled:   (B, H)
        weights:  (B, T)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, seq: torch.Tensor):
        # ---------------------------------------------------------
        # Attention scores
        # ---------------------------------------------------------

        # (B, T, H) -> (B, T, 1)
        scores = self.attn(seq)

        # ---------------------------------------------------------
        # Normalize across time
        # ---------------------------------------------------------

        # (B, T, 1)
        weights = torch.softmax(scores, dim=1)

        # ---------------------------------------------------------
        # Weighted temporal pooling
        # ---------------------------------------------------------

        # (B, T, H) * (B, T, 1)
        # -> sum over T
        # -> (B, H)
        pooled = (seq * weights).sum(dim=1)

        return pooled, weights.squeeze(-1)


class BiLSTMAttentionBackbone(nn.Module):
    """
    Bidirectional LSTM followed by additive temporal attention.

    Parameters
    ----------
    d_model:
        Input feature dimension from the adaptive fusion module.

    hidden_dim:
        Hidden dimension of each LSTM direction.

    num_layers:
        Number of stacked BiLSTM layers.

    dropout:
        Dropout probability.

    Input
    -----
    F:
        (B, T, d_model)

    Output
    ------
    dict:
        pooled:
            (B, 2 * hidden_dim)

        temporal_attn:
            (B, T)
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Bidirectional LSTM
        # ---------------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ---------------------------------------------------------
        # BiLSTM output dimension
        #
        # One direction = hidden_dim
        # Two directions = 2 * hidden_dim
        # ---------------------------------------------------------

        self.out_dim = hidden_dim * 2

        # ---------------------------------------------------------
        # Temporal attention pooling
        # ---------------------------------------------------------

        self.pool = TemporalAttentionPool(
            hidden_dim=self.out_dim
        )

    def forward(self, F: torch.Tensor):
        """
        Parameters
        ----------
        F:
            Fused sequence representation.

            Shape:
                (B, T, D_MODEL)

        Returns
        -------
        dict
            pooled:
                (B, out_dim)

            temporal_attn:
                (B, T)
        """

        # ---------------------------------------------------------
        # BiLSTM
        # ---------------------------------------------------------

        seq_out, _ = self.lstm(F)

        # seq_out:
        # (B, T, 2 * hidden_dim)

        # ---------------------------------------------------------
        # Temporal attention
        # ---------------------------------------------------------

        pooled, temporal_attn = self.pool(seq_out)

        return {
            "pooled": pooled,
            "temporal_attn": temporal_attn,
        }