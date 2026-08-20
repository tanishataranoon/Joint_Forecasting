"""
Residual MLP branch encoder.

Applied independently at every time step with shared weights across
the T=30 day sequence.

Projects each branch's input features into the common D_MODEL
representation used by the fusion module.

Architecture:
    Linear
    -> LayerNorm
    -> GELU
    -> Dropout
    -> Linear
    -> LayerNorm
    -> GELU
    -> Residual addition

A linear shortcut is used so the residual connection works when
in_dim != out_dim.
"""

import torch
import torch.nn as nn


class BranchEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Residual/shortcut projection:
        # (B, T, in_dim) -> (B, T, out_dim)
        self.shortcut = nn.Linear(in_dim, out_dim)

        # Main MLP path
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, in_dim)

        Returns:
            Tensor of shape (B, T, out_dim)
        """

        # Residual path
        residual = self.shortcut(x)

        # Main path
        h = self.fc1(x)
        h = self.ln1(h)
        h = self.act1(h)
        h = self.drop(h)

        h = self.fc2(h)
        h = self.ln2(h)
        h = self.act2(h)

        # Residual addition
        return h + residual
