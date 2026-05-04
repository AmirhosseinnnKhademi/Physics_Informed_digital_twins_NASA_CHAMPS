"""Shared PyTorch model definitions."""
import torch
import torch.nn as nn
import numpy as np


class CNNGRUModel(nn.Module):
    """1D-CNN feature extractor followed by a GRU temporal model for RUL regression.

    Architecture (roadmap recommendation):
      Conv1D x 2  ->  GRU x num_layers  ->  FC head  ->  scalar RUL

    The 1D-CNN extracts local degradation patterns within the window; the GRU
    captures the temporal ordering of those patterns across the window.
    """

    def __init__(self, input_size: int, cnn_channels: int = 64,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            cnn_channels, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, F)
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, W, cnn_channels)
        out, _ = self.gru(x)                                 # (B, W, hidden)
        return self.head(out[:, -1, :]).squeeze(-1)          # (B,)


class RULWindowDataset(torch.utils.data.Dataset):
    """Sliding-window RUL dataset over a set of engines.

    Pass a DataFrame already filtered to the desired engines (train or val or
    test). Windows are built per engine over all available cycles.
    Window i covers cycles [i-W, i) and targets RUL at cycle i-1.
    """

    def __init__(self, cycle_df, feature_cols, window_size):
        self.X, self.y = [], []
        for _, unit_df in cycle_df.groupby("unit", sort=True):
            unit_df = unit_df.sort_values("cycle").reset_index(drop=True)
            feats   = unit_df[feature_cols].values.astype(np.float32)
            ruls    = unit_df["RUL"].values.astype(np.float32)
            n       = len(unit_df)
            for i in range(window_size, n + 1):
                self.X.append(feats[i - window_size: i])
                self.y.append(ruls[i - 1])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class SimpleGRUModel(nn.Module):
    """Lightweight GRU for RUL regression on small datasets.

    Predicts normalised RUL (target / rul_cap) so gradients stay in [0, 1].
    Architecture: GRU(hidden) → Dropout → Linear(dense) → ReLU → Linear(1)
    """

    def __init__(self, input_size: int, hidden_size: int = 32,
                 dense_size: int = 32, dropout: float = 0.2):
        super().__init__()
        self.gru  = nn.GRU(input_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden_size, dense_size)
        self.fc2  = nn.Linear(dense_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)                    # h: (1, B, hidden)
        h = self.drop(h.squeeze(0))           # (B, hidden)
        h = torch.relu(self.fc1(h))
        return self.fc2(h).squeeze(-1)        # (B,)
