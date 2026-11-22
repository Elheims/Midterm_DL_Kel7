"""Task 2, part (d): deep sequence models. Input: a batch of windows (batch, W, F). Output: RUL/150.

All models end with dropout + a small dense head, so the same MC-dropout procedure (part f) works
for every architecture.
"""
from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    """1D convolutional network (in the spirit of Babu et al. 2016; Li et al. 2018).

    - Conv layers slide filters of `k` cycles along the time axis; each filter sees all sensors and
      learns a local pattern (e.g. "s04 rising while s12 rises"). Stacking layers combines simple
      patterns into larger ones; 'same' padding keeps the time length W.
    - head="flatten": the feature map (channels x W) is flattened, so the head knows *when* in the
      window a pattern occurred; most parameters sit in this layer (channels*W*hidden).
    - head="gap": global average pooling - each filter's output is averaged over time, giving one
      number per filter ("how strongly is this pattern present in the window"). Far fewer
      parameters, so less room to memorise individual machines.
    """

    def __init__(self, n_feat: int, W: int, channels=(32, 64, 64), k: int = 5, hidden: int = 64,
                 dropout: float = 0.2, head: str = "flatten"):
        super().__init__()
        layers, c_in = [], n_feat
        for c in channels:
            layers += [nn.Conv1d(c_in, c, k, padding=k // 2), nn.ReLU()]
            c_in = c
        self.conv = nn.Sequential(*layers)
        if head == "flatten":
            pool, n_in = [nn.Flatten()], c_in * W
        elif head == "gap":
            pool, n_in = [nn.AdaptiveAvgPool1d(1), nn.Flatten()], c_in
        else:
            raise ValueError(head)
        self.head = nn.Sequential(
            *pool, nn.Dropout(dropout), nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (batch, W, F)
        h = self.conv(x.transpose(1, 2))                    # Conv1d wants (batch, channels=F, time=W)
        return self.head(h).squeeze(-1)


class GRUNet(nn.Module):
    """Recurrent network with GRU cells (Cho et al. 2014; LSTM version for RUL: Zheng et al. 2017).

    - Reads the window one cycle at a time, in order. It keeps a hidden state (`hidden` numbers),
      a running memory of what it has seen. At every cycle two gates decide how much of the old
      memory to keep (update gate) and how much of it to use when writing new information
      (reset gate).
    - Two GRU layers are stacked: the second reads the sequence of memories of the first.
    - After the last cycle of the window, the final memory goes through the same small dense head
      as the CNN to give one number.
    A GRU is a lighter LSTM (2 gates instead of 3) with similar accuracy in practice.
    """

    def __init__(self, n_feat: int, W: int, hidden: int = 64, layers: int = 2, head_hidden: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, head_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (batch, W, F)
        out, _ = self.gru(x)                                # out: (batch, W, hidden), memory after each cycle
        return self.head(out[:, -1]).squeeze(-1)            # memory after the last cycle


def build(arch: str, n_feat: int, W: int, **kw) -> nn.Module:
    if arch == "cnn":
        return CNN1D(n_feat, W, **kw)
    if arch == "gru":
        return GRUNet(n_feat, W, **kw)
    raise ValueError(f"unknown architecture {arch}")


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
