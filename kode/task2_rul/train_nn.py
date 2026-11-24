"""Task 2, part (d): shared training code for the deep models.

- GPUWindows: keeps the stored rows on the GPU and cuts windows out per batch (same layout as
  windows.Windows, so no window is ever copied in full).
- fit(): mini-batch training with Adam on MSE of RUL/150, early stopping on held-aside training
  machines (never on the ruler machines), learning-rate halving on plateaus, best epoch restored.
- predict(): RUL in cycles, clipped to [0, 150]. mc=True keeps dropout on (used in part f).
- overfit_one_batch(): the sanity check - a correct network must memorise a handful of windows.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from task2_rul.labels import RUL_CAP
from task2_rul.windows import Windows

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GPUWindows:
    def __init__(self, win: Windows, device=DEVICE):
        self.W = win.W
        self.data = torch.from_numpy(win.data).to(device)
        self.ends = torch.from_numpy(win.ends.astype(np.int64)).to(device)
        self.y = None if win.y is None else torch.from_numpy(win.y / RUL_CAP).float().to(device)
        self.offs = torch.arange(-win.W + 1, 1, device=device)
        self.y_cycles = win.y

    def __len__(self) -> int:
        return len(self.ends)

    def X(self, idx: torch.Tensor) -> torch.Tensor:
        return self.data[self.ends[idx][:, None] + self.offs]       # (batch, W, F)


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch: int = 512
    max_epochs: int = 40
    patience: int = 6           # early stopping: epochs without improvement
    lr_patience: int = 3        # halve the learning rate after this many epochs without improvement
    weight_decay: float = 0.0


@torch.no_grad()
def predict(model: nn.Module, data: GPUWindows, batch: int = 8192, mc: bool = False) -> np.ndarray:
    model.train(mc)  # mc=True: dropout stays on (MC dropout); otherwise eval mode
    out = []
    for s in range(0, len(data), batch):
        idx = torch.arange(s, min(s + batch, len(data)), device=data.ends.device)
        out.append(model(data.X(idx)).float().cpu().numpy())
    model.eval()
    return np.clip(np.concatenate(out) * RUL_CAP, 0, RUL_CAP)


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def fit(model: nn.Module, train: GPUWindows, stop: GPUWindows, cfg: TrainConfig, seed: int,
        log=print) -> tuple[nn.Module, list[dict]]:
    """Train; early-stop on RMSE (cycles) of the `stop` windows. Returns best model and history."""
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=cfg.lr_patience)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    loss_fn = nn.MSELoss()
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(train), device=DEVICE, generator=gen)
        tot, n = 0.0, 0
        for s in range(0, len(perm), cfg.batch):
            idx = perm[s:s + cfg.batch]
            loss = loss_fn(model(train.X(idx)), train.y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        train_rmse = float(np.sqrt(tot / n)) * RUL_CAP          # with dropout on
        stop_rmse = rmse(predict(model, stop), stop.y_cycles)
        sched.step(stop_rmse)
        hist.append({"epoch": ep, "train_rmse": round(train_rmse, 3), "stop_rmse": round(stop_rmse, 3),
                     "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)})
        log(f"    epoch {ep:2d}  train RMSE {train_rmse:6.2f}  early-stop RMSE {stop_rmse:6.2f}  "
            f"lr {opt.param_groups[0]['lr']:.1e}  {time.time() - t0:4.1f}s")
        if stop_rmse < best - 1e-3:
            best, best_state, wait = stop_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


def overfit_one_batch(model: nn.Module, train: GPUWindows, n: int = 64, steps: int = 1500,
                      lr: float = 1e-3, seed: int = 0) -> list[float]:
    """Train on one fixed batch of n windows with dropout off; returns RMSE (cycles) per step."""
    model.to(DEVICE)
    for m in model.modules():                 # dropout off: we test capacity + gradient flow only
        if isinstance(m, nn.Dropout):
            m.p = 0.0
        if isinstance(m, nn.RNNBase):         # the GRU's built-in dropout between its layers
            m.dropout = 0.0
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    idx = torch.randperm(len(train), device=DEVICE, generator=g)[:n]
    x, y = train.X(idx), train.y[idx]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    model.train()
    for _ in range(steps):
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        curve.append(float(loss.item()) ** 0.5 * RUL_CAP)
    return curve
