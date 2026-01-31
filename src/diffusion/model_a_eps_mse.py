## Step A0.3 — Implement FINAL Model A in `src/diffusion/model_a_eps_mse.py` (sequence-aware 1D-CNN)

#Model A:
#- Training objective: ε-prediction (DDPM forward noising).
#- Architecture: lightweight temporal Conv1D network (sequence-aware).
#- Score: mean ε-MSE over selected timesteps (efficiency-controlled).
#- Output: one anomaly score per window (aligned to window end time via meta in notebook).


from __future__ import annotations

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from .schedules import cosine_beta_schedule


class TimestepEmbed(nn.Module):
    """
    Simple embedding table for discrete diffusion timesteps.
    """
    def __init__(self, max_T: int, emb_dim: int):
        super().__init__()
        self.emb = nn.Embedding(int(max_T), int(emb_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.emb(t)  # (B, emb_dim)


class EpsilonNet1D(nn.Module):
    """
    Sequence-aware epsilon predictor using Conv1D over time.

    Input:
      x_t: (B, L, 1)
      t:   (B,)
    Output:
      eps_hat: (B, L, 1)

    Conditioning:
      timestep embedding is expanded over the time axis and concatenated as extra channels.
    """
    def __init__(self, L: int, T: int, width: int, depth: int, t_emb_dim: int, dropout: float):
        super().__init__()
        self.L = int(L)
        self.T = int(T)
        self.t_emb_dim = int(t_emb_dim)

        self.t_embed = TimestepEmbed(max_T=max(self.T, 1), emb_dim=self.t_emb_dim)

        in_ch = 1 + self.t_emb_dim
        hidden = int(width)
        n_blocks = int(depth)

        layers = []
        for i in range(n_blocks):
            layers.append(nn.Conv1d(in_ch if i == 0 else hidden, hidden, kernel_size=3, padding=1))
            layers.append(nn.SiLU())
            if dropout and float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))

        # project back to 1 channel (epsilon per time step)
        layers.append(nn.Conv1d(hidden, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t: (B, L, 1) -> (B, 1, L)
        x = x_t.permute(0, 2, 1)

        # t embedding: (B, t_emb_dim) -> (B, t_emb_dim, L)
        t_emb = self.t_embed(t).unsqueeze(-1).repeat(1, 1, x.shape[-1])

        # concat channels -> (B, 1+t_emb_dim, L)
        h = torch.cat([x, t_emb], dim=1)

        y = self.net(h)  # (B, 1, L)
        return y.permute(0, 2, 1)  # (B, L, 1)


def _prep_windows(X: np.ndarray) -> np.ndarray:
    """
    Ensure windows are float32 with shape (n, L, 1).
    Accepts (n, L) or (n, L, 1).
    """
    X_in = X
    if X_in.ndim == 2:
        X_in = X_in[:, :, None]
    if X_in.ndim != 3 or X_in.shape[-1] != 1:
        raise ValueError(f"Expected X shape (n, L) or (n, L, 1); got {X_in.shape}")
    return np.asarray(X_in, dtype=np.float32)


def train_model_A(X_train: np.ndarray, config: dict):
    """
    Train Model A (sequence-aware DDPM ε-prediction).

    Required config keys:
      - device ("cpu" or "cuda")
      - seed
      - n_diffusion_steps (T)
      - schedule ("cosine" supported here)
      - epochs
      - batch_size
      - learning_rate
      - model_width
      - model_depth
      - t_emb_dim
      - dropout
    """
    device = str(config.get("device", "cpu"))
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    T = int(config["n_diffusion_steps"])
    schedule = str(config.get("schedule", "cosine"))
    if schedule != "cosine":
        raise ValueError("Model A currently supports schedule='cosine' only (for parity and simplicity).")

    betas = cosine_beta_schedule(T)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)

    # tensors for q(x_t | x_0)
    alphas_cumprod_t = torch.tensor(alphas_cumprod, dtype=torch.float32, device=device)  # (T,)

    X_np = _prep_windows(X_train)
    n, L, _ = X_np.shape

    X = torch.tensor(X_np, dtype=torch.float32, device=device)  # (n, L, 1)

    model = EpsilonNet1D(
        L=L,
        T=T,
        width=int(config.get("model_width", 64)),
        depth=int(config.get("model_depth", 3)),
        t_emb_dim=int(config.get("t_emb_dim", 32)),
        dropout=float(config.get("dropout", 0.0)),
    ).to(device)

    lr = float(config.get("learning_rate", 1e-3))
    opt = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    epochs = int(config.get("epochs", 10))
    batch_size = int(config.get("batch_size", 256))

    history_rows = []
    model.train()

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        losses = []

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            x0 = X[idx]  # (B, L, 1)

            B = x0.shape[0]
            t = torch.randint(low=0, high=T, size=(B,), device=device, dtype=torch.long)  # (B,)
            eps = torch.randn_like(x0)  # (B, L, 1)

            a_t = alphas_cumprod_t[t].view(B, 1, 1)  # (B, 1, 1)
            x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1.0 - a_t) * eps

            eps_hat = model(x_t, t)
            loss = loss_fn(eps_hat, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            losses.append(float(loss.detach().cpu().item()))

        history_rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "n_windows": int(n), "T": int(T)})

    history_df = pd.DataFrame(history_rows)

    # attach minimal metadata for router/backward-compat
    model.model_variant = "A"
    model.score_timesteps = list(config.get("score_timesteps", [int(0.2*(T-1)), int(0.4*(T-1)), int(0.6*(T-1)), int(0.8*(T-1)), int(0.95*(T-1))]))
    model.n_diffusion_steps = int(T)
    model.schedule = schedule
    model.seed = seed

    return model, history_df


@torch.no_grad()
def score_model_A(model: nn.Module, X: np.ndarray, batch_size: int, config: dict | None = None) -> np.ndarray:
    """
    Score windows by mean ε-MSE over selected timesteps.

    Uses:
      - timesteps from config["score_timesteps"] if provided
      - else from model.score_timesteps if attached during training
    """
    device = next(model.parameters()).device

    # get T and timesteps
    if config is not None:
        T = int(config.get("n_diffusion_steps", getattr(model, "n_diffusion_steps", 50)))
        timesteps = list(config.get("score_timesteps", getattr(model, "score_timesteps", [10, 20, 30, 40, 49])))
        schedule = str(config.get("schedule", getattr(model, "schedule", "cosine")))
    else:
        T = int(getattr(model, "n_diffusion_steps", 50))
        timesteps = list(getattr(model, "score_timesteps", [10, 20, 30, 40, 49]))
        schedule = str(getattr(model, "schedule", "cosine"))

    if schedule != "cosine":
        raise ValueError("Model A currently supports schedule='cosine' only.")

    betas = cosine_beta_schedule(T)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    alphas_cumprod_t = torch.tensor(alphas_cumprod, dtype=torch.float32, device=device)  # (T,)

    X_np = _prep_windows(X)
    n, L, _ = X_np.shape
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)  # (n, L, 1)

    model.eval()

    scores_out = np.zeros(n, dtype=np.float32)
    bs = int(batch_size)

    # deterministic-ish scoring RNG (repeatable within a run)
    #gen = torch.Generator(device=device)
   # gen.manual_seed(12345)
    torch.manual_seed(12345)

    idx0 = 0
    for i in range(0, n, bs):
        x0 = X_t[i : i + bs]
        B = x0.shape[0]

        per_t = []
        for t_val in timesteps:
            t_val = int(t_val)
            if t_val < 0 or t_val > (T - 1):
                raise ValueError(f"score_timesteps contains invalid t={t_val} for T={T}")

            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            #eps = torch.randn_like(x0, generator=gen)
            eps = torch.randn_like(x0)

            a_t = alphas_cumprod_t[t].view(B, 1, 1)
            x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1.0 - a_t) * eps

            eps_hat = model(x_t, t)
            mse = torch.mean((eps_hat - eps) ** 2, dim=(1, 2))  # (B,)
            per_t.append(mse)

        score_batch = torch.stack(per_t, dim=1).mean(dim=1)  # (B,)
        scores_out[idx0 : idx0 + B] = score_batch.detach().cpu().numpy()
        idx0 += B

    return scores_out
