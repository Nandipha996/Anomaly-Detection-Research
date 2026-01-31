

import numpy as np

def cosine_beta_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """
    Cosine beta schedule (Nichol & Dhariwal style), returns betas shape (T,).
    """
    steps = np.arange(T + 1, dtype=np.float64)
    alphas_cumprod = np.cos(((steps / T) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-5, 0.999)
