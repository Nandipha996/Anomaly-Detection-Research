# src/diffusion_model.py

# Step A0.4 — Convert `src/diffusion_model.py` into a thin router (stable notebook API)

#This keeps the notebook import unchanged:
#`from src.diffusion_model import train_diffusion, score_diffusion`

#It also keeps scoring backward-compatible:
#- notebook may call `score_diffusion(model, X, batch_size)` without passing config
#- config is optional and only needed later when multiple variants exist



from __future__ import annotations

from src.diffusion.model_a_eps_mse import train_model_A, score_model_A


def train_diffusion(X_train, config):
    """
    Stable notebook API.
    Dispatches to the selected diffusion model variant.
    """
    variant = str(config.get("model_variant", "A")).upper()

    if variant == "A":
        return train_model_A(X_train, config)

    raise NotImplementedError(f"Unknown diffusion model variant: {variant}")


def score_diffusion(model, X, batch_size, config=None):
    """
    Stable notebook API.
    Config is optional for backward compatibility.
    """
    # Prefer config routing when provided; else fall back to model tag
    if config is not None:
        variant = str(config.get("model_variant", "A")).upper()
    else:
        variant = str(getattr(model, "model_variant", "A")).upper()

    if variant == "A":
        return score_model_A(model, X, batch_size, config=config)

    raise NotImplementedError(f"Unknown diffusion model variant: {variant}")
