# Diffusion Model Roadmap – Thesis Design Plan (Draft v1)

This section outlines the planned diffusion-based models evaluated in this study,
their conceptual role, and how they relate to the research objectives of
efficiency, robustness under drift, and anomaly event detection.

---

## Model A — Denoising diffusion (ε-prediction) as anomaly scorer
**Primary diffusion baseline**

### Concept
Train a denoising diffusion model to predict noise ε added to normal time-series windows.
Anomaly score is defined as the average ε-prediction error across selected diffusion timesteps.

### Characteristics
- Forward process: standard DDPM noising schedule (cosine or linear).
- Training: normal windows only.
- Scoring: no reverse sampling; direct denoising difficulty.
- Architecture: lightweight sequence model (1D Conv / temporal encoder).
- Complexity: O(NT) forward only (no sampling loops).

### Role in thesis
This model represents the **core contribution**:
> Efficient use of diffusion as a diagnostic anomaly detector.

It is:
- computationally feasible,
- event-level evaluable,
- directly aligned with business time-series monitoring.

---

## Model B — DDIM-based reconstruction diffusion
**Generative extension model**

### Concept
Use DDIM-style deterministic reverse sampling to reconstruct input windows
from noisy states and measure reconstruction deviation from original signal.

### Characteristics
- Same training objective as Model A.
- Reverse process uses K << T steps.
- Scoring: reconstruction error after partial reverse sampling.
- More expensive than Model A.
- Provides a more “generative” interpretation of anomalies.

### Role in thesis
Model B tests:
> Whether explicit generative reconstruction adds value beyond simple denoising difficulty.

This model addresses the proposal’s discussion of:
- DDPM vs DDIM efficiency trade-offs.

---

## Optional Model C — Conditional diffusion (time-aware)
**Context-conditioned diffusion**

### Concept
Extend Model A by conditioning diffusion on:
- hour of day,
- day of week,
- or regime indicators.

### Role
Explores whether contextual conditioning improves detection under drift.

---

## Optional Model D — Multivariate diffusion (future work)
If resources permit:
- joint diffusion across multiple KPIs.
- out-of-scope for current dissertation but noted as extension.

---

## Experimental structure

For each case study (A–D):

1. Tier 0–2 baselines establish classical performance.
2. Model A is evaluated as the primary diffusion method.
3. Model B is compared as a generative extension.
4. Results analysed at:
   - event-level,
   - time-segment level,
   - efficiency (runtime vs gain).

---

## Research positioning

The study does not aim to invent new diffusion theory.
Instead, it evaluates:

> How different *usages* of diffusion models perform as anomaly detectors
> under realistic compute and drift constraints.

This framing prioritises:
- reproducibility,
- interpretability,
- and operational relevance over architectural novelty.
