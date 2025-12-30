# Modelling and Evaluation Charter – Diffusion-Based Anomaly Detection in Dynamic Business Time-Series (Next Phase)

This charter defines the **modelling**, **evaluation**, and **reporting** standards for the next research phase, following completion of preprocessing for Case Studies A–D.

It is designed to be consistent with:

* the **Application Proposal – Nandipha Mehlo** (ground-truth research framing: diffusion focus, efficiency constraints, drift-aware evaluation),
* the **Preprocessing Charter – Dynamic Business Time-Series Case Studies** (ground-truth data preparation rules and split logic), and
* the **Processed Data Inventory – Cases A–D** (canonical processed artefacts and known per-case characteristics).

The charter is intentionally **method-first**: it specifies a reproducible evaluation harness and a staged modelling plan so that comparisons between diffusion-based methods and baselines are fair across datasets with very different sampling resolutions and anomaly rates.

---

## 0) Confirmed design decisions (locked for this phase)

These decisions have been confirmed for the modelling and evaluation phase. They replace all prior placeholders.

1. **Primary objective (headline evaluation): event / window-level detection**

   * Event-level detection is the primary objective.
   * Point-wise scores and metrics are treated as secondary diagnostics.

2. **Primary representation: sequence windows**

   * Sequence windows are the default representation for modelling and scoring.
   * Tabular window summaries may be used only for selected baselines as an optional extension.

3. **Diffusion formulation: reconstruction-based diffusion (in-scope)**

   * The main diffusion approach is reconstruction/denoising-error based.
   * Likelihood/score-based diffusion variants are acknowledged but treated as out-of-primary-scope for this dissertation phase.

4. **Synthetic experiments: secondary experiments after real-data results**

   * Main evidence comes from real labelled anomalies (NAB + AIOps KPI labels).
   * Synthetic-on-real anomaly injection is run after real-data experiments to stress-test and interpret behaviour, especially where labels are sparse.

---

## 1) Ground-truth constraints and guiding principles

### 1.1 Compute and feasibility constraints (proposal-aligned)

The modelling plan must remain feasible under limited compute (personal CPU resources and a light cloud GPU option). Therefore:

* Prefer **lightweight baselines** as anchors (simple heuristics + classical baselines).
* Use **small, well-justified hyperparameter grids**.
* Use **early stopping** and **reproducible seeds**.
* Store intermediate artefacts so that experiments can be resumed without re-running everything.

### 1.2 Drift-aware evaluation as a first-class requirement

The proposal motivates evaluation under **dynamic business time-series** conditions (concept drift / regime shifts). Therefore:

* Do not report only one aggregate metric.
* Report performance **by time segment** (train/validation/test and within-test breakdowns where relevant).
* Treat Case D (AIOps KPI) as an explicit stress test due to its dense incident regime and high anomaly concentration.

### 1.3 Separation of concerns: real preprocessing vs synthetic modifications

* Preprocessing notebooks produce canonical processed datasets (see Processed Data Inventory).
* Any synthetic anomaly injection or synthetic-on-real augmentation must occur **later**, on **copies** of processed data, and must not overwrite canonical artefacts.

---

## 2) Inputs to modelling (canonical datasets)

### 2.1 Canonical processed files

Models must be trained and evaluated only on the canonical processed artefacts:

* Case A: `data/processed/ambient/ambient_processed_full.csv` (+ splits)
* Case B: `data/processed/nyc_taxi/nyc_taxi_full.csv` (+ splits)
* Case C: `data/processed/cpu_utilisation/cpu_utilisation_full.csv` (+ splits)
* Case D: `data/processed/aiops_kpi/aiops_kpi_full.csv` (+ splits)

### 2.2 Standard columns available

All models may assume the following fields exist and have consistent meaning:

* `time`, `value`, `value_scaled`, `is_anomaly`, `split`, `case_study`
* time features: `hour_of_day`, `day_of_week`, `is_weekend`

### 2.3 Case-specific data realities that must be respected

* Case A: extremely sparse anomalies (2 labels).
* Case B: sparse anomalies (5 labels) with strong seasonality.
* Case C: extremely sparse anomalies (2 labels) and regime changes near series end.
* Case D: higher anomaly density overall and a concentrated incident regime; 1-minute sampling with occasional gaps.

These differences require:

* careful choice of metrics (avoid misleading accuracy),
* threshold calibration strategies that do not leak test information,
* evaluation summaries that explicitly report class imbalance.

---

## 3) Problem formulation and scoring outputs

### 3.1 Common modelling output across all methods

Regardless of model family, every method must output, for each timestamp in validation and test:

* `anomaly_score` (higher = more anomalous), and
* `pred_is_anomaly` after applying a threshold chosen using validation only.

A standard per-split prediction file is required:

* `<case>_<model>_val_predictions.csv`
* `<case>_<model>_test_predictions.csv`

Minimum columns:

* `time`, `anomaly_score`, `pred_is_anomaly`, `is_anomaly`, `split`, `case_study`

### 3.2 Thresholding rule (to avoid leakage)

Threshold selection must use **validation only**.

Supported threshold selection strategies (choose at least one and apply consistently):

1. **Quantile-based threshold** on validation scores (e.g., top q% flagged).
2. **Metric-optimised threshold**: choose threshold that maximises a validation metric aligned to `<primary_objective>`.

For each trained model run, the chosen threshold must be saved:

* `threshold_value`
* `threshold_strategy`
* validation metric at the chosen threshold

---

## 4) Windowing and feature design

Because diffusion models and LSTM autoencoders are sequence-based, and classical baselines may operate either on sequences or summary features, modelling is organised into two parallel input modes.

### 4.1 Mode A: Sliding window sequences (recommended default for diffusion + LSTM AE)

**Definition**

* For each time t, form a window of length `L` covering `[t-L+1, ..., t]`.
* Input features for the window are typically `value_scaled` (and optionally time features).

**Contiguity enforcement (critical for Case D gaps)**

* A window is valid only if the time differences inside the window match the dominant sampling interval.
* Windows that cross a gap are excluded from training and scoring.

**Candidate window lengths (to be validated, not assumed)**

Window lengths must be justified per sampling resolution. Candidate sets:

* Case A (hourly): `L ∈ {24, 48, 72, 168}` (1–7 days)
* Case B (30-min): `L ∈ {48, 96, 336}` (1–7 days)
* Case C (5-min): `L ∈ {72, 144, 288, 576}` (6–48 hours)
* Case D (1-min): `L ∈ {60, 120, 360, 1440}` (1 hour–1 day)

Selection rule:

* Choose `L` using validation performance and compute feasibility.
* Record a short sensitivity table: `L`, train runtime, validation metric.

### 4.2 Mode B: Tabular features (recommended for Isolation Forest and OC-SVM)

Two defensible options exist:

1. **Flattened window**: use the last `L` values (and optionally time features) as features.
2. **Summary window features**: mean, std, min, max, slope, rolling quantiles, and seasonal indicators.

Preference under compute limits:

* summary window features reduce dimensionality and can be more stable.

Feature sets must be kept consistent across datasets where possible.

---

## 5) Model families to be evaluated (proposal-aligned)

The modelling phase is organised into three tiers to match the proposal’s baseline requirements and diffusion focus.

### 5.1 Tier 0: Simple heuristic baselines (required)

Purpose:

* Provide transparent anchors that establish whether sophisticated methods are meaningfully improving on simple rules.

Candidate heuristics (apply using validation-only thresholding):

1. **Static z-score threshold** on `value_scaled`.
2. **Rolling z-score** (rolling mean/std on train; apply to val/test).
3. **Rolling median + MAD** (robust alternative).

Case B (NYC taxi) may require a seasonal-aware heuristic:

* a **seasonal baseline** (e.g., compare to same time-of-week historical window) can be considered if already feasible; otherwise keep the heuristics above and document the limitation.

### 5.2 Tier 1: Classical baseline models (required)

These are explicitly listed in the proposal and must be implemented:

* **Isolation Forest (IF)**
* **One-Class SVM (OC-SVM)**

Implementation guidance:

* Train using the training split only.
* Use window-based tabular features (Section 4.2).
* Produce anomaly scores on validation and test.

### 5.3 Tier 2: Deep baseline (required)

* **LSTM Autoencoder (LSTM AE)**

Implementation guidance:

* Use sliding windows (Section 4.1).
* Train to reconstruct the input sequence.
* Anomaly score can be defined as reconstruction error (e.g., MSE) aggregated over the window.

### 5.4 Tier 3: Diffusion-based anomaly detection (primary research contribution)

This tier operationalises the proposal’s core direction: evaluating diffusion models for anomaly detection under feasibility constraints.

Two diffusion-compatible scoring approaches are valid; selection depends on `<diffusion_formulation>`:

1. **Reconstruction-based diffusion**

   * Train diffusion to model normal windows.
   * Score anomalies by reconstruction error or denoising difficulty.

2. **Likelihood/score-based diffusion**

   * Use model likelihood proxies or score magnitudes as anomaly scores.

Fairness requirement:

* Diffusion models must be trained on the same training data and windowing rules as LSTM AE.
* Hyperparameter search must be constrained and documented.

---

## 6) Training and tuning protocol

### 6.1 Data usage rules

* Train: training split only.
* Tune/Select: validation split only.
* Final report: test split only.

### 6.2 Hyperparameter tuning (minimal but principled)

A small grid is required for each model family.

Example tuning knobs (to be finalised per implementation):

* IF: number of estimators, contamination proxy (used only for score scaling, not for thresholding).
* OC-SVM: kernel, nu, gamma.
* LSTM AE: hidden size, layers, dropout, learning rate, epochs.
* Diffusion: number of steps, model width, learning rate, epochs.

Tuning outputs must include:

* chosen hyperparameters,
* runtime summary,
* validation metric summary.

### 6.3 Reproducibility

For every run record:

* random seeds,
* library versions,
* hardware summary (CPU/GPU),
* configuration file or dictionary.

---

## 7) Evaluation design

### 7.1 Core metrics (point-wise)

Because class imbalance is severe in multiple cases, the following must be reported:

* Precision, Recall, F1
* PR-AUC (recommended under imbalance)
* ROC-AUC (reported but not treated as primary under severe imbalance)

### 7.2 Event-aware evaluation (recommended)

If `<primary_objective>` includes incident detection, event-level evaluation should be added.

A defensible approach:

* define anomaly events as contiguous runs of `is_anomaly == 1`.
* evaluate whether predicted anomalies overlap each true event (event recall), and how many predicted events are false (event precision).

If an explicit “detection delay” metric is desired, define it clearly (minutes/hours after event start) and report it.

### 7.3 Time-segment evaluation (drift-aware)

In addition to aggregate metrics on the full test split, report:

* metrics by calendar segment where meaningful (e.g., monthly for Case B),
* metrics by regime segments (Case C boundary regimes; Case D incident vs post-incident windows),
* score distribution plots by split (train/val/test) to visualise drift.

### 7.4 Case-specific evaluation cautions

* Cases A and C have only 2 labelled anomalies. Point-wise metrics can become unstable.

  * Report counts and qualitative plots alongside metrics.
  * Treat results as illustrative evidence of feasibility and regime sensitivity rather than purely statistical superiority.

* Case D has a dense incident regime with many anomalies concentrated in test.

  * PR-AUC and event-based metrics become particularly informative.

---

## 8) Reporting artefacts (required outputs)

### 8.1 Per run (per model × case)

Save the following:

1. `config.json` (or `.yaml`): hyperparameters + seeds + window length + feature set.
2. `threshold.json`: threshold strategy + value + validation metric.
3. Prediction files:

   * `<case>_<model>_val_predictions.csv`
   * `<case>_<model>_test_predictions.csv`
4. Metrics tables:

   * `<case>_<model>_val_metrics.csv`
   * `<case>_<model>_test_metrics.csv`
5. At least one diagnostic figure:

   * anomaly score timeline with true anomalies overlaid (validation and test).

### 8.2 Cross-model summary tables

For each case, produce a single summary table comparing models:

* model name
* window length / feature set
* validation PR-AUC (or chosen primary validation metric)
* test PR-AUC
* test F1 at chosen threshold
* runtime (train + score)

A final cross-case table should summarise performance patterns across A–D.

---

## 9) Planned execution order (recommended)

1. Implement Tier 0 heuristics for all cases (fast baseline anchors).
2. Implement Tier 1 classical baselines (IF, OC-SVM) using a consistent window-feature pipeline.
3. Implement Tier 2 LSTM AE as the deep baseline.
4. Implement Tier 3 diffusion model(s) with a minimal tuning plan.
5. Add drift-aware breakdowns and event-aware evaluation.
6. Only after real-data results are stable: decide whether to run synthetic-on-real experiments (`<synthetic_experiment_scope>`).

---

## 10) Charter-to-document mapping (how this supports writing up)

* **Methodology (Chapter 3):** Sections 3–8 define the modelling pipeline, leakage controls, and evaluation standards.
* **Results (later chapter):** Section 8 ensures every result has a reproducible artefact trail and consistent comparison tables.
* **Discussion:** Sections 1–7 ensure conclusions can be grounded in drift-aware evidence rather than only aggregate metrics.

---

## 11) Immediate next step

Confirm the four placeholder decisions in Section 0. Once confirmed, proceed to implement:

* a shared windowing utility (contiguity-aware for Case D),
* a baseline evaluation harness that outputs the standard artefacts for Tier 0 and Tier 1.
