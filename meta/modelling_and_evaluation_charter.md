# Modelling and Evaluation Charter – Diffusion-Based Anomaly Detection in Dynamic Business Time-Series

This charter defines the **modelling**, **evaluation**, and **reporting** standards for the next research phase, following completion of preprocessing for Case Studies A–D.

It is designed to be consistent with:

* the **Application Proposal – Nandipha Mehlo** (diffusion focus, efficiency constraints, drift-aware evaluation, hybrid quantitative–qualitative design),
* the **Preprocessing Charter – Dynamic Business Time-Series Case Studies** (data preparation rules and split logic), and
* the **Processed Data Inventory – Cases A–D** (canonical processed artefacts and per-case characteristics).

The charter is intentionally **method-first**: it specifies a reproducible evaluation harness and a staged modelling plan so that comparisons between diffusion-based methods and baselines are fair across datasets with very different sampling resolutions and anomaly rates.

---

## 0) Confirmed design decisions (locked for this phase)

These decisions are now fixed for the modelling and evaluation phase and replace all earlier placeholders.

1. **Primary objective (headline evaluation): event-level detection**

   * The **headline question** is whether a model correctly detects **anomaly events** (incidents, regimes, episodes).
   * **Event-level metrics** (event precision, event recall, event F1, and optionally detection delay) are the **primary evaluation signal**.
   * Point-wise and window-wise scores/metrics are used as **supporting diagnostics**.

2. **Primary representation: sequence windows**

   * Sequence windows are the default representation for modelling and scoring.
   * Each window yields an anomaly score associated with a **window end time**.
   * Tabular window summaries may be used for selected baselines as an optional extension, but they must still produce scores that can be mapped back to timestamps and events.

3. **Diffusion formulation: reconstruction-based diffusion (in-scope)**

   * The main diffusion approach is reconstruction/denoising-error based.
   * Likelihood/score-based diffusion variants are acknowledged but treated as out-of-primary-scope for this dissertation phase.

4. **Synthetic experiments: secondary experiments after real-data results**

   * Main evidence comes from real labelled anomalies (NAB + AIOps KPI labels).
   * Synthetic-on-real anomaly injection is run **after** real-data experiments to stress-test and interpret behaviour, especially where labels are sparse.

---

## 1) Ground-truth constraints and guiding principles

### 1.1 Compute and feasibility constraints (proposal-aligned)

The modelling plan must remain feasible under limited compute (personal CPU resources and a light cloud GPU option). Therefore:

* Prefer **lightweight baselines** as anchors (simple heuristics + classical baselines).
* Use **small, well-justified hyperparameter grids**.
* Use **early stopping** and **reproducible seeds** where applicable.
* Store intermediate artefacts so that experiments can be resumed without re-running everything.

### 1.2 Drift-aware evaluation as a first-class requirement

The proposal motivates evaluation under **dynamic business time-series** conditions (concept drift / regime shifts). Therefore:

* Do not report only one aggregate metric.
* Where feasible, report performance **by time segment** (e.g. early vs late, pre-incident vs incident vs post-incident), not just over the full test period.
* Treat Case D (AIOps KPI) as an explicit **stress test** due to its dense incident regime and high anomaly concentration.

### 1.3 Separation of concerns: real preprocessing vs synthetic modifications

* Preprocessing notebooks produce **canonical processed datasets** (see Preprocessing Charter and Processed Data Inventory).
* Any synthetic anomaly injection or synthetic-on-real augmentation must occur **later**, on **copies** of processed data, and must not overwrite canonical artefacts.

---

## 2) Inputs to modelling (canonical datasets)

### 2.1 Canonical processed files

Models must be trained and evaluated only on the canonical processed artefacts:

* Case A: `data/processed/ambient/ambient_processed_full.csv` (+ splits).
* Case B: `data/processed/nyc_taxi/nyc_taxi_full.csv` (+ splits).
* Case C: `data/processed/cpu_utilisation/cpu_utilisation_full.csv` (+ splits).
* Case D: `data/processed/aiops_kpi/aiops_kpi_full.csv` (+ splits).

### 2.2 Standard columns available

All models may assume the following fields exist and have consistent meaning:

* `time`, `value`, `value_scaled`, `is_anomaly`, `split`, `case_study`
* time features: `hour_of_day`, `day_of_week`, `is_weekend`

### 2.3 Case-specific data realities that must be respected

* Case A: extremely sparse anomalies (2 events).
* Case B: sparse anomalies (5 events) with strong seasonality.
* Case C: extremely sparse anomalies (2 events) and regime changes near series end.
* Case D: higher anomaly density overall and a concentrated incident regime; 1-minute sampling with occasional gaps.

These differences require:

* careful choice of metrics (avoid misleading accuracy),
* threshold calibration strategies that do not leak test information,
* evaluation summaries that explicitly report **class imbalance** and **number of labelled events**.

---

## 3) Problem formulation and scoring outputs

### 3.1 Common modelling output across all methods

Regardless of model family, every method must output, for each timestamp in validation and test:

* `anomaly_score` (higher = more anomalous), and
* `pred_is_anomaly` after applying a chosen threshold.

A standard per-split prediction file is required:

* `<case>_<model>_val_predictions.csv`
* `<case>_<model>_test_predictions.csv`

Minimum columns:

* `time`, `anomaly_score`, `pred_is_anomaly`, `is_anomaly`, `split`, `case_study`

**Note:** event-level metrics will be computed **from these point-level predictions**, using consistent event definitions (Section 7.1).

### 3.2 Thresholding rule (to avoid leakage and align with event-level objective)

Threshold selection must use **validation only** and must align with the **event-level objective** as far as possible.

Supported threshold selection strategies:

1. **Primary strategy: event-F1 on validation**

   * Convert validation labels into **true events** (contiguous runs of `is_anomaly == 1`).
   * For a grid of candidate thresholds, compute **event precision**, **event recall**, and **event F1**.
   * Choose the threshold that **maximises event F1** (with event recall as a tie-breaker).

2. **Fallback strategy for extremely sparse events**

   * In cases with too few events for a meaningful event-F1 curve (e.g. Case A and possibly Case C), use **point-wise F1** on validation as a fallback.
   * Document explicitly when the fallback is used.

For each trained model run, the chosen threshold must be saved:

* `threshold_value`
* `threshold_strategy` (e.g. `"event_f1"`, `"point_f1_fallback"`, `"fixed_quantile_q=0.995"`)
* the **validation metric(s)** at the chosen threshold (event-F1 plus supporting metrics).

---

## 4) Windowing and feature design

Because diffusion models and LSTM autoencoders are sequence-based, and classical baselines may operate either on sequences or summary features, modelling is organised into two parallel input modes.

### 4.1 Mode A: Sliding window sequences (recommended default for diffusion + LSTM AE)

**Definition**

* For each time step (t), form a window of length (L) covering ([t-L+1, \dots, t]).
* Input features for the window are typically `value_scaled` (with optional time features).

**Contiguity enforcement (critical for Case D gaps)**

* A window is valid only if the time differences inside the window **match the dominant sampling interval** for that case.
* Windows that cross a gap are **excluded** from training and scoring.

**Candidate window lengths (to be validated, not assumed)**

Window lengths must be justified per sampling resolution. Candidate sets:

* Case A (hourly): `L ∈ {24, 48, 72, 168}` (1–7 days)
* Case B (30-min): `L ∈ {48, 96, 336}` (1–7 days)
* Case C (5-min): `L ∈ {72, 144, 288, 576}` (6–48 hours)
* Case D (1-min): `L ∈ {60, 120, 360, 1440}` (1 hour–1 day)

Selection rule:

* Choose `L` using validation performance and compute feasibility.
* Record a short sensitivity table: `L`, train runtime, event-level validation metrics.

### 4.2 Mode B: Tabular features (recommended for Isolation Forest and OC-SVM)

Two defensible options exist:

1. **Flattened window**: use the last `L` values (and optionally time features) as features.
2. **Summary window features**: e.g. mean, std, min, max, simple slope, and a small set of seasonal indicators.

Preference under compute limits:

* summary window features reduce dimensionality and can be more stable.
* feature sets should be kept **as consistent as possible** across datasets, while respecting different sampling resolutions.

Regardless of representation, models must still produce **timestamp-indexed anomaly scores** for event-level evaluation.

---

## 5) Model families to be evaluated (proposal-aligned)

The modelling phase is organised into tiers to match the proposal’s baseline requirements and diffusion focus.

### 5.1 Tier 0: Simple heuristic baselines (required)

**Purpose**

* Provide transparent anchors that establish whether sophisticated methods are meaningfully improving on simple rules.

**Candidate heuristics** (validation-based thresholding, evaluated at event level):

1. **Static z-score threshold** on `value_scaled`.
2. **Rolling z-score** (rolling mean/std computed on train; applied to val/test).
3. **Rolling median + MAD** (robust alternative).

Case B (NYC taxi) may also consider a **seasonal heuristic** (e.g. comparing current value to expected seasonal baseline for that time-of-week). If that is not implemented, the limitation should be documented.

### 5.2 Tier 1: Classical baseline models (required)

As stated in the proposal:

* **Isolation Forest (IF)`**
* **One-Class SVM (OC-SVM)`**

Implementation guidance:

* Train using the **training split only**.
* Use window-based tabular features (Section 4.2).
* Produce anomaly scores on validation and test.
* Evaluate using:

  * point-wise metrics (Precision, Recall, F1, AUROC, PR-AUC), and
  * event-level metrics (Section 7.1), with event-F1 as the primary metric where possible.

### 5.3 Tier 2: Deep baseline (required)

* **LSTM Autoencoder (LSTM AE)**

Implementation guidance:

* Use sliding windows (Section 4.1).
* Train to reconstruct the input sequence.
* Define anomaly score as a reconstruction-error measure aggregated over the window (e.g. mean squared error).
* Evaluate as for Tier 1, with the same thresholding protocol.

### 5.4 Tier 3: Diffusion-based anomaly detection (primary research contribution)

This tier operationalises the proposal’s core direction: evaluating diffusion models for anomaly detection under feasibility constraints.

Within the reconstruction-based scope:

* Train diffusion models on **normal windows** from the training split.
* Use reconstruction error / denoising difficulty as the anomaly score at each window end time.
* Evaluate with exactly the same event-aligned metrics and thresholding strategy used for Tier 1–2, to allow fair comparison.

Fairness requirement:

* Diffusion models must be trained on the **same training data and window rules** as LSTM AE.
* Hyperparameter search must be constrained and documented.

---

## 6) Training and tuning protocol

### 6.1 Data usage rules

* Train: **training** split only.
* Tune / select thresholds and hyperparameters: **validation** split only.
* Final report: **test** split only.

### 6.2 Hyperparameter tuning (minimal but principled)

A small grid is required for each model family.

Example tuning knobs (to be finalised per implementation):

* IF: number of estimators, max samples, contamination proxy (for score scaling only).
* OC-SVM: kernel, nu, gamma.
* LSTM AE: hidden size, number of layers, dropout, learning rate, epochs.
* Diffusion: number of diffusion steps, model width, learning rate, epochs.

Tuning outputs must include:

* chosen hyperparameters,
* runtime summary,
* validation metrics (including **event-F1** where possible).

### 6.3 Reproducibility

For every run record:

* random seeds,
* library versions,
* hardware summary (CPU/GPU),
* configuration file or dictionary with all choices.

---

## 7) Evaluation design

### 7.1 Event-level evaluation (primary)

**Event definition**

* A **true event** is a contiguous run of `is_anomaly == 1` in the ground-truth labels.
* A **predicted event** is a contiguous run of `pred_is_anomaly == 1` at a given threshold.

**Event matching**

* A true event is considered **detected** if at least one predicted event overlaps it in time.
* A predicted event is considered **false** if it overlaps no true event.

**Primary event metrics**

For each model and case:

* **Event recall**: proportion of true events that were detected at least once.
* **Event precision**: proportion of predicted events that match at least one true event.
* **Event F1**: harmonic mean of event precision and event recall.

Optional but desirable:

* **Detection delay**: time difference between event start and first correct prediction, summarised across events (e.g. median delay).

These event-level metrics are the **headline results** for the study, especially in Case B and Case D where events are more numerous. For cases with only 2 events (A and C), metrics are still computed but interpreted cautiously and supported with qualitative plots.

### 7.2 Point-wise metrics (supporting diagnostics)

To characterise discrimination at the point level, and to remain aligned with the research proposal:

* **Precision, Recall, and point-wise F1** at the chosen threshold,
* **AUROC (Area Under the ROC Curve)**,
* **PR-AUC (Area Under the Precision–Recall Curve)**.

Interpretation guidance:

* Under heavy imbalance (all four cases, especially A, C, and D), **PR-AUC** is usually more informative than AUROC, but AUROC remains part of the reported metric set because it is explicitly listed in the proposal.
* If AUROC is high but event-level recall is poor, the discussion must highlight this mismatch.

### 7.3 Time-segment evaluation (drift-aware)

In addition to aggregate metrics over the full test split, report:

* Metrics by **calendar segment** where meaningful (e.g. monthly for Case B).

* Metrics by **regime segments** where regimes have been identified in preprocessing:

  * Case C: early baseline vs unstable vs low-CPU regimes.
  * Case D: pre-incident vs incident vs post-incident.

* Score distribution plots (train/val/test) to visualise drift and calibration shifts.

### 7.4 Case-specific evaluation cautions

* **Cases A and C**: only 2 labelled events each.

  * Event metrics are based on very few events and are therefore unstable.
  * Results must be supported by plots and narrative descriptions rather than treated as purely statistical evidence.

* **Case D**: dense incident regime with many anomalies concentrated in test.

  * Event-level metrics and PR-AUC are particularly informative here.
  * The incident regime should receive a dedicated sub-section in the results and discussion chapters.

---

## 8) Reporting artefacts (required outputs)

### 8.1 Per run (per model × case)

Save the following:

1. `config.json` (or `.yaml`): hyperparameters, seeds, window length, feature set.

2. `threshold.json`: threshold strategy, value, and validation metrics (including event-F1 or fallback metric).

3. Prediction files:

   * `<case>_<model>_val_predictions.csv`
   * `<case>_<model>_test_predictions.csv`

4. Metrics tables:

   * `<case>_<model>_val_metrics.csv` (point + event metrics)
   * `<case>_<model>_test_metrics.csv` (point + event metrics)

5. Diagnostic figures (at least):

   * anomaly score timeline with true anomalies overlaid (validation and test),
   * histogram / density of scores for normal vs anomalous points (test),
   * optional event-level visualisations for key incidents (particularly in Case D).

### 8.2 Cross-model summary tables

For each case, produce a single summary table comparing models, including:

* model name,
* window length / feature set,
* validation event-F1 (or fallback metric),
* test event-F1,
* point-wise PR-AUC and AUROC on test,
* runtime (train + score).

A final cross-case table should summarise performance patterns across A–D, highlighting where diffusion models add value relative to simple and classical baselines.

---

## 9) Charter-to-document mapping (how this supports writing up)

* **Chapter 3 – Methodology**
  Sections 2–7 translate directly into the modelling pipeline, thresholding rules, event definitions, leakage controls, and evaluation metrics.

* **Chapter 4 – Results / Findings**
  Sections 7–8 define the structure of per-case result subsections and cross-model comparison tables, including event-level and point-wise metrics.

* **Chapter 5 – Discussion**
  Sections 1, 2, and 7 guide how to interpret successes and limitations, especially:

  * efficiency vs performance trade-offs,
  * robustness under drift,
  * differences between event-level and point-level views.

* **Hybrid design (quantitative + qualitative)**
  The same artefacts and metrics feed into the **Qualitative Reflection Log**, allowing the narrative to comment on:

  * patterns across models and cases,
  * where diffusion struggles or succeeds,
  * how these insights map back to responsible and accessible AI principles in the proposal.
