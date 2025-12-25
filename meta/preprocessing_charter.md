# Preprocessing Charter – Dynamic Business Time-Series Case Studies

This charter defines how raw time-series data for the four case studies are prepared for anomaly-detection experiments. It links:

* the **exploratory data overviews** (Chapter 4 descriptions), and
* the **formal methodology** (Chapter 3),

by specifying how each raw series becomes a model-ready dataset.

Implementation status per case:

* **Case A (Ambient)** – preprocessing **implemented and saved**.
* **Case B (NYC taxi)** – preprocessing **implemented and saved**.
* **Case C (CPU)** – preprocessing **implemented and saved**.
* **Case D (AIOps KPI)** – preprocessing **planned, not yet implemented**.

---

## 1. Shared preprocessing principles (all case studies)

### 1.1 Overall aim

Prepare each time series in a **consistent, transparent** way so that:

* models receive clean, well-structured inputs, and
* comparisons between diffusion-based methods and baselines are **fair across datasets**.

### 1.2 Core schema

All processed datasets share this **core schema and column meanings**:

* `time` – timestamp (`datetime64[ns]`), sorted in ascending order; natural sampling preserved.
* `value` – main measurement in **interpretable units**:

  * Ambient: temperature in °C (converted from °F).
  * NYC taxi: number of trips per 30 minutes.
  * CPU: CPU utilisation in percent.
  * AIOps KPI: anonymised KPI value (unitless but consistent).
* `value_scaled` – standardised version of `value` (z-score) using **train-only** mean and std.
* `is_anomaly` – binary label (0 = normal, 1 = anomaly) from NAB / AIOps labels.
* `case_study` – string identifier: `"ambient"`, `"nyc_taxi"`, `"cpu"`, `"aiops_kpi"`.
* `split` – `"train"`, `"validation"`, or `"test"`; assigned chronologically.

Optional **time-derived features** (if enabled, they must appear in all processed datasets):

* `hour_of_day` (0–23)
* `day_of_week` (0–6, Monday–Sunday)
* `is_weekend` (0/1, based on `day_of_week`)

Datasets may have additional analysis-only columns later (e.g. incident flags), but the core fields above must exist and have consistent meaning everywhere.

### 1.3 Time handling and sanity checks

* Parse all timestamps into `datetime64[ns]` and **sort by time**.
* Retain each dataset’s **native sampling resolution**:

  * Ambient: hourly
  * NYC taxi: 30-minute
  * CPU: 5-minute
  * AIOps KPI: 1-minute with occasional longer gaps
* Longer gaps (if any) are treated as **missing periods in the time axis**, not as evidence of a different sampling rate.
* **Step 3 in each preprocessing notebook** performs:

  * **Missing values check** (per-column counts and percentages).
  * **Duplicate rows check** (exact duplicates across all columns).
* For Ambient and NYC taxi the outcome is:

  * 0 missing values in any core column.
  * 0 duplicate rows.

### 1.4 Scaling (global rule)

* Each dataset’s `value` column is standardised with **`sklearn.preprocessing.StandardScaler`**.
* The scaler is **fitted only on the training split** (`train_df["value"]`) to avoid information leakage from validation/test.
* The learned **mean and standard deviation** are stored as a small dictionary per case:

  * Ambient: `ambient_scaler_params = {"mean": ..., "std": ...}`
  * NYC taxi: `nyc_taxi_scaler_params = {"mean": ..., "std": ...}`
  * CPU: `cpu_scaler_params = {"mean": ..., "std": ...}`
  * AIOps KPI: `aiops_kpi_scaler_params = {"mean": ..., "std": ...}` (to be created when preprocessing is implemented)
* The fitted scaler is applied to the **full processed dataframe** (train, validation, test) to produce `value_scaled`.

This gives:

* `value_scaled` with mean ≈ 0 and std ≈ 1 on the **training split**,
* validation and test on the same scale but with their own means/variances (reflecting drift).

### 1.5 Splits and non-destructive design

* Splits are always **chronological**; no shuffling.

* For each case, there is **one full processed dataframe** (e.g. `*_full.csv`) that retains **all rows**.

* Split-specific dataframes (`train_df`, `validation_df`, `test_df`) are created with boolean masks on `df["split"]` and `.copy()`.

* Integrity checks confirm:

  ```text
  len(df) == len(train_df) + len(validation_df) + len(test_df)
  ```

  so that no rows are dropped or duplicated across splits.

* Split design goals:

  * Training captures **normal behaviour** and earlier regimes.
  * Validation and test include labelled anomalies and, where possible, **regime changes** or drift.
  * At least some anomalies must appear in the test split for final evaluation.

#### Split validity documentation (cross-case)

Split validity is documented using outputs already produced by the notebooks:

* a split summary table (start/end times, rows, anomaly counts, proportions),
* an integrity check confirming non-destructive split creation, and
* an anomaly placement preview confirming where labelled anomalies fall.

Most datasets naturally satisfy validity expectations under their split rules (Ambient anomaly-based boundaries; NYC taxi calendar-month boundaries). The CPU case requires an additional dataset-specific constraint to prevent the evaluation windows from collapsing due to the short end-of-series regime tail.

### 1.6 Outputs and directory structure

Under a project-level `data/processed/` directory, each case has its own subfolder:

* `data/processed/ambient/`
* `data/processed/nyc_taxi/`
* `data/processed/cpu_utilisation/`
* `data/processed/aiops_kpi/`

Each subfolder contains:

* one **full processed file** (`*_full.csv`), and
* three **split files** (`*_train.csv`, `*_validation.csv`, `*_test.csv`),

all with the standard schema.

---

## 2. Case Study A – Ambient temperature (NAB sensor failure)

### 2.1 Raw data summary

* Source file: `ambient_temperature_system_failure.csv` (NAB `realKnownCause`).
* Interpretation: ambient temperature in a controlled environment.
* ~7 267 hourly readings; strongly regular sampling.
* Two labelled anomalies (warm spike and regime shift) from NAB labels.

### 2.2 Implemented preprocessing steps (Ambient A)

**Time and labels**

* Load raw CSV and NAB `combined_labels.json`.
* Extract labels for key `"realKnownCause/ambient_temperature_system_failure.csv"`.
* Convert both data and label timestamps to `datetime`, sort by `time`.
* Create `is_anomaly` via `df["time"].isin(ambient_label_times).astype(int)`.
* Class balance: 7 265 normal points, 2 anomalies.

**Core schema standardisation**

* Rename `timestamp → time`.
* Add `case_study = "ambient"`.
* Initial core columns:

  `time, value, is_anomaly, case_study`.

**Missing/duplicate checks**

* Missing-values table confirms 0 missing in all columns.
* Duplicate-rows table confirms 0 exact duplicates.

**Unit conversion and time features**

* Convert Fahrenheit to Celsius:

  ```python
  df["value"] = (df["value"] - 32) * 5/9
  ```

* Add time features:

  * `hour_of_day = df["time"].dt.hour`
  * `day_of_week = df["time"].dt.weekday`
  * `is_weekend = df["day_of_week"].isin([5, 6]).astype(int)`

**Splits (using anomaly-based boundaries)**

* Anomaly times:

  * First anomaly: 2013-12-22 20:00
  * Second anomaly: 2014-04-13 09:00

* Boundaries chosen as **7 days before each anomaly**:

  * `training_end_time = first_anomaly_time - 7 days`
  * `validation_end_time = second_anomaly_time - 7 days`

* Split assignment:

  * `train`: dataset start → `training_end_time`
  * `validation`: just after `training_end_time` → `validation_end_time`
  * `test`: just after `validation_end_time` → dataset end

* Summary tables record:

  * start/end times, row counts, anomaly counts, and proportions,
  * integrity check confirming split sizes add up.

**Scaling**

* Fit `StandardScaler` using only `train_df["value"]` (Celsius).
* Store `ambient_scaler_params = {"mean": ..., "std": ...}`.
* Apply scaler to full dataframe to create `value_scaled`.

**Final column order**

* Ambient processed schema:

  `time, value, value_scaled, is_anomaly, hour_of_day, day_of_week, is_weekend, split, case_study`.

### 2.3 Ambient outputs

Saved under `data/processed/ambient/`:

* `ambient_processed_full.csv` – full processed ambient dataset.
* `ambient_train.csv` – `split == "train"`.
* `ambient_validation.csv` – `split == "validation"`.
* `ambient_test.csv` – `split == "test"`.

These files are the canonical processed artefacts for Case Study A.

---

## 3. Case Study B – NYC taxi demand (NAB nyc_taxi)

### 3.1 Raw data summary

* Source file: `nyc_taxi.csv` (NAB).
* Interpretation: number of taxi trips per 30 minutes in New York City.
* 10 320 rows from 2014-07-01 00:00 to 2015-01-31 23:30.
* Strong daily and weekly seasonality; clear weekday–weekend differences.
* Five labelled anomaly timestamps from NAB.

### 3.2 Implemented preprocessing steps (NYC taxi B)

**Step 1 – Load dataset and attach anomaly labels**

* Load `nyc_taxi.csv` and NAB `combined_labels.json`.
* Extract labels for `"realKnownCause/nyc_taxi.csv"`.
* Convert label list to datetimes and create `is_anomaly` via `.isin(...)`.

**Step 2 – Standardise core columns (`time`, `value`, `is_anomaly`, `case_study`)**

* Convert `timestamp` to `datetime`, sort chronologically.
* Rename `timestamp → time`.
* Add `case_study = "nyc_taxi"`.

**Step 3 – Missing values and duplicate-rows sanity check**

* Missing-values summary: 0 missing values in all core columns.
* Duplicate-rows summary: 0 exact duplicates.

**Step 4 – Time-based features**

* Add:

  * `hour_of_day = df["time"].dt.hour`
  * `day_of_week = df["time"].dt.weekday`
  * `is_weekend = df["day_of_week"].isin([5, 6]).astype(int)`

**Step 5 – Inspect anomalies and overall time span**

* Anomaly summary table lists anomaly IDs, timestamps, and trip counts.
* Overall span summary lists start/end time, row count, and duration.

**Step 6 – Define chronological train / validation / test boundaries**

* Calendar-month split design:

  * Training: July–October 2014
  * Validation: November–December 2014
  * Test: January 2015

* Implemented via:

  * `training_end_time = "2014-10-31 23:30:00"`
  * `validation_end_time = "2014-12-31 23:30:00"`

* Anomaly-versus-split preview confirms:

  * 0 anomalies in training
  * 3 anomalies in validation
  * 2 anomalies in test

**Step 7 – Assign split labels and summarise segments**

* Create `df["split"]` from the boundary timestamps.
* Summary table reports start/end times, rows, anomalies, and proportions.

**Step 8 – Create split-specific dataframes and integrity check**

* Create split dataframes via masks and `.copy()`.
* Integrity table confirms split sizes add up to the full dataset.

**Step 9 – Standardise NYC taxi demand**

* Fit `StandardScaler` on `train_df[["value"]]` only.
* Store parameters in `nyc_taxi_scaler_params`.
* Apply to full dataframe to produce `value_scaled`.

**Step 10 – Column order and recreation of splits**

* Reorder to common schema.
* Recreate split dataframes from final `df` so that all splits include `value_scaled` and the final column order.

### 3.3 NYC taxi outputs

Saved under `data/processed/nyc_taxi/`:

* `nyc_taxi_full.csv` – full processed NYC taxi dataset.
* `nyc_taxi_train.csv` – `split == "train"`.
* `nyc_taxi_val.csv` – `split == "validation"`.
* `nyc_taxi_test.csv` – `split == "test"`.

All splits are derived directly from `nyc_taxi_full.csv`.

---

## 4. Case Study C – CPU utilisation (ASG misconfiguration, NAB)

> **Status:** Preprocessing **implemented and saved**.

### 4.1 Raw data summary

* Source file: `cpu_utilization_asg_misconfiguration.csv` (NAB `realKnownCause`).
* Interpretation: CPU utilisation (%) for an autoscaling group under misconfiguration conditions.
* Sampling: 5-minute observations.
* Label source: NAB `combined_labels.json` with key `"realKnownCause/cpu_utilization_asg_misconfiguration.csv"`.
* Labelled anomalies: 2 timestamps.

### 4.2 Implemented preprocessing steps (CPU C)

**Step 1 – Load dataset and attach anomaly labels**

* Load raw CSV and NAB `combined_labels.json`.
* Extract labels for `"realKnownCause/cpu_utilization_asg_misconfiguration.csv"`.
* Convert timestamps to `datetime`, sort chronologically.
* Create `is_anomaly` via `df["timestamp"].isin(cpu_label_times).astype(int)`.
* Class balance is extreme (two labelled anomalies in the full series).

**Step 2 – Standardise core columns (`time`, `value`, `is_anomaly`, `case_study`)**

* Convert `timestamp` to `datetime` and rename `timestamp → time`.
* Add `case_study = "cpu"`.
* The `value` field represents CPU utilisation in percent.

**Step 3 – Missing values and duplicate-rows sanity check**

* Missing-values summary table confirms 0 missing in core columns.
* Duplicate-rows summary confirms 0 exact duplicate rows.

**Step 4 – Time-based features**

* Add:

  * `hour_of_day`
  * `day_of_week`
  * `is_weekend`

**Step 5 – Inspect anomalies and overall time span (plus sampling confirmation)**

* Anomaly summary table lists the two labelled anomaly timestamps and their CPU utilisation values.
* Time span summary table reports dataset start/end, row count, and duration.
* A sampling check confirms 5-minute spacing as the most common interval.

**Step 6 – Define chronological boundaries using regime structure (CPU-specific design)**

The CPU split design uses a regime-informed approach because the labelled anomalies occur near the end of the series and align with late-stage changes in behaviour.

* **Baseline reference window (7-day cutoff)**

  * A baseline period is defined as the early part of the series ending 7 days before the first labelled anomaly.
  * A sensitivity check compares candidate baseline cutoffs (7, 14, 21 days) and shows similar mean and threshold behaviour across options, supporting a 7-day cutoff while retaining more baseline data.

* **Reference thresholds from baseline behaviour**

  * High reference threshold: 99.5th percentile of baseline hourly values.
  * Low reference threshold: 5th percentile of baseline hourly values.

* **Hourly overview series (analysis-only for boundary selection)**

  * An hourly view is created to support robust boundary selection and visual interpretation of sustained changes.
  * A 24-hour moving median is used to smooth the hourly series to focus on sustained transitions rather than isolated spikes.

* **Regime change points and split boundaries**

  * The start of the higher/unstable period is identified when the smoothed hourly series first rises above the high reference threshold.
  * The start of the later lower period is identified when the smoothed hourly series first falls below the low reference threshold.

* **Buffers tied to smoothing window**

  * Buffers are applied so that split boundaries do not sit directly on the detected transition points.
  * With a 24-hour smoothing window:

    * training buffer: 48 hours
    * validation buffer: 12 hours

* **Minimum test duration constraint (CPU-specific split validity)**

  * A minimum test duration is enforced to prevent the evaluation tail from collapsing.
  * The minimum duration is set to 2 days under 5-minute sampling.
  * If necessary, `validation_end_time` is adjusted earlier to maintain the minimum test coverage while preserving `training_end_time`.

**Step 7 – Assign split labels and summarise segments**

* Assign `split` using the boundary timestamps.
* Summary table reports:

  * start/end times,
  * row counts and durations,
  * anomaly counts,
  * split proportions.

**Step 8 – Create split dataframes and integrity checks**

* Create `train_df`, `validation_df`, `test_df` by masking `df["split"]`.
* Integrity checks confirm:

  * each split is time-sorted,
  * train ends before validation starts,
  * validation ends before test starts,
  * the combined split lengths match the full dataframe.

**Step 9 – Standardise CPU utilisation**

* Fit `StandardScaler` on `train_df[["value"]]` only.

* Store parameters:

  ```python
  cpu_scaler_params = {"mean": ..., "std": ...}
  ```

* Apply to full dataframe to create `value_scaled`.

* A training-only summary confirms mean ≈ 0 and std ≈ 1 on the training split.

**Final column order**

* CPU processed schema:

  `time, value, value_scaled, is_anomaly, hour_of_day, day_of_week, is_weekend, split, case_study`.

### 4.3 CPU outputs

Saved under `data/processed/cpu_utilisation/`:

* `cpu_utilisation_full.csv` – full processed CPU utilisation dataset.
* `cpu_utilisation_train.csv` – `split == "train"`.
* `cpu_utilisation_validation.csv` – `split == "validation"`.
* `cpu_utilisation_test.csv` – `split == "test"`.

All split-specific files are derived directly from the full processed file.

---

## 5. Case Study D – AIOps KPI (KPI ID da403e4e3f87c9e0)

> **Status:** Preprocessing **planned, not yet implemented**. Data overview and modelling notes exist.

Planned preprocessing:

* Convert Unix `time` to `datetime`, preserve 1-minute resolution with occasional gaps.
* Apply the shared core schema and time features.
* Fit StandardScaler on training split; store `aiops_kpi_scaler_params`.
* Splits designed to:

  * train on earlier lower-anomaly periods,
  * test across the high-anomaly incident window and post-incident period.

Planned outputs:

* `data/processed/aiops_kpi/aiops_kpi_full.csv`
* `aiops_kpi_train.csv`, `aiops_kpi_validation.csv`, `aiops_kpi_test.csv`

(This section will be updated once preprocessing is actually run.)

---

## 6. Cross-case story so far – linking the four case studies

This section captures the ongoing narrative of the research as each case study moves through preprocessing.

### 6.1 Ambient temperature (Case A) – controlled environment with sensor issues

* Represents a managed physical environment (building/room) where temperature must stay within safe bounds.
* Anomalies capture:

  * a short warm spike, and
  * a longer shift to a different baseline (regime change).
* Preprocessing decisions (Celsius conversion, anomaly-based split boundaries, standardised values) emphasise interpretability and clear chronological separation between normal behaviour and later drift/anomalies.

### 6.2 NYC taxi (Case B) – operational demand with strong seasonality

* Represents a dynamic service-demand signal with clear daily/weekly structure.
* Anomalies align with event periods (holidays, storms, disruptions) on top of a strongly recurring baseline.
* Calendar-based splits align with an intuitive operational interpretation while preserving strict chronology.

### 6.3 CPU utilisation (Case C) – infrastructure performance under misconfiguration

* Represents infrastructure performance where sustained regime changes and tail-end instability can occur under misconfiguration.
* Labelled anomalies occur near the end of the series, motivating a regime-informed split design.
* A baseline reference window and threshold-based regime detection are used to set defensible boundary timestamps.
* A minimum test-duration constraint is required to prevent the end-of-series evaluation window from collapsing.

### 6.4 How A, B, and C jointly support the research aim

Together, Ambient (A), NYC taxi (B), and CPU utilisation (C) provide:

* distinct business-relevant signals spanning physical environment control, service demand, and infrastructure performance,
* contrasting anomaly structures (regime changes, event-driven anomalies, and misconfiguration-driven instability), and
* a shared preprocessing pipeline (common schema, chronological splitting, train-only scaling, and saved full-plus-split artefacts).

As the AIOps KPI (D) is brought into the same pipeline, the portfolio will cover:

* dense incident windows,
* higher anomaly rates relative to NAB cases,
* and concept drift patterns under operational disturbance,

supporting comparative evaluation of diffusion-based anomaly detection across different domains, sampling resolutions, and drift structures under a consistent preprocessing framework.
