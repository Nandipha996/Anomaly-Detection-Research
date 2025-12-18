# Preprocessing Charter – Dynamic Business Time-Series Case Studies

This charter defines how raw time-series data for the four case studies are prepared for anomaly-detection experiments. It links:

* the **exploratory data overviews** (Chapter 4 descriptions), and
* the **formal methodology** (Chapter 3),

by specifying how each raw series becomes a model-ready dataset.

Implementation status per case:

* **Case A (Ambient)** – preprocessing **implemented and saved**.
* **Case B (NYC taxi)** – preprocessing **implemented and saved**.
* **Case C (CPU)** – preprocessing **planned, not yet implemented**.
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
  * CPU / AIOps KPI: analogous dictionaries to be created when preprocessing is implemented.

* The fitted scaler is then applied to the **full processed dataframe** (train, validation, test) to produce `value_scaled`.

This gives:

* `value_scaled` with mean ≈ 0 and std ≈ 1 on the **training split**,
* validation and test on the same scale but with their own means/variances (reflecting drift).

### 1.5 Splits and non-destructive design

* Splits are always **chronological**; no shuffling.

* For each case, there is **one full processed dataframe** (e.g. `*_full.csv`) that retains **all rows**.

* Split-specific dataframes (`train_df`, `validation_df`, `test_df`) are created with **boolean masks** on `df["split"]` and `.copy()`, e.g.:

  ```python
  train_df = df[df["split"] == "train"].copy()
  ```

* **Integrity checks confirm**:

  ```text
  len(df) == len(train_df) + len(validation_df) + len(test_df)
  ```

  so that no rows are dropped or duplicated across splits.

* Split design goals:

  * Training captures **normal behaviour** and earlier regimes.
  * Validation and test include labelled anomalies and, where possible, **regime changes** or drift.
  * At least some anomalies must appear in the test split for final evaluation.

### 1.6 Outputs and directory structure

Under a project-level `data/processed/` directory, each case has its own subfolder:

* `data/processed/ambient/`
* `data/processed/nyc_taxi/`
* `data/processed/cpu/`
* `data/processed/aiops_kpi/`

Each subfolder contains:

* one **full processed file** (`*_full.csv` or `*_processed_full.csv`), and
* three **split files** (`*_train.csv`, `*_val.csv` or `*_validation.csv`, `*_test.csv`),

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

* Anomaly times (from summary table):

  * First anomaly: 2013-12-22 20:00
  * Second anomaly: 2014-04-13 09:00

* Boundaries chosen as **7 days before each anomaly**:

  * `training_end_time = first_anomaly_time - 7 days`
  * `validation_end_time = second_anomaly_time - 7 days`

* Split assignment:

  * `train` : dataset start (2013-07-04 00:00) → `training_end_time`
  * `validation` : just after `training_end_time` → `validation_end_time`
  * `test` : just after `validation_end_time` → dataset end

* `split_summary` records for each split:

  * `Start_time`, `End_time`,
  * `Rows`, `Anomalies`,
  * `Proportion (%)` of total rows.
  * Result: 1 anomaly in validation, 1 in test, none in train.

**Scaling**

* Fit `StandardScaler` using **only** `train_df["value"]` (Celsius).
* Store parameters in `ambient_scaler_params = {"mean": ..., "std": ...}`.
* Apply scaler to full dataframe to create `value_scaled`.
* Training split: `value_scaled` mean ≈ 0, std ≈ 1.
* Split-level summary (`mean`, `std`, `min`, `max`) documents drift across validation and test.

**Final column order**

* Ambient processed schema:

  `time, value, value_scaled, is_anomaly, hour_of_day, day_of_week, is_weekend, split, case_study`.

### 2.3 Ambient outputs

Saved under `data/processed/ambient/`:

* `ambient_processed_full.csv` – full processed ambient dataset.
* `ambient_train.csv` – `split == "train"`.
* `ambient_validation.csv` – `split == "validation"`.
* `ambient_test.csv` – `split == "test"`.

These files are the **canonical processed artefacts** for Case Study A.

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
* Class balance:

  * 10 315 normal points
  * 5 anomalies (≈ 0.048% of series).

**Step 2 – Standardise core columns (`time`, `value`, `is_anomaly`, `case_study`)**

* Convert `timestamp` to `datetime`, sort chronologically.
* Rename `timestamp → time`.
* Add `case_study = "nyc_taxi"`.
* Core view: `time, value, is_anomaly, case_study`.

**Step 3 – Missing values and duplicate-rows sanity check**

* Missing-values summary: 0 missing values in all core columns.
* Duplicate-rows summary: 0 exact duplicates.

No imputation or deduplication is required.

**Step 4 – Time-based features**

* Add:

  * `hour_of_day = df["time"].dt.hour`
  * `day_of_week = df["time"].dt.weekday`
  * `is_weekend = df["day_of_week"].isin([5, 6]).astype(int)`

These features capture strong daily and weekly structure.

**Step 5 – Inspect anomalies and overall time span**

* Anomaly summary table:

  * Anomaly ID (1–5)
  * `time` (timestamp)
  * `value` (trip count)

* Overall span summary:

  * Start time: 2014-07-01 00:00
  * End time: 2015-01-31 23:30
  * Number of rows: 10 320
  * Total duration: 214 days

These are later used to justify split design and coverage.

**Step 6 – Define chronological train / validation / test boundaries**

* Use **calendar months**:

  * Training: July–October 2014
  * Validation: November–December 2014
  * Test: January 2015

* Implemented via timestamps:

  * `training_end_time = "2014-10-31 23:30:00"`
  * `validation_end_time = "2014-12-31 23:30:00"`

* Boundary summary table lists start/end of dataset and these boundaries.

* Anomaly-versus-split preview confirms:

  * 0 anomalies in training
  * 3 anomalies in validation
  * 2 anomalies in test

**Step 7 – Assign split labels and summarise segments**

* Define `assign_split(time_value)` using the boundaries to return `"train"`, `"validation"`, or `"test"`.
* Create `df["split"]` by applying this function to `df["time"]`.
* `split_summary` (rows ordered train → validation → test) includes:

  * `Start_time`, `End_time`
  * `Rows`, `Anomalies`
  * `Proportion (%)`

Current segment sizes:

* Train: 5 904 rows (≈ 57.2%), 0 anomalies
* Validation: 2 928 rows (≈ 28.4%), 3 anomalies
* Test: 1 488 rows (≈ 14.4%), 2 anomalies

**Step 8 – Create split-specific dataframes and integrity check**

* Non-destructive splits:

  ```python
  train_df = df[df["split"] == "train"].copy()
  validation_df = df[df["split"] == "validation"].copy()
  test_df = df[df["split"] == "test"].copy()
  ```

* Integrity table shows:

  * `len(df)` = 10 320
  * `len(train_df) + len(validation_df) + len(test_df)` = 10 320
  * No rows lost or duplicated.

**Step 9 – Standardise NYC taxi demand**

* Fit `StandardScaler` on `train_df[["value"]]` only.

* Store parameters:

  ```python
  nyc_taxi_scaler_params = {
      "mean": float(scaler.mean_[0]),
      "std": float(scaler.scale_[0]),
  }
  ```

* Apply to full dataframe:

  ```python
  df["value_scaled"] = scaler.transform(df[["value"]])
  ```

* Training split check:

  * `value_scaled` mean ≈ 0.0000
  * `value_scaled` std ≈ 1.0001

* Split-level summary (`mean`, `std`, `min`, `max` of `value_scaled`) documents how validation and test differ from training while remaining on the same scale.

**Step 10 – Column order and recreation of splits**

* Reorder NYC taxi dataframe to common schema:

  `time, value, value_scaled, is_anomaly, hour_of_day, day_of_week, is_weekend, split, case_study`.

* Recreate split dataframes from this final `df` using `split` masks, so that `train_df`, `validation_df`, `test_df` all contain `value_scaled` and the final column order.

### 3.3 NYC taxi outputs

Saved under `data/processed/nyc_taxi/`:

* `nyc_taxi_full.csv` – full processed NYC taxi dataset with all rows and columns.
* `nyc_taxi_train.csv` – `split == "train"`; used for model fitting and scaling.
* `nyc_taxi_val.csv` – `split == "validation"`; used for tuning and early comparison of methods.
* `nyc_taxi_test.csv` – `split == "test"`; held-out set for final evaluation.

All three splits are derived directly from `nyc_taxi_full.csv`, ensuring a reproducible link between preprocessing and experiments.

---

## 4. Case Study C – CPU utilisation (ASG misconfiguration, NAB)

> **Status:** Preprocessing **planned, not yet implemented**. Data overview and notes already exist.

Planned preprocessing (to be implemented following the same global rules):

* Core schema and time features identical to Ambient/NYC taxi.
* Convert timestamps to `datetime`, retain 5-minute sampling.
* StandardScaler fitted on CPU training split only; store `cpu_scaler_params`.
* Splits designed around regime structure (moderate → high unstable → low) so that training focuses on early regimes and evaluation covers transitions and anomaly windows.
* Target outputs:

  * `data/processed/cpu/cpu_full.csv`
  * `cpu_train.csv`, `cpu_val.csv`, `cpu_test.csv`.

(This section will be updated once CPU preprocessing is implemented.)

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
* `aiops_kpi_train.csv`, `aiops_kpi_val.csv`, `aiops_kpi_test.csv`.

(This section will be updated once preprocessing is actually run.)

---

## 6. Cross-case story so far – linking Ambient and NYC taxi

This section captures the **ongoing narrative** of the research as each case study moves through preprocessing.

### 6.1 Ambient temperature (Case A) – controlled environment with sensor issues

* Represents a **managed physical environment** (building/room) where temperature must stay within safe bounds.
* Anomalies capture:

  * a short warm spike, and
  * a longer shift to a different baseline (regime change).
* Preprocessing decisions (Celsius conversion, anomaly-based split boundaries, standardised values) emphasise:

  * interpretability in a South African context, and
  * clear separation between pre-failure normal behaviour and later drift/anomalies.
* This dataset serves as a compact demonstration of **drift and rare sensor-type failures** in a relatively simple univariate series.

### 6.2 NYC taxi (Case B) – operational demand with strong seasonality

* Represents a **dynamic service-demand signal** with clear daily/weekly structure.
* Anomalies are linked to specific event periods (holidays, storms, etc.) and sit on top of strong recurring patterns.
* Calendar-based splits (train on July–October, validate on November–December, test on January) align with:

  * operator intuition (thinking in months), and
  * the goal of testing models on **later months with different event patterns and seasonal phases**.
* Standardised `value_scaled` allows comparison of demand behaviour across splits and eventually across case studies, without raw units dominating optimisation.

### 6.3 How A and B jointly support the research aim

Together, Ambient A and NYC taxi B already provide:

* Two distinct types of **business-relevant time series**:

  * environmental control (ambient temperature), and
  * urban mobility demand (NYC taxi).
* Contrasting anomaly structures:

  * regime shift vs. isolated events on a strongly seasonal baseline.
* A shared, fully implemented preprocessing pipeline:

  * identical core schema,
  * chronological splits designed with anomalies and drift in mind,
  * `value_scaled` based on train-only statistics,
  * clear saved artefacts (full + splits) ready for baselines and diffusion models.

As CPU (Case C) and AIOps KPI (Case D) are brought into the same pipeline, this narrative will extend to:

* infrastructure performance under misconfiguration (CPU), and
* service-level KPIs with dense incident windows (AIOps),

creating a portfolio of four case studies that collectively test diffusion-based anomaly detection across different **operational domains, sampling resolutions, and drift patterns**, all under a consistent preprocessing and evaluation framework.
