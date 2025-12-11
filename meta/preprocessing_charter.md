# Preprocessing Charter: Dynamic Business Time-Series Case Studies

This charter defines how raw time-series data for the four case studies are prepared for anomaly detection experiments. It connects the exploratory data overviews in Chapter 4 with the formal methodology in Chapter 3 by describing how raw series become model-ready inputs.

The same shared principles apply to all case studies, with additional dataset-specific decisions noted below.

---

## Shared preprocessing principles (all case studies)

**Overall aim**

- Prepare each time series in a consistent, transparent way so that:
  - models receive clean, well-structured inputs, and  
  - comparisons between diffusion-based methods and baselines are fair across datasets.

**Core data structure**

All processed datasets will follow a common schema:

- `time` – datetime index (standard name for all datasets)  
- `value` – main measurement (temperature, trip count, CPU utilisation, KPI value)  
- `is_anomaly` – binary label (0 = normal, 1 = anomalous)  
- `case_study` – identifier string, e.g. `"ambient"`, `"nyc_taxi"`, `"cpu"`, `"aiops_kpi"`

Optional derived features (added where helpful):

- `hour_of_day` (0–23)  
- `day_of_week` (0–6)  
- `is_weekend` (0/1)  
- dataset-specific flags (e.g. incident window indicators) if needed for analysis or evaluation.

**Time handling**

- Parse all timestamps into a standard datetime format and sort by time.  
- Use each dataset’s natural sampling resolution:
  - hourly (ambient),
  - 30-minute (NYC taxi),
  - 5-minute (CPU),
  - 1-minute (AIOps KPI).
- Treat occasional longer gaps (where they exist) as missing periods in the time axis, not as changes in sampling frequency.
- Resampling to coarser resolutions (e.g. hourly) is reserved for visual summaries, not for the core modelling inputs.

**Scaling**

- Apply scaling to `value` **per case study**, using only the training portion for that case.
- Standardisation (subtract mean, divide by standard deviation) will be the default, so:
  - large raw ranges (e.g. taxi trip counts) do not dominate training, and  
  - models see values on comparable scales across datasets.

**Labels and evaluation**

- Use `is_anomaly` as a 0/1 label attached to each time step.  
- Labels are used primarily for:
  - evaluation (precision, recall, F1, event-level metrics), and  
  - descriptive analysis of anomaly shapes and regimes.
- Given the strong class imbalance in all four case studies, event- or window-based evaluation around labelled anomalies will be preferred over raw point-wise accuracy.

**Train / validation / test splits**

- Define splits **chronologically** for each dataset to respect time order and concept drift:
  - earlier periods used for training,
  - later periods reserved for validation and testing.
- Ensure that:
  - at least some labelled anomalies appear in the test region, and  
  - drift or regime changes are reflected in the way splits are drawn.
- Exact split boundaries will be documented in the Research Methodology chapter and mirrored in the preprocessing notebooks.

**Outputs and file structure**

- Processed files will be saved under `data/processed/` with a clear layout, for example:
  - `data/processed/ambient_temperature/…`
  - `data/processed/nyc_taxi/…`
  - `data/processed/cpu_utilization/…`
  - `data/processed/aiops_kpi/…`
- Each folder will contain:
  - processed series (train/validation/test, or a unified file with split markers),
  - the standard columns described above.

---

## Case Study A – Ambient temperature (NAB sensor failure)

**Raw data summary**

- File: `ambient_temperature_system_failure.csv`  
- Columns: `timestamp`, `value` (°F), `is_anomaly` (0/1)  
- ~7 267 hourly readings, no missing values, no duplicate timestamps  
- Two labelled anomalies: one warm spike and one regime-change event

**Preprocessing plan**

- **Time handling**
  - Convert `timestamp` to datetime and sort.
  - Treat as an hourly series with occasional missing blocks (longer gaps).
  - Keep gaps as-is; no artificial filling at this stage.

- **Unit conversion and scaling**
  - Convert temperature from Fahrenheit (°F) to Celsius (°C) for better alignment with the South African context:
    - `value_celsius = (value_fahrenheit - 32) * 5/9`.
  - Use the Celsius values as the main `value` column in the processed dataset.
  - Apply standardisation to the Celsius series (on the training portion).

- **Labels**
  - Retain `is_anomaly` from NAB labels.
  - Use anomalies both as:
    - point-level markers, and  
    - anchors for window-based event detection (warm spike vs regime shift).

- **Splits**
  - Choose a chronological split that keeps:
    - sufficient normal data before anomalies for training, and  
    - the anomaly events in validation/test segments for evaluation.

---

## Case Study B – NYC taxi demand (NAB)

**Raw data summary**

- File: `nyc_taxi.csv`  
- Columns: `timestamp`, `value` (trip counts per 30 minutes)  
- 10 320 half-hourly readings from 2014-07-01 to 2015-01-31  
- No missing values, no duplicate timestamps  
- Five labelled anomaly timestamps from NAB

**Preprocessing plan**

- **Time handling**
  - Convert `timestamp` to datetime and sort.
  - Preserve the native 30-minute frequency; no resampling for core modelling.

- **Feature preparation**
  - Add time-based features:
    - `hour_of_day`,
    - `day_of_week`,
    - `is_weekend`.
  - These features help capture strong daily and weekly seasonality.

- **Scaling**
  - Standardise the `value` series (trips per 30 minutes) using training data only.

- **Labels**
  - Integrate NAB labels into `is_anomaly` (0/1).
  - Use labels for:
    - event-level evaluation around each anomaly timestamp, and  
    - characterising spike- and dip-type anomalies in zoomed views.

- **Splits**
  - Define early months as training and later months as validation/test, ensuring:
    - time order is respected,
    - some anomalies fall into the test region for evaluation of rare-event detection.

---

## Case Study C – CPU utilisation (ASG misconfiguration, NAB)

**Raw data summary**

- File: `cpu_utilization_asg_misconfiguration.csv`  
- Columns: `timestamp`, `value` (CPU utilisation %)  
- 18 050 readings at a regular 5-minute interval  
- No missing values, no duplicate timestamps  
- Distinct regimes: moderate-load, high-load unstable, and low-load after a sharp drop  
- Two labelled anomaly timestamps from NAB

**Preprocessing plan**

- **Time handling**
  - Convert `timestamp` to datetime and sort.
  - Keep the native 5-minute sampling as the modelling resolution.
  - Use hourly aggregates only for plots and diagnostics, not as model input.

- **Scaling**
  - Standardise the CPU percentage `value` based on the training portion.

- **Optional time features**
  - Add `hour_of_day` and `day_of_week` if daily structure is found to be relevant, while keeping the focus on regime changes.

- **Labels**
  - Create `is_anomaly` (0/1) for the two labelled points.
  - Treat anomalies as indicators of regime-change events rather than isolated spikes.

- **Splits**
  - Design splits that train primarily on the long moderate-load regime and test on:
    - the high-load unstable regime, and  
    - the low-load regime after the drop.
  - This explicitly supports analysis of concept drift and transitions between regimes.

---

## Case Study D – AIOps KPI (KPI ID da403e4e3f87c9e0)

**Raw data summary**

- Source: 2018 AIOps KPI anomaly dataset (`train.csv`, filtered by KPI ID)  
- Columns for the chosen KPI: `time` (Unix seconds), `value` (unitless KPI), `label` (0/1)  
- 129 035 observations at mostly 1-minute resolution, with a small number of longer gaps  
- Clear normal band around 2–3 units, rare excursions up to around 19  
- Anomalies include spikes, dips and dense bursts, with a pronounced high-anomaly incident window

**Preprocessing plan**

- **Time handling**
  - Convert Unix `time` to datetime and sort.
  - Treat as a 1-minute series with occasional longer gaps.
  - Keep gaps in the index; optional gap flags can be added later if needed for analysis.

- **Scaling**
  - Standardise the unitless `value` using the training portion.

- **Time-derived features**
  - Add `hour_of_day` and `day_of_week` to support modelling of daily patterns.
  - Optionally add an incident-window flag for analysis (not required as model input).

- **Labels**
  - Use `label` as `is_anomaly` (0/1).
  - Use both point-wise labels and aggregated daily anomaly ratios when designing evaluation schemes.

- **Splits**
  - Arrange splits to train on earlier, lower-anomaly periods and test across:
    - the high-anomaly incident window, and  
    - the post-incident period.
  - This directly supports evaluation of robustness under changing anomaly intensity and concept drift.

---

## Link to dissertation chapters

- **Chapter 3 – Research Methodology**
  - This charter underpins the “Data preparation / preprocessing” section.
  - The shared principles and dataset-specific plans will be summarised and justified here.

- **Chapter 4 – Results/Findings**
  - The data overviews for each case study describe the raw series and anomaly structure.
  - All reported modelling results will be based on datasets processed according to this charter.

This document serves as the reference for all preprocessing notebooks and ensures that implementation remains consistent with the methodological choices described in the dissertation.
