# Preprocessing Charter: Dynamic Business Time Series Case Studies (Updated)

This charter defines how raw time series data for the four case studies are prepared for anomaly
detection experiments. It connects the exploratory data overviews in Chapter 4 with the formal
methodology in Chapter 3 by describing how raw series become model ready inputs.

The same shared principles apply to all case studies, with additional dataset specific decisions
noted below. Case Study A (Ambient) has now been fully prototyped and provides the template for
subsequent case studies.

---

## 1. Shared preprocessing principles (all case studies)

### 1.1 Overall aim

Prepare each time series in a consistent and transparent way so that:

- models receive clean, well structured inputs
- comparisons between diffusion based methods and baselines are fair across datasets
- preprocessing decisions (units, scaling, splits) can be clearly explained in Chapter 3 and
  reused consistently in Chapter 4 experiments

Each case study remains an operational business time series, with:

- clear anomaly labels
- preserved concept drift and regime changes
- splits that reflect realistic past versus future usage

---

### 1.2 Common schema for processed datasets

All four case studies share the following core schema in their processed form:

- `time`  
  - timestamp as `datetime64[ns]`, sorted in ascending order  
  - natural resolution kept per dataset (hourly, 30 minute, 5 minute, 1 minute)

- `value`  
  - main measurement in interpretable units per case study:  
    - Ambient: temperature in Celsius (converted once from Fahrenheit)  
    - NYC taxi: number of trips per 30 minutes  
    - CPU: CPU utilisation in percent  
    - AIOps KPI: anonymised KPI units

- `value_scaled`  
  - standardised version of `value` (z score)  
  - obtained by subtracting the training mean and dividing by the training standard deviation  
  - computed separately per case study using the StandardScaler pattern described in Section 1.5

- `is_anomaly`  
  - binary label, 0 = normal, 1 = labelled anomaly  
  - taken from NAB labels or AIOps labels without modification

- `split`  
  - categorical field indicating the chronological segment: `"train"`, `"validation"`, `"test"`

- `case_study`  
  - identifier string, for example `"ambient"`, `"nyc_taxi"`, `"cpu"`, `"aiops_kpi"`

Optional but common time features (when enabled, they should appear in all processed datasets):

- `hour_of_day` (0–23)
- `day_of_week` (0–6, with the convention documented once, for example 0 = Monday)
- `is_weekend` (0/1)

Datasets may have additional analysis specific columns, but these core fields must exist and have the
same meaning everywhere.

---

### 1.3 Time handling

- Parse all timestamps into a standard pandas datetime format and sort by `time`.
- Use each dataset’s natural sampling resolution:
  - hourly (Ambient)
  - 30 minute (NYC taxi)
  - 5 minute (CPU)
  - 1 minute (AIOps KPI)
- Treat occasional longer gaps as missing periods in the time axis, not as changes in sampling
  frequency.
- Resampling to coarser resolutions (for example hourly averages) is allowed for visual summaries
  and descriptive figures, but not used as the core modelling input resolution.

---

### 1.4 Missing values and duplicates

For each case study:

- Perform a missing values check and report:
  - count and percentage of missing values per column in an examiner style table.
- Check for duplicate rows:
  - report total number of rows, number of duplicate rows, and percentage of duplicates.
- The aim is documentation and reassurance:
  - if no missing or duplicate values are found, this is recorded explicitly
  - any non trivial issues would be addressed in a documented way before modelling

These checks do not change the data when there are no anomalies; they serve as a traceable sanity
check in the preprocessing notebooks.

---

### 1.5 Scaling: value and value_scaled

Scaling is applied per case study and follows a common pattern:

- Scaling is always performed **per dataset** using **only the training split**.
- `StandardScaler` (or equivalent) is used with the following procedure:

  1. Create `train_df` as the subset of rows where `split == "train"`.
  2. Fit a scaler on `train_df[["value"]]`, that is the raw interpretable units:
     - do not include validation or test rows in the fit.
  3. Store the learned training statistics in a small dictionary, for example  
     `{"mean": ..., "std": ...}`:
     - these statistics can also be saved in a separate metadata file for reproducibility.
  4. Apply the fitted scaler to the entire dataframe `df[["value"]]` to create `value_scaled`
     for all rows (train, validation, test).

- After scaling:
  - On the training split, `value_scaled` has mean approximately 0 and standard deviation
    approximately 1.
  - On validation and test, `value_scaled` reflects how those periods differ from the training
    distribution, which is relevant for concept drift analysis.

- The original `value` column remains in interpretable units and is never overwritten by the
  scaler.

This pattern was prototyped and verified on Ambient A and will be reused for NYC taxi, CPU, and
AIOps KPI to ensure consistent preprocessing across case studies.

---

### 1.6 Labels and evaluation

- `is_anomaly` is used as a 0/1 label attached to each time step.
- Labels are used for:
  - descriptive analysis of anomaly shapes, regimes, and incident windows
  - evaluation metrics:
    - point level metrics where they make sense
    - event based or window based metrics around anomaly intervals given strong class imbalance

- The extreme rarity of anomalies is explicitly preserved:
  - no oversampling or label modification takes place in preprocessing
  - train, validation, and test are designed to preserve this imbalance for realistic evaluation

---

### 1.7 Train, validation, and test splits and non destructive design

Splits are defined chronologically and in a non destructive manner:

- For each case study there is a full cleaned dataframe `df` that retains **all rows**.
- A `split` column is created in `df` by applying a rule that maps `time` to one of three values:
  `"train"`, `"validation"`, `"test"`.
- `train_df`, `validation_df`, and `test_df` are then created by boolean masks on `df["split"]`, for
  example:

  - `train_df = df[df["split"] == "train"].copy()`
  - `validation_df = df[df["split"] == "validation"].copy()`
  - `test_df = df[df["split"] == "test"].copy()`

- The original `df` is never shrunk by filtering to a single split. It always remains the canonical
  processed dataframe.

Chronological principles:

- Splits respect time order:
  - training covers earlier periods
  - validation and test cover later periods
- Training is designed to be free of labelled anomaly windows where this is reasonably possible.
- Validation and test contain labelled anomalies for model selection and final evaluation.
- The exact split boundaries are chosen per dataset, guided by:
  - anomaly locations
  - regime structures
  - series length

Non destructive integrity:

- For each case study, a simple integrity check verifies that:

  - `len(df) == len(train_df) + len(validation_df) + len(test_df)`

- This is recorded in the preprocessing notebook to show that no rows are lost or duplicated by
  splitting.

---

### 1.8 Outputs and file structure

Processed files are saved under `data/processed/` with a clear layout. A consistent pattern is used,
for example:

- `data/processed/ambient/ambient_processed_full.csv`
- `data/processed/ambient/ambient_train.csv`
- `data/processed/ambient/ambient_validation.csv`
- `data/processed/ambient/ambient_test.csv`

and similarly for:

- `data/processed/nyc_taxi/...`
- `data/processed/cpu/...`
- `data/processed/aiops_kpi/...`

For each case study:

- `*_processed_full.csv` is the canonical processed dataset containing:
  - all rows
  - all core columns (`time`, `value`, `value_scaled`, `is_anomaly`, `split`, `case_study`,
    time features)
- `*_train.csv`, `*_validation.csv`, `*_test.csv` are convenience subsets filtered from the full
  file using the `split` column.

---

## 2. Case Study A – Ambient temperature (NAB sensor failure)

### 2.1 Raw data summary

- Source file: `data/NAB-master/realKnownCause/ambient_temperature_system_failure.csv`
- Original columns:
  - `timestamp` – hourly timestamps in a controlled environment
  - `value` – temperature in Fahrenheit
- Anomaly labels:
  - Provided in `data/NAB-master/labels/combined_labels.json`
  - Two labelled anomaly timestamps for this file:
    - one warm spike above the normal band
    - one regime shift where the baseline level changes
- Data quality:
  - No duplicate timestamps
  - No missing `value` entries in the raw file
- Time range:
  - Hourly timestamps spanning mid 2013 to 2014 (exact start and end timestamps are recorded in the
    ambient preprocessing notebook in the time span summary table)

The series is an operational business style temperature control signal. Concept drift is present in
the form of both a spike and a regime shift. Anomalies are rare and remain correctly labelled.

---

### 2.2 Columns after cleaning and schema alignment

After preprocessing, the Ambient A dataframe `df` has the following columns:

- `time`  
  - hourly timestamps, converted from `timestamp`, sorted in ascending order

- `value`  
  - ambient temperature in Celsius  
  - obtained by converting the original Fahrenheit `value` using  
    `value_celsius = (value_fahrenheit - 32) * 5 / 9`

- `value_scaled`  
  - z score scaled version of `value` based on training data statistics  
  - used as the main numeric feature in many models

- `is_anomaly`  
  - 0/1 labels created by matching timestamps against the anomaly list extracted from
    `combined_labels.json` for this file  
  - two anomaly points

- `hour_of_day`, `day_of_week`, `is_weekend`  
  - derived from `time`  
  - provide simple calendar structure for potential modelling and analysis

- `split`  
  - `"train"`, `"validation"`, `"test"` based on chronological boundaries defined using anomaly
    times and buffer periods

- `case_study`  
  - always `"ambient"`

Column order in `ambient_processed_full.csv` follows:

1. `time`
2. `value`
3. `value_scaled`
4. `is_anomaly`
5. `split`
6. `hour_of_day`
7. `day_of_week`
8. `is_weekend`
9. `case_study`

---

### 2.3 Splits (chronological design and status)

Splits are defined chronologically using the two anomaly timestamps:

- The two anomaly times are extracted into a small `anomaly_summary` table.
- Training and validation end times are defined by subtracting a buffer (for example 7 days) from
  the anomaly times:
  - `training_end_time` = some days before the first anomaly
  - `validation_end_time` = some days before the second anomaly
- A helper function maps each timestamp to one of:
  - `train` – earliest period ending before `training_end_time`
  - `validation` – middle period between `training_end_time` and `validation_end_time`
  - `test` – most recent period after `validation_end_time`

These rules are applied to create the `split` column in `df`. An examiner style summary table shows
for each split:

- start and end times
- number of rows
- number of anomalies
- proportion of total rows

This design ensures:

- one anomaly is reserved for validation
- one anomaly is reserved for test
- training focuses on clearly normal behaviour
- no rows are discarded

The exact timestamps of `training_end_time`, `validation_end_time` and the anomaly times are
documented in the Ambient A preprocessing notebook and can be cited directly in the methodology.

---

### 2.4 Scaling for Ambient A

Scaling is implemented as the template for all case studies:

- A dataframe `train_df` is created as `df[df["split"] == "train"].copy()`.
- A `StandardScaler` instance is fitted on `train_df[["value"]]` (Celsius values only).
- The learned training mean and standard deviation are stored in a dictionary, for example:

  - `ambient_scaler_params = {"mean": ..., "std": ...}`

- The same scaler is then applied to all rows of `df`:

  - `df["value_scaled"] = scaler.transform(df[["value"]])`

- Checks are recorded:

  - On the training split, `value_scaled` has mean approximately 0 and standard deviation
    approximately 1.
  - A summary by split shows mean, standard deviation, minimum, and maximum of `value_scaled` for
    train, validation, and test.

This confirms that:

- Standardisation uses training only statistics
- Concept drift is preserved in validation and test through changes in the distribution of
  `value_scaled`

---

### 2.5 Outputs for Ambient A

Ambient A outputs are saved under:

- `data/processed/ambient/`

with the following CSV files:

- `ambient_processed_full.csv`  
  - canonical processed ambient dataset  
  - includes all rows and all core columns

- `ambient_train.csv`  
  - rows where `split == "train"`

- `ambient_validation.csv`  
  - rows where `split == "validation"`

- `ambient_test.csv`  
  - rows where `split == "test"`

Each of the split specific files includes the full common schema:

- `time`, `value`, `value_scaled`, `is_anomaly`, `split`, `case_study`, `hour_of_day`,
  `day_of_week`, `is_weekend`

---

## 3. Case Study B – NYC taxi demand (NAB)

### 3.1 Raw data summary

- Source file: `nyc_taxi.csv` (NAB)
- Columns:
  - `timestamp` – 30 minute timestamps
  - `value` – number of taxi trips per 30 minutes
- Time range:
  - from mid 2014 to early 2015 (exact range recorded in the NYC taxi overview notebook)
- Five labelled anomaly timestamps from NAB
- No missing values and no duplicate timestamps in the raw file

The series represents an operational demand signal for urban transport. Daily and weekly patterns are
present, and anomalies correspond to unusual events such as holidays or disruptions.

---

### 3.2 Preprocessing plan for NYC taxi

- Time handling:
  - convert `timestamp` to datetime and sort
  - keep native 30 minute resolution for modelling
- Schema alignment:
  - rename `timestamp` to `time`
  - keep `value` as trip count per 30 minutes
  - add `case_study = "nyc_taxi"`
- Features:
  - add `hour_of_day`, `day_of_week`, `is_weekend`
- Labels:
  - build `is_anomaly` using NAB anomaly timestamps
- Splits:
  - define chronological training, validation, and test segments
  - training on earlier months with normal behaviour
  - validation and test covering periods with labelled anomalous events
- Scaling:
  - fit a StandardScaler on `train_df[["value"]]` (trip counts)
  - store scaler parameters
  - compute `value_scaled` for all rows using the training based parameters
- Outputs:
  - `nyc_taxi_processed_full.csv` plus split specific files, following the same pattern as Ambient A

Details will be finalised after NYC taxi preprocessing is implemented.

---

## 4. Case Study C – CPU utilisation (ASG misconfiguration, NAB)

### 4.1 Raw data summary

- Source file: `cpu_utilization_asg_misconfiguration.csv` (NAB)
- Columns:
  - `timestamp` – 5 minute timestamps
  - `value` – CPU utilisation percentage
- Around 18 050 observations at regular 5 minute intervals
- Two labelled anomaly timestamps from NAB
- Clear regime structure:
  - moderate load regime
  - high load unstable regime
  - low load regime after a sharp drop

The series is an operational cloud infrastructure KPI. Concept drift appears as transitions between
regimes and the behaviour leading into and out of anomaly events.

---

### 4.2 Preprocessing plan for CPU

- Time handling:
  - convert `timestamp` to datetime and sort
  - keep native 5 minute resolution
- Schema alignment:
  - rename `timestamp` to `time`
  - keep `value` as CPU utilisation percentage
  - add `case_study = "cpu"`
- Features:
  - optional `hour_of_day` and `day_of_week` where relevant
- Labels:
  - create `is_anomaly` from NAB labels
- Splits:
  - design splits that:
    - train mainly on the long moderate load regime
    - validate and test on high load and low load regimes
    - preserve the two anomaly events in validation and test
- Scaling:
  - fit StandardScaler on `train_df[["value"]]` (CPU percentage)
  - store scaler parameters
  - compute `value_scaled` for all rows
- Outputs:
  - `cpu_processed_full.csv` and split specific files, consistent with the Ambient A pattern

Final details will be set once CPU preprocessing is implemented.

---

## 5. Case Study D – AIOps KPI (KPI ID da403e4e3f87c9e0)

### 5.1 Raw data summary

- Source: 2018 AIOps KPI anomaly dataset (`train.csv`, filtered by KPI ID)
- Columns for the chosen KPI:
  - `time` (Unix seconds)
  - `value` (unitless KPI)
  - `label` (0/1)
- Around 129 035 observations at mostly 1 minute resolution with some longer gaps
- Normal operating band around 2–3 units, with spikes up to about 19
- Anomalies include:
  - isolated spikes
  - dips towards zero
  - dense bursts during a high anomaly incident window

This series represents an operational KPI in an AIOps setting, with realistic noise, drift, and
changing anomaly intensity.

---

### 5.2 Preprocessing plan for AIOps KPI

- Time handling:
  - convert Unix `time` to datetime and sort
  - treat as a 1 minute series with occasional longer gaps
- Schema alignment:
  - keep `value` as KPI level in unitless scale
  - rename `label` to `is_anomaly`
  - add `case_study = "aiops_kpi"`
- Features:
  - add `hour_of_day`, `day_of_week`, `is_weekend`
  - optional incident window indicators for analysis (not required as model input)
- Splits:
  - arrange splits so that:
    - training focuses on earlier, lower anomaly periods
    - validation and test cover the high anomaly incident window and the post incident period
- Scaling:
  - fit StandardScaler on `train_df[["value"]]`
  - store scaler parameters
  - compute `value_scaled` for all rows, allowing comparison of distributions across splits
- Outputs:
  - `aiops_kpi_processed_full.csv` and split specific files, consistent with the Ambient A pattern

---

## 6. Link to dissertation chapters

- **Chapter 3 – Research Methodology**
  - This charter underpins the "Data preparation and preprocessing" section.
  - It documents shared rules, dataset specific choices, and the rationale for Celsius conversion,
    scaling, and chronological splits.
- **Chapter 4 – Results and Findings**
  - Data overviews for each case study describe the raw series, anomaly structure, and concept
    drift.
  - All reported modelling results are based on datasets processed according to this charter.
  - Event level and window based evaluations will draw directly on the `is_anomaly`, `split`, and
    `value_scaled` structure defined here.

This updated charter reflects the implemented preprocessing for Ambient A and sets the template for
NYC taxi, CPU, and AIOps KPI. It should next be revised once Case Study B preprocessing has been
completed and checked against the same global rules.
