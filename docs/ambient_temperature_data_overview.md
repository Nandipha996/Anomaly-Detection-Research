## Case Study A – NAB ambient temperature: data overview

### Dataset description and context

- Case Study A uses the `ambient_temperature_system_failure.csv` file from the NAB (Numenta Anomaly Benchmark) dataset.  
- The series is interpreted as an operational business time series, similar to what a building management system, warehouse or other controlled environment would log to support:
  - energy efficiency decisions,
  - asset and equipment protection,
  - environmental control and comfort.
- The file is treated as a time-dependent signal that allows investigation of:
  - how anomalies appear in practice,
  - how rare events compare to normal behaviour,
  - how slow changes in the environment relate to concept drift.
- All analysis in this data overview is performed on the **raw temperature values in degrees Fahrenheit (°F)**. Any conversion to Celsius (°C) is deferred to a later preprocessing stage.

---

### Structure, length and sampling

- Source file: `ambient_temperature_system_failure.csv` from the NAB dataset.  
- Final working dataframe columns:
  - `timestamp`: datetime index of each reading,
  - `value`: ambient temperature in °F,
  - `is_anomaly`: binary label (0 = normal, 1 = anomaly).
- Approximate size:
  - about **7 267 hourly readings**,
  - **3 columns**.

- Data types:
  - `timestamp` stored as a proper datetime type,
  - `value` stored as floating point,
  - `is_anomaly` stored as an integer class indicator.

- Anomaly labels:
  - labels come from the NAB `combined_labels.json` file,
  - there are **2 labelled anomaly timestamps** for this file,
  - labels are joined back onto the main dataframe using `timestamp` as the key so each row knows whether it is normal or anomalous.

- Time index and sampling regularity:
  - the dataframe is sorted by `timestamp` and index reset so that row order follows time,
  - the dominant gap between readings is **exactly 1 hour**,
  - a small number of longer gaps (for example 1, 2 or more days) appear only a few times,
  - there are **no duplicated timestamps**.

- Interpretation:
  - the series behaves as an **hourly time series** with occasional missing chunks,
  - longer gaps likely reflect periods where the sensor or logging system was offline,
  - this is consistent with a realistic operational system where data can disappear in blocks rather than forming a perfectly continuous stream,
  - the clean, unique time index is useful later when constructing windows for baselines and diffusion-based models.

---

### Missing values

- A simple count and percentage check per column shows:
  - **no missing values** in `timestamp`, `value` or `is_anomaly`.

- Interpretation:
  - data quality issues are not about nulls inside the series, but about **gaps at the time index level**,
  - this simplifies preprocessing because no value imputation is required at this early stage.

---

### Value distribution and normal operating band

Summary statistics for the `value` (temperature in °F), rounded for interpretation:

- Count: ~7 267 readings  
- Mean: ≈ 71.2 °F  
- Standard deviation: ≈ 4.3 °F  
- Minimum: ≈ 57.5 °F  
- 25th percentile (Q1): ≈ 68.4 °F  
- Median (50%): ≈ 71.9 °F  
- 75th percentile (Q3): ≈ 74.4 °F  
- Maximum: ≈ 86.2 °F  

A histogram with 40 bins shows:

- a **unimodal and fairly compact** distribution, consistent with a controlled indoor environment,
- most temperatures lying between about **68 and 75 °F**, which forms the **normal operating band**,
- a **rare upper tail** extending into the **80+ °F** region, up to around 86 °F.

Interpretation:

- The system spends most of its time in a comfortable range, with temperatures clustered around the low 70s °F.
- Values consistently outside this band, especially sustained excursions on the high side, are unusual and naturally align with the notion of “warm spike” anomalies.
- This separation between a dense central band and a sparse upper tail provides a clear notion of normal versus extreme behaviour for later modelling.

---

### Full-series behaviour and labelled anomalies

A full time-series plot was produced with:

- a blue line for the raw hourly temperature values,
- red markers at timestamps where `is_anomaly == 1`.

At the full-series level this shows that:

- the signal is not random noise; there are **visible patterns and cycles** over time,
- labelled anomalies are **very rare** compared with normal points.

To understand the labelled anomalies more clearly, the series was zoomed around each anomaly timestamp using small local windows.

- **Anomaly 1 – warm spike behaviour**
  - appears as a **sharp, local spike** where temperature rises well above the normal operating band and then returns to normal shortly afterwards,
  - is consistent with a short-term overheating event or sudden disturbance,
  - behaves like a **point-type anomaly** embedded in an otherwise stable neighbourhood.

- **Anomaly 2 – change in regime**
  - corresponds more to a **change in behaviour** than to a single isolated point,
  - around this time the average level shifts and the pattern after the anomaly looks different from the pattern before,
  - resembles a **regime shift** in the system, linking naturally to the idea of concept drift in an operational environment.

Together, these two labelled anomalies provide two important shapes for later analysis:

- a **spike-type anomaly**, and  
- a **shift/regime-change anomaly**.

These shapes are useful reference cases when evaluating whether different models can handle both local spikes and longer-term changes.

---

### Moving average and slower changes over time

To explore slower background shifts in temperature:

- a **7-day moving average** (7 × 24 hours) was calculated over the series,
- the smoothed curve was plotted together with the raw hourly data.

This view shows:

- the raw hourly line capturing all short-term variability and noise,  
- the 7-day moving average highlighting the **baseline level** at which the system operates over weeks.

Interpretation:

- longer-term ups and downs in the baseline become easier to see without being distracted by every small spike or dip,
- the moving average provides a visual link between this dataset and the idea of **gradual drift** in the underlying environment,
- this view will be useful when designing splits for drift experiments or when describing how the ambient conditions evolve over time.

---

### Anomaly summary and class imbalance

A simple anomaly summary shows:

- Total points: ~7 267  
- Number of labelled anomalies: **2**  
- Number of normal points: the remainder of the series  
- Anomalies as a percentage of all points: roughly **0.03%**

Interpretation:

- this is a **strongly imbalanced** setting where anomalies are extremely rare,
- the pattern matches realistic monitoring scenarios, where most of the time systems operate normally,
- anomaly detection methods must be evaluated in a way that respects this imbalance and does not simply favour the majority class,
- labels provide event markers rather than a balanced supervised training set.

---

### Implications for preprocessing and modelling

Key implications from the data overview for later stages of the project:

- **Time index and units**
  - use the hourly datetime index as the foundation for later windowing and splitting,
  - treat longer time gaps as missing periods rather than changes in sampling rate,
  - convert °F to °C cleanly in a dedicated preprocessing step if needed, keeping the original series intact in this overview.

- **Scaling and representation**
  - the compact central band suggests that simple scaling (for example standardisation) will stabilise values for diffusion models and baselines,
  - the structure of the series supports a univariate treatment focused on the temperature signal, with time used to define windows and periods.

- **Use of labels**
  - labels should be used primarily for **evaluation and case analysis** rather than as a large training dataset,
  - both spike-type and regime-change anomalies should be considered when interpreting model performance.

- **Concept drift and regimes**
  - the gradual baseline movements and the second anomaly’s regime shift make this series suitable for experiments that distinguish between point anomalies and longer-term changes,
  - the 7-day moving average view will be particularly useful when defining pre- and post-drift periods.

---

### Role within the case study portfolio

Within the broader collection of case studies, this ambient temperature series contributes:

- a **physical environment / equipment monitoring** example,  
- a mostly stable hourly signal with a **narrow normal operating band**,  
- **extremely rare anomalies** that include both a local warm spike and a regime shift,  
- a clear opportunity to discuss **gradual environmental drift** via moving averages.

It therefore serves as a natural starting point for the project: a relatively simple, low-dimensional operational time series that introduces key ideas about anomaly shapes, class imbalance and concept drift before moving on to higher-frequency and more complex business KPIs in the other case studies.
