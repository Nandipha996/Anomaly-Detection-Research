## Case Study C – NAB CPU utilisation (ASG misconfiguration): data overview

### Dataset description and context

- Case Study C uses the `cpu_utilization_asg_misconfiguration.csv` file from the Numenta Anomaly Benchmark (NAB).  
- The series records CPU utilisation (%) over time for a group of cloud servers in an auto-scaling group (ASG). Each row corresponds to the CPU usage at a specific timestamp.  
- NAB documentation describes this file as involving an **auto-scaling misconfiguration** that leads to abnormal CPU behaviour, making it a natural example of an operational incident in a cloud environment.  

- In the thesis, this dataset serves as the **cloud operations / reliability** component of the “dynamic business time series” portfolio, complementing:
  - Case Study A: ambient temperature sensor failure (physical environment),
  - Case Study B: NYC taxi demand (urban demand and events).

It represents an operational time series where anomalies correspond to misconfiguration-driven incidents and regime changes in system load.

---

### Structure, length and sampling

- After loading, the main working DataFrame (`cpu_data`) has:
  - **18 050 rows**,
  - **2 columns**.

- Columns:
  - `timestamp`: time of the reading (initially text, converted to `datetime64[ns]`),
  - `value`: CPU utilisation as a percentage (`float64`).

- The original dataset is purely univariate, with no additional feature columns; the `value` column carries a clear operational meaning as the percentage of CPU in use.

- Time range:
  - Start: **2014-05-14 01:14:00**  
  - End: **2014-07-15 17:19:00**

- Sampling frequency:
  - The difference between consecutive timestamps is almost always **0 days 00:05:00** (5 minutes),
  - This 5-minute gap appears 18 049 times, which matches the number of gaps between 18 050 rows.

- Interpretation:
  - The series is recorded at a **regular 5-minute interval**,
  - There are **no missing rows** in the implied time grid between start and end,
  - For modelling, it is appropriate to retain the native 5-minute resolution and use hourly aggregates only for higher-level visualisation.

---

### Data quality: missing values and duplicates

- Missing value checks show:
  - 0 missing values in `timestamp`,
  - 0 missing values in `value`.

- Duplicate checks show:
  - 0 duplicated timestamps.

- Interpretation:
  - The series is **complete and unique** in terms of time stamps,
  - There are no gaps due to missing rows and no repeated time points that would require aggregation or removal,
  - Preprocessing can focus on type conversion, scaling and feature preparation rather than heavy cleaning.

---

### Value distribution and normal operating band

Summary statistics for `value` (CPU utilisation in %):

- Count: 18 050 readings  
- Mean: ≈ 38.28%  
- Median: ≈ 32.00%  
- 25th percentile (Q1): ≈ 30.79%  
- 75th percentile (Q3): ≈ 35.66%  
- Standard deviation: ≈ 15.64 percentage points  
- Minimum: ≈ 11.53%  
- Maximum: 100%  

Interpretation:

- The dataset provides a dense, two-month log of CPU usage with regular 5-minute sampling.  
- The average utilisation is about 38%, while the median is about 32%, indicating that the system spends much of its time at relatively **moderate CPU levels**, with higher-CPU periods pulling the mean upward.  
- The middle 50% of values (Q1–Q3) lie in a **narrow band between about 31% and 36%**, showing that normal behaviour is tightly centred around a moderate load level.  
- The minimum around 11.5% shows that the system is rarely idle, while the maximum of 100% indicates **periods of full CPU saturation**.

---

### Histogram and normal operating range

A histogram of CPU utilisation values shows that:

- The **highest bars** are concentrated in the **30–40%** range,
- These bins contain thousands of readings each, reflecting the tight normal band seen in the summary statistics,
- Bins corresponding to **very low utilisation** (around 10–20%) and **high utilisation** (above 50%) contain much fewer readings and appear shorter.

Interpretation:

- The system spends most of its time in a **stable, moderate-load state**, consistent with a production environment operating near a safe region.
- Very high CPU levels and very low CPU levels are relatively rare.  
- From a modelling perspective, the **30–40% region** can reasonably be treated as the **normal operating band**, and methods should pay attention to periods where utilisation moves **sustainedly away** from this band.

---

### Hourly-average view and operating regimes

To obtain a high-level picture, an **hourly-average CPU time series** was constructed from the 5-minute data.

Main observations from the hourly plot:

- For most of the period, the hourly average sits in the **35–40% range**, with gentle variation and occasional short spikes.  
- There is a **gradual upward drift** in CPU levels in parts of the series, but the behaviour remains within a moderate band for a long portion of the timeline.  
- Near the end of the series, a clear change in regime appears:
  - CPU jumps into a **higher sustained level** (around 50–65%) for a period,  
  - Afterwards, utilisation drops sharply into a **low range** (about 10–20%).

Interpretation:

- The series exhibits **distinct operating regimes**:
  - a long **moderate-load regime**,  
  - a **higher-load unstable regime** near the end,  
  - followed by a **low-load regime** after the sharp drop.
- This supports describing the CPU dataset as a **regime-changing operational time series**, rather than one dominated by isolated point anomalies.

---

### Anomaly labels and class imbalance

- Anomaly labels are taken from NAB’s `combined_labels.json` file under the key:
  - `"realKnownCause/cpu_utilization_asg_misconfiguration.csv"`.

- The label list contains **2 anomaly timestamps**:

  - 2014-07-12 02:04:00  
  - 2014-07-14 21:44:00  

- These labels are converted into a small DataFrame and joined to the CPU series, creating an `is_anomaly` indicator (1 for labelled timestamps, 0 otherwise).

Class balance:

- Normal points: **18 048**  
- Anomalous points: **2**  

Interpretation:

- The series is **extremely imbalanced**, with only two labelled anomalies in more than 18 000 readings.  
- From an evaluation perspective:
  - simple accuracy metrics are not informative,  
  - focus should be on **event-level behaviour**, such as whether an anomaly detector raises an alarm in a time window around each labelled anomaly and how many extra alarms it produces elsewhere.

---

### Location of labelled anomalies and local patterns

When the two labelled timestamps are overlaid on the **hourly-average CPU plot**:

- The **first anomaly** (2014-07-12 02:04:00):
  - lies within the **high-CPU regime** near the end of the series,
  - occurs during a period where hourly average utilisation is elevated.

- The **second anomaly** (2014-07-14 21:44:00):
  - lies in the **low-CPU region** after the sharp drop in utilisation.

To understand the local behaviour more precisely, two 24-hour windows around the anomalies were plotted using the original 5-minute readings.

**Anomaly 1 window**

- The 24-hour window around 2014-07-12 02:04:00 shows:
  - a very **spiky pattern**, with frequent jumps up towards 100% utilisation,
  - between spikes, CPU often returns to around 30–40%, but overall the pattern is unstable, with rapid switches between medium and very high levels.
- The labelled anomaly point:
  - sits inside this **unstable region**, at a CPU level in roughly the mid-range of the spikes,
  - is one point within a **cluster of high-activity readings**, not a single isolated outlier.
- Interpretation:
  - this anomaly is associated with a period where the system is persistently busy and unstable, consistent with a problematic or misconfigured state.

**Anomaly 2 window**

- The 24-hour window around 2014-07-14 21:44:00 shows:
  - initially, a high and noisy region with utilisation often between 50% and 80% and repeated spikes,
  - then, a **clear downward break**, where CPU drops from this high, noisy pattern into a much lower band around 10–20%,
  - after the drop, CPU remains low and relatively stable, with only small fluctuations compared to the earlier spikes.
- The labelled anomaly point:
  - sits just after the drop, inside the **new low-CPU regime**.
- Interpretation:
  - this anomaly aligns with a **transition** from a high-load, unstable regime to a low-load, quiet regime.

**Overall interpretation of anomalies**

- The two labelled anomalies lie within or immediately after **qualitatively different local patterns**:
  - Anomaly 1: inside a **high-activity, unstable regime**,  
  - Anomaly 2: at the start of a **new low-activity regime**.
- This supports describing anomalies in this dataset as **regime-change events**, rather than purely one-point spikes.

---

### Preprocessing and modelling implications

From the data overview, the following choices are appropriate for preprocessing and modelling:

- **Time handling**
  - Keep the native **5-minute sampling** for modelling, to preserve the operational resolution.
  - Use **hourly resampling** only for visual summaries and high-level diagnostics, not as the primary modelling frequency.

- **Data cleaning**
  - No missing values or duplicate timestamps were found, so no heavy cleaning is required.
  - Preprocessing focuses on timestamp parsing, sorting and indexing by time.

- **Scaling**
  - Given the range (≈ 11.5% to 100%) and the tight normal band (≈ 31–36%), apply scaling (for example standardisation) to the `value` column before feeding the series into diffusion models and other algorithms.

- **Feature preparation**
  - Although the core series is univariate, additional time-based features such as `hour_of_day` and `day_of_week` can be derived if needed to help methods capture daily patterns.

- **Label integration and evaluation**
  - Use an `is_anomaly` flag (0/1) derived from the two labelled timestamps.
  - Because anomalies are extremely rare, employ **window-based evaluation** around the labelled times and report metrics such as event-level recall and precision, rather than relying on pointwise accuracy.

---

### Role within the case study portfolio

Within the broader collection of case studies, the CPU utilisation series contributes:

- a **cloud infrastructure and operations** example of a dynamic business time series,  
- a dense, regularly sampled **5-minute operational KPI** with a tight moderate-load baseline,  
- **extremely rare but meaningful anomalies** that correspond to changes between high-load unstable and low-load quiet regimes,  
- a clear illustration of **regime changes** driven by an ASG misconfiguration.

It therefore plays a key role in testing whether diffusion-based anomaly detectors and baseline methods can:

- learn a tight, stable normal baseline,  
- detect transitions into sustained high-load or low-load states,  
- and operate effectively under extreme class imbalance in a realistic cloud operations setting.
