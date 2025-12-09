## Case Study D – AIOps KPI anomaly dataset: data overview

### Dataset description and context

Case Study D is based on a single univariate KPI from the 2018 AIOps anomaly detection dataset. The original data are provided in a stacked training file (`train.csv`), which contains multiple KPIs. Case Study D is obtained by filtering this file on one specific `KPI ID` and treating the resulting series as a separate case study.

Each row in the stacked dataset contains:

* a Unix timestamp in seconds,
* a KPI value on an anonymised (unitless) scale,
* a binary label indicating normal (0) or anomalous (1) behaviour,
* a `KPI ID` identifying the underlying time series.

The authors do not disclose the exact business meaning or physical units of each KPI. For this series, the values are therefore interpreted as an anonymised operational metric (for example load, throughput or error rate) measured on a unitless scale. The focus is on relative behaviour (normal band versus excursions) and temporal patterns rather than on physical units.

This KPI is treated as a minute-by-minute operational “heartbeat” of an online system, with anomaly labels supplied by the dataset creators. It forms one of the main business-style case studies used to evaluate diffusion-based anomaly detection methods against baseline models under class imbalance and concept drift.

---

### Structure, length and sampling

After filtering on the chosen `KPI ID` and sorting by timestamp, the resulting series contains:

* 129 035 observations on an anonymised value scale.
* Timestamps spanning several months of continuous monitoring.
* One KPI measurement per row.

Time differences between consecutive timestamps were analysed to characterise the sampling behaviour:

* The dominant time difference is 60 seconds (1 minute), and at least 75% of all gaps are exactly one minute.
* A small number of larger gaps appear (for example 2–4 minutes, 10 minutes and several longer gaps on the order of hours). These gaps correspond to short missing blocks where one or more expected readings are not present.

Overall, the series behaves as a regular 1-minute time series with a few missing periods. This structure matches typical operational monitoring setups and is suitable for time-series anomaly detection. For modelling purposes, the KPI can be treated as a regularly sampled 1-minute series, while the rare longer gaps can either be left as breaks in the time axis or explicitly flagged as missing periods if required.

---

### Value distribution and normal operating band

Summary statistics for the KPI values show the following pattern:

* Count: 129 035 observations
* Mean ≈ 2.45
* Median ≈ 2.25
* Standard deviation ≈ 1.15
* 25th percentile (Q1) ≈ 1.92
* 75th percentile (Q3) ≈ 3.08
* Minimum = 0.00
* Maximum ≈ 19.25

A histogram of the KPI values, restricted to the 0–6 range on the x-axis, reveals a unimodal distribution with:

* a strong concentration of values between roughly 1.5 and 3.5,
* a clear peak around 2–3, and
* rapidly decreasing counts as values move toward 4–5.

Only a small number of observations occur above approximately 5–6, indicating that large excursions are rare.

Together, these statistics and the histogram identify a normal operating band centred around 2–3 units on the anonymised scale. Most 1-minute observations lie within this central band, while excursions towards 0 or beyond 5–6 represent unusual behaviour. This separation between a dense central band and sparse extremes supports the interpretation of the KPI as a mostly stable operational metric with occasional extreme deviations that align with labelled anomalies.

---

### Full-series behaviour and anomaly patterns

A full-series plot of the KPI over time, with normal points shown as a blue curve and anomalies highlighted as red markers, provides a high-level view of the temporal structure:

* Under normal conditions, the series exhibits a regular daily pattern with repeated peaks and troughs, consistent with usage cycles over a 24-hour period. These cycles generally remain within the central operating band identified by the histogram and summary statistics.
* Anomalies appear as:

  * high spikes well above the normal daily peaks,
  * low dips close to zero,
  * and bursts of consecutive points where many timestamps are labelled anomalous.

A zoomed view focusing on the 0–6 value range shows that:

* most normal points oscillate between roughly 1.5 and 4.5, with many values in the 2–3 range,
* anomalies can sit both above and below this band,
* some anomaly clusters appear as dense segments where the usual daily waveform is strongly disrupted.

This combination of a clear daily baseline and varied anomaly shapes (spikes, dips, and bursts) makes the series a rich test case for anomaly detection methods.

---

### Class balance and anomaly intensity

Class balance for the KPI is strongly skewed towards normal behaviour:

* Normal points (label = 0): 121 369 observations (≈ 94.1%).
* Anomalous points (label = 1): 7 666 observations (≈ 5.9%).

The series is therefore imbalanced, reflecting the realistic setting where faults and incidents are rare compared with normal operation. At the same time, the anomaly class is represented by several thousand observations, which is substantially richer than datasets that contain only a handful of labelled anomalies.

This pattern supports the use of the series as a case study where anomaly detectors must operate under class imbalance but still have enough labelled events to support a meaningful evaluation. Labels are most naturally used for evaluation and event analysis rather than as a balanced supervised training signal.

---

### Daily anomaly ratio and regime changes

To examine how anomaly behaviour changes over time, the fraction of anomalous points per calendar day was computed and plotted:

* For most of the monitoring period, the daily anomaly ratio remains close to zero, indicating that almost all 1-minute observations on those days are labelled as normal.
* Around the middle of the observation period, the daily anomaly ratio increases sharply and reaches values close to 1.0 on some days. Ratios near 1.0 correspond to days where almost every minute is labelled anomalous, marking a sustained incident period rather than isolated outliers.
* After this high-activity window, the daily anomaly ratio returns to low levels, with only occasional smaller increases.

This daily view provides clear evidence of non-stationary anomaly behaviour and regime changes in the KPI:

* an initial stable regime with rare anomalies,
* a pronounced incident regime with dense anomalies,
* and a subsequent regime where anomaly activity drops back toward earlier levels.

Such dynamics are important for the broader research aim of studying diffusion-based anomaly detection under concept drift and evolving anomaly structure, and they motivate a focus on window-based and event-level evaluation rather than only point-wise accuracy.

---

### Implications for preprocessing and modelling

Based on the data overview, the following preprocessing and modelling considerations are identified for Case Study D:

* **Time index and resampling**

  * Use the pandas datetime index at the original 1-minute resolution.
  * Treat the series as regularly sampled, with rare longer gaps considered as missing periods rather than changes in sampling rate.

* **Scaling and feature representation**

  * Apply standardisation or normalisation to the KPI values (for example, z-score scaling on a training split) to stabilise the value range before fitting diffusion models and baseline methods.
  * Optional time-derived features such as hour-of-day or day-of-week may be constructed if needed for specific baselines, but the primary analysis can treat the KPI as a univariate sequence.

* **Use of labels**

  * Retain the binary labels for evaluation and analysis of detection performance.
  * Emphasise metrics that handle imbalance (precision, recall, F-score) and event-level detection quality, particularly around the high-anomaly incident window.

* **Focus on regime changes and drift**

  * Design experiments to examine how models behave across different regimes: early stable period, incident window, and later period.
  * Consider evaluation schemes that reflect temporal order (for example training on earlier data and testing on later periods) to study robustness to drift and changing anomaly intensity.

---

### Role within the case study portfolio

In the overall project, Case Study D complements the other case studies (ambient temperature, NYC taxi demand and NAB CPU utilisation) by providing:

* a high-frequency, 1-minute operational KPI with a clear daily cycle,
* a substantial number of anomaly labels distributed across multiple regimes,
* and a distinct high-anomaly incident window suitable for studying concept drift and evolving anomaly behaviour.

Together, these properties make Case Study D a strong operational test case for evaluating diffusion-based anomaly detection methods against baseline models under realistic monitoring conditions.
