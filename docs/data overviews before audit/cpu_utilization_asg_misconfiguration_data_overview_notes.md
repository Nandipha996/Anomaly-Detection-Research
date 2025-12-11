# CPU Utilisation (ASG Misconfiguration) – Data Overview

## 1. Dataset context and role in the project

* This dataset records **CPU utilisation (%) over time** for a group of cloud servers in an **auto-scaling group (ASG)**.
* Each row corresponds to the **CPU usage at a specific 5-minute timestamp**.
* The dataset is part of the Numenta Anomaly Benchmark (NAB) and is documented as involving an **auto-scaling misconfiguration** that leads to abnormal CPU behaviour.
* In the thesis, this dataset serves as **Case Study C**, complementing:

  * **Case Study A:** Ambient temperature sensor failure (physical / environment).
  * **Case Study B:** NYC taxi demand (urban demand / events).
* The CPU dataset represents the **cloud operations / reliability** angle of “dynamic business time series,” where anomalies correspond to operational incidents and configuration problems.

## 2. Basic structure: shape and data types

* After loading, the main DataFrame (named `cpu_data`) has:

  * **18 050 rows** and **2 columns**.
  * Columns:

    * `timestamp` – time of the reading (initially text, converted to `datetime64[ns]`).
    * `value` – CPU utilisation as a percentage (`float64`).
* There are no additional feature columns in the raw file; it is a **univariate time series** with a clear operational meaning (percentage of CPU in use).

## 3. Time index and sampling frequency

* After conversion and sorting, the series runs from:

  * **Start:** 2014‑05‑14 01:14:00
  * **End:**   2014‑07‑15 17:19:00
* The difference between consecutive timestamps is almost always:

  * **0 days 00:05:00** (5 minutes), appearing **18 049 times**, which matches the number of gaps between 18 050 rows.
* This confirms that:

  * The series is recorded at a **regular 5-minute interval**.
  * There are no missing rows in the time grid.
* For later modelling, this means we can **retain the native 5‑minute resolution** and optionally create **hourly aggregates** for high-level visualisation, without needing time-index repair.

## 4. Data quality: missing values and duplicates

* A missing value check shows:

  * **0 missing values** in `timestamp`.
  * **0 missing values** in `value`.
* A duplicate check on `timestamp` shows:

  * **0 duplicated timestamps**.
* Therefore, the series is **complete and unique** in terms of time stamps:

  * No gaps due to missing rows.
  * No repeated time points that would require aggregation or removal.
* Preprocessing can focus on **type conversion, scaling, and feature preparation**, rather than heavy cleaning.

## 5. Summary statistics: understanding CPU levels

Summary statistics for the `value` (CPU utilisation %):

* **Count:** 18 050 readings.
* **Mean:** ≈ **38.28%**.
* **Median:** ≈ **32.00%**.
* **Q1 (25%):** ≈ **30.79%**.
* **Q3 (75%):** ≈ **35.66%**.
* **Standard deviation:** ≈ **15.64 percentage points**.
* **Minimum:** ≈ **11.53%**.
* **Maximum:** **100%**.

Key interpretations:

* The dataset provides a **dense, two‑month log of CPU usage** with regular 5‑minute sampling.
* The **average** utilisation is about **38%**, while the **median** is about **32%**.

  * This indicates that the system spends a large amount of time at relatively **moderate CPU levels**, but there are **higher‑CPU periods** that pull the mean upward.
* The **middle 50% of values (Q1–Q3)** lie in a **narrow band between about 31% and 36%**, showing that normal behaviour is **tightly centred** around a moderate load level.
* The **standard deviation** of about **15.6** suggests that most readings cluster near the 30–40% band, with fewer readings further away.
* The **minimum** around 11.5% shows the system is **rarely idle**, and the **maximum of 100%** indicates periods of **full CPU saturation**.

For the research story:

* This dataset describes a system that typically runs at **one‑third to two‑fifths capacity**, with occasional periods of **much higher load**.
* Unlike NYC taxi (which has a wide spread of normal values), CPU here has a **tight “healthy” band** and **sparser high‑utilisation episodes**, making it a good test case for detecting **departures from a well‑defined baseline**.

## 6. Value distribution: histogram insights

* The histogram of CPU utilisation shows that:

  * The highest bars are concentrated in the **30–40%** range.
  * These bins contain **thousands of readings each**, reflecting the tight normal band seen in the summary statistics.
  * Bins corresponding to very low utilisation (around 10–20%) and high utilisation (above 50%) contain **much fewer readings**, so their bars appear shorter.
* This confirms that:

  * The system spends most of its time in a **stable, moderate‑load state**.
  * **Very high CPU levels** and **very low CPU levels** are relatively **rare**, which is consistent with a real production system that usually runs near a safe operating region.

From a modelling perspective, this suggests that:

* The 30–40% region should be treated as the **normal operating band**.
* Diffusion models and baselines should pay attention to periods where utilisation moves **sustainedly away** from this band.

## 7. Visual overview: hourly average CPU time series

* A line plot of the **hourly average CPU utilisation** provides a clearer high‑level view than plotting every 5‑minute point.
* Main observations:

  * For most of the period, the hourly average sits in the **35–40%** range, with gentle variation and occasional short spikes.
  * There is a **gradual upward drift** in CPU levels in some parts of the series, but the overall behaviour remains within the moderate band.
  * Near the end of the series, there is a **clear change**:

    * CPU jumps into a **higher sustained level** (around **50–65%**) for a period.
    * Afterwards, utilisation **drops sharply** into a **low range** (about **10–20%**).
* This behaviour suggests the presence of **distinct operating regimes**:

  * a long **moderate‑load regime**,
  * a **higher‑load unstable regime** near the end,
  * followed by a **low‑load regime** after the sharp drop.

For the thesis, this visual supports describing the CPU dataset as a **regime‑changing operational time series**, rather than one with only isolated point anomalies.

## 8. Anomaly labels from NAB

* Anomaly labels are loaded from NAB’s `combined_labels.json` file.
* For key `"realKnownCause/cpu_utilization_asg_misconfiguration.csv"`, the label list contains **2 anomaly timestamps**.
* These timestamps are:

  * **2014‑07‑12 02:04:00**
  * **2014‑07‑14 21:44:00**
* These labels are converted into a small DataFrame `cpu_labels_df` with a `timestamp` column using `pd.to_datetime`.

These timestamps indicate times when the NAB designers consider the CPU behaviour to be **anomalous due to the ASG misconfiguration**.

## 9. CPU series with labelled anomalies (hourly overview)

* When the two labelled timestamps are overlaid as red points on the **hourly average CPU plot**:

  * The **first anomaly** lies inside the **high‑CPU regime** near the end of the series, where average utilisation is elevated.
  * The **second anomaly** lies in the **low‑CPU region** after the sharp drop.
* Taken together, the labels do not simply mark isolated spikes; they mark times that sit within or just after **substantial changes in the operating regime**.
* Given over **18 000 readings**, having only **2 anomaly labels** emphasises the **rare‑event nature** of this dataset.

## 10. Class balance: normal vs anomalies

* By adding an `is_anomaly` column (1 for labelled timestamps, 0 otherwise), we obtain the following class counts:

  * **normal:** 18 048 points
  * **anomaly:** 2 points
* This shows a **very strong class imbalance**.
* For evaluation, this implies that:

  * Simple accuracy metrics are not informative.
  * We should focus on **event‑level measures** such as whether the model raises an alarm in a **time window around each labelled anomaly** and how many extra alarms it produces elsewhere.

## 11. Zoomed views of anomaly regions (5‑minute resolution)

Two 24‑hour windows around the labelled anomalies were plotted using the original 5‑minute readings.

### Anomaly 1 window

* The 24‑hour window around **2014‑07‑12 02:04:00** shows:

  * A **very spiky pattern**, with frequent jumps up towards **100% utilisation**.
  * Between spikes, CPU often returns to around **30–40%**, but overall the pattern is **unstable**, with rapid switches between medium and very high levels.
* The labelled anomaly point:

  * Sits inside this unstable region, at a CPU level in roughly the **mid‑range of the spikes**.
  * It is one point within a **cluster of high‑activity readings**, not a single isolated outlier.
* Interpretation:

  * This anomaly is associated with a period where the system is **persistently busy and unstable**, consistent with a problematic or misconfigured state.

### Anomaly 2 window

* The 24‑hour window around **2014‑07‑14 21:44:00** shows:

  * Initially, a **high and noisy CPU region**, with utilisation often between **50% and 80%** and repeated spikes.
  * Then, a **clear downward break**, where CPU drops from this high, noisy pattern into a much **lower band** around **10–20%**.
  * After the drop, CPU remains low and relatively stable, with only small fluctuations compared to the earlier spikes.
* The labelled anomaly point:

  * Sits just **after the drop**, inside the new low‑CPU regime.
* Interpretation:

  * This anomaly is aligned with a **transition from a high‑load, unstable regime to a low‑load, quiet regime**.

### Overall view of anomalies

* The two labelled anomalies lie within or immediately after **qualitatively different local patterns**:

  * Anomaly 1: inside a **high‑activity, unstable regime**.
  * Anomaly 2: at the start of a **new low‑activity regime**.
* This supports describing anomalies in this dataset as **regime‑change events**, rather than simple one‑point spikes.

## 12. Preprocessing and modelling implications

Based on the overview, the following choices are appropriate for preprocessing and modelling:

* **Time handling**

  * Keep the native **5‑minute sampling** for modelling.
  * Use **hourly resampling** as needed for visual summaries, not as the main modelling frequency.

* **Data cleaning**

  * No missing values or duplicate timestamps → no heavy cleaning required.
  * Preprocessing focuses on **timestamp parsing**, **sorting**, and **indexing by time**.

* **Scaling**

  * Given the range (≈11.5% to 100%) and tight normal band (≈31–36%), apply **scaling** (e.g., standardisation) to the `value` column before feeding the series into diffusion models and other algorithms.

* **Feature preparation**

  * Although the core series is univariate, additional **time‑based features** such as `hour_of_day` and `day_of_week` can be derived if needed, to help methods capture daily patterns.

* **Label integration and evaluation**

  * Use an `is_anomaly` flag (0/1) derived from the two labelled timestamps.
  * Because anomalies are extremely rare, employ **window‑based evaluation** around the labelled times and report metrics such as **event‑level recall and precision**, rather than relying solely on pointwise accuracy.

* **Role in the diffusion‑based anomaly detection study**

  * This dataset tests whether diffusion models can learn a **tight, stable normal baseline** and detect **regime shifts** where CPU utilisation moves into sustained high or low states.
  * It complements the other case studies by representing **cloud infrastructure and operations**, rounding out the notion of “dynamic business time series” across physical environment (ambient temperature), demand (NYC taxi), and system performance (CPU utilisation with misconfiguration).

In the final thesis, this overview will appear in the **Data Exploration** section of Chapter 4 as the operations‑focused case study, providing context and justification for the preprocessing pipeline and the design of diffusion‑based and baseline anomaly detection experiments on this dataset.
