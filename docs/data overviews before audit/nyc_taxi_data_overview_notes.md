# NYC Taxi – Data Overview (Case Study B)

## 1. Dataset context and role in the project

* This dataset records **half-hourly NYC taxi demand**.
* Each row represents **one 30-minute time slot** between **2014-07-01 00:00** and **2015-01-31 23:30**.
* The `value` column is the **number of taxi trips** completed in that half hour across New York City.
* This dataset is treated as a **dynamic operational business time series**, similar to metrics that transport or mobility services would monitor for planning and operations.
* In the thesis, NYC taxi acts as **Case Study B**, complementing:

  * **Case Study A:** `ambient_temperature_system_failure` (sensor / equipment-style failure), and
  * **Case Study C:** `cpu_utilization_asg_misconfiguration` (cloud / infrastructure operations).
* Together, these case studies support the project objective of **evaluating diffusion models for anomaly detection in dynamic business time series**, where normal behaviour is noisy and anomalies are rare but important.

## 2. Basic structure: shape and data types

* The loaded DataFrame has **10 320 rows** and **2 columns**:

  * `timestamp` – initially loaded as text (`object`), later converted to `datetime64[ns]`.
  * `value` – the number of trips per 30 minutes (`int64`).
* There are **no extra metadata columns** in this raw version; the dataset is a **clean univariate time series**.
* This compact structure makes it a suitable input for time-series models and for later transformations into sliding windows or feature matrices.

## 3. Time index and sampling frequency

* After conversion, the `timestamp` column is **strictly increasing**, with no backward jumps.
* The dominant time gap between consecutive timestamps is **30 minutes**.
* This confirms that the series is recorded at a **regular half-hourly frequency** across the full period.
* Because the time grid is already clean and regular:

  * No resampling is needed at this stage.
  * The native half-hour resolution can be preserved for downstream anomaly detection.
* This regular structure supports the project’s goal of creating **reusable preprocessing pipelines**, since models do not need to handle irregular or missing time stamps for this dataset.

## 4. Data quality: missing values and duplicates

* A simple missing-value summary shows **0 missing values** in both columns (`timestamp` and `value`).
* A duplicate check on `timestamp` reports **0 duplicated timestamps**.
* This means:

  * The time series is complete over its defined range (no gaps due to missing rows).
  * There are no repeated time points that would need aggregation or correction.
* For this dataset, preprocessing can focus on **standardisation and feature preparation**, rather than heavy cleaning.

## 5. Summary statistics: understanding demand levels

Summary statistics for the `value` column:

* **Count:** 10 320 observations (half-hour slots).
* **Mean:** ≈ **15 138** trips per 30 minutes.
* **Median:** ≈ **16 778** trips per 30 minutes.
* **Q1 (25%):** ≈ **10 262** trips.
* **Q3 (75%):** ≈ **19 839** trips.
* **Standard deviation:** ≈ **6 939** trips.
* **Minimum:** **8** trips.
* **Maximum:** ≈ **39 197** trips.

Key interpretations:

* The dataset provides a **substantial time window** (over seven months) of continuous demand, giving enough observations for stable model training and evaluation.
* The **mean** and **median** are not identical: the median is higher than the mean.

  * This suggests the presence of **some very low-demand periods** that pull the average down.
  * For anomaly detection, this highlights that **low values** (sharp dips) may be as important as **high values** (spikes).
* The **interquartile range (Q1–Q3)** is wide (roughly 10k–20k trips), indicating **high natural variability** even within the middle 50% of values.
* The **standard deviation** is large relative to the mean, reinforcing that **normal behaviour is noisy** rather than tightly clustered.
* The **minimum (8)** and **maximum (~39k)** show the presence of **extreme lows and highs**.

  * These extremes are likely tied to major events or disruptions (e.g., holidays, severe weather, city events), which are precisely the kinds of behaviours we want to test anomaly detectors on.

For the thesis, these statistics support the narrative that NYC taxi demand is a **dynamic and noisy business time series** where:

* simple fixed thresholds will struggle, and
* more flexible, data-driven methods (including diffusion models) are justified.

## 6. Value distribution: histogram insights

* A histogram of `value` shows that:

  * Most half-hour periods fall in a **broad band** roughly between **10k and 22k trips**.
  * There is a noticeable cluster of **lower-demand periods** (a few thousand trips), likely corresponding to late-night or early-morning hours.
  * The distribution has a **long right tail**, with fewer but significant counts at very high values (above **30k trips**).
* This shape confirms that:

  * Normal demand already covers a **wide spread**.
  * The dataset contains **rare but large peaks**, which are good candidates for anomaly labels.
* In the broader project, this distribution will motivate:

  * Using **scaling or normalisation** before model training.
  * Careful choice of baselines and evaluation metrics, since “normal” is not centred tightly around a single level.

## 7. Visual overview: full-series pattern

* A line plot of `value` over time shows:

  * Clear **daily cycles**: demand rises during the day and falls at night.
  * **Weekly patterns**: some days appear systematically busier than others, suggesting weekend vs weekday differences.
  * Several **visually prominent spikes** that stand above the general pattern, likely corresponding to special events or disruptions.
* The series is **visually dense**, but thinner lines and a light grid make the pattern interpretable for readers.
* This visual confirms that NYC taxi demand is:

  * **Seasonal** (daily and weekly),
  * **Event-driven**, and
  * **Well-suited to concepts of dynamic normality and drift**, which are core to the project’s focus on “dynamic business time series.”

## 8. Local pattern: example week

* Zooming into a single week (e.g., 2014-07-01 to 2014-07-08) reveals:

  * A **strong wave pattern within each day**, with low values at night and peaks during busy periods.
  * Consistent daily structure that repeats across the week, aligning with intuitive expectations about city transport demand.
  * Some days appearing overall busier than others, hinting at **weekday–weekend effects**.
* This local view makes the **repeating baseline structure** more obvious and underlines that anomaly detection should be aware of:

  * time-of-day and
  * day-of-week effects.

For the thesis, this supports the idea that any anomaly detection method evaluated on this series should be tested on its ability to **respect seasonal patterns** rather than mis-labelling every peak or trough as an anomaly.

## 9. Anomaly information: labelled NAB events

* NAB provides anomaly labels for NYC taxi via the `labels.json` structure.
* For the key `"realKnownCause/nyc_taxi.csv"`, there are **5 labelled anomaly timestamps**.
* These labelled times represent events where the benchmark designers believe the series exhibits **unusual behaviour** (e.g., major holidays, storms, or city-scale events).
* When these timestamps are overlaid on the full time series:

  * They appear as **distinct peaks or unusual points** relative to the surrounding pattern.
  * They are clearly **rare** compared to the large number of normal observations.

These 5 labelled anomalies will act as **ground truth** when comparing different anomaly detection methods, including diffusion-based models and simpler baselines.

## 10. Class balance: normal vs anomalies

* After joining labels to the main DataFrame:

  * Each timestamp can be marked as **anomaly (1)** or **normal (0)**.
  * The resulting class counts show a **strong imbalance**:

    * **Normal points:** just over 10 300.
    * **Anomaly points:** **5**.
* This confirms that NYC taxi is a **rare-event detection** problem:

  * Most of the time, behaviour is normal.
  * Anomalies are **very rare**, but potentially important.
* For the research, this imbalance will influence:

  * The choice of **evaluation metrics** (e.g., precision, recall, F1 on anomaly windows).
  * The interpretation of results, since high overall accuracy can be misleading in such settings.

## Zoomed anomaly regions – local views around labelled events

* For each labelled anomaly in the NYC taxi series, a 24-hour window (one day before and one day after the anomaly timestamp) is plotted.
* Within each window, the main line shows **half-hourly taxi demand**, and the labelled anomaly appears as a **red point**.
* These zoomed plots help show whether an anomaly behaves like:

  * a clear **spike** above the usual daily pattern,
  * a deep **drop** in demand compared to neighbouring points, or
  * a more **structural change** where the whole local pattern looks different.
* The 24-hour context makes it easier to see how each anomaly compares to:

  * the typical daily cycle (rise during the day, fall at night), and
  * nearby days that may be busier or quieter.
* In Chapter 4 (Data Exploration), these views will support short descriptions of each anomaly type (spike, dip, or pattern change), and will help justify why NYC taxi is a realistic testbed for diffusion-based anomaly detection under **rare but impactful events**.

## 11. Preprocessing and modelling notes

Based on the overview, the following design choices are reasonable for the preprocessing phase (`02_preprocessing.ipynb`):

* **Time handling**

  * Keep the native **30-minute frequency**.
  * No resampling is necessary for this dataset, though rescaling time windows (e.g., to daily aggregates) may be explored in additional experiments.

* **Data cleaning**

  * No missing values or duplicated timestamps means no heavy cleaning is required.
  * Preprocessing will focus on **type conversion**, **sorting**, and **merging labels**.

* **Feature preparation**

  * Consider adding derived features such as:

    * **Time-of-day** (hour)
    * **Day-of-week**
    * Possibly flags for weekends or holidays
  * These features can help baselines and certain models encode the strong seasonal structure.

* **Scaling**

  * Because values span from 8 to ~39k with a wide normal band, applying scaling (e.g., standardisation) to the `value` series will likely make model training more stable and comparable across datasets.

* **Label integration**

  * Store an `is_anomaly` column (0/1) alongside the main series for experiments.
  * Use these labels consistently across baseline and diffusion models for fair comparison.

## 12. Fit with overall research objectives

NYC taxi fits the broader project aims in several ways:

* It is a **real operational business time series**, representing city-wide transport demand.
* It exhibits **dynamic, noisy normal behaviour** with strong seasonal patterns, aligning with the thesis focus on **dynamic business time series** rather than static or toy examples.
* It includes **rare labelled anomaly events**, giving a realistic rare-event detection setting.
* The clear daily and weekly cycles, combined with rare spikes and dips, make it a strong test case for:

  * How diffusion-based anomaly detection methods capture complex normal patterns.
  * How they compare against **simpler baselines** (moving averages, threshold-based models, classical anomaly detection methods) under realistic variability and imbalance.

In the final dissertation structure, NYC taxi will be presented in the **Data Exploration** section of Chapter 4 as a demand-focused business case study. Its characteristics will be used to justify design choices in preprocessing, modelling, and evaluation, and to demonstrate how diffusion models handle a noisy, event-driven environment where anomalies are both rare and operationally meaningful.
