## Case Study B – NYC taxi demand: data overview

### Dataset description and context

- Case Study B uses the NAB `nyc_taxi.csv` dataset, which records **half-hourly NYC taxi demand**.  
- Each row represents a **30-minute time slot** between **2014-07-01 00:00** and **2015-01-31 23:30**.  
- The `value` column gives the **number of taxi trips completed in that half hour across New York City**.
- The series is treated as a **dynamic operational business time series**, similar to metrics that transport or mobility services would monitor for planning, staffing and operational decisions.

- Within the thesis, NYC taxi acts as Case Study B and complements:
  - Case Study A: ambient temperature (physical / sensor-style failure),
  - Case Study C: CPU utilisation (cloud / infrastructure operations).

Together, these case studies support the overall objective of evaluating diffusion models for anomaly detection in **dynamic business time series**, where normal behaviour is noisy and anomalies are rare but important.

---

### Structure, length and sampling

- The loaded DataFrame has:
  - **10 320 rows**,  
  - **2 columns**.

- Columns:
  - `timestamp`: initially loaded as text (`object`), converted to `datetime64[ns]`,
  - `value`: number of trips per 30 minutes (`int64`).

- There are no extra metadata columns in this raw version; the dataset is a **clean univariate time series**.

- Time index and sampling:
  - After conversion, `timestamp` is strictly increasing, with no backward jumps,
  - The dominant gap between consecutive timestamps is **30 minutes**,
  - This confirms that the series is recorded at a **regular half-hourly frequency** across the full period.

- Interpretation:
  - No resampling is needed at this stage,
  - The **native 30-minute resolution** can be preserved for downstream anomaly detection,
  - The regular structure supports the goal of building reusable preprocessing pipelines, since models do not need to handle irregular sampling for this dataset.

---

### Data quality: missing values and duplicates

- Missing values:
  - 0 missing values in `timestamp`,
  - 0 missing values in `value`.

- Duplicates:
  - 0 duplicated timestamps.

- Interpretation:
  - The time series is **complete over its defined range** (no gaps due to missing rows),
  - There are no repeated time points that would require aggregation or correction,
  - Preprocessing can focus on standardisation and feature preparation, rather than heavy cleaning.

---

### Value distribution and demand levels

Summary statistics for `value` (trips per 30 minutes):

- Count: 10 320 observations  
- Mean: ≈ 15 138 trips  
- Median: ≈ 16 778 trips  
- 25th percentile (Q1): ≈ 10 262 trips  
- 75th percentile (Q3): ≈ 19 839 trips  
- Standard deviation: ≈ 6 939 trips  
- Minimum: 8 trips  
- Maximum: ≈ 39 197 trips  

Key interpretations:

- The dataset provides **over seven months** of continuous demand, giving enough observations for stable model training and evaluation.
- The **median is higher than the mean**, suggesting the presence of some very low-demand periods that pull the average down.
- The interquartile range (Q1–Q3) is wide (roughly 10k–20k trips), indicating **high natural variability** even within the middle 50% of values.
- The standard deviation is large relative to the mean, reinforcing that **normal behaviour is noisy** rather than tightly clustered.
- The extreme values (minimum 8, maximum ≈ 39k) show that the series includes both very low and very high demand episodes.

---

### Histogram and normal operating band

A histogram of `value` shows that:

- Most half-hour periods fall in a **broad band roughly between 10k and 22k trips**.
- There is a clear cluster of **lower-demand periods** (a few thousand trips), likely corresponding to late-night or early-morning hours.
- The distribution has a **long right tail**, with fewer but still noticeable counts at **very high values** (above 30k trips).

Interpretation:

- Normal demand already covers a **wide spread**; there is no single narrow “typical” level.  
- The dataset contains **rare but large peaks**, which are natural candidates for anomaly labels.  
- For modelling, this supports:
  - applying scaling or normalisation before training, and
  - choosing baselines and evaluation metrics that respect the wide spread of normal values.

---

### Full-series behaviour and seasonal structure

A full-series line plot of `value` over time reveals:

- **Clear daily cycles**:
  - demand rises during the day,
  - drops at night,
  - forming a repeating wave-like pattern.
- **Weekly patterns**:
  - some days are systematically busier than others,
  - consistent with weekday–weekend differences.
- **Visually prominent spikes**:
  - some peaks stand noticeably above the regular pattern,
  - these are likely linked to special events or disruptions.

Interpretation:

- NYC taxi demand is **seasonal (daily and weekly)** and **event-driven**.
- The series is visually dense but interpretable with suitable plotting choices (thin lines, light grid).
- This confirms that NYC taxi is well aligned with the project’s focus on **dynamic normality and drift**, not static or purely synthetic behaviour.

---

### Local pattern: example week

Zooming into a single week (for example 2014-07-01 to 2014-07-08) shows:

- A strong **wave pattern** within each day:
  - low values at night,
  - peaks during busy periods.
- **Consistent daily structure** that repeats across the week, matching intuitive expectations about city transport demand.
- Some days that are overall **busier than others**, hinting at a weekday–weekend effect.

Interpretation:

- The local view makes the **repeating baseline structure** more obvious.
- Anomaly detection should be aware of **time-of-day** and **day-of-week** effects to avoid mis-labelling every peak or trough as an anomaly.
- For the thesis, this supports the argument that methods must model **seasonal patterns** rather than treating every large value as suspicious.

---

### Anomaly information and class balance

- NAB provides anomaly labels for NYC taxi via `labels.json`.  
- For the key `"realKnownCause/nyc_taxi.csv"`, there are **5 labelled anomaly timestamps**.
- These labelled times represent events where the benchmark designers consider the series to exhibit **unusual behaviour** (for example major holidays, storms or city-scale events).

After joining labels to the main DataFrame:

- Each timestamp is marked as **anomaly (1)** or **normal (0)**.
- Class counts show a strong imbalance:
  - Normal points: just over **10 300**,
  - Anomaly points: **5**.

Interpretation:

- NYC taxi is a **rare-event detection** setting:
  - most of the time, behaviour is normal,
  - anomalies are very rare but potentially important.
- High overall accuracy would be easy to achieve by predicting “normal” everywhere, so **precision, recall and F1** on anomaly windows are more informative than raw accuracy.

---

### Zoomed anomaly regions and anomaly shapes

For each labelled anomaly, a **24-hour window** (one day before and one day after the anomaly timestamp) is plotted:

- Within each window, the main line shows half-hourly taxi demand.
- The labelled anomaly point is highlighted, for example as a red marker.

These zoomed plots help reveal whether each anomaly behaves like:

- a **clear spike** above the usual daily pattern,
- a **deep drop** in demand compared to neighbouring points,
- or a **local structural change**, where the overall shape of the daily pattern looks different (e.g., damped peaks or displaced timing).

Interpretation:

- The 24-hour context makes it easier to compare each anomaly with:
  - the **typical daily cycle** (daytime rise, night-time fall), and
  - nearby days that may be busier or quieter.
- In Chapter 4, these views can support short descriptions of **anomaly types** (spike, dip or pattern change) and highlight why NYC taxi is a realistic testbed for diffusion-based anomaly detection under **rare but impactful events**.

---

### Preprocessing and modelling implications

From the data overview, the following design choices are reasonable for the preprocessing phase:

- **Time handling**
  - Keep the native **30-minute frequency**.
  - No resampling is necessary for the core experiments, though daily aggregates can be explored in additional analyses.

- **Data cleaning**
  - No missing values or duplicated timestamps → no heavy cleaning required.
  - Preprocessing focuses on type conversion, sorting and merging labels.

- **Feature preparation**
  - Derived features can include:
    - time-of-day (hour),
    - day-of-week,
    - flags for weekends or recognised holidays.
  - These features help models and baselines encode the **strong seasonal structure**.

- **Scaling**
  - Values span from **8** to **≈ 39 197** with a wide normal band.
  - Applying scaling (for example standardisation) to the `value` series will likely make model training more stable and comparisons across datasets more meaningful.

- **Label integration**
  - Store an `is_anomaly` column (0/1) alongside the main series for experiments.
  - Use these labels consistently across baseline and diffusion models for fair comparison.

---

### Role within the case study portfolio

Within the overall case study portfolio, NYC taxi demand contributes:

- a **demand-focused business time series** representing city-wide mobility behaviour,  
- a **noisy, seasonal** operational signal with strong daily and weekly structure,  
- **rare labelled anomaly events** that likely correspond to real-world disruptions or special events,  
- a setting where simple fixed thresholds are unlikely to work and more **flexible, data-driven methods** are justified.

It complements:

- the **physical/environmental** perspective of ambient temperature, and  
- the **infrastructure/operations** perspective of CPU utilisation,

and provides a realistic test case for how diffusion-based anomaly detection methods and simpler baselines handle complex normal patterns, rare events and strong seasonality in a business-relevant context.
