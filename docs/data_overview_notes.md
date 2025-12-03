# Data Overview Notes – NAB Ambient Temperature Series

## 1. Context of this dataset in my project

This notebook works with the **ambient_temperature_system_failure** subset from the NAB dataset. I am treating it as an **operational business time series**, similar to what a building management system, warehouse, or controlled environment would log to support:

* energy efficiency decisions
* asset and equipment protection
* environmental control and comfort

In this project, it becomes my first concrete example of a time dependent signal where I can study:

* how anomalies show up in practice
* how rare events compare to normal behaviour
* how slow changes in the environment relate to concept drift

Everything in this Data Overview is done on the **raw series in Fahrenheit (°F)**. Any later conversion to Celsius (°C) will happen in a dedicated preprocessing stage.

---

## 2. Basic structure of the dataset

File used:

* `ambient_temperature_system_failure.csv` from NAB

Columns in the final dataframe:

* `timestamp`: datetime index of each reading
* `value`: ambient temperature reading in °F
* `is_anomaly`: binary label (0 = normal, 1 = anomaly)

Key points:

* Number of rows: about 7 267 hourly readings
* Number of columns: 3
* Data types are consistent with expectations:

  * `timestamp` is stored as a proper datetime type
  * `value` is a floating point numeric column
  * `is_anomaly` is an integer column used as a simple class indicator

The anomaly labels come from the NAB `combined_labels.json` file. For this file there are 2 labelled anomaly timestamps. These labels were joined back to the main dataframe using the timestamp as the key so that each row knows whether it is normal or anomalous.

---

## 3. Time index and sampling regularity

I checked how the time index behaves before doing anything more complex.

What I did:

* Sorted by `timestamp` and reset the index so that row order follows time
* Computed `timestamp.diff()` to see the gap between each reading and the previous one
* Looked at both a small sample of gaps and the most frequent gap values
* Checked for duplicated timestamps

What I found:

* The **dominant gap** between readings is exactly **1 hour**
* There are a **small number of longer gaps** (for example 1, 2, or more days) that appear only a few times
* There are **no duplicated timestamps**, so each time step is unique after sorting

Interpretation:

* This is essentially an **hourly time series** with occasional missing chunks
* Longer gaps likely reflect periods where the sensor or logging system was offline or not recording
* From a modelling perspective, this is closer to a realistic operational system, where data can disappear in blocks rather than one neat continuous stream
* The clean, unique time index will be useful later when constructing windows for baselines and diffusion based models

---

## 4. Missing values

I checked for missing values in each column using a simple count and percentage per column.

Result:

* No missing values were found in any of the three columns (`timestamp`, `value`, `is_anomaly`)

Interpretation:

* Data quality issues in this file are not about nulls inside the series, but about **time gaps** at the index level
* This simplifies preprocessing because I do not need to perform value imputation at this early stage

---

## 5. Temperature value distribution and possible outliers

I looked at basic summary statistics of the `value` column and then plotted a histogram with 40 bins.

Main statistics (rounded for interpretation):

* Count: ~7 267 readings
* Mean: about 71.2 °F
* Standard deviation: about 4.3 °F
* Minimum: about 57.5 °F
* Q1 (25%): about 68.4 °F
* Median (50%): about 71.9 °F
* Q3 (75%): about 74.4 °F
* Maximum: about 86.2 °F

Interpretation of the histogram and summary:

* The distribution is **unimodal** and fairly compact, which fits the idea of a controlled indoor environment
* Most temperatures lie between **about 68 and 75 °F**, which I can think of as the normal operating band
* Rare higher values extend into the **80+ °F** region, up to around 86 °F
* These high values live in the upper tail and are natural candidates for warm spike anomalies

This gives me a clear mental picture: the system spends most of its time in a comfortable range, and anything consistently outside that band, especially on the high side, is unusual and worth attention.

---

## 6. Visual overview of the time series and labelled anomalies

I plotted the full time series:

* blue line for the raw hourly temperature values
* red markers on points where `is_anomaly == 1`

At the full series level this already shows that:

* the signal is not just random noise, there are visible patterns and cycles
* anomalies are very rare compared to normal points

To understand the labelled anomalies better, I zoomed into each anomaly region using a small window around each anomaly timestamp and plotted the local behaviour.

### Anomaly 1: warm spike behaviour

* The first anomaly looks like a **sharp, local spike** where temperature shoots up well above the normal band and then returns to normal shortly afterwards
* This is consistent with a short term overheating event or a sudden disturbance
* It is essentially a **point like anomaly** in an otherwise stable neighbourhood

### Anomaly 2: change in regime

* The second anomaly corresponds more to a **change in behaviour** than a single point
* Around this time the level shifts and the pattern after the anomaly looks different to the pattern before
* This feels closer to a **regime shift** in the system, and links nicely to the idea of **concept drift** in an operational environment

Together, these two labelled anomalies already give me two different shapes to think about:

* a spike type anomaly
* a shift or regime change type anomaly

Later, when I compare models, I can check whether each method handles both of these shapes well.

---

## 7. 7 day moving average and slower changes over time

To see slower background shifts in temperature, I calculated a **7 day moving average** (7 × 24 hours) over the series and plotted it together with the raw hourly data.

* The raw hourly line shows all the short term variability and noise
* The 7 day moving average smooths out these fast wiggles and highlights the **baseline level** the system is operating at over weeks

What this achieved:

* It made longer term ups and downs easier to see, without being distracted by every small spike or dip
* It gives a visual link between this dataset and the idea of **gradual drift** in the underlying environment

This moving average will be useful later when deciding how to split the data into periods for drift experiments or when explaining how the environment changes over time.

---

## 8. Anomaly summary and class imbalance

I summarised the anomaly labels as follows:

* Total points: ~7 267
* Number of labelled anomalies: 2
* Number of normal points: the rest of the series
* Anomalies as a percentage of all points: roughly **0.03 percent** of the data

Interpretation:

* This is a **strongly imbalanced** setting where anomalies are extremely rare
* This matches realistic expectations in many monitoring systems where most of the time things work as expected
* It also reinforces that anomaly detection methods must be evaluated in a way that respects the imbalance and does not simply favour the majority class

Alongside this, I also noted the overall time span covered by the series. This will matter later when I design experiments that look at performance in different windows or stages of the time line, especially for concept drift.

---

## 9. How this Data Overview supports the rest of the project

This Data Overview phase has given me a grounded understanding of my first operational time series:

* I know what each column represents and how labels are attached
* I have a clear picture of the **normal operating range** and the **rare extreme values**
* I have seen two concrete anomaly shapes: a warm spike and a regime shift
* I know the sampling is hourly, with occasional gaps but no duplicate timestamps
* I know the data has no missing values in the core columns
* I have an initial sense of slow, longer term changes in the baseline via the 7 day moving average
* I understand that the current values are in °F and that any conversion to °C will be a clean preprocessing step later

All of this will feed into:

* the **data description** section in my Methodology chapter
* the design of **baseline methods** and diffusion based experiments
* how I talk about **concept drift** and operational change in later chapters

For now, Data Overview is complete for this file. The next stage will be to move into a more explicit **preprocessing notebook**, where I will start preparing this series (and possibly others) for baseline models, drift experiments, and diffusion based approaches.
