# Groundwork planning 1 – linking Data Overview to the bigger research
* Data used so far NAB - Ambient temperature

## 1. Where we are in the project

Working title: **Evaluating Diffusion Models for Efficient Anomaly Detection in Dynamic Business Time-Series**.

Big picture aims (in my own words):

* I want to see whether **diffusion-style models** can be used for anomaly detection in business time series.
* I care about **efficiency and accessibility**, not just accuracy. Models must be able to run on **modest hardware** (my laptop, basic university machines, light cloud).
* I also care about **concept drift**. Business environments change over time, so I need to see how models cope when the underlying pattern shifts.
* I want to do this in a way that fits **Responsible Data and Analytics**, not just raw performance.

Phase 1 (Data Overview) gave me:

* A grounded understanding of one **operational time series** (ambient temperature from NAB).
* A clear picture of:

  * hourly sampling
  * value range and normal band
  * two labelled anomalies with different shapes (spike and regime shift)
  * how rare anomalies are compared to normal points
* First visual hints of **slow background changes** (via the 7 day moving average).

This sets me up to move into **Phase 2: Preprocessing and experimental scaffolding**.

At the same time, I am aware of some of the weaker or sensitive spots in my proposal that we want to address early, for example:

* very few labelled anomalies in this particular series
* risk of relying too much on a single dataset
* need to be very clear and practical about what “efficient” and “responsible” mean in the experiments
* need to show that the time structure and concept drift angle are not just buzzwords but actually present in how I design and report experiments

Keeping these in mind now will help me design the next steps in a way that already strengthens the final thesis.

---

## 2. Workflow recap and how notebooks will share data

High level workflow I am following:

1. **Data Overview** (current notebook)

   * Understand the raw series and labels.
   * Build intuition for patterns, anomalies, and drift.

2. **Preprocessing and experimental scaffolding** (next notebook)

   * Create a **clean, documented version** of the dataset that all models will use.
   * Decide on **time based splits** and drift windows.
   * Decide how to turn the series into **windows and labels** that models will see.

3. **Baseline experiments**

   * Simple methods first: naive, moving average, Isolation Forest, One-Class SVM, LSTM autoencoder.
   * Use the same cleaned data and windowing so comparisons are fair.

4. **Diffusion based experiments**

   * Implement lighter diffusion style models that fit my hardware constraints.
   * Compare detection quality and efficiency against the baselines.

5. **Concept drift and robustness experiments**

   * Look at performance across different time periods or controlled drift scenarios.

6. **Qualitative and responsible AI reflection**

   * Interpret results in relation to access, resource constraints, and responsible use.

7. **Write up and integration into chapters**

   * Map all the above into Chapters 3, 4, and 5 of the dissertation.

How notebooks will share the same cleaned data:

* The idea is to have **one preprocessing notebook** (for example, `02_preprocessing.ipynb`) that:

  * loads the raw NAB file and labels
  * performs the agreed cleaning and transformations
  * saves the cleaned result to disk in a consistent place

* For example, it might save a file like:

  * `data/processed/ambient_temperature_clean.csv` or
  * `data/processed/ambient_temperature_clean.parquet`

* All later experiment notebooks (baselines and diffusion) will:

  * load this **processed file** instead of going back to the raw data
  * reuse the same helper functions for windowing and label construction

This means there is **one source of truth** for the prepared data. If I ever need to change a preprocessing decision (for example, how I handle gaps or resampling), I do it in one place, re-generate the processed file, and all experiment notebooks stay in sync.

Later, I may also move some repeated logic (like loading and windowing) into a small Python module (for example, `src/data_utils.py`), but that can come after the first round of experiments.

---

## 3. Phase 2: the three key goals of preprocessing (in simple terms)

Phase 2 is not about training fancy models yet. It is about making sure the **input to those models** is clean, fair, and aligned with my research questions.

### Goal 1: Build a clean, understandable dataset that fits my context

What this means in practice:

* Load the raw ambient temperature series and labels again.
* Convert the temperature values from **°F to °C** so that interpretation aligns with my South African context.
* Make column names clear and thesis ready, for example:

  * `timestamp`
  * `temperature_c`
  * `anomaly_label`
* Decide how to treat **time gaps**:

  * keep them and simply be aware of them, or
  * resample and fill in a documented way if that is needed for certain methods.

Why it matters for my research:

* It gives me a **single, well defined dataset** that reflects how a real operational series might look, but expressed in units and naming that are natural for my writing.
* It becomes the baseline I can describe in Chapter 3 under “Data preprocessing” without confusion.
* It helps examiners see that I have treated the raw data with care and transparency.

### Goal 2: Define time based splits and drift windows

What this means in practice:

* Decide, along the timeline, how I will split the data into:

  * training period(s)
  * validation period (if needed)
  * test period(s)
* Make these splits **respect time order**. No random shuffling.
* Identify interesting **sub periods** where behaviour looks different, for example:

  * an earlier “stable” period
  * a later period where the baseline level or variability shifts

Why it matters for my research:

* My topic is about **dynamic business time series** and **concept drift**. If the splits ignore time, I lose that story.
* Time based splits allow me to:

  * ask whether models trained earlier still perform well later
  * show how performance changes when the environment drifts
* This links directly back to my proposal, where I said I will explore robustness under changing conditions, not only static snapshots.

### Goal 3: Prepare the data format that all models will see

What this means in practice:

* Decide how each model will “see” the series. A common pattern is:

  * take a fixed length window of past values (for example, the last L hours)
  * ask the model to detect whether the **last point in that window** is anomalous
* Implement helper functions that:

  * take the cleaned time series
  * slice it into windows of shape `[window_length]`
  * attach the correct label for each window (normal vs anomaly)

Why it matters for my research:

* If all models (naive baselines, classical methods, and diffusion) are fed data in **very different ways**, comparisons will be unfair.
* Using a shared windowing scheme turns the preprocessing notebook into a **foundation for all experiments**.
* It also makes it easier to talk about computational efficiency later, because I can say clearly:

  * “Given windows of length L and this many samples, method X took Y seconds and used Z memory.”

In simple terms: Phase 2 is about **creating a clean, fair playing field**. Once that is done, the later models are not built on sand.

---

## 4. The worry about only 2 anomalies and whether my research is still viable

This is an important concern and worth being honest about now.

The ambient temperature series I am using from NAB has:

* about 7 267 points in total
* only **2 labelled anomalies**

On its own, this is not enough to fully evaluate complex anomaly detectors. However, this does not automatically make the research weak or invalid. It simply means I need to be thoughtful in how I design the broader experimental setup.

A few ways to handle this responsibly:

### A. Treat this series as one detailed case study, not the whole universe

* I can position this dataset as **one operational case** where I:

  * carefully analyse the shape of anomalies
  * look at concept drift in the context of this system
  * evaluate how models behave over time
* In the full study, I can and should:

  * include **additional time series** from NAB or other public datasets that have more anomalies
  * or construct **synthetic but realistic anomalies** on top of real background series to get richer evaluation scenarios

This way, the ambient temperature series is not carrying the whole burden of the thesis. It is one part of a broader evaluation.

### B. Emphasise the unsupervised nature and rarity of anomalies

* Anomaly detection is often **unsupervised or semi supervised** in practice.
* Many methods do not require large numbers of labelled anomalies; they learn a notion of “normal” and then flag deviations.
* In that framing, the 2 labels in this series are mainly useful for:

  * sanity checking
  * illustrating behaviour
* The real test becomes how well a method can model normal behaviour and spot rare deviations, even if there are very few labelled anomalies.

This sits comfortably with my focus on:

* model efficiency
* behaviour under drift
* responsible use in resource constrained settings

### C. Be transparent with the committee about this limitation and how I address it

* Instead of hiding the fact that there are only 2 anomalies, I can:

  * state it clearly
  * show that I understand why this is limiting
  * explain how I mitigate it by:

    * adding richer datasets
    * using synthetic anomaly injections where justified
    * focusing part of the analysis on concept drift and efficiency, not only on pure classification metrics

Examiners generally appreciate this kind of honesty and forward planning.

### D. Bottom line

* Using this dataset is still **viable** for my research, especially as:

  * a case study of an operational business time series
  * a concrete example to anchor discussions of drift and efficiency
* The key is to **not stop here**. I will expand my experimental set beyond a single series with two labels, and I will be explicit about how each dataset contributes to answering my research questions.

---

These notes are my grounding for Phase 2. The goal is to move into preprocessing with a clear sense of:

* what I am trying to achieve
* how this supports my proposal
* and where the current dataset helps or needs to be complemented by others.
