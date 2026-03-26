# Anomaly Detection Research Pipeline for Dynamic Business Time Series

A portfolio project exploring how anomaly detection methods can be designed, compared, and evaluated on operational time-series data using a structured, notebook-based research workflow.

This repository documents an earlier research attempt focused on **dynamic business time-series anomaly detection**, with experiments spanning **statistical baselines, classical machine learning, deep learning, and exploratory diffusion-based anomaly scoring**. While the project later evolved into a more focused telecommunications research direction, this version remains a strong portfolio example of applied data science, experimental design, and reproducible analytics.

---

## Project Overview

Modern organisations monitor time-series data to detect unusual behaviour that may indicate failures, instability, service degradation, or other operational issues. This project investigates how anomaly detection methods can be applied to such data in a way that is structured, comparable, and reproducible.

The work was built in Jupyter notebooks and organised as a multi-case study pipeline across several operational datasets. The goal was not only to test models, but also to build a repeatable research workflow covering:

- data preparation,
- train/validation/test splitting,
- leakage-aware preprocessing,
- baseline comparison,
- threshold selection,
- event-aware evaluation,
- experiment logging,
- and results tracking.

This repository represents an **applied research portfolio project** rather than a production system.

---

## What This Project Set Out to Do

The project was designed to explore the following question:

> How can anomaly detection methods be fairly evaluated on dynamic operational time-series data, especially when anomalies may appear as meaningful episodes rather than isolated points?

To answer that, the work focused on:

- building a reusable anomaly detection experimentation pipeline,
- comparing different families of detectors,
- using multiple case studies instead of relying on a single dataset,
- evaluating results with more care than a simple model leaderboard,
- and documenting the process in a reproducible way.

---

## Datasets / Case Studies

The pipeline was applied across multiple time-series case studies to test how methods behaved in different settings.

### Case studies explored
- **NAB Ambient Temperature**
- **NAB NYC Taxi**
- **NAB CPU Utilisation**
- **AIOps KPI Event Data**

These datasets represent operational or business-style time-series where anomalies may reflect issues such as:

- spikes or drops,
- flat lines,
- unusual bursts,
- instability,
- and event-like abnormal periods.

---

## Project Workflow

The project followed a structured notebook workflow rather than one-off model experiments.

### 1. Data understanding
Each dataset was first inspected to understand:
- time granularity,
- anomaly labels,
- missing values,
- duplicates,
- scaling needs,
- and case-specific data quality issues.

### 2. Preprocessing
A consistent preprocessing pattern was developed across case studies. This included:
- standardising timestamps,
- sorting observations in time order,
- handling duplicate records,
- checking missing values,
- generating train/validation/test splits,
- scaling values using **train-only fitting** to avoid leakage,
- and creating a shared schema for downstream modelling.

### 3. Feature engineering
For some workflows, time-based features were included, such as:
- hour of day,
- day of week,
- weekend indicators.

For sequence-based models, contiguous rolling windows were created with attention to time-order integrity.

### 4. Baseline modelling
The project intentionally included simple and classical baselines before more complex models. This helped create a more defensible comparison.

### 5. Threshold selection and evaluation
Rather than choosing arbitrary thresholds, the workflow used a validation-based thresholding approach and tracked performance on held-out data.

### 6. Experiment logging
Runs were saved with repeatable configurations, stored outputs, and consistent naming to support comparison across experiments.

---

## Models and Methods Explored

This project was not limited to one modelling approach. It explored a progression from simpler methods to more advanced approaches.

### Baselines
- naive / trivial references
- moving-average-style heuristics
- simple threshold-based approaches

### Classical machine learning
- **Isolation Forest**
- **One-Class SVM**

### Deep learning
- **LSTM Autoencoder**

### Exploratory advanced direction
- **Diffusion-based anomaly scoring**  
  This was treated as an exploratory research direction rather than a final production-ready solution.

---

## Evaluation Approach

A key strength of this project is that it went beyond “train model, report score”.

The evaluation process aimed to be more meaningful by considering:

- **validation-based threshold selection**
- **train/validation/test separation**
- **event-aware performance**
- **precision, recall, and F1**
- **AUROC / PR-AUC where relevant**
- **detection delay / timeliness**
- **saved artefacts for traceability**

This became especially important in the AIOps KPI work, where anomalies were not only treated as isolated abnormal points, but also as **incident-like windows or events**.

---

## Reproducibility and Structure

The workflow was built to be repeatable and organised.

### Practices used
- consistent file naming,
- structured case-study folders,
- reusable utilities,
- saved configuration files,
- metrics outputs,
- prediction outputs,
- threshold logs,
- plots for visual inspection,
- and experiment result tables.

This made the work more than a loose collection of notebooks. It became a research-style experimental pipeline.

---

## Example Process Covered by the Repository

Depending on the notebook or case study, the repository documents parts of the following process:

1. Load and inspect operational time-series data  
2. Clean and standardise the dataset  
3. Split data into train, validation, and test sets  
4. Scale numeric values without leaking future information  
5. Build baseline and model-specific inputs  
6. Train anomaly detection models  
7. Generate anomaly scores  
8. Select thresholds using validation data  
9. Evaluate on held-out test data  
10. Save metrics, predictions, and visualisations  

---

## Skills Demonstrated

This project highlights skills relevant to **data analyst, data scientist, ML, and analytics engineering pathways**.

### Technical skills shown
- Python-based data analysis
- Jupyter notebook experimentation
- time-series preprocessing
- anomaly detection workflow design
- train/validation/test methodology
- leakage-aware preprocessing
- model comparison and benchmarking
- experiment tracking
- visual analysis of time-series outputs
- structured project organisation

### Research and analytical skills shown
- translating a vague problem into a repeatable workflow
- comparing methods rather than relying on one technique
- thinking critically about evaluation
- documenting assumptions and limitations
- building reusable experimental structure across datasets

---

## Repository Structure

A typical structure for this project looked like this:

```text
project-root/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── preprocessing/
│   ├── baselines/
│   ├── deep_learning/
│   └── exploratory/
│
├── results/
│   ├── case_study_name/
│   │   ├── runs/
│   │   ├── metrics/
│   │   ├── predictions/
│   │   └── figures/
│
├── src/
│   ├── preprocessing utilities
│   ├── evaluation utilities
│   └── experiment helpers
│
└── README.md
