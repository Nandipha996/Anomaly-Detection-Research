## Case study portfolio overview

This project works with four real-world time series as separate case studies, all interpreted as dynamic business or operational series:

- **Case Study A – Ambient temperature (NAB sensor failure)**  
  Physical environment / equipment monitoring in a controlled space.

- **Case Study B – NYC taxi demand (NAB)**  
  City-wide transport demand with strong daily and weekly seasonality.

- **Case Study C – CPU utilisation with ASG misconfiguration (NAB)**  
  Cloud infrastructure utilisation under configuration-driven regime changes.

- **Case Study D – AIOps KPI anomaly dataset (2018 KPI challenge)**  
  An anonymised 1-minute operational KPI with labelled anomalies and an incident window.

Together, these cover different aspects of “dynamic business time series”: environment, demand, infrastructure and service-level KPIs.

---

### Comparative summary

| Case | Dataset                                   | Domain / interpretation                               | Sampling & length                          | Labelled anomalies (count, ~%)        | Drift / regime structure                                           | Main role in study                                                     |
|------|-------------------------------------------|-------------------------------------------------------|-------------------------------------------|---------------------------------------|---------------------------------------------------------------------|------------------------------------------------------------------------|
| A    | `ambient_temperature_system_failure.csv` | Ambient temperature in a controlled environment       | ~7 267 hourly readings, some missing blocks | 2 anomalies, ≈0.03% of points         | Mostly stable band with gradual baseline changes and a regime shift | Simple physical series; introduces spike vs regime-shift anomalies and gradual drift                       |
| B    | `nyc_taxi.csv`                           | City-wide half-hourly taxi trip counts                | 10 320 half-hour slots (≈7 months)        | 5 anomalies, ≈0.05% of points         | Strong daily/weekly seasonality; event-related spikes              | Demand-focused business series; tests methods under noisy normality and strong seasonality                |
| C    | `cpu_utilization_asg_misconfiguration.csv` | CPU utilisation (%) for a cloud auto-scaling group   | 18 050 readings at 5-minute resolution    | 2 anomalies, ≈0.01% of points         | Long moderate-load regime → high-load unstable regime → low-load regime after a drop | Operations / reliability series; focuses on regime changes under extreme class imbalance                  |
| D    | KPI ID `da403e4e3f87c9e0` (AIOps 2018)   | Anonymised 1-minute operational KPI (unitless scale) | 129 035 1-minute observations             | 7 666 anomalies, ≈5.9% of points      | Mostly stable daily cycles; high-anomaly incident window and post-incident period | High-frequency KPI with rich anomaly labels; main testbed for concept drift, incident windows and window-based evaluation |

---

### Cross-case coverage

- **Different domains and meanings**
  - Physical conditions (ambient temperature),
  - Urban demand (NYC taxi),
  - Cloud resource utilisation (CPU),
  - Generic service / system KPI (AIOps KPI).

- **Different sampling resolutions**
  - Hourly (A),
  - Half-hourly (B),
  - 5 minutes (C),
  - 1 minute (D).

- **Different anomaly densities**
  - Extremely sparse labels (A, B, C with 2–5 labelled anomalies),
  - A richer anomaly class (D with ≈5.9% anomalies) spread across regimes.

- **Different drift and regime-change patterns**
  - Slow drift and sensor failure (A),
  - Seasonal patterns plus rare event spikes (B),
  - Regime transitions between moderate, high and low utilisation (C),
  - A distinct incident window with dense anomalies and a recovery period (D).

These contrasts are central to the thesis: they allow diffusion-based anomaly detection methods to be evaluated across a range of realistic business time series, under different forms of concept drift, class imbalance and anomaly structure, and to be compared fairly against simpler baseline methods.
