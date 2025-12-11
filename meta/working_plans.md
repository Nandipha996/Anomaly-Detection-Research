## Stacking and Results Planning: Working Notes

### Current decisions

- **Must-have stacking (confirmed)**
  - Stacking will be used at the **results level**, not on raw data.
  - As experiments begin, model performance will be recorded in a **structured, appendable format** that can later be combined into a single results view (e.g. `all_results` concept).
  - The exact column layout and metric set for this results table will be finalised **later**, once the first experimental runs clarify what is needed.

- **Nice-to-have stacking (confirmed)**
  - A **read-only stacked view of processed data** will be created for comparison plots and portfolio-level summaries.
  - This stacked view will be created **in memory** (inside a dedicated notebook) by reading the per-case processed CSVs and concatenating them, using `case_study` to distinguish datasets.
  - This is primarily for **exploratory plots and descriptive tables**, not for model training.

---

### Relation to existing preprocessing charter

- The preprocessing charter remains valid:
  - Each case study (A–D) will be preprocessed **separately** in its own notebook.
  - Each notebook will output **processed CSVs** with a shared core schema:
    - `time`, `value`, `is_anomaly`, `case_study`,
    - plus optional time-based features (e.g. `hour_of_day`, `day_of_week`, `is_weekend`).

- **No extra preprocessing stage** is introduced between:
  - per-case processed CSVs, and  
  - modelling / experiments.
  
- Stacking does **not** replace the per-case preprocessing; it is an additional **analysis and reporting convenience** applied after processed data exist.

---

### Planned artefacts (high-level, not yet implemented)

1. **Per-case preprocessing notebooks (one per dataset)**
   - Inputs: raw NAB / AIOps files.
   - Outputs: processed CSVs with standard schema, saved under `data/processed/...`.

2. **Portfolio exploration notebook (optional but planned)**
   - Task: read processed CSVs for A, B, C, D and build a stacked in-memory DataFrame.
   - Use cases:
     - cross-case exploratory plots,
     - portfolio summary tables (sampling, anomaly ratios, lengths),
     - supporting figures for Chapter 4, Data Exploration.

3. **Results logging (concept only, details deferred)**
   - Intention: as soon as experiments begin, results will be logged systematically in a way that can later be stacked into a single results view.
   - The concrete schema (column names, exact metrics) is **intentionally not fixed yet** and will be designed once the first baseline experiments clarify what needs to be recorded.

---

### Open design items (intentionally postponed)

- **Exact structure of the results table (`all_results` concept)**
  - To be decided after:
    - at least one baseline experiment is run on one case study,
    - the main evaluation metrics (e.g. event-level F1, precision, recall, AUROC) have been finalised.

- **Final feature set used in modelling**
  - Time-based features (e.g. `hour_of_day`, `day_of_week`) will be created where useful, but:
    - whether they are fed into every model,
    - and whether they appear in every experiment,
    - will be finalised during the modelling design phase, not at preprocessing design time.

---

### Position in dissertation structure

- **Chapter 3 – Research Methodology**
  - Describes:
    - per-case preprocessing (as defined in the charter),
    - the idea of a common processed schema for A–D,
    - the principle that results are logged systematically for cross-case comparison.

- **Chapter 4 – Results / Findings**
  - Uses:
    - per-case data overviews already written,
    - cross-case plots and tables from the portfolio exploration notebook,
    - stacked results (once experiments are completed) to build comparison tables of diffusion models vs baselines across the four case studies.
