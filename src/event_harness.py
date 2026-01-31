from __future__ import annotations

# Single source of truth for the AIOps KPI event-level harness.
# This file will hold shared utilities previously defined in 03_baselines_aiops_kpi_event.

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json
import hashlib
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)

# ------------------------------------------------------------
# Paths (portable defaults)
# ------------------------------------------------------------

# Project root = folder that contains "src/"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Generic default (per-notebook code should pass a case-specific runs_root)
NOTEBOOK_RUNS_ROOT = PROJECT_ROOT / "results" / "runs"

# Performance-only master results file (portable)
ALL_RESULTS_PATH = PROJECT_ROOT / "results" / "all_results.csv"


# LOADING 

# Required columns for processed case-study splits (performance pipeline contract)
REQUIRED_COLUMNS = [
    "time", "value", "value_scaled", "is_anomaly",
    "hour_of_day", "day_of_week", "is_weekend",
    "split", "case_study",
]

def load_split_csv(path: Path) -> pd.DataFrame:
    """
    Load a processed split CSV and enforce minimal parsing + ordering.
    """
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"time column missing in {path}")
    df["time"] = pd.to_datetime(df["time"], errors="raise")
    df = df.sort_values("time").reset_index(drop=True)
    return df


# BASIC CHECKS

def basic_checks(df: pd.DataFrame, split_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Run minimal integrity checks on a processed split and return:
    - summary table (1 row)
    - missing-values table (core columns)
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[{split_name}] Missing required columns: {missing_cols}")

    # core missing values
    core_cols = ["time", "value", "value_scaled", "is_anomaly", "split", "case_study"]
    missing_counts = df[core_cols].isna().sum()

    # monotonic time check
    is_sorted = df["time"].is_monotonic_increasing

    # case study check
    case_values = df["case_study"].unique()

    # class balance
    n = len(df)
    n_anom = int((df["is_anomaly"] == 1).sum())
    anom_rate = (n_anom / n) if n > 0 else np.nan

    summary = pd.DataFrame({
        "split": [split_name],
        "rows": [n],
        "anomaly_points": [n_anom],
        "anomaly_rate": [anom_rate],
        "time_sorted": [bool(is_sorted)],
        "case_study_values": [", ".join(map(str, case_values))],
    })

    missing_table = (
        missing_counts
        .rename("missing_count")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "column"})
    )

    return summary, missing_table


# PLOTTING 

import matplotlib.dates as mdates

def plot_score_timeline(
    df_split: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    title: str,
    true_events_split: pd.DataFrame | None = None,
    pred_is_anomaly: np.ndarray | None = None,
) -> plt.Figure:
    """
    Plot anomaly score timeline with:
    - true anomaly events as shaded bands (distinct colour + hatch),
    - true anomaly points as red markers,
    - predicted anomaly points as orange markers (optional),
    - threshold reference line,
    - legend for readability.

    X-axis formatting:
    - year shown once as an annotation
    - tick labels shown as MM-DD
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    # Score line
    ax.plot(df_split["time"], scores, linewidth=1, label="Anomaly score")

    # True anomaly event bands (ground truth) - DISTINCT styling
    if true_events_split is not None and len(true_events_split) > 0:
        first_band = True
        for _, ev in true_events_split.iterrows():
            ax.axvspan(
                ev["start_time"],
                ev["end_time"],
                alpha=0.12,              # light transparency
                hatch="///",             # pattern makes it distinct even if colours look similar
                edgecolor="0.4",         # grey hatch/edge
                linewidth=0.0,
                label="True anomaly event" if first_band else None
            )
            first_band = False

    # True anomaly points (red markers)
    mask_true = df_split["is_anomaly"].astype(int).to_numpy() == 1
    ax.scatter(
        df_split.loc[mask_true, "time"],
        scores[mask_true],
        s=12,
        color="red",
        label="True anomaly point",
        zorder=3
    )

    # Predicted anomaly points (orange markers)
    if pred_is_anomaly is not None:
        mask_pred = pred_is_anomaly.astype(int) == 1
        ax.scatter(
            df_split.loc[mask_pred, "time"],
            scores[mask_pred],
            s=10,
            color="orange",
            label="Predicted anomaly",
            zorder=2
        )

    # Threshold line
    ax.axhline(threshold, linestyle="--", label="Threshold")

    # ---- X-axis formatting (clean labels) ----
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate(rotation=0, ha="center")

    # Show the year once as an annotation
    time_min = pd.to_datetime(df_split["time"].min())
    time_max = pd.to_datetime(df_split["time"].max())
    if not pd.isna(time_min) and not pd.isna(time_max) and (time_min.year == time_max.year):
        year_text = str(time_min.year)
    else:
        year_text = f"{time_min.year}–{time_max.year}"

    ax.text(
        0.01, 0.98,
        year_text,
        transform=ax.transAxes,
        va="top",
        ha="left"
    )

    # Axis labels + title + legend
    ax.set_title(title)
    ax.set_xlabel("time (MM-DD)")
    ax.set_ylabel("anomaly_score")
    ax.legend(loc="upper right")

    fig.tight_layout()
    return fig
    

# THRESHOLD SELECTION

def select_threshold_event_f1(
    df_split: pd.DataFrame,
    scores: np.ndarray,
    true_events_split: pd.DataFrame,
    n_grid: int = 60,
    q_min: float = 0.80,
    q_max: float = 0.999,
) -> Tuple[float, pd.DataFrame]:
    """
    Validation-only threshold selection by maximising event-level F1.
    Signature matches the way your notebook already calls this function:
      select_threshold_event_f1(val_df, val_scores, true_events_val, n_grid=60)
    """
    qs = np.linspace(q_min, q_max, n_grid)
    thr_grid = np.quantile(scores, qs)

    rows = []
    for thr in thr_grid:
        m = evaluate_at_threshold(df_split, scores, float(thr), true_events_split)
        rows.append({
            "threshold": float(thr),
            "val_event_precision": float(m["event_precision"]),
            "val_event_recall": float(m["event_recall"]),
            "val_event_f1": float(m["event_f1"]),
            "val_point_precision": float(m["point_precision"]),
            "val_point_recall": float(m["point_recall"]),
            "val_point_f1": float(m["point_f1"]),
            "val_auroc": float(m["auroc"]),
            "val_pr_auc": float(m["pr_auc"]),
            "val_event_delay_median_minutes": float(m["event_delay_median_minutes"]),
        })

    thr_table = pd.DataFrame(rows).sort_values(
        ["val_event_f1", "val_event_recall", "val_event_precision"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best_thr = float(thr_table.loc[0, "threshold"])
    return best_thr, thr_table


# ------------------------------------------------------------
# Event utilities (generic)
# ------------------------------------------------------------

def _contiguous_runs(mask: np.ndarray) -> np.ndarray:
    """
    Convert a 0/1 mask into run IDs for contiguous 1 segments.
    Returns an array of same length with -1 for 0s and run_id >= 0 for 1 segments.
    """
    run_ids = np.full(mask.shape[0], -1, dtype=int)
    run = -1
    for i, v in enumerate(mask):
        if v == 1 and (i == 0 or mask[i - 1] == 0):
            run += 1
        if v == 1:
            run_ids[i] = run
    return run_ids


def build_predicted_events_from_flags(
    df_split: pd.DataFrame,
    pred_is_anomaly: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """
    Build predicted event table from point-wise predicted flags.

    Output columns:
    - pred_event_id, start_time, end_time, n_points
    """
    if len(df_split) != len(pred_is_anomaly):
        raise ValueError("df_split and pred_is_anomaly must align in length.")

    # Robust conversion (works for pandas Series or numpy array)
    mask = np.asarray(pred_is_anomaly, dtype=int)

    run_ids = _contiguous_runs(mask)

    if (run_ids >= 0).sum() == 0:
        return pd.DataFrame(columns=["pred_event_id", "start_time", "end_time", "n_points"])

    times = pd.to_datetime(df_split["time"]).to_numpy()

    rows = []
    for rid in np.unique(run_ids[run_ids >= 0]):
        idx = np.where(run_ids == rid)[0]
        rows.append(
            {
                "pred_event_id": int(rid),
                "start_time": times[idx[0]],
                "end_time": times[idx[-1]],
                "n_points": int(len(idx)),
            }
        )

    return pd.DataFrame(rows)


def event_overlap(a_start: pd.Timestamp, a_end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp) -> bool:
    """Return True if [a_start, a_end] overlaps [b_start, b_end]."""
    return (a_start <= b_end) and (b_start <= a_end)


def compute_event_metrics(
    true_events: pd.DataFrame,
    pred_events: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute event precision/recall/F1 using overlap-based matching:
    - A true event is detected if it overlaps any predicted event.
    - A predicted event is correct if it overlaps any true event.
    """
    if len(true_events) == 0:
        return {"event_precision": np.nan, "event_recall": np.nan, "event_f1": np.nan}

    # True events detected
    true_detected = 0
    for _, te in true_events.iterrows():
        hit = any(
            event_overlap(te["start_time"], te["end_time"], pe["start_time"], pe["end_time"])
            for _, pe in pred_events.iterrows()
        )
        true_detected += int(hit)

    # Predicted events correct
    pred_correct = 0
    for _, pe in pred_events.iterrows():
        hit = any(
            event_overlap(pe["start_time"], pe["end_time"], te["start_time"], te["end_time"])
            for _, te in true_events.iterrows()
        )
        pred_correct += int(hit)

    event_recall = true_detected / len(true_events) if len(true_events) > 0 else np.nan
    event_precision = pred_correct / len(pred_events) if len(pred_events) > 0 else 0.0

    if event_precision + event_recall == 0:
        event_f1 = 0.0
    else:
        event_f1 = 2 * (event_precision * event_recall) / (event_precision + event_recall)

    return {"event_precision": event_precision, "event_recall": event_recall, "event_f1": event_f1}


def compute_detection_delay_minutes(
    true_events: pd.DataFrame,
    times: pd.Series,
    pred_is_anomaly: pd.Series,
) -> float:
    """
    Compute median detection delay in minutes:
    For each true event, find the first predicted anomaly time within the event interval.
    Delay = first_detection_time - event_start_time.
    Returns median delay across detected events. NaN if none detected.
    """
    delays = []
    pred_times = times[pred_is_anomaly.astype(bool)].to_numpy()

    for _, te in true_events.iterrows():
        # Select predicted times that fall within the true event interval
        in_event = pred_times[(pred_times >= te["start_time"]) & (pred_times <= te["end_time"])]
        if len(in_event) > 0:
            first_hit = pd.Timestamp(in_event.min())
            delay_min = (first_hit - te["start_time"]).total_seconds() / 60.0
            delays.append(delay_min)

    return float(np.median(delays)) if len(delays) > 0 else np.nan


def compute_detection_delay_median_minutes(
    df_split: pd.DataFrame,
    pred_is_anomaly: np.ndarray | pd.Series,
    true_events_split: pd.DataFrame,
) -> float:
    """
    Delay per detected true event = (first predicted anomaly time within event) - (event start).
    Returns median delay in minutes over detected events. If none detected, returns NaN.
    """
    if len(true_events_split) == 0:
        return float("nan")

    times = pd.to_datetime(df_split["time"]).to_numpy()

    # Robust conversion (works for pandas Series or numpy array)
    pred_mask = (np.asarray(pred_is_anomaly, dtype=int) == 1)

    delays = []
    for _, ev in true_events_split.iterrows():
        ev_s, ev_e = ev["start_time"], ev["end_time"]

        within = (times >= np.datetime64(ev_s)) & (times <= np.datetime64(ev_e))
        hit_idx = np.where(within & pred_mask)[0]

        if len(hit_idx) > 0:
            first_hit_time = times[hit_idx[0]]
            delay_minutes = (first_hit_time - np.datetime64(ev_s)) / np.timedelta64(1, "m")
            delays.append(float(delay_minutes))

    if len(delays) == 0:
        return float("nan")

    return float(np.median(delays))


def evaluate_at_threshold(
    df_split: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    true_events_split: pd.DataFrame,
    pred_is_anomaly: np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    Compute event-level + point-wise metrics at a threshold.
    AUROC/PR-AUC use continuous scores (threshold-independent).
    """
    y_true = df_split["is_anomaly"].astype(int).to_numpy()

    if pred_is_anomaly is None:
        pred_is_anomaly = (scores >= threshold).astype(int)
    else:
        pred_is_anomaly = pred_is_anomaly.astype(int)

    # Point-wise metrics at threshold
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred_is_anomaly, average="binary", zero_division=0
    )

    # Ranking metrics
    # Guard: if y_true is all one class, roc_auc_score will error
    try:
        auroc = roc_auc_score(y_true, scores)
    except ValueError:
        auroc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, scores)
    except ValueError:
        pr_auc = float("nan")

    # Event-level
    pred_events = build_predicted_events_from_flags(df_split, pred_is_anomaly)
    ev = compute_event_metrics(true_events_split, pred_events)
    delay_med = compute_detection_delay_median_minutes(df_split, pred_is_anomaly, true_events_split)

    return {
        "event_precision": ev["event_precision"],
        "event_recall": ev["event_recall"],
        "event_f1": ev["event_f1"],
        "event_delay_median_minutes": delay_med,
        "point_precision": float(p),
        "point_recall": float(r),
        "point_f1": float(f1),
        "auroc": float(auroc),
        "pr_auc": float(pr_auc),
    }

# RUN KEY

def make_run_key(
    case_study: str,
    tier: int,
    model_name: str,
    window_length_L: int,
    feature_mode: str,
    uses_time_features: int,
    contiguity_enforced: int,
    score_definition_key: str,
    threshold_strategy: str,
) -> str:
    """
    Deterministic identifier for an experiment configuration.

    IMPORTANT:
    - Include ONLY choices that can change scores/metrics.
    - Exclude plotting, figure styling, narrative markdown, etc.
    """
    parts = [
        f"case={case_study}",
        f"tier={tier}",
        f"model={model_name}",
        f"L={window_length_L}",
        f"feat={feature_mode}",
        f"timefeat={uses_time_features}",
        f"contig={contiguity_enforced}",
        f"score={score_definition_key}",
        f"thr={threshold_strategy}",
    ]
    return "|".join(parts)

# RUN ID

def make_run_id_timestamp() -> str:
    """Lightweight trace id recorded in config.json (not used for folder naming)."""
    return pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


# RUN DIR KEY

import hashlib
import re
from pathlib import Path

import hashlib
import re
from pathlib import Path

def run_dir_from_key(
    run_key: str,
    case_study: str,
    tier: int,
    model_slug: str,
    runs_root: Path | None = None,
    hash_len: int = 10,
) -> Path:
    """
    Overwrite-only run folder based on a stable short key.

    Folder name format:
      <case>__t<tier>__<model_slug>__rk-<shorthash>

    - shorthash is a stable hash of run_key (portable across machines)
    - details remain in config.json / notes (not the folder name)
    """
    if runs_root is None:
        runs_root = NOTEBOOK_RUNS_ROOT

    shorthash = hashlib.sha1(run_key.encode("utf-8")).hexdigest()[:hash_len]

    case = re.sub(r"[^a-z0-9]+", "_", case_study.strip().lower()).strip("_")
    slug = re.sub(r"[^a-z0-9]+", "_", model_slug.strip().lower()).strip("_")

    folder = f"{case}__t{int(tier)}__{slug}__rk-{shorthash}"

    run_dir = runs_root / folder
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# SAVE JSON

def save_json(obj: Dict[str, Any], path: Path) -> None:
    """Save a dict to JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


# SAVE DF

def save_df(df: pd.DataFrame, path: Path) -> None:
    """Save a dataframe as CSV (index disabled)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    

# LOAD RESULTS

def load_all_results_or_empty() -> pd.DataFrame:
    """Load all_results.csv if present; otherwise return an empty dataframe."""
    if ALL_RESULTS_PATH.exists():
        return pd.read_csv(ALL_RESULTS_PATH)
    return pd.DataFrame()

# PERFORMANCE

PERF_COLUMNS = [
    "run_key",
    "run_id",
    "run_timestamp",

    "case_study",
    "tier",
    "model_name",

    "window_length_L",
    "feature_mode",
    "uses_time_features",
    "contiguity_enforced",
    "score_definition",

    "threshold_strategy",
    "threshold_value",

    # Validation (primary + delay)
    "val_event_precision",
    "val_event_recall",
    "val_event_f1",
    "val_event_delay_median_minutes",

    # Test (primary + delay)
    "test_event_precision",
    "test_event_recall",
    "test_event_f1",
    "test_event_delay_median_minutes",

    # Test (secondary point-wise + ranking)
    "test_point_precision",
    "test_point_recall",
    "test_point_f1",
    "test_auroc",
    "test_pr_auc",

    # Runtime
    "train_seconds",
    "score_seconds",

    # Notes
    "notes",
]

def normalise_perf_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure row matches the performance-only schema and contains no artefact pointers.
    Missing columns are added as NaN.  
    """
    clean = {k: row.get(k, np.nan) for k in PERF_COLUMNS}
    return clean

# UPSERT ROW

def upsert_all_results_row(row: Dict[str, Any]) -> None:
    """
    Insert or update the performance-only master results table by run_key.
    """
    if "run_key" not in row:
        raise ValueError("Row must include 'run_key'.")
    
    row = normalise_perf_row(row)
    df = load_all_results_or_empty()

    # Ensure consistent columns
    incoming = pd.DataFrame([row])
    if df.empty:
        out = incoming.copy()
        save_df(out, ALL_RESULTS_PATH)
        return

    # Add any missing columns (both directions) to avoid concat warnings
    for col in incoming.columns:
        if col not in df.columns:
            df[col] = np.nan
    for col in df.columns:
        if col not in incoming.columns:
            incoming[col] = np.nan

    # Align column order
    incoming = incoming[df.columns]

    # UPSERT by run_key
    mask = df["run_key"].astype(str) == str(row["run_key"])
    if mask.any():
        df.loc[mask, :] = incoming.iloc[0].to_numpy()
        out = df
    else:
        out = pd.concat([df, incoming], ignore_index=True)

    save_df(out, ALL_RESULTS_PATH)


# AUTO NOTES

def auto_notes(
    val_event_precision: float,
    val_event_recall: float,
    val_event_f1: float,
    test_event_precision: float,
    test_event_recall: float,
    test_event_f1: float,
    test_pr_auc: float,
    threshold_strategy: str,
) -> str:
    """
    Create compact tags that summarise behaviour across validation and test.
    Notes are designed to be stable and human-readable for later analysis.
    """
    tags = []

    # Conservative behaviour markers
    if (test_event_precision >= 0.95) and (test_event_recall <= 0.10):
        tags.append("conservative_threshold_test")

    if (val_event_precision >= 0.95) and (val_event_recall <= 0.70):
        tags.append("conservative_threshold_val")

    # Possible calibration / regime shift marker (validation-to-test drop)
    if (val_event_f1 - test_event_f1) >= 0.30:
        tags.append("val_to_test_drop_calibration_shift_suspected")

    # High ranking signal but poor event detection
    if (test_pr_auc >= 0.90) and (test_event_recall <= 0.10):
        tags.append("high_pr_auc_low_event_recall_mismatch")

    # Threshold strategy trace
    tags.append(f"threshold={threshold_strategy}")

    return ";".join(tags)














































































































