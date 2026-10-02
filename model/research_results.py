"""Initialize and append reproducible research result artifacts."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RESULT_SCHEMAS = {
    "predictions.csv": (
        "model",
        "herb",
        "drug",
        "true_label",
        "predicted_label",
        "prediction_score",
        "probability_low",
        "probability_medium",
        "probability_high",
        "split",
    ),
    "baseline_comparison.csv": (
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc_ovr_macro",
        "test_samples",
        "status",
        "notes",
    ),
    "ablation_results.csv": (
        "experiment",
        "architecture",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc",
        "test_samples",
        "best_epoch",
        "status",
        "checkpoint",
        "notes",
    ),
    "docking_results.csv": (
        "timestamp_utc",
        "rank",
        "herb",
        "drug",
        "hgnn_score",
        "compound",
        "protein",
        "docking_score",
        "binding_pose",
        "docking_status",
        "reason",
    ),
}


def ensure_results_artifacts() -> None:
    """Create requested result directories/files without inventing rows."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, columns in RESULT_SCHEMAS.items():
        path = RESULTS_DIR / filename
        if path.exists():
            continue
        with path.open("w", newline="", encoding="utf-8") as output_file:
            csv.writer(output_file).writerow(columns)


def append_docking_results(rows: list[dict[str, Any]]) -> None:
    """Append actual docked/unavailable candidate outcomes to the CSV."""
    ensure_results_artifacts()
    path = RESULTS_DIR / "docking_results.csv"
    timestamp = datetime.now(timezone.utc).isoformat()
    columns = RESULT_SCHEMAS[path.name]
    with path.open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        for row in rows:
            record = {column: row.get(column, "") for column in columns}
            record["timestamp_utc"] = timestamp
            writer.writerow(record)