"""Compare only trained baseline checkpoints on a shared DDID test split."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.research_results import ensure_results_artifacts
from model.run_ablation_study import (
    EXPERIMENTS,
    _filter_normalized_pair_split_overlaps,
    _load_existing_training_split,
)
from model.herbdrug_model import HerbDrugInteractionModel


DATASET_PATH = PROJECT_ROOT / "datasets" / "ddid_processed.csv"
GCN_CHECKPOINT = MODEL_DIR / "herbdrug_complete_model.pth"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_PATH = RESULTS_DIR / "baseline_comparison.csv"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"
CLASS_IDS = [0, 1, 2]
CLASS_NAMES = ["LOW", "MEDIUM", "HIGH"]

METRIC_COLUMNS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc_ovr_macro",
]
CSV_COLUMNS = [
    "model",
    *METRIC_COLUMNS,
    "test_samples",
    "status",
    "notes",
]


def _not_evaluated(model_name: str, notes: str) -> dict[str, object]:
    return {
        "model": model_name,
        **{column: "" for column in METRIC_COLUMNS},
        "test_samples": "",
        "status": "Not evaluated yet",
        "notes": notes,
    }


def _metrics_from_probabilities(
    model_name: str,
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    notes: str,
) -> dict[str, object]:
    predicted_labels = probabilities.argmax(axis=1)
    return {
        "model": model_name,
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision_macro": precision_score(
            true_labels,
            predicted_labels,
            labels=CLASS_IDS,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            true_labels,
            predicted_labels,
            labels=CLASS_IDS,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            true_labels,
            predicted_labels,
            labels=CLASS_IDS,
            average="macro",
            zero_division=0,
        ),
        "roc_auc_ovr_macro": roc_auc_score(
            true_labels,
            probabilities,
            labels=CLASS_IDS,
            multi_class="ovr",
            average="macro",
        ),
        "test_samples": len(true_labels),
        "status": "Evaluated",
        "notes": notes,
    }


def _evaluate_saved_ablation(
    model_name: str,
    experiment_index: int,
) -> tuple[dict[str, object], pd.DataFrame | None, np.ndarray | None, np.ndarray | None]:
    experiment = EXPERIMENTS[experiment_index]
    checkpoint_path = PROJECT_ROOT / experiment.checkpoint
    if not checkpoint_path.is_file():
        return (
            _not_evaluated(model_name, f"Checkpoint not found: {experiment.checkpoint}"),
            None,
            None,
            None,
        )

    ddid, labels, train_idx, val_idx, test_idx = _load_existing_training_split()
    train_idx, val_idx, train_rows_removed, val_rows_removed = (
        _filter_normalized_pair_split_overlaps(
            ddid,
            train_idx,
            val_idx,
            test_idx,
        )
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("experiment") != experiment.name:
        return (
            _not_evaluated(model_name, "Checkpoint experiment metadata does not match."),
            None,
            None,
            None,
        )
    if checkpoint.get("train_alias_rows_removed") != train_rows_removed:
        return (
            _not_evaluated(model_name, "Checkpoint split preprocessing does not match."),
            None,
            None,
            None,
        )
    if checkpoint.get("validation_alias_rows_removed") != val_rows_removed:
        return (
            _not_evaluated(model_name, "Checkpoint split preprocessing does not match."),
            None,
            None,
            None,
        )
    if not torch.equal(checkpoint.get("test_indices", torch.empty(0)).long(), test_idx.long()):
        return (
            _not_evaluated(model_name, "Checkpoint test indices do not match the current split."),
            None,
            None,
            None,
        )

    test_labels = labels[test_idx].cpu().numpy()
    checkpoint_labels = checkpoint.get("test_labels")
    checkpoint_probabilities = checkpoint.get("test_probabilities")
    if checkpoint_labels is None or checkpoint_probabilities is None:
        return (
            _not_evaluated(model_name, "Checkpoint does not contain evaluated test outputs."),
            None,
            None,
            None,
        )
    if not np.array_equal(checkpoint_labels.cpu().numpy(), test_labels):
        return (
            _not_evaluated(model_name, "Checkpoint test labels do not match the current dataset."),
            None,
            None,
            None,
        )

    probabilities = checkpoint_probabilities.cpu().numpy()
    if probabilities.shape != (len(test_idx), len(CLASS_IDS)):
        return (
            _not_evaluated(model_name, "Checkpoint probability shape is incompatible."),
            None,
            None,
            None,
        )
    if not np.isfinite(probabilities).all():
        return (
            _not_evaluated(model_name, "Checkpoint contains non-finite test probabilities."),
            None,
            None,
            None,
        )

    notes = (
        f"Saved checkpoint; held-out split random_state=42; "
        f"excluded {train_rows_removed} train and {val_rows_removed} validation "
        "rows for normalized-pair leakage prevention."
    )
    metrics = _metrics_from_probabilities(
        model_name,
        test_labels,
        probabilities,
        notes,
    )
    test_rows = ddid.iloc[test_idx.numpy()]
    predicted_labels = probabilities.argmax(axis=1)
    prediction_rows = pd.DataFrame(
        {
            "model": model_name,
            "herb": test_rows["Herb"].to_numpy(),
            "drug": test_rows["Drug"].to_numpy(),
            "true_label": [CLASS_NAMES[label] for label in test_labels],
            "predicted_label": [CLASS_NAMES[label] for label in predicted_labels],
            "prediction_score": probabilities.max(axis=1),
            "probability_low": probabilities[:, 0],
            "probability_medium": probabilities[:, 1],
            "probability_high": probabilities[:, 2],
            "split": "test",
        }
    )
    return metrics, prediction_rows, test_labels, probabilities


def _evaluate_gcn() -> tuple[
    dict[str, object],
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
]:
    ddid = pd.read_csv(DATASET_PATH)
    ddid["Herb"] = ddid["Herb"].astype(str).str.strip()
    ddid["Drug"] = ddid["Drug"].astype(str).str.strip()
    labels = ddid["Label"].astype(int).to_numpy()
    indices = np.arange(len(ddid))

    _, remaining_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    _, test_indices = train_test_split(
        remaining_indices,
        test_size=0.50,
        random_state=42,
        stratify=labels[remaining_indices],
    )

    herbs = sorted(ddid["Herb"].unique())
    drugs = sorted(ddid["Drug"].unique())
    herb_to_id = {name: index for index, name in enumerate(herbs)}
    drug_to_id = {
        name: len(herbs) + index
        for index, name in enumerate(drugs)
    }

    checkpoint = torch.load(
        GCN_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    total_nodes = len(herbs) + len(drugs)
    if checkpoint["num_herbs"] != len(herbs):
        raise ValueError("GCN checkpoint herb count does not match DDID data")
    if checkpoint["num_drugs"] != total_nodes:
        raise ValueError("GCN checkpoint drug embedding size does not match DDID data")

    model = HerbDrugInteractionModel(
        num_herbs=len(herbs),
        num_drugs=total_nodes,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_rows = ddid.iloc[test_indices]
    herb_ids = torch.tensor(
        [herb_to_id[name] for name in test_rows["Herb"]],
        dtype=torch.long,
    )
    drug_ids = torch.tensor(
        [drug_to_id[name] for name in test_rows["Drug"]],
        dtype=torch.long,
    )
    with torch.inference_mode():
        logits = model(
            checkpoint["node_features"],
            checkpoint["edge_index"],
            herb_ids,
            drug_ids,
        )
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    true_labels = labels[test_indices]
    predicted_labels = probabilities.argmax(axis=1)
    metrics = {
        "model": "GCN",
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision_macro": precision_score(
            true_labels,
            predicted_labels,
            labels=CLASS_IDS,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            true_labels,
            predicted_labels,
            labels=CLASS_IDS,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            true_labels,
            predicted_labels,
            labels=CLASS_IDS,
            average="macro",
            zero_division=0,
        ),
        "roc_auc_ovr_macro": roc_auc_score(
            true_labels,
            probabilities,
            labels=CLASS_IDS,
            multi_class="ovr",
            average="macro",
        ),
        "test_samples": len(test_indices),
        "status": "Evaluated",
        "notes": "Held-out stratified test split; random_state=42.",
    }
    matrix = confusion_matrix(true_labels, predicted_labels, labels=CLASS_IDS)
    prediction_rows = pd.DataFrame(
        {
            "model": "GCN",
            "herb": test_rows["Herb"].to_numpy(),
            "drug": test_rows["Drug"].to_numpy(),
            "true_label": [CLASS_NAMES[label] for label in true_labels],
            "predicted_label": [CLASS_NAMES[label] for label in predicted_labels],
            "prediction_score": probabilities.max(axis=1),
            "probability_low": probabilities[:, 0],
            "probability_medium": probabilities[:, 1],
            "probability_high": probabilities[:, 2],
            "split": "test",
        }
    )
    return metrics, matrix, probabilities, prediction_rows, true_labels


def _save_metrics_plot(rows: list[dict[str, object]]) -> None:
    models = ["GCN", "GAT", "GraphSAGE", "Proposed HGNN"]
    display_metrics = [
        ("accuracy", "Accuracy"),
        ("precision_macro", "Macro precision"),
        ("recall_macro", "Macro recall"),
        ("f1_macro", "Macro F1"),
    ]
    by_model = {str(row["model"]): row for row in rows}
    positions = np.arange(len(models))
    bar_width = 0.18

    figure, axis = plt.subplots(figsize=(11, 6))
    for metric_index, (column, label) in enumerate(display_metrics):
        values = [
            float(by_model[model][column])
            if by_model[model]["status"] == "Evaluated"
            else np.nan
            for model in models
        ]
        axis.bar(
            positions + (metric_index - 1.5) * bar_width,
            values,
            width=bar_width,
            label=label,
        )

    for model_index, model_name in enumerate(models):
        if by_model[model_name]["status"] != "Evaluated":
            axis.text(
                positions[model_index],
                0.035,
                "Not evaluated yet",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axis.set_title("Baseline comparison on held-out data")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.05)
    axis.set_xticks(positions, models)
    axis.legend(loc="upper right")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "baseline_comparison_metrics.png", dpi=160)
    figure.savefig(PLOTS_DIR / "baseline_comparison.png", dpi=160)
    plt.close(figure)


def _save_roc_auc_plot(rows: list[dict[str, object]]) -> None:
    models = ["GCN", "GAT", "GraphSAGE", "Proposed HGNN"]
    by_model = {str(row["model"]): row for row in rows}
    positions = np.arange(len(models))
    values = [
        float(by_model[name]["roc_auc_ovr_macro"])
        if by_model[name]["status"] == "Evaluated"
        else np.nan
        for name in models
    ]

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(positions, values, width=0.55, color="#2878B5")
    for index, name in enumerate(models):
        if by_model[name]["status"] != "Evaluated":
            axis.text(
                positions[index],
                0.035,
                "Not evaluated yet",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_title("Macro one-vs-rest ROC-AUC")
    axis.set_ylabel("ROC-AUC")
    axis.set_ylim(0, 1.05)
    axis.set_xticks(positions, models)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "baseline_comparison_roc_auc.png", dpi=160)
    plt.close(figure)


def _save_roc_curve(
    evaluations: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    for model_name, (labels, probabilities) in evaluations.items():
        for class_id, class_name in enumerate(CLASS_NAMES):
            binary_labels = (labels == class_id).astype(int)
            false_positive_rate, true_positive_rate, _ = roc_curve(
                binary_labels,
                probabilities[:, class_id],
            )
            class_auc = auc(false_positive_rate, true_positive_rate)
            axis.plot(
                false_positive_rate,
                true_positive_rate,
                label=f"{model_name} {class_name} (AUC={class_auc:.3f})",
            )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Chance")
    axis.set_title("One-vs-rest ROC curves (held-out test set)")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "roc_curve.png", dpi=160)
    plt.close(figure)


def _save_gcn_confusion_matrix(matrix: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title("GCN held-out confusion matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )
    figure.colorbar(image, ax=axis, label="Sample count")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "gcn_confusion_matrix.png", dpi=160)
    figure.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def main() -> None:
    ensure_results_artifacts()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_frames: list[pd.DataFrame] = []
    evaluation_payloads: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if not GCN_CHECKPOINT.is_file():
        rows = [
            _not_evaluated("GCN", "Trained checkpoint was not found."),
        ]
        confusion = None
    else:
        gcn_metrics, confusion, probabilities, gcn_predictions, true_labels = (
            _evaluate_gcn()
        )
        rows = [gcn_metrics]
        predictions_frames.append(gcn_predictions)
        evaluation_payloads["GCN"] = (true_labels, probabilities)

    experiment_by_name = {
        experiment.name: experiment
        for experiment in EXPERIMENTS
    }
    for model_name, experiment_name in (
        ("GAT", "Experiment 1"),
        ("Proposed HGNN", "Experiment 4"),
    ):
        experiment = experiment_by_name[experiment_name]
        metric_row, prediction_frame, labels, probabilities = (
            _evaluate_saved_ablation(
                model_name,
                EXPERIMENTS.index(experiment),
            )
        )
        rows.append(metric_row)
        if prediction_frame is not None:
            predictions_frames.append(prediction_frame)
            evaluation_payloads[model_name] = (labels, probabilities)

    rows.append(
        _not_evaluated(
            "GraphSAGE",
            "No standalone GraphSAGE-only checkpoint is available; "
            "GAT + GraphSAGE ablation is not a standalone GraphSAGE baseline.",
        )
    )

    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    if predictions_frames:
        pd.concat(predictions_frames, ignore_index=True).to_csv(
            PREDICTIONS_PATH,
            index=False,
        )
    else:
        pd.DataFrame(
            columns=[
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
            ]
        ).to_csv(PREDICTIONS_PATH, index=False)

    _save_metrics_plot(rows)
    _save_roc_auc_plot(rows)
    if confusion is not None:
        _save_gcn_confusion_matrix(confusion)
    if evaluation_payloads:
        _save_roc_curve(evaluation_payloads)

    print(f"Results saved to: {RESULTS_PATH}")
    for row in rows:
        if row["status"] == "Evaluated":
            print(
                f"{row['model']}: accuracy={row['accuracy']:.4f}, "
                f"precision={row['precision_macro']:.4f}, "
                f"recall={row['recall_macro']:.4f}, "
                f"f1={row['f1_macro']:.4f}, "
                f"roc_auc={row['roc_auc_ovr_macro']:.4f}"
            )
        else:
            print(f"{row['model']}: Not evaluated yet ({row['notes']})")


if __name__ == "__main__":
    main()