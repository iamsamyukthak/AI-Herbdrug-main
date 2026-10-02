"""Train and evaluate the four HGNN ablation variants on the shared split."""

from __future__ import annotations

import csv
import argparse
import copy
import contextlib
import io
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.optim import AdamW
from torch_geometric.data import HeteroData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_graph import build_heterogeneous_graph
from model.heterogeneous_graph import normalize_entity_name
from model.hgnn import HerbDrugHGNN


RESULTS_PATH = PROJECT_ROOT / "results" / "ablation_results.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
SEED = 42
EPOCHS = 500
PATIENCE = 60
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MAX_GRAD_NORM = 2.0
CLASS_IDS = [0, 1, 2]
CLASS_NAMES = ["LOW", "MEDIUM", "HIGH"]
HERB_DRUG_RELATION = ("Herb", "interacts_with", "Drug")


@dataclass(frozen=True)
class AblationExperiment:
    name: str
    architecture: str
    use_graphsage: bool
    use_sagpool: bool
    use_mlp: bool
    checkpoint: str


EXPERIMENTS = (
    AblationExperiment(
        name="Experiment 1",
        architecture="GAT only",
        use_graphsage=False,
        use_sagpool=False,
        use_mlp=False,
        checkpoint="checkpoints/ablation_gat.pt",
    ),
    AblationExperiment(
        name="Experiment 2",
        architecture="GAT + GraphSAGE",
        use_graphsage=True,
        use_sagpool=False,
        use_mlp=False,
        checkpoint="checkpoints/ablation_gat_graphsage.pt",
    ),
    AblationExperiment(
        name="Experiment 3",
        architecture="GAT + GraphSAGE + SAGPool",
        use_graphsage=True,
        use_sagpool=True,
        use_mlp=False,
        checkpoint="checkpoints/ablation_gat_graphsage_sagpool.pt",
    ),
    AblationExperiment(
        name="Experiment 4",
        architecture="GAT + GraphSAGE + SAGPool + MLP",
        use_graphsage=True,
        use_sagpool=True,
        use_mlp=True,
        checkpoint="checkpoints/ablation_full_hgnn.pt",
    ),
)

METRIC_COLUMNS = (
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc",
)
CSV_COLUMNS = (
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
)


def build_ablation_model(
    graph: HeteroData,
    experiment: AblationExperiment,
    **model_kwargs,
) -> HerbDrugHGNN:
    """Construct an ablation variant with the shared graph and score head."""
    return HerbDrugHGNN.from_graph(
        graph,
        use_graphsage=experiment.use_graphsage,
        use_sagpool=experiment.use_sagpool,
        use_mlp=experiment.use_mlp,
        num_outputs=3,
        **model_kwargs,
    )


def _status_row(
    experiment: AblationExperiment,
    notes: str,
) -> dict[str, object]:
    return {
        "experiment": experiment.name,
        "architecture": experiment.architecture,
        **{column: "" for column in METRIC_COLUMNS},
        "test_samples": "",
        "best_epoch": "",
        "status": "Not evaluated yet",
        "checkpoint": experiment.checkpoint,
        "notes": notes,
    }


def _load_existing_training_split():
    """Reuse prepare_graph's CSV processing and stratified 80/10/10 split."""
    with contextlib.redirect_stdout(io.StringIO()):
        from model.prepare_graph import (
            ddid,
            labels,
            test_idx,
            train_idx,
            val_idx,
        )

    return ddid, labels, train_idx, val_idx, test_idx


def _seed_experiment() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _filter_normalized_pair_split_overlaps(
    ddid,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Drop earlier-split rows whose normalized pair occurs in a later split."""
    pair_keys = [
        (normalize_entity_name(herb), normalize_entity_name(drug))
        for herb, drug in zip(ddid["Herb"], ddid["Drug"])
    ]
    validation_keys = {pair_keys[index] for index in val_idx.tolist()}
    test_keys = {pair_keys[index] for index in test_idx.tolist()}

    filtered_train = [
        index
        for index in train_idx.tolist()
        if pair_keys[index] not in validation_keys | test_keys
    ]
    filtered_validation = [
        index
        for index in val_idx.tolist()
        if pair_keys[index] not in test_keys
    ]

    final_train_keys = {pair_keys[index] for index in filtered_train}
    final_validation_keys = {pair_keys[index] for index in filtered_validation}
    if final_train_keys & final_validation_keys:
        raise RuntimeError("Normalized pair leakage remains between train and validation")
    if final_train_keys & test_keys or final_validation_keys & test_keys:
        raise RuntimeError("Normalized pair leakage remains in the test split")

    return (
        torch.tensor(filtered_train, dtype=torch.long),
        torch.tensor(filtered_validation, dtype=torch.long),
        len(train_idx) - len(filtered_train),
        len(val_idx) - len(filtered_validation),
    )


def _prepare_graph_with_training_edges(
    graph,
    ddid,
    train_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    herb_mapping = graph.node_to_id["Herb"]
    drug_mapping = graph.node_to_id["Drug"]

    all_herb_ids = []
    all_drug_ids = []
    for herb_name, drug_name in zip(ddid["Herb"], ddid["Drug"]):
        herb_key = normalize_entity_name(herb_name)
        drug_key = normalize_entity_name(drug_name)
        if herb_key not in herb_mapping or drug_key not in drug_mapping:
            raise ValueError(
                f"DDID pair is missing from heterogeneous graph: "
                f"{herb_name!r}, {drug_name!r}"
            )
        all_herb_ids.append(herb_mapping[herb_key])
        all_drug_ids.append(drug_mapping[drug_key])

    herb_ids = torch.tensor(all_herb_ids, dtype=torch.long)
    drug_ids = torch.tensor(all_drug_ids, dtype=torch.long)
    train_ids = train_idx.to(dtype=torch.long)

    # Exclude every validation/test Herb-Drug edge to prevent label leakage.
    train_edges = torch.stack(
        [herb_ids[train_ids], drug_ids[train_ids]],
        dim=1,
    ).unique(dim=0, sorted=True)
    graph.data[HERB_DRUG_RELATION].edge_index = train_edges.t().contiguous()
    return herb_ids, drug_ids


def _evaluate(
    true_labels: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    predicted_labels = probabilities.argmax(axis=1)
    labels = true_labels.cpu().numpy()
    metrics = {
        "accuracy": float(accuracy_score(labels, predicted_labels)),
        "precision_macro": float(
            precision_score(
                labels,
                predicted_labels,
                labels=CLASS_IDS,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                labels,
                predicted_labels,
                labels=CLASS_IDS,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                labels,
                predicted_labels,
                labels=CLASS_IDS,
                average="macro",
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities,
                labels=CLASS_IDS,
                multi_class="ovr",
                average="macro",
            )
        ),
    }
    return metrics, predicted_labels, probabilities


def _evaluate_existing_checkpoint(
    experiment: AblationExperiment,
    graph,
    labels: torch.Tensor,
    herb_ids: torch.Tensor,
    drug_ids: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    train_alias_rows_removed: int,
    validation_alias_rows_removed: int,
) -> dict[str, object] | None:
    """Reload and re-evaluate only a checkpoint matching this exact setup."""
    checkpoint_path = PROJECT_ROOT / experiment.checkpoint
    if not checkpoint_path.is_file():
        return None

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    expected_config = {
        "hidden_channels": 32,
        "attention_heads": 2,
        "use_graphsage": experiment.use_graphsage,
        "use_sagpool": experiment.use_sagpool,
        "use_mlp": experiment.use_mlp,
        "num_outputs": 3,
    }
    if checkpoint.get("experiment") != experiment.name:
        return None
    if checkpoint.get("model_config") != expected_config:
        return None
    if checkpoint.get("random_seed") != SEED:
        return None
    if checkpoint.get("train_alias_rows_removed") != train_alias_rows_removed:
        return None
    if checkpoint.get("validation_alias_rows_removed") != validation_alias_rows_removed:
        return None
    if not checkpoint.get("training_loss") or not checkpoint.get("validation_loss"):
        return None
    for key, current_indices in (
        ("train_indices", train_idx),
        ("validation_indices", val_idx),
        ("test_indices", test_idx),
    ):
        if key not in checkpoint or not torch.equal(
            checkpoint[key].to(dtype=torch.long),
            current_indices.to(dtype=torch.long),
        ):
            return None

    model = build_ablation_model(graph.data, experiment)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        test_logits = model(
            graph.data,
            herb_ids[test_idx],
            drug_ids[test_idx],
        )
    metrics, predicted_labels, probabilities = _evaluate(
        labels[test_idx],
        test_logits,
    )
    checkpoint["test_metrics"] = metrics
    checkpoint["test_labels"] = labels[test_idx]
    checkpoint["test_predictions"] = torch.from_numpy(predicted_labels)
    checkpoint["test_probabilities"] = torch.from_numpy(probabilities)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)

    return {
        "experiment": experiment.name,
        "architecture": experiment.architecture,
        **metrics,
        "test_samples": len(test_idx),
        "best_epoch": int(checkpoint["best_epoch"]),
        "status": "Evaluated",
        "checkpoint": experiment.checkpoint,
        "notes": "Existing compatible checkpoint re-evaluated on the shared test split.",
    }


def _train_experiment(
    experiment: AblationExperiment,
    graph,
    labels: torch.Tensor,
    herb_ids: torch.Tensor,
    drug_ids: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    train_alias_rows_removed: int,
    validation_alias_rows_removed: int,
) -> dict[str, object]:
    _seed_experiment()
    model = build_ablation_model(graph.data, experiment)

    training_counts = torch.bincount(labels[train_idx], minlength=3).float()
    if torch.any(training_counts == 0):
        raise ValueError("Training split is missing at least one risk class")
    class_weights = len(train_idx) / (3.0 * training_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_validation_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    training_loss_history = []
    validation_loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        train_logits = model(
            graph.data,
            herb_ids[train_idx],
            drug_ids[train_idx],
        )
        loss = criterion(train_logits, labels[train_idx])
        training_loss_history.append(float(loss.detach().item()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            validation_logits = model(
                graph.data,
                herb_ids[val_idx],
                drug_ids[val_idx],
            )
            validation_loss = criterion(
                validation_logits,
                labels[val_idx],
            )
            validation_loss_history.append(float(validation_loss.item()))
            validation_predictions = validation_logits.argmax(dim=1)
            validation_f1 = float(
                f1_score(
                    labels[val_idx].cpu().numpy(),
                    validation_predictions.cpu().numpy(),
                    labels=CLASS_IDS,
                    average="macro",
                    zero_division=0,
                )
            )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"{experiment.name} epoch {epoch + 1}: "
                f"loss={loss.item():.4f}, val_macro_f1={validation_f1:.4f}"
            )
        if patience_counter >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")

    model.load_state_dict(best_state)
    checkpoint_path = PROJECT_ROOT / experiment.checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "experiment": experiment.name,
        "architecture": experiment.architecture,
        "model_config": {
            "hidden_channels": model.hidden_channels,
            "attention_heads": 2,
            "use_graphsage": experiment.use_graphsage,
            "use_sagpool": experiment.use_sagpool,
            "use_mlp": experiment.use_mlp,
            "num_outputs": 3,
        },
        "random_seed": SEED,
        "train_indices": train_idx,
        "validation_indices": val_idx,
        "test_indices": test_idx,
        "train_alias_rows_removed": train_alias_rows_removed,
        "validation_alias_rows_removed": validation_alias_rows_removed,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_validation_f1,
        "training_loss": training_loss_history,
        "validation_loss": validation_loss_history,
        "training_config": {
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "loss": "weighted CrossEntropyLoss",
            "optimizer": "AdamW",
        },
        "node_feature_names": {
            node_type: list(graph.data[node_type].feature_names)
            for node_type in graph.data.node_types
        },
    }
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint_data, temporary_path)
    temporary_path.replace(checkpoint_path)

    # Evaluate the serialized checkpoint, not a transient in-memory state.
    saved_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(saved_checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        test_logits = model(
            graph.data,
            herb_ids[test_idx],
            drug_ids[test_idx],
        )
    metrics, predicted_labels, probabilities = _evaluate(
        labels[test_idx],
        test_logits,
    )
    saved_checkpoint["test_metrics"] = metrics
    saved_checkpoint["test_labels"] = labels[test_idx]
    saved_checkpoint["test_predictions"] = torch.from_numpy(predicted_labels)
    saved_checkpoint["test_probabilities"] = torch.from_numpy(probabilities)
    torch.save(saved_checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)

    return {
        "experiment": experiment.name,
        "architecture": experiment.architecture,
        **metrics,
        "test_samples": len(test_idx),
        "best_epoch": best_epoch,
        "status": "Evaluated",
        "checkpoint": experiment.checkpoint,
        "notes": (
            f"Held-out stratified test split; random_state={SEED}; "
            f"best validation macro-F1={best_validation_f1:.6f}; "
            f"effective train/validation/test sizes="
            f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}; excluded "
            f"{train_alias_rows_removed} train and "
            f"{validation_alias_rows_removed} validation rows whose "
            "normalized pairs occur in later splits."
        ),
    }


def _write_results(rows: list[dict[str, object]]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _save_ablation_plots(rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = PROJECT_ROOT / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 6))
    plotted = False
    for experiment in EXPERIMENTS:
        checkpoint_path = PROJECT_ROOT / experiment.checkpoint
        if not checkpoint_path.is_file():
            continue
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        losses = checkpoint.get("training_loss")
        if losses:
            axis.plot(range(1, len(losses) + 1), losses, label=experiment.architecture)
            plotted = True
    axis.set_title("Ablation training loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Weighted cross-entropy loss")
    axis.grid(alpha=0.25)
    if plotted:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "Not evaluated yet", ha="center", va="center")
    figure.tight_layout()
    figure.savefig(plots_dir / "training_loss.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    plotted = False
    for experiment in EXPERIMENTS:
        checkpoint_path = PROJECT_ROOT / experiment.checkpoint
        if not checkpoint_path.is_file():
            continue
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        losses = checkpoint.get("validation_loss")
        if losses:
            axis.plot(range(1, len(losses) + 1), losses, label=experiment.architecture)
            plotted = True
    axis.set_title("Ablation validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Weighted cross-entropy loss")
    axis.grid(alpha=0.25)
    if plotted:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "Not evaluated yet", ha="center", va="center")
    figure.tight_layout()
    figure.savefig(plots_dir / "validation_loss.png", dpi=160)
    plt.close(figure)

    evaluated_rows = [row for row in rows if row.get("status") == "Evaluated"]
    figure, axis = plt.subplots(figsize=(10, 6))
    if evaluated_rows:
        names = [str(row["architecture"]) for row in evaluated_rows]
        positions = np.arange(len(names))
        width = 0.24
        for offset, (metric, label) in enumerate(
            (
                ("accuracy", "Accuracy"),
                ("f1_macro", "Macro F1"),
                ("roc_auc", "Macro OVR ROC-AUC"),
            )
        ):
            axis.bar(
                positions + (offset - 1) * width,
                [float(row[metric]) for row in evaluated_rows],
                width=width,
                label=label,
            )
        axis.set_xticks(positions, names, rotation=15, ha="right")
        axis.legend()
        axis.set_ylim(0, 1.05)
    else:
        axis.text(0.5, 0.5, "Not evaluated yet", ha="center", va="center")
    axis.set_title("Ablation comparison on held-out data")
    axis.set_ylabel("Score")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots_dir / "ablation_comparison.png", dpi=160)
    plt.close(figure)


def main(force_train: bool = False) -> None:
    initial_rows = [
        _status_row(experiment, "Training/evaluation pending.")
        for experiment in EXPERIMENTS
    ]
    _write_results(initial_rows)

    try:
        ddid, labels, train_idx, val_idx, test_idx = _load_existing_training_split()
        train_idx, val_idx, train_alias_rows_removed, validation_alias_rows_removed = (
            _filter_normalized_pair_split_overlaps(
                ddid,
                train_idx,
                val_idx,
                test_idx,
            )
        )
        graph = build_heterogeneous_graph()
        herb_ids, drug_ids = _prepare_graph_with_training_edges(
            graph,
            ddid,
            train_idx,
        )
    except Exception as error:
        rows = [
            _status_row(
                experiment,
                f"Required shared preprocessing/splits unavailable: "
                f"{type(error).__name__}: {error}",
            )
            for experiment in EXPERIMENTS
        ]
        _write_results(rows)
        print(f"Ablation preprocessing failed: {error}")
        print(f"Status rows saved to: {RESULTS_PATH}")
        return

    rows = []
    for experiment in EXPERIMENTS:
        try:
            row = None
            if not force_train:
                row = _evaluate_existing_checkpoint(
                    experiment,
                    graph,
                    labels,
                    herb_ids,
                    drug_ids,
                    train_idx,
                    val_idx,
                    test_idx,
                    train_alias_rows_removed,
                    validation_alias_rows_removed,
                )
            if row is None:
                row = _train_experiment(
                    experiment,
                    graph,
                    labels,
                    herb_ids,
                    drug_ids,
                    train_idx,
                    val_idx,
                    test_idx,
                    train_alias_rows_removed,
                    validation_alias_rows_removed,
                )
            print(
                f"{experiment.name} test metrics: "
                f"accuracy={row['accuracy']:.4f}, "
                f"precision={row['precision_macro']:.4f}, "
                f"recall={row['recall_macro']:.4f}, "
                f"f1={row['f1_macro']:.4f}, "
                f"roc_auc={row['roc_auc']:.4f}"
            )
        except Exception as error:
            row = _status_row(
                experiment,
                f"Training/evaluation failed: {type(error).__name__}: {error}",
            )
            print(f"{experiment.name}: Not evaluated yet ({row['notes']})")
        rows.append(row)
        _write_results(rows + initial_rows[len(rows):])
        _save_ablation_plots(rows + initial_rows[len(rows):])

    print(f"Ablation results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--force-train",
        action="store_true",
        help="Train every variant from scratch, replacing compatible checkpoints.",
    )
    arguments = argument_parser.parse_args()
    main(force_train=arguments.force_train)