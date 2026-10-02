"""Evaluate the saved production GCN on its deterministic test split."""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from herbdrug_model import HerbDrugInteractionModel


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODEL_DIR)
DATASET_PATH = os.path.join(PROJECT_ROOT, "datasets", "ddid_processed.csv")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "herbdrug_complete_model.pth")
CLASS_NAMES = ["LOW", "MEDIUM", "HIGH"]


def main():
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
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    expected_drug_embedding_size = len(herbs) + len(drugs)
    if checkpoint["num_herbs"] != len(herbs):
        raise ValueError("Checkpoint herb count does not match the DDID dataset.")
    if checkpoint["num_drugs"] != expected_drug_embedding_size:
        raise ValueError(
            "Checkpoint drug embedding size does not match the DDID node mapping."
        )

    model = HerbDrugInteractionModel(
        num_herbs=len(herbs),
        num_drugs=expected_drug_embedding_size,
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

    with torch.no_grad():
        logits = model(
            checkpoint["node_features"],
            checkpoint["edge_index"],
            herb_ids,
            drug_ids,
        )
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    true_labels = labels[test_indices]
    predicted_labels = predictions.numpy()
    probability_values = probabilities.numpy()

    print("GCN BASELINE: HELD-OUT TEST RESULTS")
    print(f"Test samples: {len(test_indices)} / {len(ddid)}")
    print("Split: stratified 80/10/10, random_state=42")
    print(f"Accuracy: {accuracy_score(true_labels, predicted_labels):.4f}")
    print(
        "Macro F1: "
        f"{f1_score(true_labels, predicted_labels, average='macro'):.4f}"
    )
    print("\nClassification report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0,
            digits=4,
        )
    )
    print("Confusion matrix (rows=true, columns=predicted; LOW, MEDIUM, HIGH):")
    print(confusion_matrix(true_labels, predicted_labels, labels=[0, 1, 2]))

    print("\nHeld-out pair probabilities (%):")
    print("Herb | Drug | True | Predicted | Score | LOW | MEDIUM | HIGH")
    for position, (_, row) in enumerate(test_rows.iterrows()):
        predicted_index = int(predicted_labels[position])
        pair_probabilities = probability_values[position] * 100
        print(
            f"{row['Herb']} | {row['Drug']} | "
            f"{CLASS_NAMES[int(true_labels[position])]} | "
            f"{CLASS_NAMES[predicted_index]} | "
            f"{pair_probabilities[predicted_index]:.2f} | "
            f"{pair_probabilities[0]:.2f} | "
            f"{pair_probabilities[1]:.2f} | "
            f"{pair_probabilities[2]:.2f}"
        )


if __name__ == "__main__":
    main()