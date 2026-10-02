import os
import sys
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from model.biological_features import (
    get_herb_features,
    get_drug_features,
    get_pair_evidence
)
from model.unseen_predictor import predict_unseen


BASE = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT = os.path.join(
    BASE, "model", "herbdrug_complete_model.pth"
)

DATASET = os.path.join(
    BASE, "datasets", "ddid_processed.csv"
)


# ============================================================
# GNN MODEL — EXACT ARCHITECTURE USED DURING TRAINING
# ============================================================

class HerbDrugInteractionModel(nn.Module):

    def __init__(self, num_herbs, num_drugs):
        super().__init__()

        embedding_dim = 32

        self.herb_embedding = nn.Embedding(
            num_herbs,
            embedding_dim
        )

        self.drug_embedding = nn.Embedding(
            num_drugs,
            embedding_dim
        )

        self.conv1 = GCNConv(
            2,
            32
        )

        self.conv2 = GCNConv(
            32,
            embedding_dim
        )

        self.fc1 = nn.Linear(
            embedding_dim * 4,
            64
        )

        self.fc2 = nn.Linear(
            64,
            32
        )

        self.fc3 = nn.Linear(
            32,
            3
        )

        self.dropout = nn.Dropout(
            0.25
        )

    def get_node_embeddings(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x

    def forward(
        self,
        x,
        edge_index,
        herb_ids,
        drug_ids
    ):

        graph_embeddings = self.get_node_embeddings(
            x,
            edge_index
        )

        graph_herb = graph_embeddings[
            herb_ids
        ]

        graph_drug = graph_embeddings[
            drug_ids
        ]

        learned_herb = self.herb_embedding(
            herb_ids
        )

        learned_drug = self.drug_embedding(
            drug_ids
        )

        herb = graph_herb + learned_herb
        drug = graph_drug + learned_drug

        product = herb * drug

        difference = torch.abs(
            herb - drug
        )

        combined = torch.cat(
            [
                herb,
                drug,
                product,
                difference
            ],
            dim=1
        )

        combined = self.fc1(
            combined
        )

        combined = torch.relu(
            combined
        )

        combined = self.dropout(
            combined
        )

        combined = self.fc2(
            combined
        )

        combined = torch.relu(
            combined
        )

        return self.fc3(
            combined
        )

# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

x = checkpoint["node_features"]
edge_index = checkpoint["edge_index"]

model = HerbDrugInteractionModel(
    num_herbs=checkpoint["num_herbs"],
    num_drugs=checkpoint["num_drugs"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# LOAD DDID DATA
# ============================================================

import pandas as pd

ddid = pd.read_csv(DATASET)

herbs = sorted(
    ddid["Herb"].astype(str).str.strip().unique()
)

drugs = sorted(
    ddid["Drug"].astype(str).str.strip().unique()
)

herb_to_id = {
    name: i
    for i, name in enumerate(herbs)
}

drug_to_id = {
    name: 658 + i
    for i, name in enumerate(drugs)
}


def find_name(mapping, name):

    for actual in mapping:

        if actual.lower() == name.lower():
            return actual

    return None


# ============================================================
# GNN PREDICTION
# ============================================================

def gnn_predict(herb, drug):

    actual_herb = find_name(
        herb_to_id,
        herb
    )

    actual_drug = find_name(
        drug_to_id,
        drug
    )

    if actual_herb is None or actual_drug is None:

        return None

    herb_id = torch.tensor(
        [herb_to_id[actual_herb]],
        dtype=torch.long
    )

    drug_id = torch.tensor(
        [drug_to_id[actual_drug]],
        dtype=torch.long
    )

    with torch.no_grad():

        logits = model(
            x,
            edge_index,
            herb_id,
            drug_id
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        prediction = torch.argmax(
            probabilities
        ).item()

    labels = [
        "LOW",
        "MEDIUM",
        "HIGH"
    ]

    return {
        "risk": labels[prediction],
        "confidence": float(
            probabilities[prediction] * 100
        ),
        "low": float(probabilities[0] * 100),
        "medium": float(probabilities[1] * 100),
        "high": float(probabilities[2] * 100)
    }


# ============================================================
# DDID EVIDENCE
# ============================================================

def get_ddid_record(herb, drug):

    rows = ddid[
        (ddid["Herb"].str.lower() == herb.lower()) &
        (ddid["Drug"].str.lower() == drug.lower())
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


# ============================================================
# DISPLAY
# ============================================================

def run_prediction(herb, drug):

    print()
    print("=" * 70)
    print("AI HERB–DRUG INTERACTION PREDICTION")
    print("=" * 70)

    print(f"Herb : {herb}")
    print(f"Drug : {drug}")

    # --------------------------------------------------------
    # GNN
    # --------------------------------------------------------

    gnn = gnn_predict(
        herb,
        drug
    )

    print()
    print("---------- GNN PREDICTION ----------")

    if gnn:

        print(
            f"Risk       : {gnn['risk']}"
        )

        print(
            f"Confidence : {gnn['confidence']:.2f}%"
        )

        print(
            f"LOW        : {gnn['low']:.2f}%"
        )

        print(
            f"MEDIUM     : {gnn['medium']:.2f}%"
        )

        print(
            f"HIGH       : {gnn['high']:.2f}%"
        )

    else:

        print(
            "GNN: Herb/drug not present in trained dataset."
        )

    # --------------------------------------------------------
    # DDID
    # --------------------------------------------------------

    record = get_ddid_record(
        herb,
        drug
    )

    print()
    print("---------- DDID EVIDENCE ----------")

    if record is not None:

        print(
            "Recorded Risk :",
            record["Risk"]
        )

        print(
            "Effects       :",
            record["Effects"]
        )

        print(
            "Evidence Count:",
            record["Evidence_Count"]
        )

        print(
            "PMIDs         :",
            record["PMIDs"]
        )

    else:

        print(
            "No exact DDID record found."
        )

    # --------------------------------------------------------
    # BIOLOGICAL EVIDENCE
    # --------------------------------------------------------

    herb_features = get_herb_features(
        herb
    )

    drug_features = get_drug_features(
        drug
    )

    evidence = get_pair_evidence(
        herb,
        drug
    )

    print()
    print("---------- BIOLOGICAL EVIDENCE ----------")

    print(
        "Herb compounds:",
        herb_features["compounds"]
    )

    print(
        "Herb targets:",
        herb_features["proteins"]
    )

    print(
        "Drug targets:",
        drug_features["targets"]
    )

    common_targets = sorted(
        set(herb_features["proteins"]) &
        set(drug_features["targets"])
    )

    print(
        "Common targets:",
        common_targets
    )

    if evidence:

        print()
        print("Exact biological evidence:")

        for e in evidence:

            print(
                f"Compound: {e['compound']} | "
                f"Target: {e['target']} | "
                f"Effect: {e['effect']} | "
                f"PMID: {e['pmid']}"
            )

    else:

        print(
            "No exact biological evidence found."
        )

    # --------------------------------------------------------
    # UNSEEN / DOCKING PIPELINE
    # --------------------------------------------------------

    print()
    print("---------- BIOLOGICAL + DOCKING ANALYSIS ----------")

    try:

        predict_unseen(
            herb,
            drug
        )

    except Exception as e:

        print(
            "Biological/docking analysis error:",
            e
        )

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    print()
    print("=" * 70)
    print(" AI HERB-DRUG INTERACTION PREDICTION SYSTEM")
    print("=" * 70)

    while True:
        herb = input("\nEnter herb name (or 'exit'): ").strip()

        if herb.lower() == "exit":
            break

        drug = input("Enter drug name: ").strip()

        if not drug:
            print("Drug name cannot be empty.")
            continue

        run_prediction(herb, drug)

        again = input(
            "\nTry another interaction? (y/n): "
        ).strip().lower()

        if again != "y":
            break