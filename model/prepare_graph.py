import os

import pandas as pd
import torch

from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOCATE PROJECT AND DATASET
# ============================================================

current_dir = os.path.dirname(
    os.path.abspath(__file__)
)

project_root = os.path.dirname(
    current_dir
)

dataset_path = os.path.join(
    project_root,
    "datasets"
)


# ============================================================
# 2. LOAD DDID DATASET
# ============================================================

ddid = pd.read_csv(
    os.path.join(
        dataset_path,
        "ddid_processed.csv"
    )
)


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print()
print("=" * 60)
print("             DDID KNOWLEDGE GRAPH")
print("=" * 60)

print(
    "Total DDID samples :",
    len(ddid)
)


# ============================================================
# 4. CREATE UNIQUE HERB / DRUG NODES
# ============================================================

unique_herbs = sorted(
    ddid["Herb"].unique()
)

unique_drugs = sorted(
    ddid["Drug"].unique()
)


# ============================================================
# 5. NODE → ID MAPPING
# ============================================================

node_to_id = {}

node_types = {}


# ------------------------------------------------------------
# HERB NODES
# ------------------------------------------------------------

for herb in unique_herbs:

    node_id = len(node_to_id)

    node_to_id[herb] = node_id

    node_types[node_id] = "Herb"


# ------------------------------------------------------------
# DRUG NODES
# ------------------------------------------------------------

for drug in unique_drugs:

    node_id = len(node_to_id)

    node_to_id[drug] = node_id

    node_types[node_id] = "Drug"


# ============================================================
# 6. NODE FEATURES
# ============================================================

node_features = []


for node_id in range(
    len(node_to_id)
):

    node_type = node_types[node_id]


    if node_type == "Herb":

        feature = [1.0, 0.0]

    else:

        feature = [0.0, 1.0]


    node_features.append(
        feature
    )


x = torch.tensor(
    node_features,
    dtype=torch.float
)


# ============================================================
# 7. MAP ALL DDID PAIRS
# ============================================================

herb_ids = []

drug_ids = []

labels = []


for _, row in ddid.iterrows():

    herb_ids.append(
        node_to_id[row["Herb"]]
    )

    drug_ids.append(
        node_to_id[row["Drug"]]
    )

    labels.append(
        int(row["Label"])
    )


herb_ids = torch.tensor(
    herb_ids,
    dtype=torch.long
)

drug_ids = torch.tensor(
    drug_ids,
    dtype=torch.long
)

labels = torch.tensor(
    labels,
    dtype=torch.long
)


# ============================================================
# 8. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

indices = torch.arange(
    len(ddid)
)


train_indices, temp_indices = train_test_split(
    indices.numpy(),
    test_size=0.20,
    random_state=42,
    stratify=labels.numpy()
)


val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=42,
    stratify=labels.numpy()[temp_indices]
)


train_idx = torch.tensor(
    train_indices,
    dtype=torch.long
)

val_idx = torch.tensor(
    val_indices,
    dtype=torch.long
)

test_idx = torch.tensor(
    test_indices,
    dtype=torch.long
)


# ============================================================
# 9. BUILD TRAINING GRAPH ONLY
# ============================================================

numeric_edges = []


for index in train_idx:

    herb_node = herb_ids[index].item()

    drug_node = drug_ids[index].item()


    # Forward edge
    numeric_edges.append(
        (
            herb_node,
            drug_node
        )
    )


    # Reverse edge
    numeric_edges.append(
        (
            drug_node,
            herb_node
        )
    )


edge_index = torch.tensor(
    numeric_edges,
    dtype=torch.long
).t().contiguous()


# ============================================================
# 10. DISPLAY GRAPH INFORMATION
# ============================================================

print(
    "Unique herbs       :",
    len(unique_herbs)
)

print(
    "Unique drugs       :",
    len(unique_drugs)
)

print(
    "Total nodes        :",
    len(node_to_id)
)

print(
    "Training edges     :",
    len(train_idx)
)

print(
    "GNN edges          :",
    len(numeric_edges)
)


# ============================================================
# 11. NODE TYPES
# ============================================================

herb_count = sum(
    1
    for node_type in node_types.values()
    if node_type == "Herb"
)

drug_count = sum(
    1
    for node_type in node_types.values()
    if node_type == "Drug"
)


print()
print("=" * 60)
print("                NODE TYPES")
print("=" * 60)

print(
    "Herb nodes         :",
    herb_count
)

print(
    "Drug nodes         :",
    drug_count
)


# ============================================================
# 12. TENSOR SHAPES
# ============================================================

print()
print("=" * 60)
print("              TENSOR SHAPES")
print("=" * 60)

print(
    "Node features      :",
    x.shape
)

print(
    "Edge index         :",
    edge_index.shape
)

print(
    "Herb IDs           :",
    herb_ids.shape
)

print(
    "Drug IDs           :",
    drug_ids.shape
)

print(
    "Labels             :",
    labels.shape
)


# ============================================================
# 13. SPLIT INFORMATION
# ============================================================

print()
print("=" * 60)
print("             DATA SPLIT")
print("=" * 60)

print(
    "Training samples   :",
    len(train_idx)
)

print(
    "Validation samples :",
    len(val_idx)
)

print(
    "Testing samples    :",
    len(test_idx)
)


# ============================================================
# 14. LABEL DISTRIBUTION
# ============================================================

print()
print("=" * 60)
print("             LABEL DISTRIBUTION")
print("=" * 60)

all_counts = torch.bincount(
    labels,
    minlength=3
)

train_counts = torch.bincount(
    labels[train_idx],
    minlength=3
)

val_counts = torch.bincount(
    labels[val_idx],
    minlength=3
)

test_counts = torch.bincount(
    labels[test_idx],
    minlength=3
)


print()
print("                ALL     TRAIN    VAL    TEST")

print(
    f"LOW       :   "
    f"{all_counts[0].item():4d}    "
    f"{train_counts[0].item():4d}    "
    f"{val_counts[0].item():4d}    "
    f"{test_counts[0].item():4d}"
)

print(
    f"MEDIUM    :   "
    f"{all_counts[1].item():4d}    "
    f"{train_counts[1].item():4d}    "
    f"{val_counts[1].item():4d}    "
    f"{test_counts[1].item():4d}"
)

print(
    f"HIGH      :   "
    f"{all_counts[2].item():4d}    "
    f"{train_counts[2].item():4d}    "
    f"{val_counts[2].item():4d}    "
    f"{test_counts[2].item():4d}"
)


# ============================================================
# 15. SAMPLE MAPPINGS
# ============================================================

print()
print("=" * 60)
print("             SAMPLE MAPPINGS")
print("=" * 60)


for i in range(
    min(10, len(ddid))
):

    print(
        f"{ddid.iloc[i]['Herb']:30}"
        f" -> {herb_ids[i]:4d} | "
        f"{ddid.iloc[i]['Drug']:20}"
        f" -> {drug_ids[i]:4d} | "
        f"Label: {labels[i]}"
    )


# ============================================================
# 16. COMPLETE
# ============================================================

print()
print("=" * 60)
print("        GRAPH PREPARATION COMPLETE")
print("=" * 60)