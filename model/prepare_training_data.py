import os
import pandas as pd
import torch


# ============================================================
# 1. LOCATE PROJECT AND DATASET
# ============================================================

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)

dataset_path = os.path.join(
    project_root,
    "datasets"
)


# ============================================================
# 2. LOAD DDID DATA
# ============================================================

ddid_file = os.path.join(
    dataset_path,
    "ddid_herb_drug_training.csv"
)

df = pd.read_csv(ddid_file)


# ============================================================
# 3. LOAD GRAPH NODE INFORMATION
# ============================================================

# These are the original datasets used to build our graph.

herbs_df = pd.read_csv(
    os.path.join(dataset_path, "herbs.csv")
)

drugs_df = pd.read_csv(
    os.path.join(dataset_path, "drugs.csv")
)

compound_protein_df = pd.read_csv(
    os.path.join(dataset_path, "compound_protein.csv")
)

proteins_df = pd.read_csv(
    os.path.join(dataset_path, "proteins.csv")
)


# ============================================================
# 4. RECREATE NODE → ID MAPPING
# ============================================================

import networkx as nx

G = nx.Graph()


# ------------------------------------------------------------
# Herb → Compound
# ------------------------------------------------------------

for _, row in herbs_df.iterrows():

    herb = row["Herb"]
    compound = row["Active_Compound"]

    G.add_node(
        herb,
        type="Herb"
    )

    G.add_node(
        compound,
        type="Compound"
    )

    G.add_edge(
        herb,
        compound
    )


# ------------------------------------------------------------
# Compound → Protein
# ------------------------------------------------------------

for _, row in compound_protein_df.iterrows():

    compound = row["Compound"]
    protein = row["Protein"]

    G.add_node(
        compound,
        type="Compound"
    )

    G.add_node(
        protein,
        type="Protein"
    )

    G.add_edge(
        compound,
        protein
    )


# ------------------------------------------------------------
# Drug → Protein
# ------------------------------------------------------------

for _, row in drugs_df.iterrows():

    drug = row["Drug"]
    protein = row["Target_Protein"]

    G.add_node(
        drug,
        type="Drug"
    )

    G.add_node(
        protein,
        type="Protein"
    )

    G.add_edge(
        drug,
        protein
    )


# ------------------------------------------------------------
# Protein → Pathway
# ------------------------------------------------------------

for _, row in proteins_df.iterrows():

    protein = row["Protein"]
    pathway = row["Pathway"]

    G.add_node(
        protein,
        type="Protein"
    )

    G.add_node(
        pathway,
        type="Pathway"
    )

    G.add_edge(
        protein,
        pathway
    )


# ============================================================
# 5. NODE → ID MAPPING
# ============================================================

node_to_id = {
    node: index
    for index, node in enumerate(G.nodes())
}


print("\n==============================================")
print("          GRAPH INFORMATION")
print("==============================================")

print("Total graph nodes :", len(node_to_id))


# ============================================================
# 6. MAP HERBS AND DRUGS TO NODE IDs
# ============================================================

herb_ids = []
drug_ids = []
labels = []

missing_herbs = []
missing_drugs = []


for _, row in df.iterrows():

    herb = row["Herb"]
    drug = row["Drug"]

    # -------------------------
    # Herb
    # -------------------------

    if herb in node_to_id:

        herb_ids.append(
            node_to_id[herb]
        )

    else:

        herb_ids.append(-1)

        missing_herbs.append(herb)


    # -------------------------
    # Drug
    # -------------------------

    if drug in node_to_id:

        drug_ids.append(
            node_to_id[drug]
        )

    else:

        drug_ids.append(-1)

        missing_drugs.append(drug)


    # -------------------------
    # Risk label
    # -------------------------

    if row["Risk"] == "Low":
        labels.append(0)

    elif row["Risk"] == "Medium":
        labels.append(1)

    elif row["Risk"] == "High":
        labels.append(2)


# ============================================================
# 7. DISPLAY MAPPING RESULTS
# ============================================================

print("\n==============================================")
print("          NODE MAPPING RESULTS")
print("==============================================")

print("Total samples :", len(df))

print(
    "Missing herbs :",
    len(set(missing_herbs))
)

print(
    "Missing drugs :",
    len(set(missing_drugs))
)


# ============================================================
# 8. DISPLAY SOME EXAMPLES
# ============================================================

print("\n==============================================")
print("             MAPPING EXAMPLES")
print("==============================================")


for i in range(min(10, len(df))):

    print(
        f"{df.iloc[i]['Herb']:30} "
        f"-> {herb_ids[i]:4}"
    )

    print(
        f"{df.iloc[i]['Drug']:30} "
        f"-> {drug_ids[i]:4}"
    )

    print(
        f"Risk: {df.iloc[i]['Risk']} "
        f"-> Label: {labels[i]}"
    )

    print("----------------------------------------------")


# ============================================================
# 9. CONVERT TO PYTORCH TENSORS
# ============================================================

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
# 10. SAVE PROCESSED DATA
# ============================================================

output_file = os.path.join(
    dataset_path,
    "ddid_training_tensors.pt"
)

torch.save(
    {
        "herb_ids": herb_ids,
        "drug_ids": drug_ids,
        "labels": labels
    },
    output_file
)


# ============================================================
# 11. FINAL OUTPUT
# ============================================================

print("\n==============================================")
print("       TRAINING DATA READY")
print("==============================================")

print(
    "Herb IDs shape :",
    herb_ids.shape
)

print(
    "Drug IDs shape :",
    drug_ids.shape
)

print(
    "Labels shape   :",
    labels.shape
)

print("\nSaved to:")
print(output_file)