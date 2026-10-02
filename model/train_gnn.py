import os
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


# ============================================================
# 1. PATHS
# ============================================================

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)

dataset_path = os.path.join(project_root, "datasets")


# ============================================================
# 2. LOAD DATA
# ============================================================

ddid = pd.read_csv(
    os.path.join(dataset_path, "ddid_processed.csv")
)


# ============================================================
# 3. REBUILD KNOWLEDGE GRAPH
# ============================================================

herbs = pd.read_csv(
    os.path.join(dataset_path, "herbs.csv")
)

drugs = pd.read_csv(
    os.path.join(dataset_path, "drugs.csv")
)

proteins = pd.read_csv(
    os.path.join(dataset_path, "proteins.csv")
)

compound_protein = pd.read_csv(
    os.path.join(dataset_path, "compound_protein.csv")
)


G = nx.Graph()


# Herb → Compound
for _, row in herbs.iterrows():

    herb = row["Herb"]
    compound = row["Active_Compound"]

    G.add_node(herb, type="Herb")
    G.add_node(compound, type="Compound")

    G.add_edge(herb, compound)


# Compound → Protein
for _, row in compound_protein.iterrows():

    compound = row["Compound"]
    protein = row["Protein"]

    G.add_node(compound, type="Compound")
    G.add_node(protein, type="Protein")

    G.add_edge(compound, protein)


# Drug → Protein
for _, row in drugs.iterrows():

    drug = row["Drug"]
    protein = row["Target_Protein"]

    G.add_node(drug, type="Drug")
    G.add_node(protein, type="Protein")

    G.add_edge(drug, protein)


# Protein → Pathway
for _, row in proteins.iterrows():

    protein = row["Protein"]
    pathway = row["Pathway"]

    G.add_node(protein, type="Protein")
    G.add_node(pathway, type="Pathway")

    G.add_edge(protein, pathway)


# Add DDID herbs/drugs
for herb in ddid["Herb"].unique():

    if herb not in G:
        G.add_node(herb, type="Herb")


for drug in ddid["Drug"].unique():

    if drug not in G:
        G.add_node(drug, type="Drug")


# ============================================================
# 4. NODE → ID
# ============================================================

node_to_id = {
    node: i
    for i, node in enumerate(G.nodes())
}


# ============================================================
# 5. NODE FEATURES
# ============================================================

node_type_features = {

    "Herb":     [1, 0, 0, 0, 0],
    "Compound": [0, 1, 0, 0, 0],
    "Protein":  [0, 0, 1, 0, 0],
    "Drug":     [0, 0, 0, 1, 0],
    "Pathway":  [0, 0, 0, 0, 1]
}


node_features = []

for node in G.nodes():

    node_type = G.nodes[node]["type"]

    node_features.append(
        node_type_features[node_type]
    )


x = torch.tensor(
    node_features,
    dtype=torch.float
)


# ============================================================
# 6. EDGE INDEX
# ============================================================

edges = []

for source, target in G.edges():

    s = node_to_id[source]
    t = node_to_id[target]

    edges.append((s, t))
    edges.append((t, s))


edge_index = torch.tensor(
    edges,
    dtype=torch.long
).t().contiguous()


# ============================================================
# 7. CREATE TRAINING SAMPLES
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
# 8. TRAIN / TEST SPLIT
# ============================================================

torch.manual_seed(42)

num_samples = len(labels)

indices = torch.randperm(num_samples)

train_size = int(0.8 * num_samples)

train_indices = indices[:train_size]
test_indices = indices[train_size:]


# ============================================================
# 9. GNN MODEL
# ============================================================

class HerbDrugGNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = GCNConv(
            5,
            32
        )

        self.conv2 = GCNConv(
            32,
            16
        )

        self.fc1 = nn.Linear(
            32,
            32
        )

        self.fc2 = nn.Linear(
            32,
            3
        )


    def forward(
        self,
        x,
        edge_index,
        herb_ids,
        drug_ids
    ):

        # GCN layer 1
        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)


        # GCN layer 2
        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)


        # Get herb embeddings
        herb_embeddings = x[herb_ids]


        # Get drug embeddings
        drug_embeddings = x[drug_ids]


        # Combine
        pair_embeddings = torch.cat(
            [
                herb_embeddings,
                drug_embeddings
            ],
            dim=1
        )


        # Classifier
        out = self.fc1(
            pair_embeddings
        )

        out = F.relu(out)

        out = self.fc2(
            out
        )

        return out


# ============================================================
# 10. CREATE MODEL
# ============================================================

model = HerbDrugGNN()


# ============================================================
# 11. LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# ============================================================
# 12. TRAINING
# ============================================================

print("\n==============================================")
print("             GNN TRAINING")
print("==============================================")

print("Total samples :", num_samples)
print("Training      :", len(train_indices))
print("Testing       :", len(test_indices))

print("\nTraining started...")


for epoch in range(1, 301):

    model.train()

    optimizer.zero_grad()


    # Forward pass
    output = model(
        x,
        edge_index,
        herb_ids[train_indices],
        drug_ids[train_indices]
    )


    # Calculate loss
    loss = criterion(
        output,
        labels[train_indices]
    )


    # Backpropagation
    loss.backward()

    optimizer.step()


    # Training accuracy
    predictions = output.argmax(
        dim=1
    )

    accuracy = (
        predictions ==
        labels[train_indices]
    ).float().mean()


    if epoch % 25 == 0:

        print(
            f"Epoch {epoch:3}/300 | "
            f"Loss: {loss.item():.4f} | "
            f"Train Accuracy: "
            f"{accuracy.item() * 100:.2f}%"
        )


# ============================================================
# 13. TESTING
# ============================================================

model.eval()

with torch.no_grad():

    test_output = model(
        x,
        edge_index,
        herb_ids[test_indices],
        drug_ids[test_indices]
    )

    test_predictions = test_output.argmax(
        dim=1
    )


test_accuracy = (
    test_predictions ==
    labels[test_indices]
).float().mean()


# ============================================================
# 14. TEST RESULTS
# ============================================================

print("\n==============================================")
print("             TEST RESULTS")
print("==============================================")

print("\nActual labels:")

print(
    labels[test_indices]
)


print("\nPredicted labels:")

print(
    test_predictions
)


print(
    f"\nTest Accuracy : "
    f"{test_accuracy.item() * 100:.2f}%"
)


# ============================================================
# 15. SAMPLE PREDICTIONS
# ============================================================

risk_names = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH"
}


print("\n==============================================")
print("          TEST PREDICTIONS")
print("==============================================")


for i in range(
    min(10, len(test_indices))
):

    original_index = test_indices[i].item()

    herb = ddid.iloc[original_index]["Herb"]

    drug = ddid.iloc[original_index]["Drug"]

    actual = labels[test_indices[i]].item()

    predicted = test_predictions[i].item()


    print(
        f"\nSample {original_index}"
    )

    print(
        f"Herb       : {herb}"
    )

    print(
        f"Drug       : {drug}"
    )

    print(
        f"Actual Risk: {risk_names[actual]}"
    )

    print(
        f"Predicted  : {risk_names[predicted]}"
    )


# ============================================================
# 16. SAVE MODEL
# ============================================================

model_path = os.path.join(
    dataset_path,
    "herb_drug_gnn.pth"
)


torch.save(
    model.state_dict(),
    model_path
)


print("\n==============================================")
print("       MODEL SAVED SUCCESSFULLY")
print("==============================================")

print(
    "Saved to:",
    model_path
)