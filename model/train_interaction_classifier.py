import torch
import pandas as pd
import os

from prepare_graph import x, edge_index
from hgnn_model import HerbDrugHGNN
from interaction_classifier import InteractionClassifier

# Load graph model
graph_model = HerbDrugHGNN()

# Generate embeddings
graph_model.eval()

with torch.no_grad():
    embeddings = graph_model(x, edge_index)

# Read training data
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)

training_data = pd.read_csv(
    os.path.join(project_root,
                 "datasets",
                 "training_data.csv")
)

print(training_data)

print("\nEmbeddings Shape:")
print(embeddings.shape)
X_herb = []
X_drug = []
y = []

for _, row in training_data.iterrows():

    herb_id = int(row["Herb_ID"])
    drug_id = int(row["Drug_ID"])
    label = int(row["Label"])

    X_herb.append(embeddings[herb_id])
    X_drug.append(embeddings[drug_id])

    y.append(label)

X_herb = torch.stack(X_herb)

X_drug = torch.stack(X_drug)

y = torch.tensor(y)

print("\nHerb Embeddings:")
print(X_herb.shape)

print("\nDrug Embeddings:")
print(X_drug.shape)

print("\nLabels:")
print(y)
import torch.nn as nn
import torch.optim as optim

# ============================================
# Create Classifier
# ============================================

classifier = InteractionClassifier()

# ============================================
# Loss Function
# ============================================

criterion = nn.CrossEntropyLoss()

# ============================================
# Optimizer
# ============================================

optimizer = optim.Adam(
    classifier.parameters(),
    lr=0.001
)

epochs = 200

print("\n========== TRAINING STARTED ==========\n")

# ============================================
# Training Loop
# ============================================

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = classifier(
        X_herb,
        X_drug
    )

    loss = criterion(
        outputs,
        y
    )

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 20 == 0:

        print(
            f"Epoch {epoch+1}/{epochs} | Loss = {loss.item():.4f}"
        )
# ============================================
# Evaluate Model
# ============================================

with torch.no_grad():

    predictions = classifier(
        X_herb,
        X_drug
    )

    predicted_labels = torch.argmax(
        predictions,
        dim=1
    )

    correct = (
        predicted_labels == y
    ).sum().item()

    accuracy = (
        correct / len(y)
    ) * 100

print("\n========== MODEL EVALUATION ==========\n")

print("Predicted Labels:")
print(predicted_labels)

print("\nActual Labels:")
print(y)

print(f"\nTraining Accuracy : {accuracy:.2f}%")        
# ============================================
# Save Trained Classifier
# ============================================

torch.save(
    classifier.state_dict(),
    "model/interaction_classifier.pth"
)

print("\nClassifier saved successfully!")