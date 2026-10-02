import os
import torch
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from herbdrug_model import HerbDrugInteractionModel


# ============================================================
# 1. PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    CURRENT_DIR
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "ddid_processed.csv"
)

MODEL_PATH = os.path.join(
    CURRENT_DIR,
    "herbdrug_complete_model.pth"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

ddid = pd.read_csv(
    DATASET_PATH
)

unique_herbs = sorted(
    ddid["Herb"].unique()
)

unique_drugs = sorted(
    ddid["Drug"].unique()
)

num_herbs = len(unique_herbs)
num_drugs = len(unique_drugs)

num_nodes = num_herbs + num_drugs


herb_to_id = {
    herb: i
    for i, herb in enumerate(unique_herbs)
}


drug_to_id = {
    drug: num_herbs + i
    for i, drug in enumerate(unique_drugs)
}


# ============================================================
# 3. LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

x = checkpoint["node_features"]
edge_index = checkpoint["edge_index"]


# ============================================================
# 4. CREATE MODEL
# ============================================================

model = HerbDrugInteractionModel(
    num_herbs=num_herbs,
    num_drugs=num_nodes
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# 5. PREPARE ALL PAIRS
# ============================================================

herb_ids = torch.tensor(
    [
        herb_to_id[h]
        for h in ddid["Herb"]
    ],
    dtype=torch.long
)

drug_ids = torch.tensor(
    [
        drug_to_id[d]
        for d in ddid["Drug"]
    ],
    dtype=torch.long
)

true_labels = torch.tensor(
    ddid["Label"].values,
    dtype=torch.long
)


# ============================================================
# 6. RUN PREDICTIONS
# ============================================================

with torch.no_grad():

    outputs = model(
        x,
        edge_index,
        herb_ids,
        drug_ids
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predictions = torch.argmax(
        probabilities,
        dim=1
    )


y_true = true_labels.numpy()
y_pred = predictions.numpy()


# ============================================================
# 7. BASIC METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0
    )
)


macro_f1 = f1.mean()


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print()
print("=" * 70)
print(
    "                 MODEL EVALUATION"
)
print("=" * 70)

print()

print(
    f"Dataset samples : {len(ddid)}"
)

print(
    f"Unique herbs    : {num_herbs}"
)

print(
    f"Unique drugs    : {num_drugs}"
)

print()

print(
    f"Accuracy        : {accuracy * 100:.2f}%"
)

print(
    f"Macro F1        : {macro_f1:.4f}"
)

print()


# ============================================================
# 9. CLASS-WISE METRICS
# ============================================================

class_names = [
    "LOW",
    "MEDIUM",
    "HIGH"
]

print("-" * 70)
print(
    "                 CLASS-WISE METRICS"
)
print("-" * 70)

print()

for i, name in enumerate(class_names):

    print(
        f"{name:7} | "
        f"Precision: {precision[i]:.4f} | "
        f"Recall: {recall[i]:.4f} | "
        f"F1: {f1[i]:.4f} | "
        f"Support: {support[i]}"
    )


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print()
print("-" * 70)
print(
    "                 CLASSIFICATION REPORT"
)
print("-" * 70)

print()

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2]
)

print("-" * 70)
print(
    "                 CONFUSION MATRIX"
)
print("-" * 70)

print()

print(
    "                 Predicted"
)

print(
    "              LOW  MED  HIGH"
)

print(
    f"Actual LOW   {cm[0,0]:4d} {cm[0,1]:4d} {cm[0,2]:5d}"
)

print(
    f"Actual MED   {cm[1,0]:4d} {cm[1,1]:4d} {cm[1,2]:5d}"
)

print(
    f"Actual HIGH  {cm[2,0]:4d} {cm[2,1]:4d} {cm[2,2]:5d}"
)


# ============================================================
# 12. CONFIDENCE ANALYSIS
# ============================================================

max_probabilities = (
    probabilities.max(
        dim=1
    ).values.numpy()
)

average_confidence = (
    max_probabilities.mean()
)

high_confidence = (
    max_probabilities >= 0.90
).sum()

correct_predictions = (
    y_true == y_pred
)

high_confidence_correct = (
    correct_predictions &
    (max_probabilities >= 0.90)
).sum()

high_confidence_wrong = (
    (~correct_predictions) &
    (max_probabilities >= 0.90)
).sum()


print()
print("-" * 70)
print(
    "                 CONFIDENCE ANALYSIS"
)
print("-" * 70)

print()

print(
    f"Average confidence       : "
    f"{average_confidence * 100:.2f}%"
)

print(
    f"Predictions >= 90%       : "
    f"{high_confidence}"
)

print(
    f">=90% confidence correct : "
    f"{high_confidence_correct}"
)

print(
    f">=90% confidence wrong   : "
    f"{high_confidence_wrong}"
)


# ============================================================
# 13. FIND HIGH-CONFIDENCE ERRORS
# ============================================================

wrong_indices = [
    i
    for i in range(len(ddid))
    if y_true[i] != y_pred[i]
]

wrong_indices.sort(
    key=lambda i: max_probabilities[i],
    reverse=True
)


print()
print("-" * 70)
print(
    "           HIGH-CONFIDENCE WRONG PREDICTIONS"
)
print("-" * 70)

print()

if not wrong_indices:

    print(
        "No incorrect predictions."
    )

else:

    count = min(
        10,
        len(wrong_indices)
    )

    for i in wrong_indices[:count]:

        predicted_name = class_names[
            y_pred[i]
        ]

        actual_name = class_names[
            y_true[i]
        ]

        confidence = (
            max_probabilities[i] * 100
        )

        print(
            f"{i + 1:3}. "
            f"{ddid.iloc[i]['Herb']} + "
            f"{ddid.iloc[i]['Drug']}"
        )

        print(
            f"     Actual: {actual_name} | "
            f"Predicted: {predicted_name} | "
            f"Confidence: {confidence:.2f}%"
        )

        print()


# ============================================================
# 14. FINISH
# ============================================================

print("=" * 70)
print(
    "                 EVALUATION COMPLETE"
)
print("=" * 70)