import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from prepare_graph import (
    x,
    edge_index,
    herb_ids,
    drug_ids,
    labels
)

from herbdrug_model import (
    HerbDrugInteractionModel
)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

torch.manual_seed(42)


# ============================================================
# 2. REMOVE INVALID SAMPLES
# ============================================================

valid_mask = (
    (herb_ids >= 0) &
    (drug_ids >= 0)
)

herb_ids = herb_ids[valid_mask]
drug_ids = drug_ids[valid_mask]
labels = labels[valid_mask]


print()
print("=" * 60)
print("                 TRAINING DATA")
print("=" * 60)

print("Valid samples:", len(labels))


# ============================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

indices = torch.arange(
    len(labels)
)


train_idx, temp_idx = train_test_split(
    indices.numpy(),
    test_size=0.20,
    random_state=42,
    stratify=labels.numpy()
)


val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=labels.numpy()[temp_idx]
)


train_idx = torch.tensor(
    train_idx,
    dtype=torch.long
)

val_idx = torch.tensor(
    val_idx,
    dtype=torch.long
)

test_idx = torch.tensor(
    test_idx,
    dtype=torch.long
)


print()
print("=" * 60)
print("                   DATA SPLIT")
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
# 4. NUMBER OF HERBS / DRUGS
# ============================================================

num_herbs = int(
    herb_ids.max().item()
) + 1

num_drugs = int(
    drug_ids.max().item()
) + 1


print()
print("=" * 60)
print("               ENTITY COUNTS")
print("=" * 60)

print(
    "Number of herbs :",
    num_herbs
)

print(
    "Number of drugs :",
    num_drugs
)


# ============================================================
# 5. CLASS DISTRIBUTION
# ============================================================

class_counts = torch.bincount(
    labels[train_idx],
    minlength=3
)


print()
print("=" * 60)
print("             CLASS DISTRIBUTION")
print("=" * 60)

print(
    "LOW    :",
    class_counts[0].item()
)

print(
    "MEDIUM :",
    class_counts[1].item()
)

print(
    "HIGH   :",
    class_counts[2].item()
)


# ============================================================
# 6. CLASS WEIGHTS
# ============================================================

class_weights = (
    len(train_idx)
    /
    (
        3.0 *
        class_counts.float()
    )
)


print()
print("Class weights:")
print(class_weights)


# ============================================================
# 7. MODEL
# ============================================================

model = HerbDrugInteractionModel(
    num_herbs=num_herbs,
    num_drugs=num_drugs
)


print()
print("=" * 60)
print("                     MODEL")
print("=" * 60)

print(model)


# ============================================================
# 8. LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# 9. OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)


# ============================================================
# 10. TRAINING SETTINGS
# ============================================================

epochs = 500

best_val_f1 = -1

best_state = None

patience = 60

patience_counter = 0


print()
print("=" * 60)
print("                TRAINING STARTED")
print("=" * 60)


# ============================================================
# 11. TRAINING LOOP
# ============================================================

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    outputs = model(
        x,
        edge_index,
        herb_ids[train_idx],
        drug_ids[train_idx]
    )


    loss = criterion(
        outputs,
        labels[train_idx]
    )


    loss.backward()


    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=2.0
    )


    optimizer.step()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()


    with torch.no_grad():

        val_outputs = model(
            x,
            edge_index,
            herb_ids[val_idx],
            drug_ids[val_idx]
        )


        val_predictions = torch.argmax(
            val_outputs,
            dim=1
        )


        val_f1 = f1_score(
            labels[val_idx].numpy(),
            val_predictions.numpy(),
            average="macro"
        )


        val_accuracy = accuracy_score(
            labels[val_idx].numpy(),
            val_predictions.numpy()
        )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        best_state = copy.deepcopy(
            model.state_dict()
        )

        patience_counter = 0

    else:

        patience_counter += 1


    # --------------------------------------------------------
    # PRINT PROGRESS
    # --------------------------------------------------------

    if (
        epoch == 0
        or
        (epoch + 1) % 20 == 0
    ):

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss {loss.item():.4f} | "
            f"Val Acc {val_accuracy:.4f} | "
            f"Val F1 {val_f1:.4f}"
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if patience_counter >= patience:

        print()

        print(
            "Early stopping at epoch",
            epoch + 1
        )

        break


# ============================================================
# 12. RESTORE BEST MODEL
# ============================================================

model.load_state_dict(
    best_state
)

model.eval()


# ============================================================
# 13. FINAL TEST
# ============================================================

with torch.no_grad():

    test_outputs = model(
        x,
        edge_index,
        herb_ids[test_idx],
        drug_ids[test_idx]
    )


    test_predictions = torch.argmax(
        test_outputs,
        dim=1
    )


y_true = labels[
    test_idx
].numpy()

y_pred = test_predictions.numpy()


# ============================================================
# 14. METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro"
)


print()
print("=" * 60)
print("                  FINAL TEST")
print("=" * 60)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Macro F1 : {macro_f1:.4f}"
)


print()
print("Classification Report:")
print()


print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "LOW",
            "MEDIUM",
            "HIGH"
        ],
        zero_division=0
    )
)


print("Confusion Matrix:")
print()


print(
    confusion_matrix(
        y_true,
        y_pred
    )
)


# ============================================================
# 15. SAVE MODEL
# ============================================================

current_dir = os.path.dirname(
    os.path.abspath(__file__)
)

project_root = os.path.dirname(
    current_dir
)


model_dir = os.path.join(
    project_root,
    "model"
)


os.makedirs(
    model_dir,
    exist_ok=True
)


model_path = os.path.join(
    model_dir,
    "herbdrug_complete_model.pth"
)


torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "node_features":
            x,

        "edge_index":
            edge_index,

        "num_herbs":
            num_herbs,

        "num_drugs":
            num_drugs,

        "best_val_f1":
            best_val_f1,

        "test_accuracy":
            accuracy,

        "test_macro_f1":
            macro_f1
    },
    model_path
)


print()
print("=" * 60)
print("             MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print()

print(
    "Saved to:",
    model_path
)