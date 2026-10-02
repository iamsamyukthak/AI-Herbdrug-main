import os
import torch
import pandas as pd

from herbdrug_model import HerbDrugInteractionModel


# ============================================================
# 1. PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets"
)

MODEL_PATH = os.path.join(
    CURRENT_DIR,
    "herbdrug_complete_model.pth"
)

DDID_PATH = os.path.join(
    DATASET_PATH,
    "ddid_processed.csv"
)


# ============================================================
# 2. LOAD DDID DATASET
# ============================================================

if not os.path.exists(DDID_PATH):

    raise FileNotFoundError(
        f"\nDDID dataset not found:\n{DDID_PATH}"
    )


ddid = pd.read_csv(
    DDID_PATH
)


# ============================================================
# 3. CREATE EXACT GRAPH MAPPING
# ============================================================
#
# Current graph:
#
#   658 Herb nodes
#   108 Drug nodes
#   ----------------
#   766 Total nodes
#
# Node IDs:
#
#   Herbs:
#       0 ... 657
#
#   Drugs:
#       658 ... 765
#
# ============================================================

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
# 4. LOAD MODEL CHECKPOINT
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel checkpoint not found:\n{MODEL_PATH}"
    )


checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)


# ============================================================
# 5. LOAD GRAPH
# ============================================================

x = checkpoint["node_features"]
edge_index = checkpoint["edge_index"]


# ============================================================
# 6. VERIFY GRAPH
# ============================================================

actual_nodes = x.shape[0]

print()
print("=" * 60)
print("             LOADING TRAINED MODEL")
print("=" * 60)

print()

print(
    f"Herb nodes       : {num_herbs}"
)

print(
    f"Drug nodes       : {num_drugs}"
)

print(
    f"Expected nodes   : {num_nodes}"
)

print(
    f"Checkpoint nodes : {actual_nodes}"
)


if actual_nodes != num_nodes:

    raise ValueError(
        "\n\nGraph mismatch detected.\n"
        f"Expected {num_nodes} nodes but checkpoint "
        f"contains {actual_nodes} nodes."
    )


# ============================================================
# 7. CREATE MODEL
# ============================================================
#
# IMPORTANT:
#
# The trained model uses the TOTAL graph-node count
# for BOTH embedding tables.
#
# Therefore:
#
#   herb_embedding = 766 x 32
#   drug_embedding = 766 x 32
#
# This must match the checkpoint exactly.
#
# ============================================================

model = HerbDrugInteractionModel(
    num_herbs=num_herbs,
    num_drugs=num_nodes
)


# ============================================================
# 8. LOAD TRAINED WEIGHTS
# ============================================================

try:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

except RuntimeError as error:

    raise RuntimeError(
        "\n\nModel architecture does not match "
        "the saved checkpoint.\n\n"
        f"{error}"
    )


model.eval()


print()
print("Model weights loaded successfully.")
print()


# ============================================================
# 9. RISK LABELS
# ============================================================

risk_names = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH"
}


# ============================================================
# 10. FIND DDID EVIDENCE
# ============================================================

def find_ddid_evidence(
    herb_name,
    drug_name
):

    matches = ddid[
        (ddid["Herb"] == herb_name) &
        (ddid["Drug"] == drug_name)
    ]

    if matches.empty:

        return None

    return matches.iloc[0]


# ============================================================
# 11. PROBABILITY BAR
# ============================================================

def show_probability_bar(
    name,
    probability
):

    bar_length = int(
        probability * 30
    )

    bar = "█" * bar_length

    print(
        f"{name:7} : "
        f"{bar:<30} "
        f"{probability * 100:6.2f}%"
    )


# ============================================================
# 12. PREDICT INTERACTION
# ============================================================

def predict_interaction(
    herb_name,
    drug_name
):

    # --------------------------------------------------------
    # Clean input
    # --------------------------------------------------------

    herb_name = str(
        herb_name
    ).strip()

    drug_name = str(
        drug_name
    ).strip()


    # --------------------------------------------------------
    # Empty input
    # --------------------------------------------------------

    if not herb_name:

        print(
            "\nERROR: Herb name cannot be empty."
        )

        return


    if not drug_name:

        print(
            "\nERROR: Drug name cannot be empty."
        )

        return


    # --------------------------------------------------------
    # Check herb
    # --------------------------------------------------------

    if herb_name not in herb_to_id:

        print()
        print("=" * 60)
        print(
            "ERROR: Herb not found in trained dataset."
        )
        print(
            f"Input herb: {herb_name}"
        )
        print(
            "Please enter a herb from the available dataset."
        )
        print("=" * 60)

        return


    # --------------------------------------------------------
    # Check drug
    # --------------------------------------------------------

    if drug_name not in drug_to_id:

        print()
        print("=" * 60)
        print(
            "ERROR: Drug not found in trained dataset."
        )
        print(
            f"Input drug: {drug_name}"
        )
        print(
            "Please enter a drug from the available dataset."
        )
        print("=" * 60)

        return


    # ========================================================
    # GRAPH IDs
    # ========================================================

    herb_id = herb_to_id[
        herb_name
    ]

    drug_id = drug_to_id[
        drug_name
    ]


    # ========================================================
    # TENSORS
    # ========================================================

    herb_tensor = torch.tensor(
        [herb_id],
        dtype=torch.long
    )

    drug_tensor = torch.tensor(
        [drug_id],
        dtype=torch.long
    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    with torch.no_grad():

        output = model(
            x,
            edge_index,
            herb_tensor,
            drug_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

        predicted_class = torch.argmax(
            probabilities
        ).item()


    # ========================================================
    # RESULT
    # ========================================================

    predicted_risk = risk_names[
        predicted_class
    ]

    confidence = (
        probabilities[
            predicted_class
        ].item()
        * 100
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print()
    print("=" * 60)
    print(
        "             HERB-DRUG PREDICTION"
    )
    print("=" * 60)

    print()

    print(
        f"Herb              : {herb_name}"
    )

    print(
        f"Drug              : {drug_name}"
    )

    print()

    print(
        f"Predicted Risk    : {predicted_risk}"
    )

    print(
        f"Model Confidence  : {confidence:.2f}%"
    )

    print()

    print(
        "Class probabilities:"
    )

    print()

    show_probability_bar(
        "LOW",
        probabilities[0].item()
    )

    show_probability_bar(
        "MEDIUM",
        probabilities[1].item()
    )

    show_probability_bar(
        "HIGH",
        probabilities[2].item()
    )


    # ========================================================
    # DDID EVIDENCE
    # ========================================================

    evidence = find_ddid_evidence(
        herb_name,
        drug_name
    )

    print()
    print("-" * 60)
    print(
        "                 DDID EVIDENCE"
    )
    print("-" * 60)


    if evidence is None:

        print()

        print(
            "This exact herb-drug pair is not present "
            "in the DDID labeled dataset."
        )

        print()

        print(
            "No direct DDID evidence is available "
            "for this pair."
        )


    else:

        print()

        print(
            f"Recorded Risk    : {evidence['Risk']}"
        )

        print(
            f"Effects          : {evidence['Effects']}"
        )

        print(
            f"Evidence Count   : {evidence['Evidence_Count']}"
        )

        print(
            f"PMIDs            : {evidence['PMIDs']}"
        )


        # ----------------------------------------------------
        # Compare prediction with DDID
        # ----------------------------------------------------

        recorded_risk = str(
            evidence["Risk"]
        ).strip().upper()


        if predicted_risk == recorded_risk:

            print()

            print(
                "Model vs DDID    : MATCH"
            )

        else:

            print()

            print(
                "Model vs DDID    : DIFFER"
            )

            print(
                "The model prediction differs from "
                "the recorded DDID risk."
            )


    # ========================================================
    # CONFIDENCE NOTE
    # ========================================================

    print()
    print("-" * 60)

    print(
        "NOTE:"
    )

    print(
        "Model confidence is the softmax probability "
        "of the predicted class."
    )

    print(
        "It is not a clinically calibrated probability."
    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    print()

    print(
        "DISCLAIMER:"
    )

    print(
        "This is a machine-learning research prototype."
    )

    print(
        "Predictions should not be treated as medical advice."
    )

    print(
        "Always consult a qualified healthcare professional "
        "for real medication decisions."
    )

    print("=" * 60)


# ============================================================
# 13. START INTERACTIVE PREDICTOR
# ============================================================

print("=" * 60)
print(
    "        AI HERB-DRUG INTERACTION PREDICTOR"
)
print("=" * 60)

print()

print(
    f"Available herbs : {num_herbs}"
)

print(
    f"Available drugs : {num_drugs}"
)

print()

print(
    "Type 'exit' to stop."
)

print()


# ============================================================
# 14. INTERACTIVE LOOP
# ============================================================

while True:

    herb_name = input(
        "Enter herb name: "
    ).strip()


    if herb_name.lower() == "exit":

        print(
            "\nExiting predictor."
        )

        break


    drug_name = input(
        "Enter drug name: "
    ).strip()


    if drug_name.lower() == "exit":

        print(
            "\nExiting predictor."
        )

        break


    predict_interaction(
        herb_name,
        drug_name
    )

    print()