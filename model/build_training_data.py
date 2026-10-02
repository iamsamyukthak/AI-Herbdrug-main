import pandas as pd
import os

from prepare_graph import node_to_id

# ============================================
# Locate Dataset
# ============================================

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)

dataset_path = os.path.join(
    project_root,
    "datasets"
)

# ============================================
# Read DDID Dataset
# ============================================

ddid = pd.read_csv(
    os.path.join(
        dataset_path,
        "ddid_processed.csv"
    )
)

# ============================================
# Build Training Data
# ============================================

training_data = []

for _, row in ddid.iterrows():

    herb = row["Herb"]
    drug = row["Drug"]
    label = int(row["Label"])

    herb_id = node_to_id[herb]
    drug_id = node_to_id[drug]

    training_data.append(
        [herb_id, drug_id, label]
    )

# ============================================
# Create DataFrame
# ============================================

training_df = pd.DataFrame(
    training_data,
    columns=[
        "Herb_ID",
        "Drug_ID",
        "Label"
    ]
)

# ============================================
# Display Information
# ============================================

print("\n==============================================")
print("       TRAINING DATA CREATED")
print("==============================================")

print("\nNumber of samples :", len(training_df))

print("\nFirst 10 rows:")
print(training_df.head(10))

print("\nLabel distribution:")
print(training_df["Label"].value_counts())

# ============================================
# Save
# ============================================

training_df.to_csv(
    os.path.join(
        dataset_path,
        "training_data.csv"
    ),
    index=False
)

print("\nTraining data saved successfully!")