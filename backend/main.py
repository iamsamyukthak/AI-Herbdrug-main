import pandas as pd
import os
# Get the project root directory
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)

# Dataset folder path
dataset_path = os.path.join(project_root, "datasets")

# Read CSV files
herbs = pd.read_csv(os.path.join(dataset_path, "herbs.csv"))
drugs = pd.read_csv(os.path.join(dataset_path, "drugs.csv"))
proteins = pd.read_csv(os.path.join(dataset_path, "proteins.csv"))
interactions = pd.read_csv(
    os.path.join(dataset_path, "interactions.csv")
)

# Display datasets
print("\n========== HERBS DATA ==========\n")
print(herbs)

print("\n========== DRUGS DATA ==========\n")
print(drugs)

print("\n========== PROTEINS DATA ==========\n")
print(proteins)
print("\n========== INTERACTION DATA ==========\n")
print(interactions)
print("\n✅ All datasets loaded successfully!")