import pandas as pd
import os

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)

dataset_path = os.path.join(project_root, "datasets")

interactions = pd.read_csv(
    os.path.join(dataset_path, "interactions.csv")
)

risk_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

interactions["Risk_Label"] = interactions["Risk"].map(risk_mapping)

print(interactions)