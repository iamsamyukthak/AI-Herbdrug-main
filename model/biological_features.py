import os
import pandas as pd
from collections import defaultdict

BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets"
)

BIO_FILE = os.path.join(BASE, "external_bio_evidence.csv")

bio = pd.read_csv(BIO_FILE)

# -----------------------------
# Herb → compounds
# -----------------------------
herb_to_compounds = defaultdict(set)

for _, r in bio.iterrows():
    herb_to_compounds[str(r["Herb"]).strip()].add(
        str(r["Compound"]).strip()
    )

# -----------------------------
# Herb → compounds → targets
# -----------------------------
herb_to_targets = defaultdict(set)

for _, r in bio.iterrows():
    herb_to_targets[str(r["Herb"]).strip()].add(
        str(r["Target"]).strip()
    )

# -----------------------------
# Drug → targets
# -----------------------------
drug_to_targets = defaultdict(set)

for _, r in bio.iterrows():
    drug_to_targets[str(r["Drug"]).strip()].add(
        str(r["Target"]).strip()
    )

# -----------------------------
# Exact biological evidence
# -----------------------------
pair_evidence = defaultdict(list)

for _, r in bio.iterrows():

    key = (
        str(r["Herb"]).strip().lower(),
        str(r["Drug"]).strip().lower()
    )

    pair_evidence[key].append({
        "compound": str(r["Compound"]).strip(),
        "target": str(r["Target"]).strip(),
        "effect": str(r["Effect"]).strip(),
        "pmid": str(r["PMID"]).strip()
    })


def find_name(mapping, name):

    for actual in mapping:

        if actual.lower() == name.lower():
            return actual

    return None


def get_herb_features(herb):

    actual = find_name(herb_to_compounds, herb)

    if actual is None:
        return {
            "herb": herb,
            "compounds": [],
            "proteins": [],
            "pathways": []
        }

    return {
        "herb": actual,
        "compounds": sorted(herb_to_compounds[actual]),
        "proteins": sorted(herb_to_targets[actual]),
        "pathways": []
    }


def get_drug_features(drug):

    actual = find_name(drug_to_targets, drug)

    if actual is None:
        return {
            "drug": drug,
            "targets": [],
            "pathways": []
        }

    return {
        "drug": actual,
        "targets": sorted(drug_to_targets[actual]),
        "pathways": []
    }


def get_pair_evidence(herb, drug):

    key = (herb.lower(), drug.lower())

    return pair_evidence.get(key, [])


if __name__ == "__main__":

    print(get_herb_features("Acacia nilotica"))
    print(get_drug_features("Midazolam"))

    print("\nExact evidence:")
    print(get_pair_evidence("Acacia nilotica", "Midazolam"))