import os
import subprocess
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "datasets")
DOCKING_DIR = os.path.join(BASE_DIR, "docking")
LIGAND_DIR = os.path.join(DOCKING_DIR, "ligands")
RECEPTOR_DIR = os.path.join(DOCKING_DIR, "receptors")
RESULT_DIR = os.path.join(DOCKING_DIR, "results")

VINA_EXE = os.path.join(BASE_DIR, "vina_1.2.7_win.exe")

DDID_FILE = os.path.join(DATA_DIR, "ddid_processed.csv")


# ============================================================
# MODEL
# ============================================================

class HerbDrugInteractionModel(nn.Module):

    def __init__(self, num_herbs, num_drugs, embedding_dim=32):

        super().__init__()

        self.herb_embedding = nn.Embedding(
            num_herbs,
            embedding_dim
        )

        self.drug_embedding = nn.Embedding(
            num_drugs,
            embedding_dim
        )

        self.conv1 = GCNConv(2, 32)

        self.conv2 = GCNConv(
            32,
            embedding_dim
        )

        self.fc1 = nn.Linear(
            embedding_dim * 4,
            64
        )

        self.fc2 = nn.Linear(
            64,
            32
        )

        self.fc3 = nn.Linear(
            32,
            3
        )

        self.dropout = nn.Dropout(0.25)

    def get_node_embeddings(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x

    def forward(
        self,
        x,
        edge_index,
        herb_ids,
        drug_ids
    ):

        graph_embeddings = self.get_node_embeddings(
            x,
            edge_index
        )

        graph_herb = graph_embeddings[herb_ids]

        graph_drug = graph_embeddings[drug_ids]

        learned_herb = self.herb_embedding(
            herb_ids
        )

        learned_drug = self.drug_embedding(
            drug_ids
        )

        herb = graph_herb + learned_herb

        drug = graph_drug + learned_drug

        product = herb * drug

        difference = torch.abs(
            herb - drug
        )

        combined = torch.cat(
            [
                herb,
                drug,
                product,
                difference
            ],
            dim=1
        )

        combined = self.fc1(
            combined
        )

        combined = F.relu(
            combined
        )

        combined = self.dropout(
            combined
        )

        combined = self.fc2(
            combined
        )

        combined = F.relu(
            combined
        )

        return self.fc3(
            combined
        )


# ============================================================
# FILE HELPERS
# ============================================================

def find_file(folder, names):

    if isinstance(names, str):
        names = [names]

    files = {}

    if os.path.exists(folder):

        for f in os.listdir(folder):

            files[f.lower()] = os.path.join(
                folder,
                f
            )

    for name in names:

        path = files.get(
            name.lower()
        )

        if path and os.path.exists(path):
            return path

    return None


# ============================================================
# DDID DATA
# ============================================================

def load_ddid():

    if not os.path.exists(DDID_FILE):
        return None

    return pd.read_csv(
        DDID_FILE
    )


def get_ddid_record(
    herb,
    drug
):

    df = load_ddid()

    if df is None:
        return None

    matches = df[
        (df["Herb"].astype(str).str.lower() == herb.lower())
        &
        (df["Drug"].astype(str).str.lower() == drug.lower())
    ]

    if len(matches) == 0:
        return None

    return matches.iloc[0].to_dict()


# ============================================================
# BIOLOGICAL EVIDENCE
# ============================================================

def load_biological_evidence():

    bio_file = os.path.join(
        DATA_DIR,
        "external_bio_evidence.csv"
    )

    if not os.path.exists(bio_file):
        return None

    return pd.read_csv(
        bio_file
    )


def get_biological_evidence(
    herb,
    drug
):

    df = load_biological_evidence()

    if df is None:
        return []

    matches = df[
        (df["Herb"].astype(str).str.lower() == herb.lower())
        &
        (df["Drug"].astype(str).str.lower() == drug.lower())
    ]

    evidence = []

    for _, row in matches.iterrows():

        evidence.append(
            {
                "compound": str(
                    row.get(
                        "Compound",
                        ""
                    )
                ),

                "target": str(
                    row.get(
                        "Target",
                        ""
                    )
                ),

                "effect": str(
                    row.get(
                        "Effect",
                        ""
                    )
                ),

                "pmid": str(
                    row.get(
                        "PMID",
                        ""
                    )
                )
            }
        )

    return evidence


def get_common_targets(
    herb,
    drug
):

    evidence = get_biological_evidence(
        herb,
        drug
    )

    targets = set()

    for item in evidence:

        target = item["target"].strip()

        if target:
            targets.add(
                target.upper()
            )

    return sorted(targets)


# ============================================================
# RISK
# ============================================================

def determine_risk(
    evidence,
    common_targets,
    ddid_record
):

    # DDID exact pair has highest priority
    if ddid_record is not None:

        label = str(
            ddid_record.get(
                "Risk",
                ""
            )
        ).upper()

        if label in [
            "LOW",
            "MEDIUM",
            "HIGH"
        ]:

            return (
                label,
                "DDID recorded risk for this herb-drug pair."
            )

    effects = [
        str(
            item.get(
                "effect",
                ""
            )
        ).lower()
        for item in evidence
    ]

    if any(
        effect in [
            "negative",
            "harmful"
        ]
        for effect in effects
    ):

        return (
            "HIGH",
            "External biological evidence contains negative/harmful interaction evidence."
        )

    if effects and all(
        effect == "no effect"
        for effect in effects
    ):

        return (
            "LOW",
            "External evidence reports no effect for the available evidence."
        )

    if any(
        effect == "possible"
        for effect in effects
    ):

        return (
            "MEDIUM",
            "External evidence reports a possible interaction."
        )

    if effects and all(
        effect == "positive"
        for effect in effects
    ):

        return (
            "LOW",
            "External evidence reports positive effects."
        )

    if any(
        target in [
            "CYP3A4",
            "CYP3A5"
        ]
        for target in common_targets
    ):

        return (
            "MEDIUM",
            "Common CYP3A4/CYP3A5 biological target detected."
        )

    return (
        "LOW",
        "No high-risk biological evidence detected."
    )


# ============================================================
# DOCKING CONFIGURATION
# ============================================================

def get_docking_box(target):

    target = target.upper()

    if target == "CYP3A4":

        return {
            "center_x": -17.114,
            "center_y": -24.427,
            "center_z": -11.868,
            "size_x": 30,
            "size_y": 30,
            "size_z": 30
        }

    if target == "CYP3A5":

        return {
            "center_x": -9.17428,
            "center_y": -48.73258,
            "center_z": 23.56622,
            "size_x": 30,
            "size_y": 30,
            "size_z": 30
        }

    return None


def get_receptor(target):

    target = target.upper()

    if target == "CYP3A4":

        return find_file(
            RECEPTOR_DIR,
            [
                "CYP3A4_final.pdbqt"
            ]
        )

    if target == "CYP3A5":

        return find_file(
            RECEPTOR_DIR,
            [
                "CYP3A5.pdbqt"
            ]
        )

    return None


# ============================================================
# CURCUMIN LIGAND
# ============================================================

def get_ligand(compound):

    compound = compound.strip().lower()

    # Curcumin now uses the verified PubChem 3D structure
    if compound == "curcumin":

        return find_file(
            LIGAND_DIR,
            [
                "curcumin_pubchem.pdbqt"
            ]
        )

    # Generic fallback
    filename = compound.replace(
        " ",
        "_"
    ) + ".pdbqt"

    return find_file(
        LIGAND_DIR,
        [
            filename
        ]
    )


# ============================================================
# DOCKING
# ============================================================

def run_docking(
    ligand,
    receptor,
    target
):

    if ligand is None:
        return None

    if receptor is None:
        return None

    if not os.path.exists(VINA_EXE):
        return None

    box = get_docking_box(
        target
    )

    if box is None:
        return None

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    output_name = (
        "dock_"
        + os.path.basename(ligand)
        + "_"
        + target
        + ".pdbqt"
    )

    output_file = os.path.join(
        RESULT_DIR,
        output_name
    )

    command = [
        VINA_EXE,

        "--receptor",
        receptor,

        "--ligand",
        ligand,

        "--center_x",
        str(box["center_x"]),

        "--center_y",
        str(box["center_y"]),

        "--center_z",
        str(box["center_z"]),

        "--size_x",
        str(box["size_x"]),

        "--size_y",
        str(box["size_y"]),

        "--size_z",
        str(box["size_z"]),

        "--exhaustiveness",
        "8",

        "--num_modes",
        "9",

        "--out",
        output_file
    ]

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=BASE_DIR
        )

    except Exception as e:

        print(
            f"Docking execution error: {e}"
        )

        return None

    print(
        result.stdout
    )

    if result.stderr:

        print(
            result.stderr
        )

    if result.returncode != 0:
        return None

    if not os.path.exists(output_file):
        return None

    # Parse first affinity from Vina output
    affinity = None

    lines = result.stdout.splitlines()

    for line in lines:

        parts = line.split()

        if len(parts) >= 2:

            try:

                mode = int(
                    parts[0]
                )

                if mode == 1:

                    affinity = float(
                        parts[1]
                    )

                    break

            except ValueError:
                continue

    return {
        "affinity": affinity,
        "output": output_file,
        "target": target,
        "ligand": ligand,
        "receptor": receptor
    }


# ============================================================
# DOCKING SELECTION
# ============================================================

def select_docking_pair(
    evidence,
    common_targets
):

    # Exact CYP3A4/CYP3A5 evidence
    for item in evidence:

        target = item["target"].upper()

        if target in [
            "CYP3A4",
            "CYP3A5"
        ]:

            return (
                item["compound"],
                target
            )

    # Generic CYP3A evidence -> use CYP3A4
    for item in evidence:

        target = item["target"].upper()

        if target == "CYP3A":

            return (
                item["compound"],
                "CYP3A4"
            )

    # Common target fallback
    for target in common_targets:

        target_upper = target.upper()

        if target_upper in [
            "CYP3A4",
            "CYP3A5"
        ]:

            return (
                "Curcumin",
                target_upper
            )

        if target_upper == "CYP3A":

            return (
                "Curcumin",
                "CYP3A4"
            )

    return (
        None,
        None
    )


# ============================================================
# UNSEEN PREDICTION
# ============================================================

def predict_unseen(
    herb,
    drug
):

    print()
    print("=" * 60)
    print("BIOLOGICAL + DOCKING ANALYSIS")
    print("=" * 60)

    evidence = get_biological_evidence(
        herb,
        drug
    )

    common_targets = get_common_targets(
        herb,
        drug
    )

    ddid_record = get_ddid_record(
        herb,
        drug
    )

    # --------------------------------------------------------
    # DDID
    # --------------------------------------------------------

    if ddid_record is not None:

        print()
        print("DDID RECORD:")

        print(
            "Risk:",
            ddid_record.get(
                "Risk"
            )
        )

        print(
            "Effects:",
            ddid_record.get(
                "Effects"
            )
        )

        print(
            "PMID:",
            ddid_record.get(
                "PMIDs"
            )
        )

    else:

        print()
        print(
            "DDID RECORD: Not available"
        )

    # --------------------------------------------------------
    # Biological evidence
    # --------------------------------------------------------

    print()
    print("BIOLOGICAL EVIDENCE:")

    if evidence:

        for item in evidence:

            print(
                f"Compound: {item['compound']} | "
                f"Target: {item['target']} | "
                f"Effect: {item['effect']} | "
                f"PMID: {item['pmid']}"
            )

    else:

        print(
            "No exact external biological evidence found."
        )

    print()
    print(
        "COMMON TARGETS:",
        ", ".join(common_targets)
        if common_targets
        else "None"
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    risk, reason = determine_risk(
        evidence,
        common_targets,
        ddid_record
    )

    print()
    print(
        "BIOLOGICAL PREDICTION:",
        risk
    )

    print(
        "Reason:",
        reason
    )

    # --------------------------------------------------------
    # Docking
    # --------------------------------------------------------

    compound, target = select_docking_pair(
        evidence,
        common_targets
    )

    print()
    print(
        "DOCKING TARGET:",
        target if target else "None"
    )

    print(
        "DOCKING COMPOUND:",
        compound if compound else "None"
    )

    docking_result = None

    if compound and target:

        ligand = get_ligand(
            compound
        )

        receptor = get_receptor(
            target
        )

        print()
        print(
            "Ligand:",
            ligand
            if ligand
            else "Not found"
        )

        print(
            "Receptor:",
            receptor
            if receptor
            else "Not found"
        )

        if ligand and receptor:

            print()
            print(
                "Running AutoDock Vina..."
            )

            docking_result = run_docking(
                ligand,
                receptor,
                target
            )

    print()
    print("=" * 60)
    print("FINAL BIOLOGICAL RESULT")
    print("=" * 60)

    print(
        "Risk:",
        risk
    )

    print(
        "Reason:",
        reason
    )

    if docking_result is not None:

        print(
            "Docking affinity:",
            docking_result["affinity"],
            "kcal/mol"
            if docking_result["affinity"] is not None
            else "Not available"
        )

        print(
            "Docking target:",
            docking_result["target"]
        )

    else:

        print(
            "Docking affinity: Not available"
        )

    print("=" * 60)

    return {
        "risk": risk,
        "reason": reason,
        "ddid": ddid_record,
        "biological_evidence": evidence,
        "common_targets": common_targets,
        "docking": docking_result
    }