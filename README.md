# AI Herb-Drug Interaction Prediction System

A research prototype for exploring herb-drug pairs using a graph convolutional network (GCN), DDID records, biological evidence, and optional molecular docking.

## Requirements

- Windows, macOS, or Linux
- Python 3.13 (used for the current project setup)
- The model checkpoint and CSV data files included in this repository

## Setup

From the repository root, create and activate a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

For Command Prompt, activate with `.venv\Scripts\activate.bat` instead. Install the application dependencies:

```console
python -m pip install --upgrade pip
python -m pip install numpy torch pandas streamlit torch-geometric
```

Some model-training and knowledge-graph scripts also use these packages:

```console
python -m pip install matplotlib networkx scikit-learn
```

If you use a CUDA-enabled GPU, install the PyTorch build appropriate for your CUDA version before installing `torch-geometric`. The application also runs with CPU-only PyTorch.

## Run the application

From the repository root, with the environment activated:

```console
python -m streamlit run frontend/app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. Select an herb and a drug, then choose **Analyse interaction**. The page presents the model's LOW/MEDIUM/HIGH output, any exact DDID record, and available biological evidence. Enable the optional biological analysis and docking checkbox to request that additional pipeline.

The **Potential Interaction Candidates** panel ranks unobserved Herb–Drug pairs and lets you choose Top K (1–100). Its scalar scoring head has no compatible trained checkpoint, so those rankings are exploratory only; they are not confirmed interactions or scientifically validated predictions. The trained ablation checkpoints are three-class classifiers and are not loaded by that scalar ranking panel.

The frontend also provides separate Model Evaluation, Ablation Study, and Knowledge Graph Explorer sections. Evaluation views use saved measured CSV results; the explorer displays observed graph relations and source provenance.

There is also a terminal-based interface:

```console
python main.py
```

Run the GCN held-out baseline evaluation (test-set probabilities and classification metrics):

```console
python model/evaluate_gcn_baseline.py
```

Run the unified baseline comparison and generate metric plots:

```console
python model/evaluate_baselines.py
```

Results are written to `results/baseline_comparison.csv` with comparison plots in `results/`. Models without trained, compatible checkpoints are marked **Not evaluated yet**.

Train missing ablation variants and evaluate all compatible checkpoints with:

```console
python model/run_ablation_study.py
```
Research artifacts are stored under `results/`: `predictions.csv`, `baseline_comparison.csv`, `ablation_results.csv`, and `docking_results.csv`, with plots under `results/plots/`. Run the baseline evaluator and ablation runner to refresh their measured outputs. Docking rows are appended only when candidate docking is actually attempted; unavailable rows have blank score/pose fields and an explanation.

See [docs/experiment_protocol.md](docs/experiment_protocol.md) for dataset provenance, preprocessing, feature schema, split/leakage handling, model settings, metrics, and docking configuration. Unknown external dataset citations, negative sampling, or unperformed experiments are explicitly not inferred.

The script re-evaluates compatible checkpoints, trains variants without checkpoints, and writes measured results to `results/ablation_results.csv`. Use `python model/run_ablation_study.py --force-train` to train all four variants from scratch.

## Project layout

- `frontend/app.py` — Streamlit user interface
- `main.py` — GCN inference, DDID lookup, and terminal interface
- `model/` — GCN/HGNN definitions, candidate ranking, training/evaluation scripts, checkpoints, and biological/docking prediction logic
- `datasets/` — herb, drug, protein, interaction, training, DDID, and external biological-evidence data
- `knowledge_graph/` — typed PyTorch Geometric heterogeneous graph builder
- `docking/` — ligand and receptor structures plus Vina configuration files
- `backend/main.py` — prints the core herb, drug, protein, and interaction tables

Inspect the normalized heterogeneous graph from the repository root:

```console
python -m knowledge_graph.graph_builder
```

The Streamlit app expects `model/herbdrug_complete_model.pth`, `datasets/ddid_processed.csv`, and `datasets/external_bio_evidence.csv` at their repository-relative paths. Optional docking depends on compatible ligand/receptor files and the bundled `vina_1.2.7_win.exe`; a docking result is not available for every pair.

## Important

This is a research prototype, not a clinical decision tool. Model probabilities are not clinically calibrated, and a LOW prediction or missing evidence does not establish that a combination is safe. Consult a qualified healthcare professional about medication decisions.