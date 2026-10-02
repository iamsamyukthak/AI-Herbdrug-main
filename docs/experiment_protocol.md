# Experiment Protocol

## Scope and traceability

This protocol describes the repository's measured GCN baseline and Phase 10 HGNN ablations. Results are written to `results/baseline_comparison.csv`, `results/ablation_results.csv`, and held-out pair predictions to `results/predictions.csv`. Each ablation checkpoint under `checkpoints/` stores model configuration, random seed, train/validation/test row indices, best epoch, validation macro-F1, per-epoch training/validation loss histories, test labels/predictions/probabilities, and test metrics.

The repository includes `datasets/ddid_processed.csv` (1,614 labeled herb-drug rows with LOW/MEDIUM/HIGH labels). No upstream dataset URL, release identifier, or publication citation is recorded in the repository; the CSV is therefore identified here by its repository path, not by an inferred external citation. The existing Vina binary is `vina_1.2.7_win.exe` (named 1.2.7 in the binary filename).

## Data preprocessing

- Entity strings are normalized using Unicode NFKC, leading/trailing whitespace trimming, and case-folding for heterogeneous graph ID matching. This is exact normalized string matching, not fuzzy alias resolution.
- Node features come from the repository CSV metadata and graph relationships: 12 Herb, 4 Compound, 10 Drug, and 20 Protein features. Count features use `log1p` followed by per-node-type z-score normalization; categorical indicators remain binary.
- The model graph uses Herb, Compound, Drug, and Protein node stores with typed Herb-Compound, Compound-Protein, Drug-Protein, and Herb-Drug relations.
- For ablation supervision, `model/prepare_graph.py` provides the existing risk labels and deterministic stratified row split. The ablation runner replaces the graph's Herb-Drug message-passing edges with training rows only; validation and test Herb-Drug edges are not exposed to message passing.
- The existing row split is 80% train, 10% validation, and 10% test using `train_test_split(..., random_state=42, stratify=labels)`. Because case-folding merges some differently capitalized duplicate pairs, the runner excludes earlier-split rows whose normalized pair occurs in a later split: 6 train and 1 validation rows are removed. The held-out test set remains 162 rows; effective counts are 1,285 train, 160 validation, and 162 test.
- No synthetic negatives or other negative sampling are performed. All supervised rows and labels come from `ddid_processed.csv`; the protocol does not assign labels to unobserved pairs.

## Models

The ablation classifier reuses `model/hgnn.py` with three output logits for LOW/MEDIUM/HIGH and weighted CrossEntropyLoss:

1. GAT only: relation-specific heterogeneous GAT, followed by a linear three-class readout.
2. GAT + GraphSAGE: adds relation-specific heterogeneous GraphSAGE, followed by the same linear readout.
3. GAT + GraphSAGE + SAGPool: adds the documented deterministic SAGPool projection, followed by the same linear readout.
4. Full HGNN: adds the MLP readout to GAT + GraphSAGE + SAGPool.

The default scalar HGNN (`num_outputs=1`, sigmoid) is a separate inference mode and is not the classifier trained in these ablations. The existing production GCN and its checkpoint are unchanged.

## Training configuration

- Random seed: 42 for Python, NumPy, and PyTorch, reset before each variant.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 0.0001.
- Loss: weighted CrossEntropyLoss; class weights are calculated from training split label counts using `N_train / (3 * class_count)`.
- Epoch limit: 500.
- Early stopping: patience of 60 epochs on validation macro-F1; the checkpoint stores the best validation-F1 epoch.
- Gradient clipping: maximum norm 2.0.
- Dropout: 0.2 in the HGNN attention/readout paths.
- Attention heads: 2.
- SAGPool ratio: 0.5.
- Batch size: full-batch graph training. Each optimizer step processes all 1,285 effective labeled training pairs against the heterogeneous graph; there is no mini-batch sampler.

Run `python model/run_ablation_study.py` to re-evaluate compatible checkpoints and train missing or history-incomplete variants. Use `--force-train` to retrain all four from scratch. Checkpoints are `checkpoints/ablation_gat.pt`, `checkpoints/ablation_gat_graphsage.pt`, `checkpoints/ablation_gat_graphsage_sagpool.pt`, and `checkpoints/ablation_full_hgnn.pt`.

## Evaluation

All variants use the same 162-row held-out test split. Reported metrics are accuracy, macro precision, macro recall, macro F1, and macro one-vs-rest ROC-AUC using the three class probabilities from softmax. ROC-AUC is computed from actual test labels and model probabilities; no values are filled for an unevaluated model.

`model/evaluate_baselines.py` evaluates the production GCN checkpoint and re-evaluates the compatible Experiment 1 GAT-only and Experiment 4 full-HGNN checkpoints on the same test rows. Standalone GraphSAGE is still `Not evaluated yet`; Experiment 2 includes GAT and is not a GraphSAGE-only baseline. Current measured rows are:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Macro OVR ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| GCN | 0.8025 | 0.7858 | 0.8072 | 0.7946 | 0.9039 |
| GAT | 0.7901 | 0.7733 | 0.8077 | 0.7828 | 0.9329 |
| GraphSAGE | Not evaluated yet | Not evaluated yet | Not evaluated yet | Not evaluated yet | Not evaluated yet |
| Proposed HGNN | 0.8642 | 0.8506 | 0.8774 | 0.8559 | 0.9728 |

`results/predictions.csv` contains the held-out pair predictions/probabilities for the three evaluated models (162 rows per model, 486 rows total). The scalar sigmoid output used by candidate ranking is not the three-class ablation classifier and remains untrained.

## Docking configuration

Docking preserves `model/unseen_predictor.py` and uses the evidence-mapped `model/docking_workflow.py` orchestration. Only graph-supported Herb→Compound, Compound→Protein, and explicit Drug→Protein paths are considered. No missing molecular feature, relationship, docking score, or pose is inferred.

- Executable: repository-root `vina_1.2.7_win.exe`.
- Receptors: `docking/receptors/CYP3A4_final.pdbqt` and `docking/receptors/CYP3A5.pdbqt`.
- Ligands: exact existing PDBQT files under `docking/ligands/`; Curcumin uses `curcumin_pubchem.pdbqt`; other compounds require a matching file.
- CYP3A4 box: center `(-17.114, -24.427, -11.868)`, size `(30, 30, 30)` Angstrom.
- CYP3A5 box: center `(-9.17428, -48.73258, 23.56622)`, size `(30, 30, 30)` Angstrom.
- Exhaustiveness: 8. Requested output modes: 9.
- Score: first Vina mode affinity parsed from actual standard output, in kcal/mol. Binding pose is the actual Vina output PDBQT path.
- Unavailable runs store status and a reason; score and pose remain blank/null unless Vina actually returns them.

## Output artifacts

- `results/predictions.csv`: actual held-out GCN, GAT, and Proposed HGNN rows, labels, predictions, and class probabilities.
- `results/baseline_comparison.csv`: measured GCN/GAT/Proposed HGNN metrics and explicit standalone GraphSAGE not-evaluated status.
- `results/ablation_results.csv`: measured ablation metrics, checkpoint, and training notes.
- `results/docking_results.csv`: actual candidate docking attempts and unavailable reasons. The current log has eight sourced paths: one completed Vina result and seven unavailable paths.
- `results/plots/training_loss.png`, `validation_loss.png`, `roc_curve.png`, `confusion_matrix.png`, `baseline_comparison.png`, and `ablation_comparison.png`: generated from saved predictions, training histories, and measured metric CSV rows. Missing measurements are left blank rather than plotted as substituted values.

A model probability is computational evidence, a Vina affinity is structure-based computational evidence, and a DDID row is database evidence. None establishes clinical safety, toxicity, dose, timing, pharmacokinetic/pharmacodynamic effects, or patient outcomes.
