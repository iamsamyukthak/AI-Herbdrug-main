# Current Architecture Inspection

**Status:** Phase 0–10 inspection and implementation record. The production GCN remains unchanged; the heterogeneous graph, feature pipeline, proposed HGNN, candidate ranking, baseline evaluation, and ablation training are separate additions.

## Existing Architecture

The runnable Streamlit application remains a three-class GCN predictor with evidence display and an optional, separate biological-analysis/docking path. The proposed GAT → GraphSAGE → SAGPool → MLP architecture is implemented separately in `model/hgnn.py` and is not yet wired into the Streamlit app.

### Production GCN

`main.py` defines and loads `HerbDrugInteractionModel`; `frontend/app.py` imports its prediction and DDID lookup functions. The architecture used by this loader is:

1. Two `GCNConv` layers: node features 2 → 32 → 32, with ReLU and dropout after the first layer.
2. Gather the herb and drug graph embeddings for each requested pair and add separate learned herb/drug ID embeddings (32 dimensions each).
3. Concatenate herb, drug, elementwise product, and absolute difference (128 dimensions).
4. MLP: 128 → 64 → 32 → 3, with ReLU and dropout between hidden layers.

The loaded checkpoint is `model/herbdrug_complete_model.pth`. It contains the model state, node features, edge index, node-count fields, and stored validation/test metrics. The current source preserves the GCN checkpoint architecture; it is the production baseline to retain.

### Other Existing Models

- `model/herbdrug_model.py` contains another implementation of the production-style GCN classifier.
- `model/train_gnn.py` defines a separate GCN over a five-feature, typed-node biomedical graph and writes `datasets/herb_drug_gnn.pth`.
- `model/hgnn_model.py` is named HGNN, but its layers are both `GCNConv` (5 → 16 → 8). It has no attention, GraphSAGE, or pooling layer.
- `model/hgnn.py` is the separate proposed heterogeneous model: relation-aware GAT → heterogeneous GraphSAGE → SAGPooling → MLP → sigmoid pair score.
- `model/interaction_classifier.py` is an MLP classifier over concatenated 8-dimensional herb/drug embeddings.
- `model/herbdrug_hgnn.pth` belongs to the older two-GCN `model/hgnn_model.py` and is not a compatible checkpoint for `model/hgnn.py`.

The `model/train_model.py` path uses random synthetic targets rather than interaction labels, and its configured 5-feature GCN input does not match the two-feature tensor exported by `model/prepare_graph.py`. The embedding/classifier scripts are exploratory rather than a verified training pipeline. They should not be treated as completed comparative baselines.

## Existing Data Flow

### CSV data and graph construction

The biomedical tables are separate CSV files:

- `datasets/herbs.csv`: herb, scientific name, active compound (10 rows).
- `datasets/drugs.csv`: drug, class, target protein (10 rows).
- `datasets/proteins.csv`: protein, gene, pathway (10 rows).
- `datasets/compound_protein.csv`: compound–protein pairs (11 rows).
- `datasets/interactions.csv`: small herb–drug interaction/risk table (10 rows).
- `datasets/ddid_processed.csv`: DDID herb–drug pairs, evidence fields, and integer `Label` (1,614 rows; 658 herbs, 108 drugs).
- `datasets/ddid_herb_drug_training.csv`: DDID training rows with risk/evidence fields (1,614 rows).
- `datasets/external_bio_evidence.csv`: herb, drug, compound, target, effect, PMID records (19,689 rows).
- `datasets/herbs_ddid_edges.csv`, `datasets/drugs_ddid_edges.csv`, and `datasets/training_data.csv`: derived edge/training tables.

`knowledge_graph/graph_builder.py` is the public PyTorch Geometric heterogeneous graph builder. It creates distinct Herb, Compound, Drug, and Protein node stores and typed directed edge stores. The graph can be built with `build_heterogeneous_graph()` or summarized with `python -m knowledge_graph.graph_builder`; the normalized source/provenance loader remains in `model/heterogeneous_graph.py` for compatibility.

The built graph has typed relations Herb→Compound (`contains`), Compound→Protein (`associated_with`), Drug→Protein (`targets`), and Herb→Drug (`interacts_with`). Direct Drug→Protein assertions come only from the explicit `Target_Protein` column. External evidence is retained with its row provenance but is not used to infer a drug target from co-occurrence.

### Node Features (Phase 4)

`knowledge_graph/node_features.py` attaches a float32 `x` matrix to each node store and records `feature_names` and numeric normalization statistics on the returned `HeteroData` object. Numerical count columns are transformed with `log1p` and then z-score normalized per node type; categorical indicators remain 0/1 multi-hot values.

| Node type | Dimensions | Feature columns |
| --- | ---: | --- |
| Herb | 12 | 2 normalized counts (unique compound degree, external-evidence rows) + 10 scientific-name indicators |
| Compound | 4 | 3 normalized counts (herb degree, protein degree, external-evidence rows) + exact ligand-file availability indicator |
| Drug | 10 | 2 normalized counts (explicit target degree, external-evidence rows) + 8 drug-class indicators |
| Protein | 20 | 2 normalized counts (compound degree, explicit drug-target degree) + 10 gene and 8 pathway indicators |

Categorical vocabularies come from the current CSV metadata, so nodes without matching metadata have zero values in those categorical columns. DDID risk labels are not node features, and DDID Herb–Drug edge degree is not used as a feature. No RDKit package or SMILES/descriptor table is present; the current docking directory contains exact SDF/PDBQT files only for a small subset of compounds. Accordingly, the compound feature indicates exact structure-file availability and does not fabricate molecular descriptors. Drug molecular descriptors are unavailable.

## Proposed HGNN (Phase 5)

`model/hgnn.py` defines `HerbDrugHGNN`. Construct it with `HerbDrugHGNN.from_graph(graph.data)`, then call it with the same `HeteroData` plus herb and drug node-ID tensors. Its GraphSAGE node embeddings are fused with the type-specific SAGPool context to form learned herb/drug embeddings `z_h` and `z_d`. The score head applies an MLP to `concat(z_h, z_d)` followed by sigmoid and returns one continuous score per pair.

## SAGPool Projection (Phase 6)

PyG `SAGPooling` accepts one node-feature matrix, one `edge_index`, and one batch vector; it does not accept a `HeteroData` node dictionary or relation-specific edge stores. The model therefore applies heterogeneous GAT and GraphSAGE before pooling, then makes a temporary homogeneous topology solely for SAGPool. This is a documented pooling-stage projection, not a claim that the complete model is heterogeneous at every stage.

The conversion is implemented by `project_heterogeneous_graph()` in `model/hgnn.py` and is reproducible: nodes concatenate in `NODE_TYPES` order (`Herb`, `Compound`, `Drug`, `Protein`), preserving each store's local node-ID order. The returned `SAGPoolProjection` contains node-type IDs, local IDs, names, and type offsets. Edges concatenate in `RELATION_TYPES` order; each relation's forward edges are followed by their reverse edges to form an undirected pooling topology. Every projected edge retains its relation ID and direction ID. These identifiers map to `RELATION_TYPES` and direction IDs `0` (forward) / `1` (reverse).

The conversion creates new tensors and does not modify the source `HeteroData`. After selection, the score uses one pooled mean per original node type, keeping type summaries distinct rather than using one undifferentiated pooled mean. To inspect the actual selected nodes and induced edges with their retained relation/direction IDs, call `model(graph, herb_ids, drug_ids, return_pooling_details=True)`; the default call still returns only the score tensor. At the pooling layer itself, edge-type labels are not consumed by SAGPool's internal GraphConv; their identities remain available as projection metadata, and no relation-aware convolution follows pooling.

The scalar-output configuration (`num_outputs=1`) still has no compatible trained checkpoint. Newly initialized scalar weights do not constitute a trained, calibrated, or scientifically meaningful interaction prediction. Phase 10 produced separate three-class ablation checkpoints; they cannot be loaded into the scalar candidate-ranking head. The legacy `model/herbdrug_hgnn.pth` is also incompatible because it belongs to a different GCN implementation.

## Interaction Prediction (Phase 7)

`HerbDrugHGNN.predict_interaction(graph, herb, drug, thresholds=...)` resolves names using the graph's normalized entity mappings and returns exactly `herb`, `drug`, `prediction_score`, and `prediction_label`. The continuous sigmoid score is the primary output. `PredictionThresholds` defaults to 0.33 and 0.66: scores below 0.33 are LOW, scores from 0.33 up to but not including 0.66 are MEDIUM, and scores at or above 0.66 are HIGH. Callers can supply different cutoffs; these labels are configurable presentation bins, not scientifically validated thresholds.

## Candidate Ranking (Phase 8)

`model/candidate_ranking.py` exposes `rank_candidates(model, graph, candidates, top_k=10, exclude_observed=True)`. It normalizes names, removes duplicate pairs, generates scores in one batched HGNN inference pass, sorts descending by continuous score, and returns `rank`, `herb`, `drug`, and `hgnn_score`. By default it excludes pairs already present in the graph's Herb–Drug relation; unknown entity names raise an error rather than being guessed.

The Streamlit **Potential Interaction Candidates** panel ranks the cross product of entities in the graph, with Top K configurable from 1 to 100. It displays `Rank | Herb | Drug | HGNN Score` and explicitly warns that results are not confirmed interactions. Its scalar-output HGNN remains randomly initialized because the trained Phase 10 checkpoints use three-class output; those ranking scores are exploratory only, not meaningful or validated predictions.

## Candidate Docking (Phase 11)

The candidate panel lets users select ranked pairs and invokes `model/docking_workflow.py`. For each pair, the workflow follows only observed Herb→Compound, Compound→Protein, and explicit Drug→Protein graph edges, then calls the existing `get_ligand`, `get_receptor`, `get_docking_box`, and `run_docking` functions in `model/unseen_predictor.py`. The Vina execution implementation is preserved.

Each compound/protein path returns `compound`, `protein`, `docking_score`, `binding_pose`, and `docking_status`, alongside the candidate identifiers and HGNN ranking score. A score is included only when Vina returns a finite parsed affinity; a pose is included only when its output file exists. Missing graph paths, unsupported proteins, absent ligand/receptor files, unavailable Vina, or failed/unparseable Vina output produce `docking_status="unavailable"`, null score/pose where not actually produced, and a reason. The existing Vina configuration currently supports CYP3A4/CYP3A5 boxes; not every sourced target has a box or receptor.

The workflow was exercised on an unobserved pair: sourced Quercetin→CYP3A4→Busulfan target mapping ran the existing Vina executable and returned an observed -7.022 kcal/mol affinity and pose file. Other unsupported paths for the same candidate were reported unavailable rather than assigned scores. This smoke result validates execution only; it is not clinical evidence. Because the scalar HGNN ranking checkpoint is still untrained, selected candidates are exploratory, and their rankings should not be interpreted as validated prioritization.

## Ablation Study (Phase 10)

`model/run_ablation_study.py` trains the following variants using the same three-class DDID target and a shared heterogeneous graph/features: GAT only with a linear readout; GAT + GraphSAGE with a linear readout; GAT + GraphSAGE + SAGPool with a linear readout; and the full GAT + GraphSAGE + SAGPool + MLP classifier.

The runner reuses the split indices from `model/prepare_graph.py`, seed 42, weighted CrossEntropyLoss, AdamW (`lr=0.001`, `weight_decay=0.0001`), gradient clipping at 2, validation macro-F1 checkpoint selection, 500-epoch maximum, and 60-epoch patience. The Herb–Drug graph relation is restricted to training pairs. Because NFKC/case-folded graph IDs merge a few differently-capitalized DDID rows across partitions, 6 training and 1 validation rows were excluded from those earlier partitions to prevent normalized-pair leakage; all 162 test rows remain. Effective split sizes are 1,285/160/162.

Run `python model/run_ablation_study.py` to re-evaluate compatible checkpoints and train missing variants. Add `--force-train` to retrain all four from scratch. Checkpoints are saved to `checkpoints/ablation_gat.pt`, `checkpoints/ablation_gat_graphsage.pt`, `checkpoints/ablation_gat_graphsage_sagpool.pt`, and `checkpoints/ablation_full_hgnn.pt`; measured test metrics are written to `results/ablation_results.csv`.

Measured held-out results (162 test rows):

| Experiment | Accuracy | Macro precision | Macro recall | Macro F1 | Macro OVR ROC-AUC | Best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GAT only | 0.7901 | 0.7733 | 0.8077 | 0.7828 | 0.9320 | 367 |
| GAT + GraphSAGE | 0.8333 | 0.8159 | 0.8485 | 0.8270 | 0.9486 | 252 |
| GAT + GraphSAGE + SAGPool | 0.8457 | 0.8276 | 0.8584 | 0.8377 | 0.9551 | 291 |
| GAT + GraphSAGE + SAGPool + MLP | 0.8642 | 0.8445 | 0.8757 | 0.8547 | 0.9710 | 365 |

`model/train_gnn.py` separately rebuilds a homogeneous NetworkX graph from those biomedical tables, encodes five node types as one-hot node features, converts the graph to `x`/`edge_index`, and trains a GCN on DDID labels. This path does not create a PyG `HeteroData` graph or relation-specific message-passing layers.

The production pipeline uses a different, DDID-only graph. `model/prepare_graph.py` builds sorted herb and drug node IDs from `ddid_processed.csv`, assigns two-channel one-hot Herb/Drug features, and creates edges only from the training partition's DDID pairs (each pair is added in both directions). Compound, protein, pathway, and external-evidence nodes/edges are not part of this checkpoint graph.

### Prediction and evidence flow

1. Streamlit offers herb/drug names from the processed DDID table.
2. `main.py` case-insensitively maps names to IDs, runs the loaded GCN, and converts its three logits to softmax probabilities.
3. The UI displays the predicted LOW/MEDIUM/HIGH class and class probabilities, and independently looks up an exact herb–drug row from `ddid_processed.csv`.
4. `model/biological_features.py` loads `external_bio_evidence.csv` and serves herb compounds/proteins, drug targets, and exact-pair evidence. These features are displayed after inference; they are not input to the production GCN.
5. If requested, `model/unseen_predictor.py` applies rule-based evidence logic and may select a compound/target for docking. The current docking selection uses exact CYP3A4/CYP3A5/CYP3A evidence when present, otherwise a supported common-target fallback. This path is independent of the GCN class score; there is no score-ranked candidate-selection stage.

## Existing Model Inputs and Outputs

### Production checkpoint facts

Runtime inspection of `model/herbdrug_complete_model.pth` confirmed:

- `node_features`: float tensor of shape `(766, 2)`; 658 Herb nodes and 108 Drug nodes, represented by `(1, 0)` and `(0, 1)` respectively.
- `edge_index`: long tensor of shape `(2, 2582)`, representing 1,291 training-pair edges in both directions.
- Each prediction call supplies one or more herb IDs and drug IDs plus these graph tensors.
- The checkpoint fields report `num_herbs = 658` and `num_drugs = 766`. The latter is the total graph-node count used to size the drug embedding table, not the number of distinct drugs; drug IDs are offset by 658.
- `forward(...)` returns unnormalized logits of shape `(batch_size, 3)`, ordered as LOW, MEDIUM, HIGH. The caller applies softmax and returns `risk`, predicted-class `confidence`, and each class probability as percentages.

The checkpoint stores `best_val_f1`, `test_accuracy`, and `test_macro_f1`; these are saved metadata, not independently validated by this inspection. `model/evaluate_model.py` currently evaluates predictions over all rows in `ddid_processed.csv`, rather than using a held-out split in that evaluation script.

The other graph training path in `model/train_gnn.py` uses five-channel one-hot node features, a homogeneous `edge_index`, herb/drug ID batches, and outputs three class logits. Its GCN layers are 5 → 32 → 16, and its pair classifier concatenates the two 16-dimensional embeddings before an MLP.

## DDID Processing and Training/Evaluation Files

- `model/load_labels.py` maps Low/Medium/High values in `interactions.csv` to integer labels for display/output; it does not prepare the production checkpoint.
- `model/prepare_graph.py` loads `ddid_processed.csv`, maps labels 0/1/2, and prepares the production-style two-channel DDID graph and stratified train/validation/test indices.
- `model/train_end_to_end.py` trains `model/herbdrug_model.py`, selects by validation macro-F1, reports held-out metrics, and writes the checkpoint consumed by `main.py`.
- `model/evaluate_model.py` loads that checkpoint and prints classification/confidence metrics.
- `model/predict.py` is a terminal inference path for the same checkpoint.
- `model/prepare_training_data.py` rebuilds a biomedical NetworkX graph and maps DDID training names to graph IDs; missing entities are represented as `-1`.
- `model/build_training_data.py` converts `ddid_processed.csv` to ID/label rows using IDs exported by `prepare_graph.py`.
- `model/train_gnn.py` and `model/train_end_to_end.py` are distinct GCN training routes with different graph construction and input features.
- `model/train_model.py`, `model/train_classifier.py`, and `model/train_interaction_classifier.py` comprise an experimental embedding/classifier route; this route is not wired into the app.

## Biological Evidence and Docking

`model/biological_features.py` reads `datasets/external_bio_evidence.csv` at module import and builds lookup tables. Pair evidence is matched by lowercase herb/drug names; these data are not merged into the production GCN node features or edges.

The functional docking implementation is in `model/unseen_predictor.py`; `model/docking.py` is empty. Docking invokes the bundled Windows executable `vina_1.2.7_win.exe` through `subprocess`, uses receptor files under `docking/receptors/`, ligand files under `docking/ligands/`, and writes outputs under `docking/results/`. Supported receptor paths are configured for CYP3A4 and CYP3A5. A compatible evidence-selected ligand/receptor pair is required, so docking is not guaranteed for every input pair. The docking result is supporting computational evidence, not a clinical interaction label.

## Frontend and Entry Points

- `frontend/app.py` is the Streamlit UI. It preserves pair selection, GCN inference, DDID lookup, biological evidence display, optional biological analysis/docking, and research/medical disclaimers.
- `frontend/app.py` also exposes Top-K candidate selection and candidate docking results.
- `main.py` is both the production GCN loader and a terminal prompt loop. Its `if __name__ == "__main__"` flow also calls the biological/docking analysis for each pair.
- `backend/main.py` loads and prints the basic herb, drug, protein, and interaction CSV tables; it is not the Streamlit backend/API.
- `knowledge_graph/graph_builder.py` exposes the typed graph builder and summary CLI.

## Repository Inventory

### Application and graph code

- `main.py`, `frontend/app.py`, `backend/main.py`
- `knowledge_graph/__init__.py`, `knowledge_graph/graph_builder.py`, `knowledge_graph/node_features.py`
- `model/biological_features.py`, `model/unseen_predictor.py`, `model/docking.py`, `model/docking_report.py`

### Model and data preparation code

- `model/herbdrug_model.py`, `model/hgnn_model.py`, `model/interaction_classifier.py`
- `model/hgnn.py` — proposed heterogeneous pair-scoring model
- `model/candidate_ranking.py` — batched continuous-score ranking with observed-pair exclusion
- `model/docking_workflow.py` — source-mapped candidate docking orchestration using the existing Vina runner
- `model/prepare_graph.py`, `model/prepare_training_data.py`, `model/build_training_data.py`, `model/load_labels.py`
- `model/train_end_to_end.py`, `model/train_gnn.py`, `model/train_model.py`, `model/train_classifier.py`, `model/train_interaction_classifier.py`
- `model/evaluate_model.py`, `model/predict.py`

### Existing model artifacts

- `model/herbdrug_complete_model.pth` — production GCN checkpoint
- `datasets/herb_drug_gnn.pth` — separate five-feature GCN checkpoint
- `model/herbdrug_hgnn.pth`, `model/interaction_classifier.pth`, `model/node_embeddings.pt` — experimental embedding/classifier artifacts
- `datasets/ddid_training_tensors.pt` — stored herb IDs, drug IDs, and labels

### Data and docking assets

- `datasets/` contains the CSV tables listed above.
- `docking/` contains Vina configuration files, ligand structures, receptor structures, and generated-result location.
- `vina_1.2.7_win.exe` is the bundled docking executable.

## Dependencies

No `requirements.txt`, `pyproject.toml`, setup file, or Conda environment file was present during inspection. Imports across the Python source require:

- Application/inference: NumPy, PyTorch, PyTorch Geometric, pandas, Streamlit.
- Graph/training/evaluation scripts: NetworkX and scikit-learn, in addition to NumPy, PyTorch/PyTorch Geometric, and pandas.
- Docking: the bundled Windows Vina executable and Python standard-library subprocess/file utilities.

The README contains the current local setup/run commands; dependency versions are not pinned by a project manifest.

## Remaining Work

- Add a standalone GraphSAGE baseline checkpoint for the Phase 9 model comparison; the trained `GAT + GraphSAGE` ablation is not a GraphSAGE-only baseline.
- Train and validate a scalar-output HGNN checkpoint compatible with the Top-K candidate-ranking API.
- Add a Streamlit model selector/comparison view and connect validated HGNN candidate ranking to the existing docking workflow.
- Pin dependency versions after settling compatible Python, PyTorch, PyTorch Geometric, and Matplotlib versions.

## What Remains Unchanged

During the planned upgrade, retain the existing GCN implementation and checkpoint as the baseline; preserve the current Streamlit pair-analysis workflow, DDID and external-evidence CSVs, the user-visible LOW/MEDIUM/HIGH categories, current biological evidence lookups, and the existing Vina executable/ligand/receptor assets. New graph/model artifacts should be separate from the baseline artifacts. No existing files or artifacts were deleted or overwritten during this inspection.

## Remaining Research Decisions

- Define whether the proposed score remains the current three-class output or becomes a separate scalar/ranking objective, and how the two outputs relate.
- Decide whether candidate ranking will use the scalar score or a separately trained class-probability head, and calibrate the presentation thresholds with held-out data.
- Decide candidate-selection and negative-pair protocols for external validation so future comparisons remain leak-free.
- Define how predictions and candidate rankings choose compounds/proteins for docking, while retaining the current evidence-based fallback.