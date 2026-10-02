"""Run existing AutoDock Vina on evidence-mapped ranked candidates."""

from __future__ import annotations

import contextlib
import io
import math
import os
from collections.abc import Iterable
from typing import Any

from model.heterogeneous_graph import HeterogeneousGraph, normalize_entity_name
from model.research_results import append_docking_results
from model.unseen_predictor import (
    VINA_EXE,
    get_docking_box,
    get_ligand,
    get_receptor,
    run_docking,
)


HERB_COMPOUND = ("Herb", "contains", "Compound")
COMPOUND_PROTEIN = ("Compound", "associated_with", "Protein")
DRUG_PROTEIN = ("Drug", "targets", "Protein")


def _unavailable_result(
    candidate: dict[str, Any],
    reason: str,
    compound: str | None = None,
    protein: str | None = None,
) -> dict[str, Any]:
    return {
        "rank": candidate.get("rank"),
        "herb": candidate.get("herb"),
        "drug": candidate.get("drug"),
        "hgnn_score": candidate.get("hgnn_score"),
        "compound": compound,
        "protein": protein,
        "docking_score": None,
        "binding_pose": None,
        "docking_status": "unavailable",
        "reason": reason,
    }


def _relation_neighbors(
    graph: HeterogeneousGraph,
    edge_type: tuple[str, str, str],
    source_id: int,
) -> set[int]:
    edge_index = graph.data[edge_type].edge_index
    return set(edge_index[1, edge_index[0] == source_id].tolist())


def _candidate_paths(
    graph: HeterogeneousGraph,
    herb: str,
    drug: str,
) -> tuple[list[tuple[str, str]], str | None]:
    herb_key = normalize_entity_name(herb)
    drug_key = normalize_entity_name(drug)
    herb_id = graph.node_to_id["Herb"].get(herb_key)
    drug_id = graph.node_to_id["Drug"].get(drug_key)
    if herb_id is None:
        return [], f"Herb {herb!r} is not present in the heterogeneous graph."
    if drug_id is None:
        return [], f"Drug {drug!r} is not present in the heterogeneous graph."

    compounds = _relation_neighbors(graph, HERB_COMPOUND, herb_id)
    if not compounds:
        return [], "No observed Herb-to-Compound mapping exists for this herb."

    drug_targets = _relation_neighbors(graph, DRUG_PROTEIN, drug_id)
    if not drug_targets:
        return [], "No explicit Drug-to-Protein target mapping exists for this drug."

    paths: set[tuple[str, str]] = set()
    has_compound_protein_mapping = False
    has_intersecting_target = False
    for compound_id in compounds:
        compound_proteins = _relation_neighbors(
            graph,
            COMPOUND_PROTEIN,
            compound_id,
        )
        if compound_proteins:
            has_compound_protein_mapping = True
        for protein_id in compound_proteins & drug_targets:
            has_intersecting_target = True
            paths.add(
                (
                    graph.node_names["Compound"][compound_id],
                    graph.node_names["Protein"][protein_id],
                )
            )

    if paths:
        return sorted(paths, key=lambda path: (path[0].casefold(), path[1].casefold())), None
    if not has_compound_protein_mapping:
        return [], "No observed Compound-to-Protein mapping exists for these herb compounds."
    if not has_intersecting_target:
        return [], (
            "No observed Compound-to-Protein target is shared with this drug's "
            "explicit Drug-to-Protein targets."
        )
    return [], "No complete sourced Herb-to-Compound-to-Protein-from-Drug path was available."


def dock_ranked_candidates(
    graph: HeterogeneousGraph,
    candidates: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Dock sourced compound/protein paths for ranked pairs using existing Vina.

    No affinity or pose is synthesized. A result is marked unavailable with a
    reason whenever the evidence path or required Vina assets are absent.
    """
    results = []
    for candidate in candidates:
        paths, path_error = _candidate_paths(
            graph,
            str(candidate.get("herb", "")),
            str(candidate.get("drug", "")),
        )
        if not paths:
            results.append(_unavailable_result(candidate, path_error or "No sourced target path."))
            continue

        for compound, protein in paths:
            docking_box = get_docking_box(protein)
            if docking_box is None:
                results.append(
                    _unavailable_result(
                        candidate,
                        f"No docking box is configured for protein {protein!r}.",
                        compound,
                        protein,
                    )
                )
                continue

            ligand = get_ligand(compound)
            if ligand is None:
                expected_ligand = compound.strip().lower().replace(" ", "_") + ".pdbqt"
                results.append(
                    _unavailable_result(
                        candidate,
                        f"No exact ligand PDBQT file is available for {compound!r} "
                        f"(expected {expected_ligand}).",
                        compound,
                        protein,
                    )
                )
                continue

            receptor = get_receptor(protein)
            if receptor is None:
                results.append(
                    _unavailable_result(
                        candidate,
                        f"No receptor PDBQT file is available for {protein!r}.",
                        compound,
                        protein,
                    )
                )
                continue

            if not os.path.isfile(VINA_EXE):
                results.append(
                    _unavailable_result(
                        candidate,
                        f"AutoDock Vina executable is unavailable at {VINA_EXE}.",
                        compound,
                        protein,
                    )
                )
                continue

            console_output = io.StringIO()
            try:
                with contextlib.redirect_stdout(console_output):
                    docking_result = run_docking(ligand, receptor, protein)
            except Exception as error:
                results.append(
                    _unavailable_result(
                        candidate,
                        f"AutoDock Vina execution failed: {type(error).__name__}: {error}",
                        compound,
                        protein,
                    )
                )
                continue

            if not docking_result:
                message = console_output.getvalue().strip()
                reason = "Vina failed, returned a non-zero exit code, or produced no pose file."
                if message:
                    reason += f" Vina output: {message[-500:]}"
                results.append(
                    _unavailable_result(candidate, reason, compound, protein)
                )
                continue

            pose_path = docking_result.get("output")
            affinity = docking_result.get("affinity")
            if not pose_path or not os.path.isfile(pose_path):
                results.append(
                    _unavailable_result(
                        candidate,
                        "Vina did not leave a readable binding-pose file.",
                        compound,
                        protein,
                    )
                )
                continue
            if not isinstance(affinity, (int, float)) or not math.isfinite(affinity):
                results.append(
                    {
                        **_unavailable_result(
                            candidate,
                            "Vina produced a pose file but no parseable finite "
                            "affinity score.",
                            compound,
                            protein,
                        ),
                        "binding_pose": pose_path,
                    }
                )
                continue

            results.append(
                {
                    "rank": candidate.get("rank"),
                    "herb": candidate.get("herb"),
                    "drug": candidate.get("drug"),
                    "hgnn_score": candidate.get("hgnn_score"),
                    "compound": compound,
                    "protein": protein,
                    "docking_score": float(affinity),
                    "binding_pose": pose_path,
                    "docking_status": "completed",
                    "reason": "",
                }
            )

    append_docking_results(results)
    return results