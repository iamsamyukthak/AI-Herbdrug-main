"""Rank unobserved herb-drug candidates using the proposed HGNN score."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from model.heterogeneous_graph import HeterogeneousGraph, normalize_entity_name
from model.hgnn import HerbDrugHGNN


HERB_DRUG_RELATION = ("Herb", "interacts_with", "Drug")


def rank_candidates(
    model: HerbDrugHGNN,
    graph: HeterogeneousGraph,
    candidates: Iterable[tuple[str, str]],
    top_k: int = 10,
    exclude_observed: bool = True,
) -> list[dict[str, str | float | int]]:
    """Return the highest-scoring herb-drug pairs.

    Scores are computed in one batched HGNN pass. By default, pairs already
    present in the graph's Herb-Drug relation are excluded to avoid presenting
    known graph edges as novel candidates. This function does not establish
    that any returned pair is a confirmed interaction.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    herb_to_id = graph.node_to_id["Herb"]
    drug_to_id = graph.node_to_id["Drug"]
    observed_pairs: set[tuple[int, int]] = set()
    if exclude_observed:
        edge_index = graph.data[HERB_DRUG_RELATION].edge_index
        observed_pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    unique_pairs: list[tuple[int, int, str, str]] = []
    seen: set[tuple[int, int]] = set()
    for candidate in candidates:
        try:
            herb, drug = candidate
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "each candidate must be a (herb_name, drug_name) pair"
            ) from exc

        herb_key = normalize_entity_name(herb)
        drug_key = normalize_entity_name(drug)
        if herb_key not in herb_to_id or drug_key not in drug_to_id:
            raise ValueError(f"Unknown herb-drug candidate: {candidate!r}")

        herb_id = herb_to_id[herb_key]
        drug_id = drug_to_id[drug_key]
        pair_ids = (herb_id, drug_id)
        if pair_ids in seen or pair_ids in observed_pairs:
            continue

        seen.add(pair_ids)
        unique_pairs.append(
            (
                herb_id,
                drug_id,
                graph.node_names["Herb"][herb_id],
                graph.node_names["Drug"][drug_id],
            )
        )

    if not unique_pairs:
        return []

    device = next(model.parameters()).device
    herb_ids = torch.tensor(
        [pair[0] for pair in unique_pairs],
        dtype=torch.long,
        device=device,
    )
    drug_ids = torch.tensor(
        [pair[1] for pair in unique_pairs],
        dtype=torch.long,
        device=device,
    )

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            scores = model.predict_score(graph.data, herb_ids, drug_ids)
    finally:
        model.train(was_training)

    order = torch.argsort(scores, descending=True, stable=True)[:top_k].tolist()
    return [
        {
            "rank": rank,
            "herb": unique_pairs[index][2],
            "drug": unique_pairs[index][3],
            "hgnn_score": float(scores[index].item()),
        }
        for rank, index in enumerate(order, start=1)
    ]