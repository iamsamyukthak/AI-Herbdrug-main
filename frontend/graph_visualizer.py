"""Interactive Knowledge Graph visualizer using Vis.js for Streamlit."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from model.heterogeneous_graph import HeterogeneousGraph, normalize_entity_name


NODE_COLORS = {
    "Herb": {
        "background": "#ecfdf5",
        "border": "#059669",
        "highlight": {"background": "#a7f3d0", "border": "#047857"},
        "color": "#065f46",
    },
    "Compound": {
        "background": "#ecfeff",
        "border": "#0891b2",
        "highlight": {"background": "#a5f3fc", "border": "#0e7490"},
        "color": "#155e75",
    },
    "Protein": {
        "background": "#f5f3ff",
        "border": "#7c3aed",
        "highlight": {"background": "#ddd6fe", "border": "#6d28d9"},
        "color": "#5b21b6",
    },
    "Drug": {
        "background": "#fffbeb",
        "border": "#d97706",
        "highlight": {"background": "#fde68a", "border": "#b45309"},
        "color": "#92400e",
    },
}

NODE_ICONS = {
    "Herb": "🌿",
    "Compound": "🧪",
    "Protein": "🧬",
    "Drug": "💊",
}


def get_entity_details(
    graph: HeterogeneousGraph,
    entity_type: str,
    entity_name: str,
) -> dict[str, Any] | None:
    """Extract actual connected nodes and direct relations for a selected entity."""
    norm_name = normalize_entity_name(entity_name)
    if not norm_name or norm_name not in graph.node_to_id.get(entity_type, {}):
        return None

    node_id = graph.node_to_id[entity_type][norm_name]

    connected_herbs = set()
    connected_compounds = set()
    connected_proteins = set()
    connected_drugs = set()
    direct_relations = []

    for edge_type, df in graph.edge_records.items():
        src_t, rel, tgt_t = edge_type
        if src_t == entity_type:
            matches = df[df["source_id"] == node_id]
            for _, r in matches.iterrows():
                target_id = int(r["target_id"])
                target_name = graph.node_names[tgt_t][target_id]
                if tgt_t == "Herb":
                    connected_herbs.add(target_name)
                elif tgt_t == "Compound":
                    connected_compounds.add(target_name)
                elif tgt_t == "Protein":
                    connected_proteins.add(target_name)
                elif tgt_t == "Drug":
                    connected_drugs.add(target_name)

                direct_relations.append({
                    "relation": rel,
                    "direction": "outgoing",
                    "related_type": tgt_t,
                    "related_entity": target_name,
                    "source_file": r["source_file"],
                    "source_row": int(r["source_row"]),
                })
        elif tgt_t == entity_type:
            matches = df[df["target_id"] == node_id]
            for _, r in matches.iterrows():
                source_id = int(r["source_id"])
                source_name = graph.node_names[src_t][source_id]
                if src_t == "Herb":
                    connected_herbs.add(source_name)
                elif src_t == "Compound":
                    connected_compounds.add(source_name)
                elif src_t == "Protein":
                    connected_proteins.add(source_name)
                elif src_t == "Drug":
                    connected_drugs.add(source_name)

                direct_relations.append({
                    "relation": rel,
                    "direction": "incoming",
                    "related_type": src_t,
                    "related_entity": source_name,
                    "source_file": r["source_file"],
                    "source_row": int(r["source_row"]),
                })

    return {
        "name": entity_name,
        "type": entity_type,
        "total_connections": len(connected_herbs) + len(connected_compounds) + len(connected_proteins) + len(connected_drugs),
        "herbs": sorted(connected_herbs),
        "compounds": sorted(connected_compounds),
        "proteins": sorted(connected_proteins),
        "drugs": sorted(connected_drugs),
        "relations": direct_relations,
    }


def find_multihop_pathways(
    graph: HeterogeneousGraph,
    entity_type: str,
    entity_name: str,
    max_paths: int = 50,
) -> list[dict[str, str]]:
    """Extract complete multi-hop biological pathways (Herb -> Compound -> Protein <- Drug)."""
    norm_name = normalize_entity_name(entity_name)
    if not norm_name or norm_name not in graph.node_to_id.get(entity_type, {}):
        return []

    node_id = graph.node_to_id[entity_type][norm_name]

    edge_stores = {
        "herb_compound": graph.data[("Herb", "contains", "Compound")].edge_index,
        "compound_protein": graph.data[("Compound", "associated_with", "Protein")].edge_index,
        "drug_protein": graph.data[("Drug", "targets", "Protein")].edge_index,
    }
    herb_compounds, compound_proteins, drug_proteins = edge_stores.values()
    hc, cp, dp = {}, {}, {}
    for h_id, c_id in zip(herb_compounds[0].tolist(), herb_compounds[1].tolist()):
        hc.setdefault(h_id, set()).add(c_id)
    for c_id, p_id in zip(compound_proteins[0].tolist(), compound_proteins[1].tolist()):
        cp.setdefault(c_id, set()).add(p_id)
    for d_id, p_id in zip(drug_proteins[0].tolist(), drug_proteins[1].tolist()):
        dp.setdefault(p_id, set()).add(d_id)

    paths = []
    for h_id, compounds in hc.items():
        if entity_type == "Herb" and h_id != node_id:
            continue
        for c_id in compounds:
            if entity_type == "Compound" and c_id != node_id:
                continue
            for p_id in cp.get(c_id, set()):
                if entity_type == "Protein" and p_id != node_id:
                    continue
                for d_id in dp.get(p_id, set()):
                    if entity_type == "Drug" and d_id != node_id:
                        continue
                    paths.append({
                        "Herb": graph.node_names["Herb"][h_id],
                        "Compound": graph.node_names["Compound"][c_id],
                        "Protein": graph.node_names["Protein"][p_id],
                        "Drug": graph.node_names["Drug"][d_id],
                    })
                    if len(paths) >= max_paths:
                        return paths
    return paths


def build_entity_subgraph(
    graph: HeterogeneousGraph,
    entity_type: str,
    entity_name: str,
    include_pathways: bool = True,
    max_neighbors_per_type: int = 15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct nodes and edges for an interactive subgraph centered at the chosen entity."""
    norm_name = normalize_entity_name(entity_name)
    if not norm_name or norm_name not in graph.node_to_id.get(entity_type, {}):
        return [], []

    center_node_key = f"{entity_type}:{entity_name}"
    nodes_dict: dict[str, dict[str, Any]] = {}
    edges_list: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    # Add the central node
    icon = NODE_ICONS.get(entity_type, "⚪")
    colors = NODE_COLORS.get(entity_type, {})
    nodes_dict[center_node_key] = {
        "id": center_node_key,
        "label": f"{icon} {entity_name}",
        "group": entity_type,
        "title": f"<b>{entity_type}</b>: {entity_name} (Central Focus)",
        "value": 35,
        "font": {"size": 15, "bold": True, "color": "#0f172a"},
        "color": {
            "background": colors.get("highlight", {}).get("background", "#ffffff"),
            "border": colors.get("highlight", {}).get("border", "#000000"),
            "highlight": {"background": "#fef08a", "border": "#ca8a04"},
        },
        "borderWidth": 3,
        "shadow": True,
    }

    # If requested and multi-hop paths exist, include them
    if include_pathways:
        paths = find_multihop_pathways(graph, entity_type, entity_name, max_paths=15)
        for p in paths:
            h_key = f"Herb:{p['Herb']}"
            c_key = f"Compound:{p['Compound']}"
            pr_key = f"Protein:{p['Protein']}"
            d_key = f"Drug:{p['Drug']}"

            # Ensure nodes exist
            for key, t, name in [
                (h_key, "Herb", p["Herb"]),
                (c_key, "Compound", p["Compound"]),
                (pr_key, "Protein", p["Protein"]),
                (d_key, "Drug", p["Drug"]),
            ]:
                if key not in nodes_dict:
                    t_colors = NODE_COLORS.get(t, {})
                    nodes_dict[key] = {
                        "id": key,
                        "label": f"{NODE_ICONS.get(t, '')} {name}",
                        "group": t,
                        "title": f"<b>{t}</b>: {name}",
                        "value": 18,
                        "font": {"size": 12, "color": "#1e293b"},
                        "color": {
                            "background": t_colors.get("background", "#ffffff"),
                            "border": t_colors.get("border", "#94a3b8"),
                        },
                        "borderWidth": 1.5,
                    }

            # Edges
            for from_id, to_id, rel in [
                (h_key, c_key, "contains"),
                (c_key, pr_key, "associated_with"),
                (d_key, pr_key, "targets"),
            ]:
                edge_sig = (from_id, to_id, rel)
                if edge_sig not in seen_edges:
                    seen_edges.add(edge_sig)
                    edges_list.append({
                        "from": from_id,
                        "to": to_id,
                        "label": rel,
                        "arrows": "to",
                        "font": {"size": 10, "color": "#64748b", "align": "middle"},
                        "color": {"color": "#94a3b8", "highlight": "#0f766e"},
                        "width": 1.5,
                        "smooth": {"type": "curvedCW", "roundness": 0.15},
                    })

    # Add direct 1-hop connections
    details = get_entity_details(graph, entity_type, entity_name)
    if details:
        for rel in details["relations"][:60]:
            rel_name = rel["relation"]
            rel_type = rel["related_type"]
            rel_ent = rel["related_entity"]
            nbr_key = f"{rel_type}:{rel_ent}"

            if nbr_key not in nodes_dict:
                t_colors = NODE_COLORS.get(rel_type, {})
                nodes_dict[nbr_key] = {
                    "id": nbr_key,
                    "label": f"{NODE_ICONS.get(rel_type, '')} {rel_ent}",
                    "group": rel_type,
                    "title": f"<b>{rel_type}</b>: {rel_ent}",
                    "value": 16,
                    "font": {"size": 12, "color": "#1e293b"},
                    "color": {
                        "background": t_colors.get("background", "#ffffff"),
                        "border": t_colors.get("border", "#94a3b8"),
                    },
                    "borderWidth": 1.5,
                }

            if rel["direction"] == "outgoing":
                from_k, to_k = center_node_key, nbr_key
            else:
                from_k, to_k = nbr_key, center_node_key

            edge_sig = (from_k, to_k, rel_name)
            if edge_sig not in seen_edges:
                seen_edges.add(edge_sig)
                edges_list.append({
                    "from": from_k,
                    "to": to_k,
                    "label": rel_name,
                    "arrows": "to",
                    "font": {"size": 10, "color": "#64748b", "align": "middle"},
                    "color": {"color": "#94a3b8", "highlight": "#0f766e"},
                    "width": 1.5,
                    "smooth": {"type": "continuous"},
                })

    return list(nodes_dict.values()), edges_list


def build_filtered_graph(
    graph: HeterogeneousGraph,
    selected_entity_types: list[str],
    selected_relations: list[str],
    max_nodes: int = 70,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct an overview graph filtered by entity and relation types."""
    nodes_dict: dict[str, dict[str, Any]] = {}
    edges_list: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    # Prioritize multi-hop complete pathways for maximum scientific demonstration value
    edge_stores = {
        "herb_compound": graph.data[("Herb", "contains", "Compound")].edge_index,
        "compound_protein": graph.data[("Compound", "associated_with", "Protein")].edge_index,
        "drug_protein": graph.data[("Drug", "targets", "Protein")].edge_index,
    }
    herb_compounds, compound_proteins, drug_proteins = edge_stores.values()
    hc, cp, dp = {}, {}, {}
    for h_id, c_id in zip(herb_compounds[0].tolist(), herb_compounds[1].tolist()):
        hc.setdefault(h_id, set()).add(c_id)
    for c_id, p_id in zip(compound_proteins[0].tolist(), compound_proteins[1].tolist()):
        cp.setdefault(c_id, set()).add(p_id)
    for d_id, p_id in zip(drug_proteins[0].tolist(), drug_proteins[1].tolist()):
        dp.setdefault(p_id, set()).add(d_id)

    # Collect connected pathway nodes
    for h_id, compounds in hc.items():
        if len(nodes_dict) >= max_nodes:
            break
        for c_id in compounds:
            for p_id in cp.get(c_id, set()):
                for d_id in dp.get(p_id, set()):
                    h_name = graph.node_names["Herb"][h_id]
                    c_name = graph.node_names["Compound"][c_id]
                    pr_name = graph.node_names["Protein"][p_id]
                    d_name = graph.node_names["Drug"][d_id]

                    entities_to_add = []
                    if "Herb" in selected_entity_types:
                        entities_to_add.append(("Herb", h_name, f"Herb:{h_name}"))
                    if "Compound" in selected_entity_types:
                        entities_to_add.append(("Compound", c_name, f"Compound:{c_name}"))
                    if "Protein" in selected_entity_types:
                        entities_to_add.append(("Protein", pr_name, f"Protein:{pr_name}"))
                    if "Drug" in selected_entity_types:
                        entities_to_add.append(("Drug", d_name, f"Drug:{d_name}"))

                    for t, name, key in entities_to_add:
                        if key not in nodes_dict and len(nodes_dict) < max_nodes:
                            t_colors = NODE_COLORS.get(t, {})
                            nodes_dict[key] = {
                                "id": key,
                                "label": f"{NODE_ICONS.get(t, '')} {name}",
                                "group": t,
                                "title": f"<b>{t}</b>: {name}",
                                "value": 20,
                                "font": {"size": 12, "color": "#0f172a"},
                                "color": {
                                    "background": t_colors.get("background", "#ffffff"),
                                    "border": t_colors.get("border", "#94a3b8"),
                                },
                                "borderWidth": 1.5,
                            }

                    # Add relations if both endpoints are active
                    hk = f"Herb:{h_name}"
                    ck = f"Compound:{c_name}"
                    pk = f"Protein:{pr_name}"
                    dk = f"Drug:{d_name}"

                    if "contains" in selected_relations and hk in nodes_dict and ck in nodes_dict:
                        sig = (hk, ck, "contains")
                        if sig not in seen_edges:
                            seen_edges.add(sig)
                            edges_list.append({
                                "from": hk,
                                "to": ck,
                                "label": "contains",
                                "arrows": "to",
                                "font": {"size": 10, "color": "#64748b", "align": "middle"},
                                "color": {"color": "#94a3b8"},
                            })

                    if "associated_with" in selected_relations and ck in nodes_dict and pk in nodes_dict:
                        sig = (ck, pk, "associated_with")
                        if sig not in seen_edges:
                            seen_edges.add(sig)
                            edges_list.append({
                                "from": ck,
                                "to": pk,
                                "label": "associated_with",
                                "arrows": "to",
                                "font": {"size": 10, "color": "#64748b", "align": "middle"},
                                "color": {"color": "#94a3b8"},
                            })

                    if "targets" in selected_relations and dk in nodes_dict and pk in nodes_dict:
                        sig = (dk, pk, "targets")
                        if sig not in seen_edges:
                            seen_edges.add(sig)
                            edges_list.append({
                                "from": dk,
                                "to": pk,
                                "label": "targets",
                                "arrows": "to",
                                "font": {"size": 10, "color": "#64748b", "align": "middle"},
                                "color": {"color": "#94a3b8"},
                            })

                    if len(nodes_dict) >= max_nodes:
                        break
                if len(nodes_dict) >= max_nodes:
                    break
            if len(nodes_dict) >= max_nodes:
                break

    return list(nodes_dict.values()), edges_list


def build_vis_network_html(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    height: int = 620,
) -> str:
    """Generate self-contained interactive Vis.js graph visualization HTML."""
    payload = {
        "nodes": nodes,
        "edges": edges,
    }
    encoded_json = json.dumps(payload).replace("</", "<\\/")

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
  <link rel="stylesheet" href="https://unpkg.com/vis-network@9.1.9/dist/dist/vis-network.min.css" />
  <style>
    :root {{
      color-scheme: dark;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}
    body {{
      margin: 0;
      background: #0f172a;
      color: #e2e8f0;
      overflow: hidden;
      user-select: none;
    }}
    .network-container {{
      border: 1px solid #334155;
      border-radius: 12px;
      overflow: hidden;
      background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
      position: relative;
    }}
    #mynetwork {{
      width: 100%;
      height: {height}px;
    }}
    .graph-hud {{
      position: absolute;
      top: 12px;
      left: 14px;
      display: flex;
      gap: 8px;
      pointer-events: none;
      z-index: 10;
    }}
    .hud-badge {{
      background: rgba(15, 23, 42, 0.88);
      backdrop-filter: blur(6px);
      border: 1px solid rgba(51, 65, 85, 0.8);
      border-radius: 20px;
      padding: 4px 12px;
      font-size: 0.78rem;
      font-weight: 600;
      color: #f8fafc;
      box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }}
    .graph-controls {{
      position: absolute;
      bottom: 12px;
      right: 14px;
      display: flex;
      gap: 6px;
      z-index: 10;
    }}
    .graph-btn {{
      background: rgba(30, 41, 59, 0.9);
      border: 1px solid #475569;
      border-radius: 6px;
      color: #e2e8f0;
      padding: 5px 10px;
      font-size: 0.76rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .graph-btn:hover {{
      background: #0f766e;
      color: #ffffff;
      border-color: #0f766e;
    }}
    .selected-node-banner {{
      position: absolute;
      bottom: 12px;
      left: 14px;
      background: rgba(15, 23, 42, 0.92);
      border: 1px solid #0f766e;
      border-radius: 8px;
      padding: 6px 12px;
      font-size: 0.8rem;
      color: #e2e8f0;
      pointer-events: none;
      z-index: 10;
      display: none;
    }}
    .hint-label {{
      position: absolute;
      top: 12px;
      right: 14px;
      font-size: 0.72rem;
      color: #94a3b8;
      pointer-events: none;
    }}
  </style>
</head>
<body>
  <div class="network-container">
    <div class="graph-hud">
      <div class="hud-badge">Nodes: <b>{len(nodes)}</b> &bull; Edges: <b>{len(edges)}</b></div>
    </div>
    <div class="hint-label">Click node to highlight &bull; Drag to reposition &bull; Scroll to zoom</div>

    <div id="mynetwork"></div>

    <div id="selection-banner" class="selected-node-banner"></div>

    <div class="graph-controls">
      <button class="graph-btn" id="btn-fit" title="Fit graph to view">🔍 Fit View</button>
      <button class="graph-btn" id="btn-stabilize" title="Stabilize layout physics">⚡ Stabilize</button>
    </div>
  </div>

  <script id="graph-data" type="application/json">{encoded_json}</script>

  <script>
    (function() {{
      const rawData = JSON.parse(document.getElementById('graph-data').textContent);
      const container = document.getElementById('mynetwork');
      const selectionBanner = document.getElementById('selection-banner');

      const data = {{
        nodes: new vis.DataSet(rawData.nodes),
        edges: new vis.DataSet(rawData.edges)
      }};

      const options = {{
        nodes: {{
          shape: 'box',
          margin: 10,
          borderRadius: 8,
          shadow: {{
            enabled: true,
            color: 'rgba(0, 0, 0, 0.25)',
            size: 6,
            x: 2,
            y: 2
          }},
          font: {{
            face: 'Inter, Segoe UI, Roboto, sans-serif',
            size: 13,
            color: '#0f172a'
          }}
        }},
        edges: {{
          arrows: {{
            to: {{
              enabled: true,
              scaleFactor: 0.65
            }}
          }},
          color: {{
            color: '#64748b',
            highlight: '#14b8a6',
            hover: '#0ea5e9'
          }},
          font: {{
            face: 'Inter, Segoe UI, Roboto, sans-serif',
            size: 10,
            color: '#94a3b8',
            align: 'middle',
            background: 'rgba(15, 23, 42, 0.7)'
          }},
          smooth: {{
            type: 'continuous',
            roundness: 0.2
          }}
        }},
        physics: {{
          solver: 'forceAtlas2Based',
          forceAtlas2Based: {{
            gravitationalConstant: -38,
            centralGravity: 0.015,
            springLength: 95,
            springConstant: 0.08,
            damping: 0.65
          }},
          stabilization: {{
            iterations: 150,
            updateInterval: 25
          }}
        }},
        interaction: {{
          hover: true,
          navigationButtons: false,
          keyboard: false,
          tooltipDelay: 150
        }}
      }};

      const network = new vis.Network(container, data, options);

      // On click node event
      network.on('click', function(params) {{
        if (params.nodes.length > 0) {{
          const nodeId = params.nodes[0];
          const nodeObj = data.nodes.get(nodeId);
          selectionBanner.style.display = 'block';
          selectionBanner.innerHTML = 'Selected: <b>' + (nodeObj ? nodeObj.label : nodeId) + '</b> (' + (nodeObj ? nodeObj.group : '') + ')';
        }} else {{
          selectionBanner.style.display = 'none';
        }}
      }});

      // Double click to focus
      network.on('doubleClick', function(params) {{
        if (params.nodes.length > 0) {{
          network.focus(params.nodes[0], {{
            scale: 1.4,
            animation: {{
              duration: 600,
              easingFunction: 'easeInOutQuad'
            }}
          }});
        }}
      }});

      document.getElementById('btn-fit').addEventListener('click', function() {{
        network.fit({{
          animation: {{
            duration: 600,
            easingFunction: 'easeInOutQuad'
          }}
        }});
      }});

      document.getElementById('btn-stabilize').addEventListener('click', function() {{
        network.stabilize();
      }});
    }})();
  </script>
</body>
</html>"""
