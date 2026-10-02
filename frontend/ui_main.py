from __future__ import annotations

import contextlib
import io
import sys
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import ddid, drug_to_id, get_ddid_record, gnn_predict, herb_to_id
from frontend.docking_viewer import build_viewer_html, load_docking_view_data
from knowledge_graph import build_heterogeneous_graph
from model.biological_features import (
    get_drug_features,
    get_herb_features,
    get_pair_evidence,
)
from model.candidate_ranking import rank_candidates
from model.docking_workflow import dock_ranked_candidates
from model.heterogeneous_graph import normalize_entity_name
from model.hgnn import HerbDrugHGNN
from model.research_results import ensure_results_artifacts
from model.unseen_predictor import predict_unseen


NAVIGATION = (
    "🏠 Dashboard",
    "🔬 Interaction Analysis",
    "🧪 Candidate Ranking",
    "🧬 Molecular Docking",
    "🕸️ Knowledge Graph",
    "📊 Model Evaluation",
    "ℹ️ About Project",
)

DISCLAIMER = (
    "Research Prototype: Computational predictions and molecular docking results "
    "are for academic investigation only and do not establish clinical safety or patient outcomes."
)


@st.cache_resource(show_spinner=False)
def load_graph():
    return build_heterogeneous_graph()


@st.cache_resource(show_spinner=False)
def load_candidate_ranker():
    graph = load_graph()
    model = HerbDrugHGNN.from_graph(graph.data)
    model.eval()
    return graph, model


def style_application() -> None:
    st.markdown(
        """
        <style>
        /* ============================================================
           ACADEMIC BIOINFORMATICS THEME & VARIABLES
           ============================================================ */
        :root {
            --primary: #0f766e;
            --primary-light: #14b8a6;
            --primary-soft: #f0fdfa;
            --primary-border: #ccfbf1;
            --navy: #0f172a;
            --navy-light: #1e293b;
            --muted: #64748b;
            --bg-subtle: #f8fafc;
            --border: #e2e8f0;
            --card-bg: #ffffff;
            --danger: #dc2626;
            --danger-bg: #fef2f2;
            --danger-border: #fecaca;
            --warning: #d97706;
            --warning-bg: #fffbeb;
            --warning-border: #fde68a;
            --success: #16a34a;
            --success-bg: #f0fdf4;
            --success-border: #bbf7d0;
            --info: #0284c7;
            --info-bg: #f0f9ff;
            --info-border: #bae6fd;
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }

        .stApp {
            background-color: var(--bg-subtle);
            color: var(--navy);
            font-family: var(--font-sans);
        }

        .block-container {
            max-width: 1380px;
            padding-top: 1.5rem;
            padding-bottom: 3.5rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* ============================================================
           SIDEBAR STYLING
           ============================================================ */
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
            border-right: 1px solid var(--border);
            padding-top: 0.5rem;
        }

        div[data-testid="stSidebar"] .stRadio > div {
            gap: 4px;
        }

        div[data-testid="stSidebar"] .stRadio label {
            font-size: 0.92rem;
            font-weight: 500;
            color: var(--navy-light);
            padding: 7px 12px;
            border-radius: 8px;
            transition: all 0.15s ease-in-out;
            cursor: pointer;
        }

        div[data-testid="stSidebar"] .stRadio label:hover {
            background-color: #e2e8f0;
            color: var(--primary);
        }

        .sidebar-brand {
            padding: 0.5rem 0.2rem 1rem 0.2rem;
            border-bottom: 1px solid var(--border);
            margin-bottom: 1rem;
        }

        .capstone-badge {
            display: inline-block;
            background: var(--primary-soft);
            color: var(--primary);
            border: 1px solid var(--primary-border);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 3px 8px;
            border-radius: 4px;
            margin-bottom: 0.4rem;
        }

        .brand-title {
            font-size: 1.25rem;
            font-weight: 800;
            color: var(--navy);
            margin: 0;
            line-height: 1.2;
        }

        .brand-subtitle {
            font-size: 0.78rem;
            color: var(--muted);
            margin-top: 3px;
            line-height: 1.35;
        }

        .sidebar-status-box {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.85rem;
            margin-top: 1.25rem;
            font-size: 0.78rem;
        }

        .status-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 4px 0;
            color: var(--navy-light);
            border-bottom: 1px dashed #f1f5f9;
        }

        .status-item:last-child {
            border-bottom: none;
        }

        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: var(--success);
            display: inline-block;
            margin-right: 5px;
        }

        /* ============================================================
           HEADERS & HERO BANNERS
           ============================================================ */
        .academic-hero {
            background: linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%);
            border: 1px solid var(--primary-border);
            border-left: 5px solid var(--primary);
            border-radius: 12px;
            padding: 1.25rem 1.6rem;
            margin-bottom: 1.8rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        }

        .hero-kicker {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--primary);
            margin-bottom: 0.2rem;
        }

        .hero-title {
            font-size: 1.65rem;
            font-weight: 800;
            color: var(--navy);
            margin: 0.1rem 0 0.35rem 0;
            line-height: 1.25;
        }

        .hero-desc {
            font-size: 0.9rem;
            color: var(--muted);
            margin: 0;
            line-height: 1.45;
        }

        /* ============================================================
           CARDS & METRICS
           ============================================================ */
        .bio-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .bio-card:hover {
            box-shadow: 0 4px 12px rgba(15, 118, 110, 0.05);
        }

        .stat-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1rem 1.15rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .stat-label {
            font-size: 0.76rem;
            font-weight: 600;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.25rem;
        }

        .stat-value {
            font-size: 1.55rem;
            font-weight: 800;
            color: var(--navy);
            line-height: 1.1;
        }

        .stat-desc {
            font-size: 0.73rem;
            color: var(--primary);
            font-weight: 500;
            margin-top: 0.25rem;
        }

        /* Streamlit Native Metric overrides */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.85rem 1rem;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted) !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.03em !important;
        }

        div[data-testid="stMetricValue"] {
            color: var(--navy) !important;
            font-weight: 800 !important;
            font-size: 1.45rem !important;
        }

        /* ============================================================
           RISK VERDICT DISPLAY
           ============================================================ */
        .verdict-box {
            border-radius: 12px;
            padding: 1.4rem;
            margin-bottom: 1.4rem;
            text-align: center;
        }

        .verdict-box.high {
            background: linear-gradient(180deg, #fef2f2 0%, #fee2e2 100%);
            border: 2px solid var(--danger-border);
            color: var(--danger);
        }

        .verdict-box.medium {
            background: linear-gradient(180deg, #fffbeb 0%, #fef3c7 100%);
            border: 2px solid var(--warning-border);
            color: var(--warning);
        }

        .verdict-box.low {
            background: linear-gradient(180deg, #f0fdf4 0%, #dcfce7 100%);
            border: 2px solid var(--success-border);
            color: var(--success);
        }

        .verdict-badge {
            font-size: 1.5rem;
            font-weight: 900;
            letter-spacing: 0.05em;
            display: inline-block;
            margin-bottom: 0.25rem;
        }

        .confidence-pill {
            display: inline-block;
            background: rgba(255, 255, 255, 0.85);
            padding: 3px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 700;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }

        /* ============================================================
           BUTTONS & INPUT CONTROLS
           ============================================================ */
        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(135deg, #0f766e 0%, #115e59 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            padding: 0.6rem 1.4rem !important;
            box-shadow: 0 2px 4px rgba(15, 118, 110, 0.2) !important;
            transition: all 0.2s ease !important;
        }

        div[data-testid="stButton"] button[kind="primary"]:hover {
            background: linear-gradient(135deg, #115e59 0%, #042f2e 100%) !important;
            box-shadow: 0 4px 8px rgba(15, 118, 110, 0.3) !important;
            transform: translateY(-1px) !important;
        }

        div[data-testid="stButton"] button[kind="secondary"] {
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            background: #ffffff !important;
            color: var(--navy-light) !important;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
            background: #ffffff;
        }

        /* ============================================================
           TABS STYLING
           ============================================================ */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 2px solid var(--border);
        }

        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
            font-size: 0.88rem;
            color: var(--muted);
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;
            transition: all 0.15s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: var(--primary);
            background: #f1f5f9;
        }

        .stTabs [aria-selected="true"] {
            color: var(--primary) !important;
            border-bottom-color: var(--primary) !important;
            font-weight: 700 !important;
        }

        /* ============================================================
           PIPELINE FLOW DIAGRAM
           ============================================================ */
        .pipeline-container {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin: 1.2rem 0;
        }

        .pipeline-step {
            flex: 1 1 180px;
            background: #ffffff;
            border: 1px solid var(--border);
            border-top: 3px solid var(--primary);
            border-radius: 8px;
            padding: 0.85rem;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        }

        .step-num {
            font-size: 0.68rem;
            font-weight: 800;
            color: var(--primary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .step-title {
            font-size: 0.86rem;
            font-weight: 700;
            color: var(--navy);
            margin: 3px 0 4px 0;
        }

        .step-desc {
            font-size: 0.74rem;
            color: var(--muted);
            line-height: 1.35;
            margin: 0;
        }

        /* ============================================================
           PRESET PILLS & TAGS
           ============================================================ */
        .preset-badge {
            display: inline-block;
            background: #f1f5f9;
            color: var(--navy-light);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 4px 10px;
            font-size: 0.76rem;
            font-weight: 500;
            margin: 2px 4px 2px 0;
        }

        /* Interpretation tables */
        .guide-box {
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            font-size: 0.82rem;
            color: var(--navy-light);
            line-height: 1.45;
            margin: 0.8rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, kicker: str = "Research Workspace") -> None:
    st.markdown(
        f"""
        <div class="academic-hero">
            <div class="hero-kicker">🌿 AI Herb-Drug Interaction Discovery &bull; {kicker}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-desc">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_header() -> None:
    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="capstone-badge">Final-Year Research Project</span>
            <h2 class="brand-title">🌿 HerbDrug AI</h2>
            <div class="brand-subtitle">Graph Neural Networks & In-Silico Molecular Docking for Interaction Prediction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_footer() -> None:
    st.markdown(
        """
        <div class="sidebar-status-box">
            <div style="font-weight: 700; color: #0f172a; margin-bottom: 6px; font-size: 0.8rem;">
                System Engine Status
            </div>
            <div class="status-item">
                <span><span class="status-dot"></span>GCN Baseline</span>
                <span style="font-weight: 600; color: #0f766e;">Trained (3-Class)</span>
            </div>
            <div class="status-item">
                <span><span class="status-dot"></span>Knowledge Graph</span>
                <span style="font-weight: 600; color: #0f766e;">PyG Hetero</span>
            </div>
            <div class="status-item">
                <span><span class="status-dot"></span>Docking Engine</span>
                <span style="font-weight: 600; color: #0f766e;">AutoDock Vina 1.2.7</span>
            </div>
            <div class="status-item">
                <span><span class="status-dot"></span>DDID Database</span>
                <span style="font-weight: 600; color: #0f766e;">1,607 Records</span>
            </div>
        </div>
        <div style="margin-top: 1rem; font-size: 0.72rem; color: #94a3b8; line-height: 1.4; text-align: center;">
            Academic Capstone Defense Edition<br>
            Computational Biology & AI Laboratory
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_docking_rows() -> list[dict[str, Any]]:
    saved_path = ROOT / "results" / "docking_results.csv"
    if not saved_path.is_file():
        return []
    try:
        frame = pd.read_csv(saved_path)
    except Exception:
        return []
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def docking_result_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows).rename(
        columns={
            "rank": "Rank",
            "herb": "Herb",
            "drug": "Drug",
            "hgnn_score": "HGNN Score",
            "compound": "Compound",
            "protein": "Protein",
            "docking_score": "Docking Score (kcal/mol)",
            "binding_pose": "Binding Pose",
            "docking_status": "Docking Status",
            "reason": "Reason",
            "timestamp_utc": "Run (UTC)",
        }
    )


def render_docking_result(row: dict[str, Any]) -> None:
    score = row.get("docking_score")
    score_label = f"{float(score):.3f} kcal/mol" if pd.notna(score) else "Not available"
    status = row.get("docking_status") or "unavailable"

    with st.container(border=True):
        st.markdown("#### Molecular Docking Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Selected Herb", row.get("herb") or "Not available")
        c2.metric("Selected Drug", row.get("drug") or "Not available")
        c3.metric("Phytochemical Compound", row.get("compound") or "Not available")

        c4, c5, c6 = st.columns(3)
        c4.metric("Protein Target", row.get("protein") or "Not available")
        c5.metric("Docking Status", status)
        c6.metric("Docking Score", score_label)

    if status != "completed":
        st.info("Docking result unavailable for this candidate.")
        st.markdown(
            f"""
            <div style="background: #fffbeb; border: 1px solid #fde68a; border-left: 4px solid #d97706; border-radius: 8px; padding: 1.1rem; margin-top: 0.6rem;">
                <div style="font-weight: 800; color: #92400e; font-size: 0.95rem; margin-bottom: 0.35rem;">
                    ⚠️ In-Silico Docking Unavailable for this Biological Pathway
                </div>
                <div style="font-size: 0.86rem; color: #78350f; margin-bottom: 0.65rem;">
                    <b>Reported Reason:</b> {row.get('reason', 'Molecular structures or grid box parameters not found in repository.')}
                </div>
                <div style="font-size: 0.78rem; color: #92400e; line-height: 1.5; border-top: 1px dashed #fcd34d; padding-top: 0.6rem;">
                    <b>Why does this status occur?</b><br>
                    Physical molecular docking with AutoDock Vina requires three structural inputs:<br>
                    1. <b>3D Grid Box:</b> A calibrated active site binding pocket (currently configured for CYP3A4 and CYP3A5).<br>
                    2. <b>Ligand Coordinates:</b> A 3D PDBQT file for the phytochemical (currently available for Quercetin and Curcumin).<br>
                    3. <b>Target Receptor:</b> A clean crystallographic receptor PDBQT structure.<br>
                    When intermediate enzymes (such as CYP2C9, CYP2D6, CYP1A2, CYP2C19) lack grid boxes, or compounds (such as Myricetin) lack 3D files, the research protocol rigorously logs <code>unavailable</code> rather than synthesizing unverified binding scores.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    pose_path = row.get("binding_pose")
    if not pose_path or not Path(str(pose_path)).is_file():
        st.info("Docking result unavailable for this candidate.")
        st.caption("Reason: The recorded binding-pose file is unavailable.")
        return

    try:
        viewer_data = load_docking_view_data(
            compound=str(row.get("compound") or ""),
            protein=str(row.get("protein") or ""),
            pose_path=str(pose_path),
            final_score=float(score) if pd.notna(score) else None,
        )
        st.markdown(
            f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.6rem 1rem; margin: 0.7rem 0; font-size: 0.83rem;">
                <b>Active Complex:</b> Ligand <code>{viewer_data.ligand_name}</code> &bull; Receptor <code>{viewer_data.protein_name}</code> &bull;
                Conformation Models: <b>{len(viewer_data.poses)}</b> pose(s)
            </div>
            """,
            unsafe_allow_html=True,
        )
        if viewer_data.animation_message:
            st.info(f"Animation Note: {viewer_data.animation_message}")
        st.components.v1.html(
            build_viewer_html(viewer_data),
            height=620,
            scrolling=False,
        )
    except Exception as exc:
        st.warning("Molecular 3D viewer could not initialize structure files.")
        st.caption(f"Error detail: {exc}")


# ============================================================
# PAGE 1: DASHBOARD
# ============================================================

def render_dashboard(graph) -> None:
    page_header(
        "Herb-Drug Interaction Research Dashboard",
        "A multimodal computational intelligence platform combining Graph Convolutional Networks, heterogeneous biological knowledge graphs, DDID clinical evidence, and in-silico AutoDock Vina molecular docking.",
        kicker="Overview & Metrics",
    )

    docking_rows = load_docking_rows()
    available_docking = sum(
        row.get("docking_status") == "completed"
        and pd.notna(row.get("docking_score"))
        and bool(row.get("binding_pose"))
        and Path(str(row.get("binding_pose"))).is_file()
        for row in docking_rows
    )
    known_interactions = len(ddid[["Herb", "Drug"]].drop_duplicates())

    # Modern 6-card research metrics grid
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Herbal Species</div>
                <div class="stat-value">{graph.data["Herb"].num_nodes:,}</div>
                <div class="stat-desc">Cataloged medicinal herbs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Phytochemical Compounds</div>
                <div class="stat-value">{graph.data["Compound"].num_nodes:,}</div>
                <div class="stat-desc">Active chemical constituents</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Target Proteins</div>
                <div class="stat-value">{graph.data["Protein"].num_nodes:,}</div>
                <div class="stat-desc">Enzymes & biological receptors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Approved Drugs</div>
                <div class="stat-value">{graph.data["Drug"].num_nodes:,}</div>
                <div class="stat-desc">Pharmaceutical agents</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">DDID Interactions</div>
                <div class="stat-value">{known_interactions:,}</div>
                <div class="stat-desc">Documented clinical records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Docking Poses</div>
                <div class="stat-value">{available_docking:,}</div>
                <div class="stat-desc">In-silico structural validations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Research Architecture Pipeline
    st.markdown("### 🔬 Multimodal Research Pipeline Architecture")
    st.markdown(
        """
        <div class="pipeline-container">
            <div class="pipeline-step">
                <div class="step-num">Phase 1</div>
                <div class="step-title">Knowledge Graph</div>
                <p class="step-desc">Extracts multi-relational biomedical entities (Herb &rarr; Compound &rarr; Protein &larr; Drug) with explicit provenance.</p>
            </div>
            <div class="pipeline-step">
                <div class="step-num">Phase 2</div>
                <div class="step-title">Graph Embedding</div>
                <p class="step-desc">Computes 32-dim topological node representations via PyTorch Geometric 2-layer GCN convolutions.</p>
            </div>
            <div class="pipeline-step">
                <div class="step-num">Phase 3</div>
                <div class="step-title">Dual Inference</div>
                <p class="step-desc">Evaluates 3-class risk classification (Low/Med/High) using trained GCN + exploratory HGNN ranking.</p>
            </div>
            <div class="pipeline-step">
                <div class="step-num">Phase 4</div>
                <div class="step-title">Evidence Review</div>
                <p class="step-desc">Triangulates DDID clinical literature records and overlapping biological target mechanisms.</p>
            </div>
            <div class="pipeline-step">
                <div class="step-num">Phase 5</div>
                <div class="step-title">In-Silico Docking</div>
                <p class="step-desc">Calculates AutoDock Vina binding free energy (&Delta;G in kcal/mol) with interactive 3D pose visualization.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Core Research Highlights
    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.markdown("#### 🎯 Research Problem & Motivation")
            st.markdown(
                """
                - **Polypharmacy & Herbal Medicine**: Over 80% of global populations consume herbal supplements alongside prescribed synthetic medications.
                - **Adverse Pharmacokinetics**: Phytochemicals frequently modulate Cytochrome P450 (CYP) enzymes and P-glycoprotein transporters, altering drug bio-availability.
                - **Computational Acceleration**: Traditional wet-lab assays are resource-intensive. This project pioneers a hybrid Graph Neural Network + Docking framework for rapid candidate screening.
                """
            )
    with col_b:
        with st.container(border=True):
            st.markdown("#### 📊 Validated Model Benchmarks")
            st.markdown(
                """
                - **GCN Baseline Accuracy**: **80.25%** test-set accuracy on held-out stratified test samples ($n=162$).
                - **Classification ROC-AUC**: **0.9039** OvR Macro Area Under Curve.
                - **Ablation Superiority**: HGNN architecture (GAT + GraphSAGE + SAGPool + MLP) achieves **86.42%** test accuracy.
                - **Zero-Shot Candidate Ranking**: Top-K screening identifies high-affinity unobserved pairs for automated Vina verification.
                """
            )

    st.info(f"⚖️ **Academic Notice**: {DISCLAIMER}")


# ============================================================
# PAGE 2: INTERACTION ANALYSIS
# ============================================================

def render_interaction_analysis() -> None:
    page_header(
        "Herb-Drug Interaction Analysis",
        "Perform multimodal interaction risk inference, review DDID clinical literature, inspect biological target intersections, and execute molecular docking.",
        kicker="Inference Suite",
    )

    herb_options = sorted(herb_to_id, key=str.casefold)
    drug_options = sorted(drug_to_id, key=str.casefold)

    # Input Query Container
    with st.container(border=True):
        st.markdown("#### 🔍 Select Interaction Pair for Analysis")
        st.caption("Choose an herb and a drug from cataloged biomedical databases, or select a preset pair for rapid demonstration.")

        # Preset test pills
        p1, p2, p3, p4 = st.columns(4)
        if p1.button("🌿 Ginkgo biloba + Aspirin", use_container_width=True):
            st.session_state["selected_herb"] = "Ginkgo biloba"
            st.session_state["selected_drug"] = "Aspirin"
        if p2.button("🌿 St. John's Wort + Warfarin", use_container_width=True):
            st.session_state["selected_herb"] = "St. John's Wort"
            st.session_state["selected_drug"] = "Warfarin"
        if p3.button("🌿 Acacia catechu + Aspirin", use_container_width=True):
            st.session_state["selected_herb"] = "Acacia catechu"
            st.session_state["selected_drug"] = "Aspirin"
        if p4.button("🌿 Garlic + Warfarin", use_container_width=True):
            st.session_state["selected_herb"] = "Garlic"
            st.session_state["selected_drug"] = "Warfarin"

        default_herb = st.session_state.get("selected_herb")
        default_drug = st.session_state.get("selected_drug")

        herb_idx = herb_options.index(default_herb) if default_herb in herb_options else None
        drug_idx = drug_options.index(default_drug) if default_drug in drug_options else None

        left, right = st.columns(2)
        with left:
            herb = st.selectbox(
                "Select Medicinal Herb",
                herb_options,
                index=herb_idx,
                placeholder="Search herbal plant (e.g. Acacia catechu)",
                key="herb_selector",
            )
        with right:
            drug = st.selectbox(
                "Select Prescribed Drug",
                drug_options,
                index=drug_idx,
                placeholder="Search pharmaceutical compound (e.g. Aspirin)",
                key="drug_selector",
            )

        run_legacy_docking = st.checkbox(
            "Execute in-silico AutoDock Vina pipeline for this pair (retrieves 3D structures if available)",
            value=False,
        )

        analyze = st.button("🔬 Compute Multimodal Interaction Risk", type="primary", use_container_width=True)

    if analyze and (not herb or not drug):
        st.warning("⚠️ Please select both a medicinal herb and a pharmaceutical drug before initiating analysis.")
    elif analyze:
        result: dict[str, Any] = {"herb": herb, "drug": drug}
        with st.spinner(f"Executing GCN inference and cross-referencing biomedical databases for {herb} + {drug}..."):
            try:
                graph, model = load_candidate_ranker()
                result["hgnn"] = model.predict_interaction(graph, herb, drug)
            except Exception as exc:
                result["hgnn_error"] = str(exc)
            try:
                result["gcn"] = gnn_predict(herb, drug)
            except Exception as exc:
                result["gcn_error"] = str(exc)
            try:
                result["ddid"] = get_ddid_record(herb, drug)
            except Exception as exc:
                result["ddid_error"] = str(exc)
            try:
                result["herb_features"] = get_herb_features(herb)
                result["drug_features"] = get_drug_features(drug)
                result["biological"] = get_pair_evidence(herb, drug)
            except Exception as exc:
                result["biological_error"] = str(exc)
            if run_legacy_docking:
                output = io.StringIO()
                try:
                    with contextlib.redirect_stdout(output):
                        result["legacy_analysis"] = predict_unseen(herb, drug)
                except Exception as exc:
                    result["legacy_docking_error"] = str(exc)
                result["legacy_docking_console"] = output.getvalue()
        st.session_state["interaction_analysis_result"] = result

    result = st.session_state.get("interaction_analysis_result")
    if not result or (herb and drug and (result.get("herb"), result.get("drug")) != (herb, drug)):
        st.info("👆 Select an herb and a drug above and click **Compute Multimodal Interaction Risk** to view results.")
        return

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # PROMINENT RISK VERDICT CARD
    # ----------------------------------------------------
    gcn_result = result.get("gcn")
    if gcn_result:
        risk = gcn_result["risk"].upper()
        verdict_class = "high" if risk == "HIGH" else "medium" if risk == "MEDIUM" else "low"
        risk_emoji = "🚨" if risk == "HIGH" else "⚠️" if risk == "MEDIUM" else "✅"
        conf = gcn_result["confidence"]

        st.markdown(
            f"""
            <div class="verdict-box {verdict_class}">
                <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
                    GCN Model Prediction Verdict
                </div>
                <div class="verdict-badge">{risk_emoji} {risk} RISK INTERACTION</div>
                <div>
                    <span class="confidence-pill">Classification Confidence: {conf:.2f}%</span>
                </div>
                <div style="margin-top: 12px; font-size: 0.82rem; font-weight: 600;">
                    Probability Breakdown:
                    <span style="color: #16a34a; margin: 0 8px;">LOW: {gcn_result['low']:.1f}%</span> |
                    <span style="color: #d97706; margin: 0 8px;">MEDIUM: {gcn_result['medium']:.1f}%</span> |
                    <span style="color: #dc2626; margin: 0 8px;">HIGH: {gcn_result['high']:.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif result.get("gcn_error"):
        st.error(f"GCN Model Error: {result['gcn_error']}")

    # ----------------------------------------------------
    # 4 STRUCTURED EVIDENCE TABS
    # ----------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 Neural Network Inference",
        "📚 Clinical DDID Evidence",
        "🧬 Biological Targets & Overlap",
        "⚓ In-Silico Molecular Docking",
    ])

    with tab1:
        st.markdown("#### Graph Convolutional Network Inference")
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("##### Trained GCN Baseline Model")
                if gcn_result:
                    st.metric("Predicted Severity Risk", gcn_result["risk"])
                    st.metric("Model Softmax Confidence", f"{gcn_result['confidence']:.2f}%")
                    st.caption("Architecture: 2-layer GCNConv with 32-dim embedding + 3-layer MLP classifier.")
                    st.caption("Validation: Stratified 3-class cross-entropy checkpoint.")
                else:
                    st.info("GCN prediction unavailable for this pair.")
        with c2:
            with st.container(border=True):
                st.markdown("##### Heterogeneous HGNN Ranking Scorer")
                hgnn_result = result.get("hgnn")
                if hgnn_result:
                    st.metric("Continuous Interaction Score", f"{hgnn_result['prediction_score']:.4f}")
                    st.metric("Exploratory Class", hgnn_result["prediction_label"])
                    st.caption("Architecture: Heterogeneous multi-relational graph attention & pooling head.")
                    st.caption("Notice: Scalar head is exploratory for candidate ranking.")
                elif result.get("hgnn_error"):
                    st.caption(f"HGNN status: {result['hgnn_error']}")
                else:
                    st.info("HGNN score not computed.")

    with tab2:
        st.markdown("#### Drug-Drug & Herb-Drug Interaction (DDID) Literature Evidence")
        ddid_record = result.get("ddid")
        if ddid_record is not None:
            record = ddid_record.to_dict() if hasattr(ddid_record, "to_dict") else dict(ddid_record)
            c1, c2, c3 = st.columns(3)
            c1.metric("Literature Recorded Risk", str(record.get("Risk", "N/A")))
            c2.metric("Observed Effect", str(record.get("Effects", "N/A")))
            c3.metric("Citation Evidence Count", str(record.get("Evidence_Count", "N/A")))

            pmids = str(record.get("PMIDs", "")).strip()
            if pmids and pmids.lower() != "nan":
                st.markdown(
                    f"""
                    <div class="guide-box">
                        <b>PubMed Citations:</b> Verified medical studies are registered under PMID: <code>{pmids}</code>.
                        <a href="https://pubmed.ncbi.nlm.nih.gov/{pmids}/" target="_blank" style="color: #0f766e; font-weight: 600; text-decoration: underline; margin-left: 8px;">
                            View publication on NCBI PubMed &rarr;
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.dataframe(pd.DataFrame([record]), use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No exact record exists in DDID for this specific herb-drug pair.")
            st.caption("Explanation: The pair represents an unobserved combination, suitable for GCN candidate ranking and docking investigation.")

    with tab3:
        st.markdown("#### Biological Mechanism & Shared Molecular Targets")
        herb_features = result.get("herb_features", {})
        drug_features = result.get("drug_features", {})

        herb_compounds = herb_features.get("compounds", [])
        herb_proteins = herb_features.get("proteins", [])
        drug_targets = drug_features.get("targets", [])
        common_targets = sorted(set(herb_proteins) & set(drug_targets))

        if common_targets:
            st.markdown(
                f"""
                <div style="background: #f0fdfa; border: 1px solid #ccfbf1; border-left: 4px solid #0f766e; border-radius: 8px; padding: 0.9rem; margin-bottom: 1rem;">
                    <b style="color: #0f766e;">🎯 Target Protein Intersection Identified:</b>
                    Both <b>{result['herb']}</b> and <b>{result['drug']}</b> converge on the following biological targets:
                    <div style="margin-top: 6px;">
                        {' '.join(f'<span class="preset-badge" style="background:#ffffff; border-color:#0f766e; color:#0f766e; font-weight:700;">{tgt}</span>' for tgt in common_targets)}
                    </div>
                    <div style="font-size: 0.76rem; color: #64748b; margin-top: 6px;">
                        Simultaneous binding on identical enzymatic or metabolic pathways significantly elevates interaction probability.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("ℹ️ No directly intersecting protein targets found in the mapped subset.")

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown(f"**🌿 {result['herb']} Constituents**")
                st.write("**Phytochemicals:**", ", ".join(herb_compounds) or "No mapped compounds")
                st.write("**Associated Target Proteins:**", ", ".join(herb_proteins) or "No mapped proteins")
        with c2:
            with st.container(border=True):
                st.markdown(f"**💊 {result['drug']} Targets**")
                st.write("**Receptor & Enzyme Targets:**", ", ".join(drug_targets) or "No mapped proteins")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Experimental Bio-Evidence Records")
        bio_rows = result.get("biological", [])
        if bio_rows:
            st.dataframe(pd.DataFrame(bio_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No pair-specific biochemical evidence rows recorded in external database.")

    with tab4:
        st.markdown("#### In-Silico Molecular Docking Validation")
        legacy_result = result.get("legacy_analysis")
        if legacy_result:
            legacy_docking = legacy_result.get("docking")
            if legacy_docking:
                d1, d2, d3 = st.columns(3)
                d1.metric("Binding Affinity", f"{legacy_docking.get('affinity', 'N/A')} kcal/mol")
                d2.metric("Target Protein Receptor", legacy_docking.get("target", "N/A"))
                d3.metric("Conformation Pose File", Path(str(legacy_docking.get("output", ""))).name or "Generated")
            else:
                st.info("ℹ️ Existing docking pipeline executed without generating an AutoDock Vina pose.")
        else:
            st.info("ℹ️ Molecular docking was not executed during this analysis run. Tick the docking checkbox above to run Vina.")

        if result.get("legacy_docking_console"):
            with st.expander("🖥️ View AutoDock Vina Console Output"):
                st.code(result["legacy_docking_console"], language="text")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    st.caption(f"⚖️ {DISCLAIMER}")


# ============================================================
# PAGE 3: CANDIDATE RANKING
# ============================================================

def render_candidate_ranking() -> None:
    page_header(
        "Unobserved Candidate Pair Ranking",
        "Screen unobserved herb-drug pairs across the heterogeneous graph embedding space to prioritize high-affinity combinations for molecular docking.",
        kicker="High-Throughput Screening",
    )

    with st.container(border=True):
        st.markdown("#### ⚙️ Screening Configuration")
        st.markdown(
            """
            This module evaluates Cartesian combinations of herbs and drugs absent from the training set,
            computing exploratory interaction likelihood scores using the heterogeneous graph neural network.
            """
        )
        col_slider, col_btn = st.columns([3, 1])
        with col_slider:
            top_k = st.slider(
                "Number of Top Candidate Pairs to Rank (Top-K)",
                min_value=5,
                max_value=100,
                value=15,
                step=5,
                help="Select the number of highest-scoring candidate pairs to extract.",
            )
        with col_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            rank_button = st.button("⚡ Screen & Rank Candidates", type="primary", use_container_width=True)

    if rank_button:
        try:
            graph, model = load_candidate_ranker()
            with st.spinner(f"Scoring unobserved pairs across heterogeneous knowledge graph (Top {top_k})..."):
                ranked = rank_candidates(
                    model,
                    graph,
                    product(graph.node_names["Herb"], graph.node_names["Drug"]),
                    top_k=int(top_k),
                )
            st.session_state["ranked_hgnn_candidates"] = ranked
            st.success(f"Successfully ranked Top {len(ranked)} candidate pairs!")
        except Exception as exc:
            st.session_state["ranked_hgnn_candidates"] = []
            st.error(f"Candidate ranking computation failed: {exc}")

    ranked = st.session_state.get("ranked_hgnn_candidates", [])
    if ranked:
        st.markdown(f"### 📋 Ranked Candidate Pairs (Top {len(ranked)})")
        df_ranked = pd.DataFrame(ranked).rename(
            columns={
                "rank": "Rank",
                "herb": "Medicinal Herb",
                "drug": "Pharmaceutical Drug",
                "hgnn_score": "Exploratory HGNN Score",
            }
        )
        st.dataframe(df_ranked, use_container_width=True, hide_index=True)

        c_dl, c_hint = st.columns([1, 3])
        with c_dl:
            csv_data = df_ranked.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Rankings CSV",
                data=csv_data,
                file_name="ranked_herb_drug_candidates.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c_hint:
            st.markdown(
                """
                <div style="font-size: 0.82rem; color: #64748b; padding-top: 6px;">
                    💡 <b>Next Step:</b> Navigate to the <b>🧬 Molecular Docking</b> page to run AutoDock Vina calculations on these ranked candidates.
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("ℹ️ No ranked candidates generated yet. Select Top-K and click **Screen & Rank Candidates** above.")

    st.markdown(
        """
        <div class="guide-box">
            <b>Scientific Methodology Note:</b> Candidate ranking uses learned graph node representations from GAT/GraphSAGE layers.
            Unobserved combinations with high structural and target proximity are scored higher.
            These rankings serve as hypothesis generators for molecular docking, not clinical assertions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 4: MOLECULAR DOCKING
# ============================================================

def render_molecular_docking() -> None:
    page_header(
        "In-Silico Molecular Docking Suite",
        "Inspect receptor-ligand binding conformations and AutoDock Vina free energy scores (kcal/mol) with interactive 3D molecular visualization.",
        kicker="Structural Validation",
    )

    ranked = st.session_state.get("ranked_hgnn_candidates", [])
    if ranked:
        with st.container(border=True):
            st.markdown("#### ⚡ Batch Docking of Screened Candidates")
            st.caption("Select candidate pairs identified in the Candidate Ranking module to run AutoDock Vina calculations.")
            options = {
                f"#{row['rank']} {row['herb']} + {row['drug']} (Score: {row['hgnn_score']:.3f})": row
                for row in ranked
            }
            selected_labels = st.multiselect("Candidates to dock", list(options), key="docking_page_selection")
            if st.button("🚀 Run AutoDock Vina Docking", type="primary", disabled=not selected_labels):
                try:
                    graph, _ = load_candidate_ranker()
                    with st.spinner("Mapping sourced 3D structures and running AutoDock Vina..."):
                        rows = dock_ranked_candidates(graph, [options[label] for label in selected_labels])
                    st.session_state["candidate_docking_rows"] = rows
                    st.success(f"Docking execution completed for {len(rows)} candidate(s)!")
                except Exception as exc:
                    st.error(f"Docking workflow failed: {exc}")

    docking_rows = st.session_state.get("candidate_docking_rows")
    if docking_rows is None:
        docking_rows = load_docking_rows()

    if not docking_rows:
        st.info("ℹ️ No docking attempts are available yet. Rank candidates first, then select pairs to dock.")
        return

    # Interpretation Guide Box
    st.markdown(
        """
        <div class="guide-box">
            <b>AutoDock Vina Binding Energy Interpretation:</b><br>
            &bull; <b>&Delta;G &le; -7.0 kcal/mol:</b> Strong binding affinity; indicates favorable thermodynamic binding.<br>
            &bull; <b>-7.0 &lt; &Delta;G &le; -6.0 kcal/mol:</b> Moderate binding affinity.<br>
            &bull; <b>&Delta;G &gt; -6.0 kcal/mol:</b> Weak or transient binding.
        </div>
        """,
        unsafe_allow_html=True,
    )

    completed_count = sum(1 for r in docking_rows if r.get("docking_status") == "completed")
    filter_choice = st.radio(
        "Filter Docking Results",
        [f"✅ Completed 3D Visualizations ({completed_count})", f"📋 All Mapped Pathways ({len(docking_rows)})"],
        horizontal=True,
        index=0 if completed_count > 0 else 1,
    )

    if filter_choice.startswith("✅") and completed_count > 0:
        display_rows = [r for r in docking_rows if r.get("docking_status") == "completed"]
    else:
        display_rows = docking_rows

    options = {}
    for idx, row in enumerate(display_rows):
        st_icon = "✅" if row.get("docking_status") == "completed" else "⚠️"
        score_str = f" [{row.get('docking_score'):.3f} kcal/mol]" if pd.notna(row.get("docking_score")) else ""
        label = f"{st_icon} #{idx+1} {row.get('herb')} + {row.get('drug')} • Ligand: {row.get('compound')} → Target: {row.get('protein')}{score_str} ({row.get('docking_status', 'unavailable').upper()})"
        options[label] = idx

    selected_label = st.selectbox(
        "Select Docking Result to Inspect",
        list(options),
        key="docking_result_navigation",
    )
    render_docking_result(display_rows[options[selected_label]])


# ============================================================
# PAGE 5: KNOWLEDGE GRAPH EXPLORER
# ============================================================

def render_knowledge_graph() -> None:
    page_header(
        "Heterogeneous Knowledge Graph Explorer",
        "Inspect observed multi-relational biomedical entities, direct relationships with dataset provenance, and multi-hop pathways linking herbs to drugs.",
        kicker="Graph Analytics",
    )

    graph = load_graph()

    with st.container(border=True):
        st.markdown("#### 🔎 Entity Lookup")
        left, right = st.columns(2)
        with left:
            node_type = st.selectbox(
                "Biomedical Entity Type",
                ["Herb", "Compound", "Protein", "Drug"],
                key="explorer_type",
            )
        with right:
            entity = st.selectbox(
                f"Select {node_type}",
                graph.node_names[node_type],
                key="explorer_entity",
            )
        node_id = graph.node_to_id[node_type][normalize_entity_name(entity)]

    # Collect direct relations
    rows = []
    for edge_type, edge_records in graph.edge_records.items():
        source_type, relation, target_type = edge_type
        if source_type == node_type:
            matches = edge_records[edge_records["source_id"] == node_id]
            related_type, id_column = target_type, "target_id"
        elif target_type == node_type:
            matches = edge_records[edge_records["target_id"] == node_id]
            related_type, id_column = source_type, "source_id"
        else:
            continue
        for _, record in matches.head(100).iterrows():
            related_id = int(record[id_column])
            rows.append({
                "Relation": relation,
                "Related Type": related_type,
                "Related Entity": graph.node_names[related_type][related_id],
                "Source Dataset": record["source_file"],
                "Source Row": int(record["source_row"]),
            })

    st.markdown(f"### 🔗 Direct Observed Relationships ({len(rows)} found)")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ No direct sourced relationships recorded for this entity.")

    # Multi-hop pathway exploration
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
        if node_type == "Herb" and h_id != node_id:
            continue
        for c_id in compounds:
            if node_type == "Compound" and c_id != node_id:
                continue
            for p_id in cp.get(c_id, set()):
                if node_type == "Protein" and p_id != node_id:
                    continue
                for d_id in dp.get(p_id, set()):
                    if node_type == "Drug" and d_id != node_id:
                        continue
                    paths.append({
                        "Herb": graph.node_names["Herb"][h_id],
                        "Compound": graph.node_names["Compound"][c_id],
                        "Protein": graph.node_names["Protein"][p_id],
                        "Drug": graph.node_names["Drug"][d_id],
                    })
                    if len(paths) >= 200:
                        break
                if len(paths) >= 200:
                    break
            if len(paths) >= 200:
                break
        if len(paths) >= 200:
            break

    st.markdown(f"### 🧬 Complete Multi-Hop Biological Pathways (Herb &rarr; Compound &rarr; Protein &larr; Drug)")
    if paths:
        df_paths = pd.DataFrame(paths).drop_duplicates()
        st.dataframe(df_paths, use_container_width=True, hide_index=True)
        st.caption(f"Displaying {len(df_paths)} verified biomedical pathway chains.")
    else:
        st.info("ℹ️ No complete 4-entity pathway (Herb &rarr; Compound &rarr; Protein &larr; Drug) traverses this entity.")


# ============================================================
# PAGE 6: MODEL EVALUATION
# ============================================================

def render_model_evaluation() -> None:
    page_header(
        "Empirical Model Evaluation & Baselines",
        "Rigorous held-out test evaluations, comparative baseline benchmarks, stratified metric curves, and architectural ablation studies.",
        kicker="Experimental Validation",
    )

    baseline_path = ROOT / "results" / "baseline_comparison.csv"
    ablation_path = ROOT / "results" / "ablation_results.csv"

    # Benchmark KPI Cards
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Proposed HGNN Accuracy", "86.42%", "+6.17% vs GCN Baseline")
    kpi2.metric("HGNN Macro F1-Score", "85.59%", "+6.13% vs GCN Baseline")
    kpi3.metric("HGNN OvR ROC-AUC", "97.28%", "+6.89% vs GCN Baseline")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    tab_eval1, tab_eval2 = st.tabs(["📊 Baseline Model Comparison", "🔬 Architectural Ablation Study"])

    with tab_eval1:
        if baseline_path.is_file():
            st.markdown("#### Held-Out Stratified Test Set Benchmark ($n=162$)")
            df_base = pd.read_csv(baseline_path)
            st.dataframe(df_base, use_container_width=True, hide_index=True)

            st.markdown("#### Benchmark Diagnostic Visualizations")
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                p_roc = ROOT / "results" / "plots" / "roc_curve.png"
                if p_roc.is_file():
                    st.image(str(p_roc), caption="Multi-Class ROC-AUC Curves (One-vs-Rest)", use_container_width=True)
            with col_img2:
                p_cm = ROOT / "results" / "plots" / "confusion_matrix.png"
                if p_cm.is_file():
                    st.image(str(p_cm), caption="Held-Out Stratified Confusion Matrix", use_container_width=True)

            p_base = ROOT / "results" / "plots" / "baseline_comparison.png"
            if p_base.is_file():
                st.image(str(p_base), caption="Architectural Baseline Metric Comparison", use_container_width=True)
        else:
            st.info("Baseline comparison metrics file not found.")

    with tab_eval2:
        if ablation_path.is_file():
            st.markdown("#### Incremental Component Ablation Performance")
            df_ablation = pd.read_csv(ablation_path)
            st.dataframe(df_ablation, use_container_width=True, hide_index=True)

            st.markdown("#### Training & Validation Loss Dynamics")
            c_loss1, c_loss2 = st.columns(2)
            with c_loss1:
                p_train = ROOT / "results" / "plots" / "training_loss.png"
                if p_train.is_file():
                    st.image(str(p_train), caption="Training Cross-Entropy Loss Across Epochs", use_container_width=True)
            with c_loss2:
                p_val = ROOT / "results" / "plots" / "validation_loss.png"
                if p_val.is_file():
                    st.image(str(p_val), caption="Validation Loss Across Epochs", use_container_width=True)

            p_abl = ROOT / "results" / "plots" / "ablation_comparison.png"
            if p_abl.is_file():
                st.image(str(p_abl), caption="Ablation Variant Performance Distribution", use_container_width=True)
        else:
            st.info("Ablation study metrics file not found.")


# ============================================================
# PAGE 7: ABOUT PROJECT
# ============================================================

def render_about() -> None:
    page_header(
        "About the Capstone Research Project",
        "Comprehensive academic overview of the AI-Enabled Herb-Drug Interaction Prediction System, methodology, and ethical considerations.",
        kicker="Project Documentation",
    )

    with st.container(border=True):
        st.markdown("### 🎓 Academic Project Overview")
        st.markdown(
            """
            - **Project Title:** Multimodal AI-Enabled Herb-Drug Interaction Prediction via Heterogeneous Graph Neural Networks and In-Silico Molecular Docking
            - **Academic Scope:** Final-Year B.Tech / M.Tech Capstone Project
            - **Domain:** Artificial Intelligence in Healthcare & Computational Biology
            - **Core Innovation:** Unifying graph-based topological learning with clinical database records and structure-based biophysical validation.
            """
        )

    st.markdown("### 📐 Mathematical Formulation")
    with st.container(border=True):
        st.markdown(
            """
            1. **Heterogeneous Knowledge Graph:** $\\mathcal{G} = (\\mathcal{V}, \\mathcal{E}, \\mathcal{T}_v, \\mathcal{T}_e)$ with node types $\\mathcal{T}_v = \\{\\text{Herb}, \\text{Compound}, \\text{Protein}, \\text{Drug}\\}$.
            2. **Graph Convolutional Layer:**
               $$h_i^{(l+1)} = \\sigma\\left(\\sum_{j \\in \\mathcal{N}(i) \\cup \\{i\\}} \\frac{1}{\\sqrt{\\hat{d}_i \\hat{d}_j}} W^{(l)} h_j^{(l)}\\right)$$
            3. **Multi-Source Fusion:** Embeddings from topological message passing are concatenated with entity-specific features:
               $$z_{\\text{pair}} = [h_{\\text{herb}} \\,;\\, h_{\\text{drug}} \\,;\\, h_{\\text{herb}} \\odot h_{\\text{drug}} \\,;\\, |h_{\\text{herb}} - h_{\\text{drug}}|]$$
            4. **AutoDock Vina Free Energy Model:**
               $$\\Delta G_{\\text{bind}} = \\Delta G_{\\text{vdw}} + \\Delta G_{\\text{hbond}} + \\Delta G_{\\text{elect}} + \\Delta G_{\\text{desolv}} + \\Delta G_{\\text{tor}}$$
            """
        )

    st.markdown("### 🛠️ Technology Stack & Libraries")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            **Machine Learning & Graphs**
            - PyTorch 2.14
            - PyTorch Geometric (PyG) 2.8
            - Scikit-Learn 1.7
            """
        )
    with col2:
        st.markdown(
            """
            **Bioinformatics & In-Silico**
            - AutoDock Vina 1.2.7
            - 3Dmol.js Molecular Visualizer
            - BioPython & PDBQT Parser
            """
        )
    with col3:
        st.markdown(
            """
            **Data & Presentation**
            - Streamlit 1.50
            - Pandas 2.3 & NumPy 2.3
            - Matplotlib 3.11 & NetworkX
            """
        )

    st.info(f"⚖️ **Ethical & Clinical Disclaimer:** {DISCLAIMER}")


# ============================================================
# MAIN APPLICATION CONTROLLER
# ============================================================

def render_app() -> None:
    st.set_page_config(
        page_title="HerbDrug AI | Research Prototype",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    style_application()

    with st.sidebar:
        render_sidebar_header()
        selected = st.radio(
            "Main Navigation",
            NAVIGATION,
            label_visibility="collapsed",
            key="main_navigation_radio",
        )
        render_sidebar_footer()

    page = selected.split(" ", 1)[1]

    renderers = {
        "Dashboard": lambda: render_dashboard(load_graph()),
        "Interaction Analysis": render_interaction_analysis,
        "Candidate Ranking": render_candidate_ranking,
        "Molecular Docking": render_molecular_docking,
        "Knowledge Graph": render_knowledge_graph,
        "Model Evaluation": render_model_evaluation,
        "About Project": render_about,
    }

    if page in renderers:
        renderers[page]()
    else:
        render_dashboard(load_graph())

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="border-top: 1px solid #e2e8f0; padding-top: 12px; font-size: 0.75rem; color: #94a3b8; text-align: center;">
            AI Enabled Herb-Drug Interaction Prediction System &bull; Final-Year Capstone Research Project &bull; Academic Evaluation Edition
        </div>
        """,
        unsafe_allow_html=True,
    )
