"""Load authentic molecular structures and build an interactive 3Dmol.js viewer with docking animation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DockingViewData:
    receptor_text: str
    receptor_format: str
    ligand_name: str
    protein_name: str
    poses: tuple[dict[str, Any], ...]
    approach_poses: tuple[dict[str, Any], ...]
    animation_available: bool
    animation_kind: str
    animation_message: str
    has_approach: bool
    final_score: float | None


_AUTODOCK_ELEMENTS = {
    "A": "C",
    "C": "C",
    "N": "N",
    "NA": "N",
    "OA": "O",
    "O": "O",
    "S": "S",
    "SA": "S",
    "P": "P",
    "F": "F",
    "CL": "Cl",
    "BR": "Br",
    "I": "I",
    "H": "H",
    "HD": "H",
    "HS": "H",
    "FE": "Fe",
    "MG": "Mg",
    "ZN": "Zn",
}


def _atom_element(line: str) -> str:
    atom_type = line.split()[-1].upper() if line.split() else ""
    if atom_type in _AUTODOCK_ELEMENTS:
        return _AUTODOCK_ELEMENTS[atom_type]

    atom_name = line[12:16].strip()
    if len(atom_name) > 1 and atom_name[:2].upper() in {"CL", "BR", "FE", "MG", "ZN"}:
        return atom_name[:2].title()
    return next((character.upper() for character in atom_name if character.isalpha()), "C")


def _pdbqt_atom_to_pdb(line: str) -> str:
    if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
        return ""
    atom_record = line[:66].ljust(66)
    return f"{atom_record}          {_atom_element(line):>2}  "


def _pdbqt_to_pdb(lines: list[str]) -> str:
    atoms = [_pdbqt_atom_to_pdb(line) for line in lines]
    atoms = [line for line in atoms if line]
    return "\n".join(["HEADER    PDBQT CONVERTED FOR MOLECULAR VIEWER", *atoms, "END"])


def _parse_vina_pose_models(text: str) -> list[dict[str, Any]]:
    poses = []
    current_atoms: list[str] = []
    current_score: float | None = None
    pending_score: float | None = None
    model_index = 1

    def append_pose() -> None:
        nonlocal model_index
        if current_atoms:
            score = current_score if current_score is not None else pending_score
            poses.append(
                {
                    "text": _pdbqt_to_pdb(current_atoms),
                    "format": "pdb",
                    "score": score,
                    "model_number": model_index,
                }
            )
            model_index += 1

    for line in text.splitlines():
        if line.startswith("MODEL "):
            append_pose()
            current_atoms = []
            current_score = None
        elif line.startswith("REMARK VINA RESULT:"):
            try:
                score = float(line.split()[3])
            except (IndexError, ValueError):
                score = None
            if current_atoms:
                current_score = score
            else:
                pending_score = score
                current_score = score
        elif line.startswith(("ATOM  ", "HETATM")):
            current_atoms.append(line)
        elif line.startswith("ENDMDL"):
            append_pose()
            current_atoms = []
            current_score = None
            pending_score = None

    append_pose()
    if poses:
        return poses

    atom_lines = [
        line for line in text.splitlines() if line.startswith(("ATOM  ", "HETATM"))
    ]
    if atom_lines:
        return [
            {
                "text": _pdbqt_to_pdb(atom_lines),
                "format": "pdb",
                "score": pending_score,
                "model_number": 1,
            }
        ]
    return []


def _parse_pdb_models(text: str, score: float | None) -> list[dict[str, Any]]:
    models = []
    current_lines = []
    has_models = False
    for line in text.splitlines():
        if line.startswith("MODEL "):
            has_models = True
            current_lines = [line]
        elif line.startswith("ENDMDL") and current_lines:
            current_lines.append(line)
            models.append("\n".join(current_lines))
            current_lines = []
        elif current_lines:
            current_lines.append(line)
    if current_lines:
        models.append("\n".join(current_lines))

    if not has_models:
        models = [text]
    return [
        {"text": model, "format": "pdb", "score": score, "model_number": idx + 1}
        for idx, model in enumerate(models)
    ]


def _load_receptor(path: Path) -> tuple[str, str]:
    """Load receptor structure, preferring crystallographic PDB if available."""
    suffix = path.suffix.lower()
    # Check if a sibling crystallographic .pdb exists
    pdb_sibling = path.with_suffix(".pdb")
    if pdb_sibling.is_file():
        try:
            pdb_text = pdb_sibling.read_text(encoding="utf-8", errors="replace")
            if pdb_text.strip():
                return pdb_text, "pdb"
        except Exception:
            pass

    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdbqt":
        converted = _pdbqt_to_pdb(text.splitlines())
        if not converted.strip():
            raise ValueError("The PDBQT receptor contains no readable atom records.")
        return converted, "pdb"
    if suffix in {".pdb", ".ent"}:
        return text, "pdb"
    if suffix in {".cif", ".mmcif"}:
        return text, "cif"
    raise ValueError(f"Unsupported receptor format: {suffix or 'unknown'}")


def _atom_data(pdb_text: str) -> tuple[list[tuple[str, str]], np.ndarray, list[dict[str, str]]]:
    identities = []
    coordinates = []
    metadata = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        try:
            coordinates.append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
        except ValueError:
            continue
        identities.append((line[6:11].strip(), line[12:16].strip()))
        metadata.append(
            {
                "chain": line[21:22].strip(),
                "resi": line[22:26].strip(),
                "resn": line[17:20].strip(),
                "element": line[76:78].strip().upper() if len(line) >= 78 else "",
            }
        )
    return identities, np.asarray(coordinates, dtype=float), metadata


def _pocket_residues(
    receptor_pdb: str,
    ligand_pdb: str,
    cutoff_angstroms: float = 5.0,
) -> list[dict[str, str]]:
    _, receptor_coordinates, receptor_atoms = _atom_data(receptor_pdb)
    _, ligand_coordinates, ligand_atoms = _atom_data(ligand_pdb)
    if not len(receptor_coordinates) or not len(ligand_coordinates):
        return []

    receptor_keep = np.asarray(
        [atom["element"] not in {"H", "D"} for atom in receptor_atoms],
        dtype=bool,
    )
    ligand_keep = np.asarray(
        [atom["element"] not in {"H", "D"} for atom in ligand_atoms],
        dtype=bool,
    )
    receptor_coordinates = receptor_coordinates[receptor_keep]
    receptor_atoms = [atom for atom, keep in zip(receptor_atoms, receptor_keep) if keep]
    ligand_coordinates = ligand_coordinates[ligand_keep]
    if not len(receptor_coordinates) or not len(ligand_coordinates):
        return []

    differences = receptor_coordinates[:, None, :] - ligand_coordinates[None, :, :]
    min_distances = np.sqrt(np.sum(differences * differences, axis=2)).min(axis=1)
    pocket = {}
    for atom, distance in zip(receptor_atoms, min_distances):
        if distance <= cutoff_angstroms:
            key = (atom["chain"], atom["resi"], atom["resn"])
            pocket[key] = {"chain": atom["chain"], "resi": atom["resi"], "resn": atom["resn"]}
    return list(pocket.values())


def _interpolate_reference_pose(
    reference_pdb: str,
    final_pdb: str,
    final_score: float | None,
    frame_count: int = 18,
) -> list[dict[str, Any]] | None:
    reference_ids, reference_coordinates, _ = _atom_data(reference_pdb)
    final_ids, final_coordinates, _ = _atom_data(final_pdb)
    if not reference_ids or len(reference_coordinates) != len(final_coordinates):
        return None
    if reference_coordinates.shape != final_coordinates.shape:
        return None

    reference_lines = [
        line for line in reference_pdb.splitlines() if line.startswith(("ATOM  ", "HETATM"))
    ]
    frames = []
    for frame_index in range(frame_count):
        # Ease-in-out interpolation curve for smooth approach
        t = frame_index / (frame_count - 1)
        fraction = t * t * (3.0 - 2.0 * t)

        coordinates = reference_coordinates + fraction * (
            final_coordinates - reference_coordinates
        )
        atom_number = 0
        output_lines = ["HEADER    SMOOTH APPROACH BETWEEN SOURCE LIGAND AND DOCKED POSE"]
        for line in reference_lines:
            x, y, z = coordinates[atom_number]
            output_lines.append(
                f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            )
            atom_number += 1
        output_lines.append("END")
        frames.append(
            {
                "text": "\n".join(output_lines),
                "format": "pdb",
                "score": final_score if frame_index == frame_count - 1 else None,
                "model_number": frame_index + 1,
                "fraction": fraction,
                "label": (
                    "Source Ligand (Reference Coordinates)"
                    if frame_index == 0
                    else "Final Docked Conformation"
                    if frame_index == frame_count - 1
                    else f"Pocket Approach: Frame {frame_index + 1}/{frame_count}"
                ),
            }
        )
    return frames


def load_docking_view_data(
    compound: str,
    protein: str,
    pose_path: str | Path,
    final_score: float | None,
) -> DockingViewData:
    """Load authentic receptor and Vina docking conformations, preparing 3Dmol.js payloads."""
    from model.unseen_predictor import get_ligand, get_receptor

    receptor_path = get_receptor(protein)
    if receptor_path is None:
        raise FileNotFoundError(f"No receptor structure is available for {protein}.")
    receptor_text, receptor_format = _load_receptor(Path(receptor_path))

    pose_file = Path(pose_path)
    if not pose_file.is_file():
        raise FileNotFoundError(f"Docked pose file is not available: {pose_file}")
    pose_suffix = pose_file.suffix.lower()
    pose_text = pose_file.read_text(encoding="utf-8", errors="replace")

    if pose_suffix == ".pdbqt":
        raw_poses = _parse_vina_pose_models(pose_text)
    elif pose_suffix in {".pdb", ".ent"}:
        raw_poses = _parse_pdb_models(pose_text, final_score)
    elif pose_suffix in {".sdf", ".mol"}:
        raw_poses = [{"text": pose_text, "format": "sdf", "score": final_score, "model_number": 1}]
    elif pose_suffix == ".mol2":
        raw_poses = [{"text": pose_text, "format": "mol2", "score": final_score, "model_number": 1}]
    else:
        raise ValueError(f"Unsupported docked pose format: {pose_suffix or 'unknown'}")

    if not raw_poses:
        raise ValueError("The docking output contains no readable molecular poses.")

    # Calculate binding pocket for all poses
    for pose in raw_poses:
        if pose["format"] == "pdb":
            pose["pocket"] = _pocket_residues(receptor_text, pose["text"])
        else:
            pose["pocket"] = []

    # Check for reference source ligand coordinates to build smooth entry animation
    reference_path = get_ligand(compound)
    approach_frames = None
    if reference_path and Path(reference_path).is_file():
        ref_file = Path(reference_path)
        ref_text = ref_file.read_text(encoding="utf-8", errors="replace")
        if ref_file.suffix.lower() == ".pdbqt":
            ref_models = _parse_vina_pose_models(ref_text)
            if ref_models:
                approach_frames = _interpolate_reference_pose(
                    ref_models[0]["text"],
                    raw_poses[0]["text"],
                    final_score,
                    frame_count=18,
                )

    if approach_frames:
        for frame in approach_frames:
            frame["pocket"] = raw_poses[0]["pocket"]

    has_approach = approach_frames is not None and len(approach_frames) > 1

    if len(raw_poses) > 1:
        animation_available = True
        animation_kind = "vina_poses"
        animation_message = (
            f"AutoDock Vina produced {len(raw_poses)} conformational binding poses. "
            "Use the controls below to animate through the poses or inspect individual binding modes."
        )
    elif has_approach:
        animation_available = True
        animation_kind = "interpolated_reference_to_final"
        animation_message = (
            "Smooth approach transition between authentic reference ligand coordinates "
            "and final docked pose."
        )
    else:
        animation_available = False
        animation_kind = "static_final_pose"
        animation_message = (
            "Interactive docking animation is unavailable for this result. "
            "The final docked pose is shown instead."
        )

    return DockingViewData(
        receptor_text=receptor_text,
        receptor_format=receptor_format,
        ligand_name=compound,
        protein_name=protein,
        poses=tuple(raw_poses),
        approach_poses=tuple(approach_frames) if approach_frames else (),
        animation_available=animation_available,
        animation_kind=animation_kind,
        animation_message=animation_message,
        has_approach=has_approach,
        final_score=final_score,
    )


def build_viewer_html(view_data: DockingViewData) -> str:
    """Return an interactive 3Dmol.js molecular visualization component with playback and trajectory controls."""
    payload = {
        "receptor": view_data.receptor_text,
        "receptor_format": view_data.receptor_format,
        "poses": list(view_data.poses),
        "approach_poses": list(view_data.approach_poses),
        "has_approach": view_data.has_approach,
        "animation_available": view_data.animation_available,
        "ligand_name": view_data.ligand_name,
        "protein_name": view_data.protein_name,
        "final_score": view_data.final_score,
    }
    encoded_json = json.dumps(payload).replace("</", "<\\/")

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://3dmol.org/build/3Dmol-min.js"></script>
  <style>
    :root {{
      color-scheme: dark;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }}
    body {{
      margin: 0;
      background: #090e14;
      color: #e2e8f0;
      user-select: none;
      -webkit-user-select: none;
    }}
    .viewer-container {{
      border: 1px solid #1e293b;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
      background: #090e14;
      position: relative;
    }}
    #viewer {{
      width: 100%;
      height: 520px;
      position: relative;
      background: radial-gradient(circle at center, #111b27 0%, #080d12 100%);
    }}
    /* Top HUD Overlay */
    .hud-overlay {{
      position: absolute;
      top: 12px;
      left: 14px;
      right: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      pointer-events: none;
      z-index: 10;
    }}
    .hud-chip {{
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      border: 1px solid rgba(51, 65, 85, 0.7);
      border-radius: 20px;
      padding: 5px 14px;
      font-size: 0.82rem;
      font-weight: 600;
      color: #f1f5f9;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .hud-chip-accent {{
      color: #14b8a6;
      font-weight: 700;
    }}
    .hud-chip-gold {{
      color: #f59e0b;
      font-weight: 700;
    }}
    /* Control Toolbar */
    .control-toolbar {{
      background: #0f172a;
      border-top: 1px solid #1e293b;
      padding: 10px 14px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }}
    .btn-group {{
      display: inline-flex;
      border-radius: 6px;
      overflow: hidden;
      border: 1px solid #334155;
    }}
    .control-btn {{
      background: #1e293b;
      border: none;
      color: #e2e8f0;
      padding: 6px 12px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }}
    .control-btn:hover:not(:disabled) {{
      background: #0f766e;
      color: #ffffff;
    }}
    .control-btn:disabled {{
      opacity: 0.4;
      cursor: not-allowed;
    }}
    .control-btn.active {{
      background: #0f766e;
      color: #ffffff;
    }}
    .timeline-wrapper {{
      flex: 1 1 180px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 0 4px;
    }}
    #timeline {{
      width: 100%;
      height: 6px;
      accent-color: #14b8a6;
      cursor: pointer;
    }}
    .mode-switch {{
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 6px;
      color: #cbd5e1;
      padding: 5px 10px;
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
    }}
    .mode-switch:hover {{
      border-color: #14b8a6;
      color: #14b8a6;
    }}
    #error-msg {{
      padding: 12px;
      background: #450a0a;
      color: #fca5a5;
      font-size: 0.85rem;
    }}
    .interaction-hint {{
      position: absolute;
      bottom: 8px;
      right: 14px;
      font-size: 0.72rem;
      color: #64748b;
      pointer-events: none;
      z-index: 5;
    }}
  </style>
</head>
<body>
  <div class="viewer-container">
    <!-- Top HUD Bar -->
    <div class="hud-overlay">
      <div class="hud-chip">
        <span>Ligand:</span>
        <span class="hud-chip-accent" id="hud-ligand">{view_data.ligand_name}</span>
        <span style="color:#64748b; margin:0 2px;">&bull;</span>
        <span>Target:</span>
        <span class="hud-chip-accent" id="hud-protein">{view_data.protein_name}</span>
      </div>

      <div class="hud-chip">
        <span id="hud-mode-label">Mode: Vina Conformations</span>
        <span style="color:#64748b; margin:0 2px;">&bull;</span>
        <span id="hud-pose">Pose 1 of {len(view_data.poses)}</span>
        <span style="color:#64748b; margin:0 2px;">&bull;</span>
        <span class="hud-chip-gold" id="hud-score">
          {f"{view_data.final_score:.3f} kcal/mol" if view_data.final_score is not None else "Score N/A"}
        </span>
      </div>
    </div>

    <!-- 3Dmol WebGL Canvas -->
    <div id="viewer"></div>
    <div class="interaction-hint">Left drag: Rotate &bull; Scroll: Zoom &bull; Right drag: Pan</div>

    <!-- Interactive Animation & Control Toolbar -->
    <div class="control-toolbar">
      <!-- Playback Controls -->
      <div class="btn-group">
        <button id="btn-play" class="control-btn" title="Play animation">▶ Play</button>
        <button id="btn-pause" class="control-btn" title="Pause animation">⏸ Pause</button>
        <button id="btn-replay" class="control-btn" title="Replay from start">🔄 Replay</button>
      </div>

      <div class="btn-group">
        <button id="btn-prev" class="control-btn" title="Previous pose">⏪ Prev</button>
        <button id="btn-next" class="control-btn" title="Next pose">⏩ Next</button>
      </div>

      <!-- Scrubber Timeline -->
      <div class="timeline-wrapper">
        <input id="timeline" type="range" min="0" value="0" />
      </div>

      <!-- Animation Sequence Switcher (if both poses and approach exist) -->
      <div class="btn-group" id="group-mode" {"" if view_data.has_approach and len(view_data.poses) > 1 else 'style="display:none;"'}>
        <button id="btn-mode-poses" class="control-btn active" title="Show Vina docking modes">🎯 9 Modes</button>
        <button id="btn-mode-approach" class="control-btn" title="Show ligand approach into pocket">🚀 Approach</button>
      </div>

      <!-- Visual Helpers -->
      <button id="btn-pocket" class="control-btn active" title="Toggle active site pocket highlight">👁️ Pocket</button>
      <button id="btn-reset" class="control-btn" title="Reset camera view">🔍 Reset View</button>
    </div>

    <div id="error-msg" hidden></div>
  </div>

  <script id="viewer-data" type="application/json">{encoded_json}</script>

  <script>
    (function() {{
      const data = JSON.parse(document.getElementById('viewer-data').textContent);
      const viewerElem = document.getElementById('viewer');
      const errElem = document.getElementById('error-msg');

      if (!window.$3Dmol) {{
        errElem.hidden = false;
        errElem.textContent = '3Dmol.js could not be loaded. Please ensure internet access to 3dmol.org CDN.';
        return;
      }}

      // Initialize 3Dmol viewer
      const viewer = $3Dmol.createViewer(viewerElem, {{
        backgroundColor: '#090e14',
        antialias: true
      }});

      // Add receptor model (Protein)
      const receptorModel = viewer.addModel(data.receptor, data.receptor_format);
      receptorModel.setStyle({{}}, {{cartoon: {{color: '#8ba2b5', opacity: 0.90}}}});

      // Active dataset
      let activePoseSet = data.poses;
      let currentMode = 'poses'; // 'poses' or 'approach'
      let currentIndex = 0;
      let isPlaying = false;
      let timer = null;
      let showPocket = true;
      let ligandModel = null;

      const timeline = document.getElementById('timeline');
      const hudPose = document.getElementById('hud-pose');
      const hudScore = document.getElementById('hud-score');
      const hudModeLabel = document.getElementById('hud-mode-label');

      function updateTimelineLimits() {{
        timeline.max = String(Math.max(0, activePoseSet.length - 1));
        timeline.value = String(currentIndex);
      }}

      function updateHud() {{
        const pose = activePoseSet[currentIndex];
        hudPose.textContent = `Pose ${{currentIndex + 1}} of ${{activePoseSet.length}}`;
        if (pose && Number.isFinite(pose.score)) {{
          hudScore.textContent = `${{pose.score.toFixed(3)}} kcal/mol`;
        }} else if (currentMode === 'approach') {{
          hudScore.textContent = (currentIndex === activePoseSet.length - 1)
            ? `${{Number(data.final_score).toFixed(3)}} kcal/mol`
            : `Entry Step ${{currentIndex + 1}}`;
        }} else {{
          hudScore.textContent = data.final_score ? `${{Number(data.final_score).toFixed(3)}} kcal/mol` : 'Score N/A';
        }}
      }}

      function renderPocketResidues(pocketList) {{
        receptorModel.setStyle({{}}, {{cartoon: {{color: '#8ba2b5', opacity: 0.90}}}});
        if (showPocket && pocketList && pocketList.length > 0) {{
          pocketList.forEach(res => {{
            const sel = {{resi: Number(res.resi)}};
            if (res.chain) sel.chain = res.chain;
            receptorModel.setStyle(
              sel,
              {{
                cartoon: {{color: '#f59e0b'}},
                stick: {{color: '#f59e0b', radius: 0.16}}
              }},
              true
            );
          }});
        }}
      }}

      function setPose(index, recenter) {{
        currentIndex = Math.max(0, Math.min(activePoseSet.length - 1, index));
        const pose = activePoseSet[currentIndex];

        if (ligandModel) {{
          viewer.removeModel(ligandModel);
        }}

        ligandModel = viewer.addModel(pose.text, pose.format);
        // Ligand sticks representation with carbon coloring
        ligandModel.setStyle({{}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.22}}}});
        // Spheres for heteroatoms
        ['O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I'].forEach(elem => {{
          ligandModel.setStyle({{elem: elem}}, {{sphere: {{scale: 0.28}}}}, true);
        }});

        renderPocketResidues(pose.pocket || []);
        timeline.value = String(currentIndex);
        updateHud();

        if (recenter) {{
          viewer.zoomTo();
        }}
        viewer.render();
      }}

      function stop() {{
        isPlaying = false;
        if (timer !== null) {{
          clearInterval(timer);
          timer = null;
        }}
      }}

      function play() {{
        stop();
        if (!data.animation_available || activePoseSet.length < 2) return;
        isPlaying = true;
        const intervalMs = (currentMode === 'approach') ? 140 : 650;
        timer = setInterval(() => {{
          if (currentIndex >= activePoseSet.length - 1) {{
            stop();
            return;
          }}
          setPose(currentIndex + 1, false);
        }}, intervalMs);
      }}

      // Event listeners
      document.getElementById('btn-play').addEventListener('click', play);
      document.getElementById('btn-pause').addEventListener('click', stop);
      document.getElementById('btn-replay').addEventListener('click', () => {{
        stop();
        setPose(0, false);
        play();
      }});

      document.getElementById('btn-prev').addEventListener('click', () => {{
        stop();
        setPose(currentIndex - 1, false);
      }});

      document.getElementById('btn-next').addEventListener('click', () => {{
        stop();
        setPose(currentIndex + 1, false);
      }});

      timeline.addEventListener('input', e => {{
        stop();
        setPose(Number(e.target.value), false);
      }});

      document.getElementById('btn-reset').addEventListener('click', () => {{
        viewer.zoomTo();
        viewer.render();
      }});

      document.getElementById('btn-pocket').addEventListener('click', e => {{
        showPocket = !showPocket;
        e.target.classList.toggle('active', showPocket);
        const pose = activePoseSet[currentIndex];
        renderPocketResidues(pose ? pose.pocket : []);
        viewer.render();
      }});

      // Mode toggling
      const btnPoses = document.getElementById('btn-mode-poses');
      const btnApproach = document.getElementById('btn-mode-approach');

      if (btnPoses && btnApproach) {{
        btnPoses.addEventListener('click', () => {{
          stop();
          currentMode = 'poses';
          activePoseSet = data.poses;
          btnPoses.classList.add('active');
          btnApproach.classList.remove('active');
          hudModeLabel.textContent = 'Mode: Vina Conformations';
          updateTimelineLimits();
          setPose(0, false);
        }});

        btnApproach.addEventListener('click', () => {{
          stop();
          currentMode = 'approach';
          activePoseSet = data.approach_poses;
          btnApproach.classList.add('active');
          btnPoses.classList.remove('active');
          hudModeLabel.textContent = 'Mode: Ligand Approach';
          updateTimelineLimits();
          setPose(0, false);
          play();
        }});
      }}

      // Initial state
      updateTimelineLimits();
      setPose(0, true);

      // Disable playback controls if only 1 pose and no animation
      const playbackBtns = ['btn-play', 'btn-pause', 'btn-replay', 'btn-prev', 'btn-next', 'timeline'];
      if (!data.animation_available || activePoseSet.length < 2) {{
        playbackBtns.forEach(id => {{
          const el = document.getElementById(id);
          if (el) el.disabled = true;
        }});
      }}
    }})();
  </script>
</body>
</html>"""