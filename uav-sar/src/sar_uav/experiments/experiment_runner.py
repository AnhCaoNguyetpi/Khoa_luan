"""Shared experiment execution and result archival logic.

Provides a single entry point for running the 5-tier experiment suite,
ensuring consistent run metadata, result archival, and reproducibility
regardless of whether invoked from CLI (`python -m sar_uav exp`) or
the standalone script (`scripts/run_experiments.py`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _get_git_info() -> Dict[str, Any]:
    """Capture comprehensive git and source repository provenance."""
    info: Dict[str, Any] = {}
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=5
        )
        if res.returncode == 0:
            full_hash = res.stdout.strip()
            info["git_hash"] = full_hash[:10]
            info["git_hash_full"] = full_hash
    except Exception:
        pass

    try:
        res = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=5
        )
        if res.returncode == 0:
            info["git_branch"] = res.stdout.strip()
    except Exception:
        pass

    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=5
        )
        if res.returncode == 0:
            info["git_is_dirty"] = bool(res.stdout.strip())
    except Exception:
        pass

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # Compute deterministic SHA256 fingerprint over active Python source tree
    try:
        h_src = hashlib.sha256()
        for py_path in sorted(PACKAGE_ROOT.rglob("*.py")):
            if "__pycache__" in py_path.parts:
                continue
            h_src.update(py_path.relative_to(PACKAGE_ROOT).as_posix().encode("utf-8"))
            try:
                h_src.update(py_path.read_bytes())
            except Exception:
                pass
        info["source_sha256"] = h_src.hexdigest()[:16]
    except Exception:
        pass

    # Compute fingerprint over scripts
    try:
        h_scripts = hashlib.sha256()
        scripts_dir = PROJECT_ROOT / "scripts"
        if scripts_dir.exists():
            for p in sorted(scripts_dir.glob("*.py")):
                h_scripts.update(p.name.encode("utf-8"))
                try:
                    h_scripts.update(p.read_bytes())
                except Exception:
                    pass
            info["scripts_sha256"] = h_scripts.hexdigest()[:16]
    except Exception:
        pass

    # Compute fingerprint over configs
    try:
        h_cfg = hashlib.sha256()
        configs_dir = PROJECT_ROOT / "configs"
        if configs_dir.exists():
            for p in sorted(configs_dir.glob("*.json")):
                h_cfg.update(p.name.encode("utf-8"))
                try:
                    h_cfg.update(p.read_bytes())
                except Exception:
                    pass
            info["configs_sha256"] = h_cfg.hexdigest()[:16]
    except Exception:
        pass

    # Composite project fingerprint
    composite = f"{info.get('source_sha256', '')}_{info.get('scripts_sha256', '')}_{info.get('configs_sha256', '')}"
    info["project_tree_sha256"] = hashlib.sha256(composite.encode("utf-8")).hexdigest()[:16]

    return info


def _build_run_metadata(
    run_id: str,
    tier: int,
    seed: int,
    quick: bool,
    elapsed: float,
    experiment_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build comprehensive metadata for a tier run."""
    git_info = _get_git_info()
    meta: Dict[str, Any] = {
        "tier": tier,
        "run_id": run_id,
        "seed": seed,
        "quick": quick,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": round(elapsed, 4),
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            **git_info
        },
    }
    if experiment_config:
        meta["experiment_config"] = experiment_config
    return meta


def run_experiment_suite(
    tier_selection: List[int],
    out_dir: Path,
    seed: int = 42,
    quick: bool = False,
) -> Dict[str, Any]:
    """Run selected experiment tiers and archive results with a unique run ID.

    Parameters
    ----------
    tier_selection : list of int
        Which tiers to run (1–5).
    out_dir : Path
        Base output directory (e.g. ``results/``).
    seed : int
        Master random seed for reproducibility.
    quick : bool
        If True, use reduced sizes for smoke testing.

    Returns
    -------
    dict
        Mapping ``tier_N`` → ``{metadata, results}`` for each completed tier.
    """
    from sar_uav.experiments import (
        run_tier1_mechanism_analysis,
        run_tier2_accuracy_benchmark,
        run_tier3_benchmark,
        run_tier4_misspecification,
        run_tier5_scalability,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    results_all: Dict[str, Any] = {}

    for t in tier_selection:
        t0 = time.time()
        log.info("Running Tier %d (Run ID: %s)...", t, run_id)

        experiment_config: Dict[str, Any] = {"seed": seed, "quick": quick}

        if t == 1:
            res = run_tier1_mechanism_analysis(seed=seed)
        elif t == 2:
            res = run_tier2_accuracy_benchmark(seed=seed)
        elif t == 3:
            n_m = 3 if quick else 15
            experiment_config.update(num_missions=n_m, H=15, num_uavs=2, max_evals=600)
            res = run_tier3_benchmark(num_missions=n_m, seed=seed)
        elif t == 4:
            n_m = 3 if quick else 10
            experiment_config.update(num_missions=n_m)
            res = run_tier4_misspecification(num_missions=n_m, seed=seed)
        elif t == 5:
            grids = [16] if quick else [16, 36, 64]
            experiment_config.update(grid_sizes=grids)
            res = run_tier5_scalability(grid_sizes=grids, seed=seed)
        else:
            log.warning("Unknown tier: %d", t)
            continue

        elapsed = time.time() - t0
        if isinstance(res, dict) and "experiment_config" in res:
            experiment_config.update(res["experiment_config"])

        meta = _build_run_metadata(run_id, t, seed, quick, elapsed, experiment_config)
        tier_payload = {"metadata": meta, "results": res}
        results_all[f"tier_{t}"] = tier_payload

        # Save latest (for plot scripts and quick access)
        latest_file = out_dir / f"tier_{t}_results.json"
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(tier_payload, f, indent=2)

        # Archive to prevent overwriting history
        archive_file = runs_dir / f"{run_id}_tier_{t}.json"
        with open(archive_file, "w", encoding="utf-8") as f:
            json.dump(tier_payload, f, indent=2)

        log.info("Tier %d saved to %s (%.2fs)", t, latest_file, elapsed)

    # Summary file
    summary_file = out_dir / "all_tiers_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2)

    # Manifest for this run
    git_info = _get_git_info()
    manifest = {
        "run_id": run_id,
        "tiers_completed": tier_selection,
        "seed": seed,
        "quick": quick,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            **git_info
        },
    }
    manifest_file = runs_dir / f"{run_id}_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log.info("All tiers completed for %s. Summary: %s, Manifest: %s",
             run_id, summary_file, manifest_file)
    return results_all
