from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.public_data_v1.build_topology_v7 import (
    PROJECT_ROOT,
    SourcePaths,
    _archive_v6,
    _build_graph_table,
    _default_sources,
    _generate_microbiome,
    _generate_survival_labels,
    _model_generated_modalities,
    _pivot_v6_graph,
    _quality_report,
    _resolve_paths,
    _sha256,
)


GENERATOR_VERSION = "topology_v7_hybrid_generator_v3"
PARENT_GENERATOR_VERSION = "topology_v7_hybrid_generator_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/research/topology_v7_generator_v3"
FORMAL_GENERATION_SEED = 20261001
PROTOCOL_LOCK_PATH = (
    PROJECT_ROOT / "experiments/topology_v7_generator_v3/generator_lock.json"
)


def _parent_v2_manifest() -> tuple[Path, dict[str, Any]]:
    path = PROJECT_ROOT / "data/research/topology_v7_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("generator_version") != PARENT_GENERATOR_VERSION:
        raise RuntimeError(f"Expected {PARENT_GENERATOR_VERSION} at {path}.")
    return path, manifest


def _quality_gate(label_metrics: dict[str, Any]) -> dict[str, Any]:
    group_values = [
        float(value)
        for value in label_metrics["generation_group_latent_risk_c_index"].values()
    ]
    checks = {
        "overall_latent_risk_c_index_0_72_to_0_80": bool(
            0.72 <= float(label_metrics["deterministic_latent_risk_c_index"]) <= 0.80
        ),
        "minimum_group_latent_risk_c_index_at_least_0_68": bool(min(group_values) >= 0.68),
        "group_latent_risk_c_index_spread_at_most_0_10": bool(
            max(group_values) - min(group_values) <= 0.10
        ),
        "event_noise_to_signal_ratio_1_10_to_1_40": bool(
            1.10 <= float(label_metrics["event_noise_to_signal_sd_ratio"]) <= 1.40
        ),
        "absolute_censor_risk_correlation_at_most_0_08": bool(
            abs(float(label_metrics["censor_log_time_latent_risk_correlation"])) <= 0.08
        ),
        "generated_event_rate_within_0_05_of_v6_target": bool(
            abs(
                float(label_metrics["generated_event_rate"])
                - float(label_metrics["target_event_rate_from_v6"])
            )
            <= 0.05
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "selection_policy": (
            "Generator parameters are selected using feature-domain harmonization and "
            "a separate development cohort. Formal-cohort subgroup thresholds are broad "
            "sampling-sanity checks and are never optimized by rerunning random seeds."
        ),
    }


def _validate_formal_protocol_lock(
    *,
    seed: int,
    local_anchor_weight: float,
    anchor_balance_searches: int,
    latent_noise_scale: float,
) -> dict[str, Any]:
    lock = json.loads(PROTOCOL_LOCK_PATH.read_text(encoding="utf-8"))
    expected = lock["locked_generator_parameters"]
    actual = {
        "formal_seed": int(seed),
        "local_anchor_weight": float(local_anchor_weight),
        "anchor_balance_searches": int(anchor_balance_searches),
        "latent_noise_scale": float(latent_noise_scale),
    }
    if actual != expected:
        raise RuntimeError(
            f"Formal generator parameters differ from the protocol lock: {actual} != {expected}"
        )
    if lock.get("status") != "locked_before_formal_generation":
        raise RuntimeError("Formal generator protocol is not locked.")
    return lock


def build_topology_v7_v3(
    *,
    sources: SourcePaths,
    output_dir: Path,
    archive_dir: Path,
    sample_count: int = 3600,
    seed: int = FORMAL_GENERATION_SEED,
    local_anchor_weight: float = 0.55,
    anchor_balance_searches: int = 20000,
    latent_noise_scale: float = 0.80,
    enforce_quality_gate: bool = True,
) -> dict[str, Any]:
    if sample_count < 100:
        raise ValueError("topology_v7 generator-v3 requires at least 100 generated samples.")
    for path in sources.__dict__.values():
        if not Path(path).exists():
            raise FileNotFoundError(path)

    protocol_lock = None
    if output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        protocol_lock = _validate_formal_protocol_lock(
            seed=seed,
            local_anchor_weight=local_anchor_weight,
            anchor_balance_searches=anchor_balance_searches,
            latent_noise_scale=latent_noise_scale,
        )

    parent_manifest_path, _ = _parent_v2_manifest()
    archived = _archive_v6(sources, archive_dir)
    public = pd.read_csv(sources.public_features)
    v6_graph = pd.read_csv(sources.v6_graph)
    v6_clinical = pd.read_csv(sources.v6_clinical)
    v6_metabolite = pd.read_csv(sources.v6_metabolite)
    v6_label = pd.read_csv(sources.v6_label)
    v6_abundance, v6_function, v6_edges = _pivot_v6_graph(v6_graph)

    abundance, oral_gut, provenance, microbiome_metrics = _generate_microbiome(
        public,
        v6_abundance,
        sample_count=sample_count,
        seed=seed,
        generation_groups=5,
        max_features_per_site=96,
        latent_components=8,
        anchor_shrinkage_to_class_mean=1.0 - local_anchor_weight,
        latent_noise_scale=latent_noise_scale,
        balance_target_by_group=True,
        balance_anchor_features=True,
        anchor_balance_searches=anchor_balance_searches,
        anchor_prior_strength=None,
        frozen_quantile_calibration=True,
    )
    observed_public_panel = np.asarray(
        microbiome_metrics.pop("observed_calibrated_panel"), dtype=float
    )
    clinical, metabolite, function_edges, modality_metrics = _model_generated_modalities(
        v6_abundance,
        v6_function,
        v6_edges,
        v6_clinical,
        v6_metabolite,
        abundance,
        provenance["generation_group_id"].to_numpy(dtype=int),
        observed_public_panel,
        seed=seed + 100,
    )
    graph = _build_graph_table(v6_graph, abundance, function_edges)
    label, label_metrics, survival_audit = _generate_survival_labels(
        v6_abundance,
        v6_function,
        v6_edges,
        v6_clinical,
        v6_metabolite,
        v6_label,
        abundance,
        function_edges,
        clinical,
        metabolite,
        provenance["generation_group_id"].to_numpy(dtype=int),
        seed=seed + 200,
        censor_location_mode="analytic_prior_calibration",
    )
    provenance = provenance.merge(
        survival_audit, on="sample_id", how="inner", validate="one_to_one"
    )
    provenance["label_source"] = "group_harmonized_multimodal_aft_survival_proxy"

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = _resolve_paths(output_dir)
    quality = _quality_report(graph, clinical, metabolite, label, provenance)
    if quality["num_samples"] != sample_count:
        raise RuntimeError("Generated sample count does not match the requested cohort size.")
    if quality["exact_duplicate_abundance_vectors"] != 0:
        raise RuntimeError("Generated microbiome panel contains exact duplicate vectors.")
    if quality["anchor_overlap_between_generation_groups"]:
        raise RuntimeError("Public anchors overlap between generation groups.")
    if not quality["finite_values"]:
        raise RuntimeError("Generated tables contain non-finite values.")

    quality_gate = _quality_gate(label_metrics)
    if enforce_quality_gate and not quality_gate["passed"]:
        failed = [name for name, passed in quality_gate["checks"].items() if not passed]
        raise RuntimeError(f"Generator-v3 quality gate failed: {failed}; metrics={label_metrics}")

    graph.to_csv(outputs.graph, index=False, float_format="%.8f")
    clinical.to_csv(outputs.clinical, index=False)
    metabolite.to_csv(outputs.metabolite, index=False, float_format="%.8f")
    label.to_csv(outputs.label, index=False)
    oral_gut.to_csv(outputs.oral_gut, index=False, float_format="%.10f")
    provenance.to_csv(outputs.provenance, index=False, float_format="%.8f")

    source_hashes = {
        Path(path).relative_to(PROJECT_ROOT).as_posix(): _sha256(Path(path))
        for path in sources.__dict__.values()
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_version": "topology_v7",
        "dataset_id": "topology_v7_generator_v3",
        "generator_version": GENERATOR_VERSION,
        "parent_generator_version": PARENT_GENERATOR_VERSION,
        "parent_v2_manifest_path": parent_manifest_path.relative_to(PROJECT_ROOT).as_posix(),
        "parent_v2_manifest_sha256": _sha256(parent_manifest_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "sample_count": int(sample_count),
        "observed_real_patient_count": int(len(public)),
        "observed_real_patient_rows_in_v7": 0,
        "dataset_class": "hybrid_model_generated_development_cohort",
        "sources": {
            "public_microbiome_anchor": "russo_crc_oral_gut_2023",
            "missing_modality_prior": "topology_v6",
            "sha256": source_hashes,
        },
        "generation": {
            "microbiome": {
                "method": (
                    "group_disjoint_class_conditional_pca_with_global_class_shrinkage"
                ),
                **microbiome_metrics,
            },
            "clinical_and_metabolite": {
                "method": "random_forest_conditional_prediction_with_oob_residual_sampling",
                **modality_metrics,
            },
            "function_score": "random_forest_prediction_from_v6_prior_targets",
            "edge_weight": "public_graphical_model_partial_association_with_sample_specific_modulation",
            "survival": {
                "method": (
                    "group_harmonized_multimodal_log_normal_aft_with_independent_noise_"
                    "and_analytic_prior_censoring"
                ),
                **label_metrics,
            },
        },
        "quality": quality,
        "quality_gate": quality_gate,
        "archive": archived,
        "prohibited_model_features": [
            column for column in survival_audit.columns if column != "sample_id"
        ],
        "limitations": [
            "All 3600 generator-v3 rows are model-generated development proxies, not observed patients.",
            "Only paired oral-gut microbiome anchors come from an open real cohort of 42 patients.",
            "Clinical variables, metabolites, function scores, topology, and outcomes inherit model priors.",
            "Generation-group balancing removes artificial simulation-batch effects; it is not a clinical claim.",
            "No audit-only latent risk, event noise, event time, or censor time may enter model features.",
            "Performance on this cohort measures generator recovery and requires external validation.",
        ],
    }
    if protocol_lock is not None:
        manifest["formal_protocol_lock"] = {
            "path": PROTOCOL_LOCK_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(PROTOCOL_LOCK_PATH),
            "development_benchmark_sha256": protocol_lock[
                "development_benchmark_sha256"
            ],
            "formal_seed_rerun_policy": protocol_lock["formal_seed_rerun_policy"],
        }
    manifest["outputs"] = {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path)
        for path in (
            outputs.graph,
            outputs.clinical,
            outputs.metabolite,
            outputs.label,
            outputs.oral_gut,
            outputs.provenance,
        )
    }
    outputs.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the independent topology_v7 generator-v3 development cohort."
    )
    parser.add_argument("--samples", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=FORMAL_GENERATION_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=PROJECT_ROOT / "archive/datasets/topology_v6",
    )
    parser.add_argument("--local-anchor-weight", type=float, default=0.55)
    parser.add_argument("--anchor-balance-searches", type=int, default=20000)
    parser.add_argument("--latent-noise-scale", type=float, default=0.80)
    parser.add_argument("--allow-quality-failure", action="store_true")
    args = parser.parse_args()

    manifest = build_topology_v7_v3(
        sources=_default_sources(),
        output_dir=args.output_dir.resolve(),
        archive_dir=args.archive_dir.resolve(),
        sample_count=args.samples,
        seed=args.seed,
        local_anchor_weight=args.local_anchor_weight,
        anchor_balance_searches=args.anchor_balance_searches,
        latent_noise_scale=args.latent_noise_scale,
        enforce_quality_gate=not args.allow_quality_failure,
    )
    print(
        json.dumps(
            {
                "dataset_id": manifest["dataset_id"],
                "generator_version": manifest["generator_version"],
                "sample_count": manifest["sample_count"],
                "quality_gate": manifest["quality_gate"],
                "survival": manifest["generation"]["survival"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
