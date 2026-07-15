from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

from ctm_fusion_experiment.evaluate_baseline_v9_oof import build_baseline_v9_oof_summary
from ctm_fusion_experiment.train import (
    _evaluate_fusion,
    _export_graph_embeddings,
    _train_fusion,
    _train_graph_encoder,
    resolve_device,
    set_seed,
)
from ctm_fusion_experiment.train_baseline_v5 import (
    _candidate_record,
    _selected_metrics,
    _selection_prediction_rows,
    _train_pairwise_baseline,
)
from ctm_fusion_experiment.utils.data_loader import (
    FoldSplit,
    FusionArraySet,
    FusionArrays,
    load_graph_dataset,
    make_cv_splits,
    prepare_fusion_arrays,
)
from ctm_fusion_experiment.utils.metrics import concordance_index
from ctm_fusion_experiment.utils.reporting import save_embeddings, write_csv, write_json
from ctm_fusion_experiment.utils.risk_ensemble_selection import (
    apply_risk_ensemble,
    choose_cindex_risk_ensemble,
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    graph_seed: int
    baseline_seed: int
    is_reference: bool = False


def run_baseline_v9_oof_experiment(config_path: str, device_arg: str = "auto") -> dict[str, object]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    device = resolve_device(device_arg)
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config_snapshot.json", config)
    bundle = load_graph_dataset(config, sample_limit=config.get("runtime", {}).get("sample_limit"))
    graph_seeds = [int(seed) for seed in config["baseline_v5"]["graph_seeds"]]
    baseline_seeds = [int(seed) for seed in config["baseline_v5"]["baseline_seeds"]]
    write_json(
        output_dir / "run_metadata.json",
        {
            "device": str(device),
            "sample_count": int(len(bundle.sample_table)),
            "dataset_version": "topology_v6",
            "dataset_is_synthetic_noisy_augmented": True,
            "flow": "baseline_v9_head_oof_stacking",
            "graph_seed_count": len(graph_seeds),
            "baseline_seed_count": len(baseline_seeds),
            "inner_oof_folds": int(config["baseline_v9"]["inner_folds"]),
        },
    )
    splits = make_cv_splits(
        bundle.sample_table,
        folds=int(config["cv"]["folds"]),
        seed=int(config["seed"]),
        val_ratio=float(config["cv"]["val_ratio"]),
    )
    max_folds = config.get("runtime", {}).get("max_folds")
    if max_folds is not None:
        splits = splits[: int(max_folds)]

    for split in splits:
        fold_dir = output_dir / f"fold_{split.fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_json(fold_dir / "split.json", split.__dict__)
        graph_summaries: list[dict[str, Any]] = []
        candidate_records: list[dict[str, Any]] = []
        candidate_specs: list[CandidateSpec] = []
        oof_risk_by_candidate: dict[str, dict[str, float]] = {}
        oof_arrays: FusionArraySet | None = None
        val_arrays: FusionArraySet | None = None
        test_arrays: FusionArraySet | None = None

        for graph_seed in graph_seeds:
            set_seed(_graph_seed_value(int(config["seed"]), split.fold, graph_seed))
            graph_dir = fold_dir / f"graph_seed_{graph_seed}"
            graph_dir.mkdir(parents=True, exist_ok=True)
            graph_model, graph_summary = _train_graph_encoder(bundle, split, config, device)
            torch.save(graph_model.state_dict(), graph_dir / "graph_encoder.pt")
            export_ids = tuple(sorted(set(split.train_ids + split.val_ids + split.test_ids)))
            embeddings = _export_graph_embeddings(
                graph_model,
                bundle,
                export_ids,
                int(config["graph_encoder"]["batch_size"]),
                device,
            )
            save_embeddings(graph_dir / "frozen_graph_embeddings.npz", embeddings)
            arrays = prepare_fusion_arrays(
                sample_table=bundle.sample_table,
                split=split,
                graph_embeddings=embeddings,
                clinical_columns=config["model"]["clinical_columns"],
                metabolite_columns=config["model"]["metabolite_columns"],
            )
            oof_arrays = arrays.train
            val_arrays = arrays.val
            test_arrays = arrays.test
            graph_summary.pop("history")
            graph_summary["graph_seed"] = graph_seed
            graph_summaries.append(graph_summary)

            if graph_seed == graph_seeds[0]:
                set_seed(int(config["seed"]) + split.fold)
                reference_model, reference_summary, reference_test_details = _train_fusion("baseline", arrays, config, device)
                reference_val_metrics, reference_val_details = _evaluate_fusion(
                    reference_model,
                    arrays.val,
                    "baseline",
                    int(config["fusion"]["batch_size"]),
                    device,
                )
                torch.save(reference_model.state_dict(), graph_dir / "reference_baseline.pt")
                write_json(graph_dir / "reference_baseline_history.json", reference_summary["history"])
                reference_summary["validation"] = reference_val_metrics
                reference_summary.pop("history")
                candidate_records.append(
                    _candidate_record(
                        name="reference",
                        graph_seed=graph_seed,
                        baseline_seed=0,
                        summary=reference_summary,
                        val_details=reference_val_details,
                        test_details=reference_test_details,
                    )
                )
                candidate_specs.append(CandidateSpec("reference", graph_seed, 0, is_reference=True))
                oof_risk_by_candidate["reference"] = _collect_oof_risk_for_spec(
                    arrays.train,
                    CandidateSpec("reference", graph_seed, 0, is_reference=True),
                    config,
                    split.fold,
                    device,
                )

            for baseline_seed in baseline_seeds:
                spec = CandidateSpec(f"g{graph_seed}_b{baseline_seed}", graph_seed, baseline_seed)
                set_seed(int(config["seed"]) + split.fold * 10000 + graph_seed * 100 + baseline_seed)
                candidate_dir = graph_dir / f"baseline_seed_{baseline_seed}"
                candidate_dir.mkdir(parents=True, exist_ok=True)
                model, summary, val_details, test_details = _train_pairwise_baseline(arrays, config, device)
                torch.save(model.state_dict(), candidate_dir / "baseline_v9.pt")
                write_json(candidate_dir / "history.json", summary["history"])
                summary.pop("history")
                candidate_records.append(
                    _candidate_record(
                        name=spec.name,
                        graph_seed=graph_seed,
                        baseline_seed=baseline_seed,
                        summary=summary,
                        val_details=val_details,
                        test_details=test_details,
                    )
                )
                candidate_specs.append(spec)
                oof_risk_by_candidate[spec.name] = _collect_oof_risk_for_spec(
                    arrays.train,
                    spec,
                    config,
                    split.fold,
                    device,
                )

        if oof_arrays is None or val_arrays is None or test_arrays is None:
            raise RuntimeError("No arrays were prepared.")

        candidate_names = [candidate["name"] for candidate in candidate_records]
        oof_matrix = torch.tensor(
            [
                [oof_risk_by_candidate[name][sample_id] for sample_id in oof_arrays.sample_ids]
                for name in candidate_names
            ],
            dtype=torch.float32,
        )
        selection = choose_cindex_risk_ensemble(
            risk_matrix=oof_matrix,
            times=torch.tensor(oof_arrays.time, dtype=torch.float32),
            events=torch.tensor(oof_arrays.event, dtype=torch.float32),
            model_names=candidate_names,
            reference_index=0,
            min_c_index_delta=float(config["baseline_v9"]["min_oof_c_index_delta"]),
            softmax_temperature=float(config["baseline_v5"]["softmax_temperature"]),
            candidate_policy=config["baseline_v5"].get("selection_policy", "ensemble_only_or_reference"),
            max_top_k=int(config["baseline_v5"].get("max_top_k", 5)),
        )
        val_matrix = torch.tensor([candidate["val_risk"] for candidate in candidate_records], dtype=torch.float32)
        val_reference = torch.tensor(candidate_records[0]["val_risk"], dtype=torch.float32)
        selected_val_risk = apply_risk_ensemble(val_matrix, selection.weights, selection.risk_means, selection.risk_stds)
        val_selected_c_index = concordance_index(val_arrays.time, val_arrays.event, selected_val_risk.detach().cpu().numpy())
        val_reference_c_index = concordance_index(val_arrays.time, val_arrays.event, val_reference.detach().cpu().numpy())
        accepted_val_c_index = val_selected_c_index
        if val_selected_c_index - val_reference_c_index < float(config["baseline_v9"]["min_validation_gate_delta"]):
            selection_weights = [0.0 for _ in candidate_records]
            selection_weights[0] = 1.0
            selected_candidate_name = "reference"
            selection_c_index = selection.reference_c_index
            accepted_val_c_index = val_reference_c_index
        else:
            selection_weights = selection.weights
            selected_candidate_name = selection.candidate_name
            selection_c_index = selection.c_index

        test_matrix = torch.tensor([candidate["test_risk"] for candidate in candidate_records], dtype=torch.float32)
        test_reference = torch.tensor(candidate_records[0]["test_risk"], dtype=torch.float32)
        selected_test_risk = apply_risk_ensemble(test_matrix, selection_weights, selection.risk_means, selection.risk_stds)
        selected_metrics, diagnostics = _selected_metrics(test_arrays, test_reference, selected_test_risk)
        write_json(
            fold_dir / "baseline_v9_oof_selection.json",
            {
                "candidate_name": selected_candidate_name,
                "weights": selection_weights,
                "oof_c_index": selection_c_index,
                "oof_reference_c_index": selection.reference_c_index,
                "validation_c_index": accepted_val_c_index,
                "validation_reference_c_index": val_reference_c_index,
                "oof_selected_validation_c_index": val_selected_c_index,
                "risk_means": selection.risk_means,
                "risk_stds": selection.risk_stds,
                "candidates": selection.candidates,
                "test": selected_metrics,
                "pair_diagnostics": diagnostics,
            },
        )
        write_csv(
            fold_dir / "test_predictions.csv",
            _selection_prediction_rows(test_arrays, candidate_records, selected_test_risk.detach().cpu().tolist()),
        )
        candidate_summaries = [
            {
                "name": candidate["name"],
                "graph_seed": candidate["graph_seed"],
                "baseline_seed": candidate["baseline_seed"],
                "training_seconds": candidate["training_seconds"],
                "parameters": candidate["parameters"],
                "validation": candidate["validation"],
                "test": candidate["test"],
            }
            for candidate in candidate_records
        ]
        write_json(
            fold_dir / "fold_summary.json",
            {
                "fold": split.fold,
                "graph_encoders": graph_summaries,
                "reference_baseline": candidate_summaries[0],
                "candidate_models": candidate_summaries,
                "baseline_v5_selected": {
                    "candidate_name": selected_candidate_name,
                    "weights": selection_weights,
                    "validation_c_index": accepted_val_c_index,
                    "validation_reference_c_index": val_reference_c_index,
                    "oof_selected_validation_c_index": val_selected_c_index,
                    "oof_c_index": selection_c_index,
                    "oof_reference_c_index": selection.reference_c_index,
                    "candidate_count": len(selection.candidates),
                    "test": selected_metrics,
                    "pair_diagnostics": diagnostics,
                },
            },
        )

    return build_baseline_v9_oof_summary(output_dir)


def _collect_oof_risk_for_spec(
    arrays: FusionArraySet,
    spec: CandidateSpec,
    config: dict[str, Any],
    outer_fold: int,
    device: torch.device,
) -> dict[str, float]:
    result: dict[str, float] = {}
    inner_folds = _make_inner_splits(
        arrays,
        folds=int(config["baseline_v9"]["inner_folds"]),
        val_ratio=float(config["baseline_v9"]["inner_val_ratio"]),
        seed=int(config["seed"]) + int(outer_fold) * 1000 + spec.graph_seed + spec.baseline_seed,
    )
    for inner_index, inner_split in enumerate(inner_folds, start=1):
        inner_arrays = _subset_arrays(arrays, inner_split)
        set_seed(int(config["seed"]) + outer_fold * 100000 + inner_index * 1000 + spec.graph_seed * 10 + spec.baseline_seed)
        if spec.is_reference:
            _, _, details = _train_fusion("baseline", inner_arrays, config, device)
        else:
            _, _, _, details = _train_pairwise_baseline(inner_arrays, config, device)
        for sample_id, risk in zip(inner_arrays.test.sample_ids, details["risk"]):
            result[sample_id] = float(risk)
    missing = set(arrays.sample_ids) - set(result)
    if missing:
        raise RuntimeError(f"OOF risk is missing {len(missing)} samples for {spec.name}.")
    return result


def _make_inner_splits(
    arrays: FusionArraySet,
    *,
    folds: int,
    val_ratio: float,
    seed: int,
) -> list[FoldSplit]:
    sample_ids = np.asarray(arrays.sample_ids)
    events = arrays.event.astype(int)
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    splits = []
    for fold, (train_val_index, test_index) in enumerate(splitter.split(sample_ids, events), start=1):
        train_val_ids = sample_ids[train_val_index]
        train_val_events = events[train_val_index]
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=float(val_ratio),
            random_state=int(seed) + fold,
            shuffle=True,
            stratify=train_val_events,
        )
        splits.append(
            FoldSplit(
                fold=fold,
                train_ids=tuple(sorted(str(sample_id) for sample_id in train_ids.tolist())),
                val_ids=tuple(sorted(str(sample_id) for sample_id in val_ids.tolist())),
                test_ids=tuple(sorted(str(sample_id) for sample_id in sample_ids[test_index].tolist())),
            )
        )
    return splits


def _subset_arrays(arrays: FusionArraySet, split: FoldSplit) -> FusionArrays:
    by_id = {sample_id: index for index, sample_id in enumerate(arrays.sample_ids)}

    def subset(sample_ids: Sequence[str]) -> FusionArraySet:
        indices = np.asarray([by_id[str(sample_id)] for sample_id in sample_ids], dtype=int)
        return FusionArraySet(
            sample_ids=tuple(str(sample_id) for sample_id in sample_ids),
            graph=arrays.graph[indices],
            clinical=arrays.clinical[indices],
            metabolite=arrays.metabolite[indices],
            time=arrays.time[indices],
            event=arrays.event[indices],
        )

    return FusionArrays(
        train=subset(split.train_ids),
        val=subset(split.val_ids),
        test=subset(split.test_ids),
    )


def _graph_seed_value(base_seed: int, fold: int, graph_seed: int) -> int:
    if graph_seed == 0:
        return base_seed + fold
    return base_seed + fold * 10000 + graph_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ctm_fusion_experiment/configs/baseline_v9_oof.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    print(json.dumps(run_baseline_v9_oof_experiment(args.config, args.device), indent=2))


if __name__ == "__main__":
    main()
