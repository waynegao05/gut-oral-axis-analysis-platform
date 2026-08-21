from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

from research.data import build_dataset_from_csv, set_seed
from research.losses import (
    build_time_bin_edges,
    combined_survival_loss,
    discrete_time_nll_loss,
    pairwise_ranking_loss,
)
from research.metrics import concordance_index
from research.model_v2 import DeepStructureAwareGATCoxModelV2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_run_provenance(config: dict, config_path: Path, split_seed: int) -> dict:
    data_path_keys = [
        "graph_csv",
        "clinical_csv",
        "metabolite_csv",
        "label_csv",
        "provenance_csv",
    ]
    data_hashes = {
        key: {
            "path": Path(config["paths"][key]).as_posix(),
            "sha256": _sha256(Path(config["paths"][key])),
        }
        for key in data_path_keys
        if config["paths"].get(key)
    }
    manifest_path = Path(config["paths"]["manifest_json"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_hashes = manifest.get("outputs", {})
    for value in data_hashes.values():
        expected = declared_hashes.get(value["path"])
        if expected is None:
            raise RuntimeError(f"Dataset manifest does not declare input file: {value['path']}")
        if expected != value["sha256"]:
            raise RuntimeError(f"Dataset input hash does not match manifest: {value['path']}")

    source_paths = [
        Path(__file__),
        Path("research/data.py"),
        Path("research/losses.py"),
        Path("research/model_v2.py"),
        Path("research/metrics.py"),
        Path("research/repeat_runs_v2.py"),
    ]
    return {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_seed": int(split_seed),
        "model_seed": int(config["seed"]),
        "git_head": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_status_short": _git_value("status", "--short", "--untracked-files=no"),
        "config": {
            "path": config_path.as_posix(),
            "sha256": _sha256(config_path),
        },
        "dataset": {
            "manifest_path": manifest_path.as_posix(),
            "manifest_sha256": _sha256(manifest_path),
            "dataset_version": manifest.get("dataset_version"),
            "generator_version": manifest.get("generator_version"),
            "declared_output_hashes_verified": True,
            "inputs": data_hashes,
        },
        "source_files": {
            path.as_posix(): _sha256(path)
            for path in source_paths
        },
    }


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in the current environment.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("CUDA auto-selection failed. Use --device cpu only for explicit CPU debugging.")


def build_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def compute_survival_losses(
    output,
    batch,
    survival_head_type: str,
    time_bin_edges: torch.Tensor | None,
    ranking_weight: float,
    ranking_margin: float,
    cox_ties_method: str = "legacy",
):
    if survival_head_type == "discrete_time":
        if time_bin_edges is None:
            raise ValueError("time_bin_edges must be provided for the discrete_time survival head.")
        discrete_time_loss = discrete_time_nll_loss(
            time_logits=output["time_logits"],
            time=batch.time,
            event=batch.event,
            time_bin_edges=time_bin_edges.to(batch.time.device),
        )
        ranking_loss = pairwise_ranking_loss(
            risk=output["risk"],
            time=batch.time,
            event=batch.event,
            margin=ranking_margin,
        )
        zero = torch.zeros((), device=batch.time.device, dtype=batch.time.dtype)
        return {
            "total": discrete_time_loss + ranking_weight * ranking_loss,
            "cox": zero,
            "ranking": ranking_loss,
            "discrete_time": discrete_time_loss,
        }

    cox_losses = combined_survival_loss(
        risk=output["risk"],
        time=batch.time,
        event=batch.event,
        ranking_weight=ranking_weight,
        ranking_margin=ranking_margin,
        cox_ties_method=cox_ties_method,
    )
    return {
        "total": cox_losses["total"],
        "cox": cox_losses["cox"],
        "ranking": cox_losses["ranking"],
        "discrete_time": torch.zeros((), device=batch.time.device, dtype=batch.time.dtype),
    }


def compute_cohort_evaluation_losses(
    *,
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    survival_head_type: str,
    time_bin_edges: torch.Tensor | None,
    time_logits: torch.Tensor | None,
    ranking_weight: float,
    ranking_margin: float,
    graph_aux_loss: float,
    node_aux_loss: float,
    graph_aux_weight: float,
    node_aux_weight: float,
    cox_ties_method: str = "legacy",
) -> dict[str, float]:
    if survival_head_type == "discrete_time":
        if time_bin_edges is None or time_logits is None:
            raise ValueError("time_bin_edges and time_logits are required for discrete-time cohort evaluation.")
        discrete_loss = discrete_time_nll_loss(
            time_logits=time_logits,
            time=time,
            event=event,
            time_bin_edges=time_bin_edges.to(time.device),
        )
        ranking_loss = pairwise_ranking_loss(risk, time, event, margin=ranking_margin)
        cox_loss = torch.zeros((), device=risk.device, dtype=risk.dtype)
        survival_loss = discrete_loss + float(ranking_weight) * ranking_loss
    else:
        losses = combined_survival_loss(
            risk=risk,
            time=time,
            event=event,
            ranking_weight=ranking_weight,
            ranking_margin=ranking_margin,
            cox_ties_method=cox_ties_method,
        )
        survival_loss = losses["total"]
        cox_loss = losses["cox"]
        ranking_loss = losses["ranking"]
        discrete_loss = torch.zeros((), device=risk.device, dtype=risk.dtype)

    total_loss = (
        survival_loss
        + float(graph_aux_weight) * float(graph_aux_loss)
        + float(node_aux_weight) * float(node_aux_loss)
    )
    return {
        "cohort_loss": float(total_loss.item()),
        "cohort_survival_loss": float(survival_loss.item()),
        "cohort_cox_loss": float(cox_loss.item()),
        "cohort_ranking_loss": float(ranking_loss.item()),
        "cohort_discrete_time_loss": float(discrete_loss.item()),
    }


def evaluate(
    model,
    loader,
    device,
    survival_head_type: str,
    time_bin_edges: torch.Tensor | None,
    graph_aux_weight: float,
    node_aux_weight: float,
    ranking_weight: float,
    ranking_margin: float,
    cox_ties_method: str = "legacy",
):
    model.eval()
    all_time, all_event, all_risk = [], [], []
    all_time_logits = []
    losses = []
    cox_losses = []
    ranking_losses = []
    discrete_time_losses = []
    graph_aux_losses = []
    node_aux_losses = []
    batch_weights = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch, compute_contrastive=False)
            survival_losses = compute_survival_losses(
                output=output,
                batch=batch,
                survival_head_type=survival_head_type,
                time_bin_edges=time_bin_edges,
                ranking_weight=ranking_weight,
                ranking_margin=ranking_margin,
                cox_ties_method=cox_ties_method,
            )
            graph_aux = output["graph_aux_loss"]
            node_aux = output["node_aux_loss"]
            total_loss = survival_losses["total"] + graph_aux_weight * graph_aux + node_aux_weight * node_aux
            losses.append(float(total_loss.item()))
            cox_losses.append(float(survival_losses["cox"].item()))
            ranking_losses.append(float(survival_losses["ranking"].item()))
            discrete_time_losses.append(float(survival_losses["discrete_time"].item()))
            graph_aux_losses.append(float(graph_aux.item()))
            node_aux_losses.append(float(node_aux.item()))
            batch_weights.append(int(batch.num_graphs))
            all_time.extend(batch.time.cpu().numpy().tolist())
            all_event.extend(batch.event.cpu().numpy().tolist())
            all_risk.extend(output["risk"].cpu().numpy().tolist())
            if output.get("time_logits") is not None:
                all_time_logits.append(output["time_logits"].detach().cpu())
            del batch, output, survival_losses, graph_aux, node_aux, total_loss

    total_weight = max(sum(batch_weights), 1)
    weighted_graph_aux = sum(value * weight for value, weight in zip(graph_aux_losses, batch_weights)) / total_weight
    weighted_node_aux = sum(value * weight for value, weight in zip(node_aux_losses, batch_weights)) / total_weight
    cohort_metrics = compute_cohort_evaluation_losses(
        risk=torch.tensor(all_risk, dtype=torch.float32, device=device),
        time=torch.tensor(all_time, dtype=torch.float32, device=device),
        event=torch.tensor(all_event, dtype=torch.float32, device=device),
        survival_head_type=survival_head_type,
        time_bin_edges=time_bin_edges,
        time_logits=(torch.cat(all_time_logits, dim=0).to(device) if all_time_logits else None),
        ranking_weight=ranking_weight,
        ranking_margin=ranking_margin,
        graph_aux_loss=weighted_graph_aux,
        node_aux_loss=weighted_node_aux,
        graph_aux_weight=graph_aux_weight,
        node_aux_weight=node_aux_weight,
        cox_ties_method=cox_ties_method,
    )
    return {
        "head_type": survival_head_type,
        "loss": sum(losses) / max(len(losses), 1),
        "cox_loss": sum(cox_losses) / max(len(cox_losses), 1),
        "ranking_loss": sum(ranking_losses) / max(len(ranking_losses), 1),
        "discrete_time_loss": sum(discrete_time_losses) / max(len(discrete_time_losses), 1),
        "graph_aux_loss": sum(graph_aux_losses) / max(len(graph_aux_losses), 1),
        "node_aux_loss": sum(node_aux_losses) / max(len(node_aux_losses), 1),
        "c_index": concordance_index(all_time, all_event, all_risk),
        **cohort_metrics,
        "loss_note": (
            "loss/cox_loss are legacy per-batch averages; cohort_loss/cohort_cox_loss are computed over the full "
            "evaluation cohort and should be used for comparisons across batch-size experiments."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/research/research_config_v2.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--validation-group", default=None)
    parser.add_argument("--test-group", default=None)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--patience-override", type=int, default=None)
    parser.add_argument("--batch-size-override", type=int, default=None)
    parser.add_argument("--output-dir-override", default=None)
    args = parser.parse_args()
    training_started = time.perf_counter()
    torch.set_float32_matmul_precision("high")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs_override is not None:
        config["train"]["epochs"] = int(args.epochs_override)
    if args.patience_override is not None:
        config["train"]["early_stop_patience"] = int(args.patience_override)
    if args.batch_size_override is not None:
        config["train"]["batch_size"] = int(args.batch_size_override)
    if args.output_dir_override is not None:
        config["paths"]["output_dir"] = str(args.output_dir_override)
    set_seed(config["seed"])
    split_seed = args.split_seed
    if split_seed is None:
        split_seed = config["train"].get("split_seed")
    validation_group = args.validation_group
    if validation_group is None:
        validation_group = config["train"].get("validation_group")
    test_group = args.test_group
    if test_group is None:
        test_group = config["train"].get("test_group")
    graph_preprocess = config.get("graph_preprocess", {})
    tabular_preprocess = config.get("tabular_preprocess", {})
    survival_head_type = str(config["train"].get("survival_head_type", "cox"))
    cox_ties_method = str(config["train"].get("cox_ties_method", "legacy"))
    num_time_bins = int(config["train"].get("num_time_bins", 12))

    dataset = build_dataset_from_csv(
        graph_csv=config["paths"]["graph_csv"],
        clinical_csv=config["paths"]["clinical_csv"],
        metabolite_csv=config["paths"]["metabolite_csv"],
        label_csv=config["paths"]["label_csv"],
        node_feature_columns=config["model"]["node_feature_columns"],
        clinical_columns=config["model"]["clinical_columns"],
        metabolite_columns=config["model"]["metabolite_columns"],
        seed=config["seed"],
        split_seed=split_seed,
        keep_top_k_edges=graph_preprocess.get("keep_top_k_edges"),
        min_edge_weight=graph_preprocess.get("min_edge_weight"),
        standardize_tabular=bool(tabular_preprocess.get("standardize", False)),
        val_ratio=config["train"]["val_ratio"],
        test_ratio=config["train"]["test_ratio"],
        validation_group=validation_group,
        test_group=test_group,
    )
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_provenance = build_run_provenance(
        config,
        Path(args.config),
        int(split_seed if split_seed is not None else config["seed"]),
    )
    split_assignments = {
        "split_seed": int(split_seed if split_seed is not None else config["seed"]),
        "split_strategy": dataset.split_summary["split_strategy"],
        "train_sample_ids": [str(item.sample_id) for item in dataset.train_set],
        "validation_sample_ids": [str(item.sample_id) for item in dataset.val_set],
        "test_sample_ids": [str(item.sample_id) for item in dataset.test_set],
        "train_groups": dataset.split_summary.get("train_groups", []),
        "validation_groups": dataset.split_summary.get("val_groups", []),
        "test_groups": dataset.split_summary.get("test_groups", []),
        "requested_validation_group": validation_group,
        "requested_test_group": test_group,
    }
    (output_dir / "run_provenance.json").write_text(
        json.dumps(run_provenance, indent=2), encoding="utf-8"
    )
    (output_dir / "split_assignments.json").write_text(
        json.dumps(split_assignments, indent=2), encoding="utf-8"
    )
    if survival_head_type == "discrete_time":
        train_times = torch.tensor([float(item.time.item()) for item in dataset.train_set], dtype=torch.float32)
        time_bin_edges = build_time_bin_edges(train_times, num_bins=num_time_bins)
    else:
        time_bin_edges = None

    train_loader = DataLoader(dataset.train_set, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset.val_set, batch_size=config["train"]["batch_size"], shuffle=False)
    test_loader = DataLoader(dataset.test_set, batch_size=config["train"]["batch_size"], shuffle=False)

    device = resolve_device(args.device)
    model = DeepStructureAwareGATCoxModelV2(
        node_feature_dim=dataset.node_feature_dim,
        clinical_dim=dataset.clinical_dim,
        metabolite_dim=dataset.metabolite_dim,
        hidden_dim=config["train"]["hidden_dim"],
        heads=config["train"]["heads"],
        dropout=config["train"]["dropout"],
        edge_hidden_dim=config["train"].get("edge_hidden_dim", 24),
        num_layers=config["train"].get("num_layers", 4),
        layer_attn_heads=config["train"].get("layer_attn_heads", 4),
        contrastive_temperature=config["train"].get("contrastive_temperature", 0.2),
        survival_head_type=survival_head_type,
        num_time_bins=num_time_bins,
        use_layer_attention=bool(config["train"].get("use_layer_attention", False)),
        num_node_types=(
            dataset.num_node_types
            if int(config["model"].get("node_identity_dim", 0)) > 0
            else 0
        ),
        node_identity_dim=int(config["model"].get("node_identity_dim", 0)),
        identity_readout_dim=int(config["model"].get("identity_readout_dim", 0)),
        pool_every_layer=bool(config["model"].get("pool_every_layer", True)),
        graph_projection_dim=int(config["model"].get("graph_projection_dim", 0)),
        tabular_projection_dim=int(config["model"].get("tabular_projection_dim", 0)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = build_scheduler(
        optimizer,
        total_epochs=config["train"]["epochs"],
        warmup_epochs=config["train"].get("warmup_epochs", 10),
    )

    graph_aux_weight = float(config["train"].get("graph_aux_weight", 0.08))
    node_aux_weight = float(config["train"].get("node_aux_weight", 0.05))
    contrastive_weight = float(config["train"].get("contrastive_weight", 0.03))
    ranking_weight = float(config["train"].get("ranking_weight", 0.0))
    ranking_margin = float(config["train"].get("ranking_margin", 0.0))
    ranking_warmup_epochs = int(config["train"].get("ranking_warmup_epochs", 0))
    grad_clip_norm = float(config["train"].get("grad_clip_norm", 1.0))

    best_val = float("-inf")
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, config["train"]["epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        epoch_survival_losses = []
        epoch_cox_losses = []
        epoch_ranking_losses = []
        epoch_discrete_time_losses = []
        epoch_graph_aux = []
        epoch_node_aux = []
        epoch_contrastive = []

        if ranking_weight > 0.0 and ranking_warmup_epochs > 0:
            ranking_scale = min(1.0, float(epoch) / float(ranking_warmup_epochs))
        else:
            ranking_scale = 1.0
        effective_ranking_weight = ranking_weight * ranking_scale

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            output = model(batch, compute_contrastive=contrastive_weight > 0.0)
            survival_losses = compute_survival_losses(
                output=output,
                batch=batch,
                survival_head_type=survival_head_type,
                time_bin_edges=time_bin_edges,
                ranking_weight=effective_ranking_weight,
                ranking_margin=ranking_margin,
                cox_ties_method=cox_ties_method,
            )
            graph_aux = output["graph_aux_loss"]
            node_aux = output["node_aux_loss"]
            contrastive = output["contrastive_loss"]

            loss = (
                survival_losses["total"]
                + graph_aux_weight * graph_aux
                + node_aux_weight * node_aux
                + contrastive_weight * contrastive
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            epoch_survival_losses.append(float(survival_losses["total"].item()))
            epoch_cox_losses.append(float(survival_losses["cox"].item()))
            epoch_ranking_losses.append(float(survival_losses["ranking"].item()))
            epoch_discrete_time_losses.append(float(survival_losses["discrete_time"].item()))
            epoch_graph_aux.append(float(graph_aux.item()))
            epoch_node_aux.append(float(node_aux.item()))
            epoch_contrastive.append(float(contrastive.item()))
            del batch, output, survival_losses, graph_aux, node_aux, contrastive, loss

        scheduler.step()
        gc.collect()

        train_survival_loss = sum(epoch_survival_losses) / max(len(epoch_survival_losses), 1)
        train_cox_loss = sum(epoch_cox_losses) / max(len(epoch_cox_losses), 1)
        train_ranking_loss = sum(epoch_ranking_losses) / max(len(epoch_ranking_losses), 1)
        train_discrete_time_loss = sum(epoch_discrete_time_losses) / max(len(epoch_discrete_time_losses), 1)
        train_graph_aux = sum(epoch_graph_aux) / max(len(epoch_graph_aux), 1)
        train_node_aux = sum(epoch_node_aux) / max(len(epoch_node_aux), 1)
        train_contrastive = sum(epoch_contrastive) / max(len(epoch_contrastive), 1)

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            survival_head_type,
            time_bin_edges,
            graph_aux_weight,
            node_aux_weight,
            ranking_weight,
            ranking_margin,
            cox_ties_method,
        )
        history.append(
            {
                "epoch": epoch,
                "head_type": survival_head_type,
                "lr": optimizer.param_groups[0]["lr"],
                "train_survival_loss": train_survival_loss,
                "train_cox_loss": train_cox_loss,
                "train_ranking_loss": train_ranking_loss,
                "train_discrete_time_loss": train_discrete_time_loss,
                "train_graph_aux_loss": train_graph_aux,
                "train_node_aux_loss": train_node_aux,
                "train_contrastive_loss": train_contrastive,
                "effective_ranking_weight": effective_ranking_weight,
                "cox_ties_method": cox_ties_method,
                "epoch_seconds": time.perf_counter() - epoch_started,
                **val_metrics,
            }
        )

        if val_metrics["c_index"] > best_val + config["train"]["min_delta"]:
            best_val = val_metrics["c_index"]
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= config["train"]["early_stop_patience"]:
                break

    if best_state is not None:
        torch.save(best_state, output_dir / "best_model.pt")
        model.load_state_dict(best_state)
    if time_bin_edges is not None:
        (output_dir / "time_bins.json").write_text(
            json.dumps(time_bin_edges.cpu().tolist(), indent=2),
            encoding="utf-8",
        )
    (output_dir / "task_definition.json").write_text(
        json.dumps(dataset.task_definition, indent=2),
        encoding="utf-8",
    )
    (output_dir / "data_summary.json").write_text(
        json.dumps(dataset.data_summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "split_summary.json").write_text(
        json.dumps(dataset.split_summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    train_metrics = evaluate(
        model,
        train_loader,
        device,
        survival_head_type,
        time_bin_edges,
        graph_aux_weight,
        node_aux_weight,
        ranking_weight,
        ranking_margin,
        cox_ties_method,
    )
    validation_metrics = evaluate(
        model,
        val_loader,
        device,
        survival_head_type,
        time_bin_edges,
        graph_aux_weight,
        node_aux_weight,
        ranking_weight,
        ranking_margin,
        cox_ties_method,
    )
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        survival_head_type,
        time_bin_edges,
        graph_aux_weight,
        node_aux_weight,
        ranking_weight,
        ranking_margin,
        cox_ties_method,
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "train_metrics.json").write_text(json.dumps(train_metrics, indent=2), encoding="utf-8")
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    training_summary = {
        "epochs_run": len(history),
        "best_validation_c_index": best_val,
        "train_c_index": train_metrics["c_index"],
        "validation_c_index": validation_metrics["c_index"],
        "test_c_index": test_metrics["c_index"],
        "train_cohort_loss": train_metrics["cohort_loss"],
        "validation_cohort_loss": validation_metrics["cohort_loss"],
        "test_cohort_loss": test_metrics["cohort_loss"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": time.perf_counter() - training_started,
        "cox_ties_method": cox_ties_method,
        "node_identity_dim": int(config["model"].get("node_identity_dim", 0)),
        "identity_readout_dim": int(config["model"].get("identity_readout_dim", 0)),
        "pool_every_layer": bool(config["model"].get("pool_every_layer", True)),
        "graph_projection_dim": int(config["model"].get("graph_projection_dim", 0)),
        "tabular_projection_dim": int(config["model"].get("tabular_projection_dim", 0)),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2), encoding="utf-8"
    )
    run_provenance["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_provenance["training_summary"] = training_summary
    (output_dir / "run_provenance.json").write_text(
        json.dumps(run_provenance, indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "device": str(device),
                "split_seed": split_seed,
                "head_type": survival_head_type,
                "best_val_c_index": best_val,
                "task_name": dataset.task_definition["task_name"],
                "dataset_version": dataset.data_summary["dataset_origin"]["dataset_version"],
                "split_strategy": dataset.split_summary["split_strategy"],
                "test_metrics": test_metrics,
                "training_summary": training_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
