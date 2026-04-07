from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch_geometric.loader import DataLoader

from research.losses import cox_ph_loss
from research.metrics import concordance_index


@dataclass
class TrainerResult:
    best_val_c_index: float
    history: List[Dict[str, float]]


class CoxTrainer:
    def __init__(self, model, optimizer, device, min_delta: float = 1e-4, patience: int = 15):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.min_delta = min_delta
        self.patience = patience

    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        losses = []
        all_time, all_event, all_risk = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                loss = cox_ph_loss(output["risk"], batch.time, batch.event)
                losses.append(float(loss.item()))
                all_time.extend(batch.time.cpu().numpy().tolist())
                all_event.extend(batch.event.cpu().numpy().tolist())
                all_risk.extend(output["risk"].cpu().numpy().tolist())
        return {
            "loss": sum(losses) / max(len(losses), 1),
            "c_index": concordance_index(all_time, all_event, all_risk),
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        best_val = float("-inf")
        best_state = None
        stop_counter = 0
        history: List[Dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_losses = []
            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = cox_ph_loss(output["risk"], batch.time, batch.event)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(float(loss.item()))

            train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            val_metrics = self._evaluate(val_loader)
            history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

            if val_metrics["c_index"] > best_val + self.min_delta:
                best_val = val_metrics["c_index"]
                best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                stop_counter = 0
            else:
                stop_counter += 1
                if stop_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return TrainerResult(best_val_c_index=best_val, history=history)

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        return self._evaluate(loader)
