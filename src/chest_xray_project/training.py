from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


@dataclass
class EvaluationResult:
    loss: float
    accuracy: float
    predictions: list[int]
    labels: list[int]


def train_one_epoch(model, loader, criterion, optimizer, device: str, grad_clip_norm: float | None = None) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += predictions.eq(labels).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device: str) -> EvaluationResult:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += predictions.eq(labels).sum().item()
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return EvaluationResult(
        loss=running_loss / total,
        accuracy=correct / total,
        predictions=all_predictions,
        labels=all_labels,
    )


def build_optimizer(model, optimizer_config):
    if optimizer_config.name != "adam":
        raise ValueError(f"Unsupported optimizer: {optimizer_config.name}")

    return torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=optimizer_config.lr,
        weight_decay=optimizer_config.weight_decay,
    )


def build_scheduler(optimizer, training_config):
    if training_config.scheduler == "none":
        return None
    if training_config.scheduler != "reduce_on_plateau":
        raise ValueError(f"Unsupported scheduler: {training_config.scheduler}")

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=training_config.scheduler_factor,
        patience=training_config.scheduler_patience,
        min_lr=training_config.min_lr,
    )


def _current_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def fit(model, train_loader, val_loader, criterion, optimizer, training_config, model_name: str):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    scheduler = build_scheduler(optimizer, training_config)

    for epoch in range(1, training_config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            training_config.device,
            training_config.grad_clip_norm,
        )
        validation = evaluate(model, val_loader, criterion, training_config.device)
        if scheduler is not None:
            scheduler.step(validation.loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(validation.loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(validation.accuracy)

        improved = validation.loss < best_val_loss - 1e-4
        if improved:
            best_val_loss = validation.loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        marker = " *" if improved else ""
        print(
            f"[{model_name}] Epoch {epoch:02d}/{training_config.epochs} | "
            f"Train {train_loss:.4f}/{train_acc:.4f} | "
            f"Val {validation.loss:.4f}/{validation.accuracy:.4f} | "
            f"LR {_current_lr(optimizer):.2e} | "
            f"Patience {wait}/{training_config.patience}{marker}"
        )

        if wait >= training_config.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    return history


def summarize_evaluation(evaluation: EvaluationResult, class_names: list[str]) -> dict:
    return {
        "loss": evaluation.loss,
        "accuracy": accuracy_score(evaluation.labels, evaluation.predictions),
        "macro_f1": f1_score(evaluation.labels, evaluation.predictions, average="macro"),
        "classification_report": classification_report(
            evaluation.labels,
            evaluation.predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(evaluation.labels, evaluation.predictions).tolist(),
    }


def run_training(config, model, datamodule, class_names: list[str], output_dir):
    start_time = time.time()
    optimizer = build_optimizer(model, config.optimizer)
    history = fit(
        model=model,
        train_loader=datamodule.train_loader,
        val_loader=datamodule.val_loader,
        criterion=datamodule.criterion,
        optimizer=optimizer,
        training_config=config.training,
        model_name=config.name,
    )
    elapsed = time.time() - start_time

    val_evaluation = evaluate(model, datamodule.val_loader, datamodule.criterion, config.training.device)
    test_evaluation = evaluate(model, datamodule.test_loader, datamodule.criterion, config.training.device)

    return {
        "history": history,
        "train_time_seconds": elapsed,
        "validation": summarize_evaluation(val_evaluation, class_names),
        "test": summarize_evaluation(test_evaluation, class_names),
    }
