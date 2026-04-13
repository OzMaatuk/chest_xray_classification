from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def save_training_curves(history: dict, title: str, output_path: str | Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(12, 4))

    loss_ax.plot(epochs, history["train_loss"], label="Train")
    loss_ax.plot(epochs, history["val_loss"], label="Val")
    loss_ax.set_title(f"{title} Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.legend()

    acc_ax.plot(epochs, history["train_acc"], label="Train")
    acc_ax.plot(epochs, history["val_acc"], label="Val")
    acc_ax.set_title(f"{title} Accuracy")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrix(confusion: list[list[int]], class_names: list[str], title: str, output_path: str | Path) -> None:
    figure, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=class_names, yticklabels=class_names)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

