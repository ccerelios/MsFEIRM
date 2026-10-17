"""Headless publication figures from saved CSV values and evaluation matrices."""

import os
from pathlib import Path
import textwrap

from msfeirm.config import PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


STYLE = {"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
         "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
         "svg.fonttype": "none", "savefig.facecolor": "white"}


def _save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=170)
    fig.savefig(path.with_suffix(".svg"), metadata={"Date": None})
    plt.close(fig)


def training_figure(training, validation, title, path):
    x = np.array([int(row["global_step"]) for row in training])
    window = min(100, len(x))
    def smooth(values):
        return np.convolve(values, np.ones(window) / window, mode="valid")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), layout="constrained")
        total = np.array([float(row["loss_total"]) for row in training])
        axes[0].plot(x, total, color="#ADB7C2", lw=.45, alpha=.45, label="Training steps")
        axes[0].plot(x[window-1:], smooth(total), color="#2166AC", lw=1.4, label=f"Training: {window}-step mean")
        axes[0].plot([int(r["global_step"]) for r in validation], [float(r["loss_total"]) for r in validation],
                     "o-", color="#D76C26", ms=3.2, lw=1.4, label="Validation")
        axes[0].set(title="Joint loss", xlabel="Optimizer step", ylabel="Loss")
        for key, label, color in [("loss_group", "Activity", "#2166AC"),
                                  ("loss_cardinality", "Cardinality", "#D76C26"),
                                  ("loss_member", "Membership", "#239080")]:
            values = np.array([float(row[key]) for row in training])
            axes[1].plot(x[window-1:], smooth(values), label=label, color=color, lw=1.3)
        axes[1].set(title=f"Training components: {window}-step mean", xlabel="Optimizer step", ylabel="Unweighted loss")
        for ax in axes:
            ax.grid(alpha=.18); ax.legend(frameon=False); ax.set_xlim(left=0)
        fig.suptitle(title)
        _save(fig, path)


def confusion_figure(matrix, labels, title, path):
    values = np.asarray(matrix, dtype=int)
    totals = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, totals, out=np.zeros_like(values, dtype=float), where=totals > 0)
    names = [textwrap.fill(name, 12) for name in labels]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.8, 7.7), layout="constrained")
        picture = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=40, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(names)), names)
        for row in range(len(names)):
            for column in range(len(names)):
                ax.text(column, row, str(values[row, column]), ha="center", va="center", fontsize=8,
                        color="white" if normalized[row, column] > .55 else "#22313F")
        ax.set(xlabel="Predicted subgroup activity", ylabel="Target subgroup activity",
               title=title + "\nMatched activity; row-normalized color, raw counts")
        fig.colorbar(picture, ax=ax, fraction=.046, pad=.025, label="Fraction within target class")
        _save(fig, path)


def member_figure(rows, path):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), layout="constrained")
        x = [int(row["members"]) for row in rows]
        for ax, columns, title in [
                (axes[0], [("segmentation_f1", "F1"), ("segmentation_map", "mAP"), ("iou_recall_auc", "IoU-recall AUC")], "Subgroup segmentation"),
                (axes[1], [("activity_f1_macro", "Macro F1"), ("activity_accuracy", "Accuracy"), ("activity_coverage", "Matching coverage")], "Matched subgroup activity")]:
            for (column, label), color in zip(columns, ["#2166AC", "#D76C26", "#239080"], strict=True):
                ax.plot(x, [row[column] for row in rows], "o-", color=color, ms=3.8, lw=1.5, label=label)
            ax.set(title=title, xlabel="Test members N", ylabel="Score", ylim=(0, 1.03), xticks=x)
            ax.grid(alpha=.18); ax.legend(frameon=False, loc="lower left")
        fig.suptitle("WBSensor · training N=15 · seed 42 · final epoch 20")
        _save(fig, path)
