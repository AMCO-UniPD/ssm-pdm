"""Generate quantile and business plots for the validation-selected fold."""

import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "src"))
sys.path.insert(0, str(ROOT.parents[1] / "src/ceruleo"))
from phm_monitor import best_fold, emit
from phm_dashboard import read_events
from ceruleo.graphics.results import plot_J_Cost, plot_unexpected_breaks, plot_unexploited_lifetime
from ceruleo.results.results import PredictionResult


def load_latest(directory):
    paths = list(directory.glob("*.pickle"))
    if not paths:
        raise FileNotFoundError(f"No predictions found in {directory}")
    with max(paths, key=lambda path: path.stat().st_mtime_ns).open("rb") as stream:
        return pickle.load(stream)


def generate(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    config, settings = manifest["config"], manifest["settings"]
    events = read_events(directory)
    folds = {event["fold"] for event in events if event["kind"] == "fold_completed"}
    if len(folds) != config["n_folds"]:
        raise ValueError("Best-fold plots require all folds to complete")
    selected = best_fold(events)
    fold = selected["fold"]
    emit("best_fold_selected", best_fold=fold, validation_loss=selected["validation_loss"])
    print(f"Selected fold {fold}: mean checkpoint validation loss {selected['validation_loss']}")
    outputs = ROOT / "combined_outputs" / settings["model_name"] / settings["failure_type"] / config["approach"] / manifest["campaign_id"] / f"fold_{fold}"
    combined = load_latest(outputs)
    plot_dir = directory / "plots"
    plot_dir.mkdir(exist_ok=True)

    def save(figure, filename, title):
        figure.savefig(plot_dir / filename, dpi=150, bbox_inches="tight")
        figure.savefig(plot_dir / Path(filename).with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)
        emit("plot", filename=filename, title=f"Fold {fold} · {title}")

    indices = selected["test_idx"]
    lives = config["plot_life_idx"]
    for last in dict.fromkeys((0, config["n_last_samples"])):
        fig, axes = plt.subplots(config["nrows"], config["ncols"], figsize=(16, 9), squeeze=False)
        if len(lives) > axes.size:
            raise ValueError("Plot grid is smaller than selected lives")
        for ax, life in zip(axes.flat, lives):
            i = indices.index(life)
            true = np.asarray(combined["y_true"][i]).reshape(-1)
            predictions = {tau: np.asarray(combined[f"pred_quantile_{tau}"][i]).reshape(-1)
                           for tau in config["quantiles"]}
            if last:
                true = true[-last:]
                predictions = {tau: pred[-last:] for tau, pred in predictions.items()}
            valid = true != 0 if not config["full_life"] else np.ones(true.shape, dtype=bool)
            ax.plot(true[valid], color="black", label="True RUL")
            for tau, pred in predictions.items():
                ax.plot(pred[valid], label=f"τ={tau}", alpha=0.85)
            low, high = min(predictions), max(predictions)
            ax.fill_between(np.arange(valid.sum()), predictions[low][valid], predictions[high][valid], alpha=0.12)
            ax.set(title=f"Life {life}", xlabel="Time step", ylabel="RUL")
            ax.legend()
        for ax in list(axes.flat)[len(lives):]:
            ax.set_visible(False)
        fig.suptitle(f"{manifest['campaign_id']} · fold {fold}")
        fig.tight_layout()
        save(fig, f"fold_{fold}_quantiles_last_{last}.png", "Quantile predictions" + (f" · last {last} samples" if last else " · full sequence"))

    results = {}
    for tau in config["quantiles"]:
        raw = load_latest(outputs / f"quantile_{tau}")
        predictions = []
        for i in config["life_idx"]:
            true = np.asarray(raw["y_true"][i]).reshape(-1)
            pred = np.asarray(raw["y_pred"][i]).reshape(-1)
            # Keep the first terminal zero, remove later padding (existing pipeline convention).
            valid = np.ones(true.shape, dtype=bool)
            valid[np.flatnonzero(true == 0)[1:]] = False
            predictions.append(PredictionResult(name=f"Life_{indices[i]}", true_RUL=true[valid], predicted_RUL=pred[valid]))
        results[f"τ={tau}"] = predictions
    common = {"max_window": config["max_windows"], "n": config["n_maintenance_windows"]}
    for function, name, title in (
        (plot_unexpected_breaks, "unexpected_breaks", "Unexpected breaks"),
        (plot_unexploited_lifetime, "unexploited_lifetime", "Unexploited lifetime"),
    ):
        ax = function(results_dict=results, **common)
        save(ax.figure, f"fold_{fold}_{name}.png", title)
    ax = plot_J_Cost(results=results, window=config["max_windows"], step=config["n_maintenance_windows"])
    save(ax.figure, f"fold_{fold}_business_cost.png", "Business cost J")


if __name__ == "__main__":
    generate(Path(os.environ["PHM_MONITOR_DIR"]))
