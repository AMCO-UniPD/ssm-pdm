"""Compact business metrics figure for use at paper publication size."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ceruleo.graphics.results import (
    plot_J_Cost,
    plot_unexpected_breaks,
    plot_unexploited_lifetime,
)


def plot_business_metrics_compact(
    exp_config,
    results_dict,
    plot_path,
    base_filename,
    figsize=None,
    font_size=8,
    legend_columns=None,
):
    """Draw UB, UL and J on one 1x3 figure, sharing a single model legend.

    Reuse the existing metric functions, including their curve styles, shading,
    window sampling and cost ratios. Dimensions are in inches; fonts are in pt.
    All three metrics are always drawn, irrespective of the legacy plot flags.
    Save vector PDF and a 300 dpi PNG preview and return the open figure.
    """
    if not results_dict:
        raise ValueError("At least one model is required to plot business metrics")
    if figsize is None:
        figsize = (7.2, 2.25 if len(results_dict) > 5 else 2.0)
    if legend_columns is None:
        legend_columns = (
            min(7, (len(results_dict) + 1) // 2)
            if len(results_dict) > 5
            else len(results_dict)
        )
    if min(figsize) <= 0 or font_size <= 1 or legend_columns < 1:
        raise ValueError(
            "Figure dimensions, font size and legend columns must be positive"
        )

    style = {
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=figsize, layout="constrained")
        axes[1].sharex(axes[0])
        common = dict(
            results_dict=results_dict,
            max_window=exp_config.max_windows,
            n=exp_config.n_maintenance_windows,
            add_shade=exp_config.add_shade,
        )
        plot_unexpected_breaks(ax=axes[0], **common)
        plot_unexploited_lifetime(ax=axes[1], log_scale=exp_config.log_scale, **common)
        plot_J_Cost(
            results=results_dict,
            window=exp_config.max_windows,
            step=exp_config.n_maintenance_windows,
            ax=axes[2],
            log_scale=exp_config.log_scale,
        )

        handles, labels = axes[0].get_legend_handles_labels()
        # Shorten display labels without changing model keys or style order.
        model_prefixes = {
            label.split("_quantile_")[0] for label in labels if "_quantile_" in label
        }
        quantiles_only = len(model_prefixes) == 1 and all(
            "_quantile_" in label for label in labels
        )

        def legend_label(label):
            if label in ("mean", "median"):
                return f"PvM {label}"
            if label.startswith("quantile_"):
                return rf"PvM $q_{{{label.removeprefix('quantile_')}}}$"
            if "_quantile_" in label:
                model, quantile = label.split("_quantile_", 1)
                prefix = "" if quantiles_only else f"{model} "
                return prefix + rf"$\tau={quantile}$"
            return label

        labels = [legend_label(label) for label in labels]
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            frameon=False,
            ncol=min(legend_columns, len(labels)),
            handlelength=2.5,
            columnspacing=1.1,
            handletextpad=0.4,
        )
        for i, (ax, title) in enumerate(
            zip(
                axes,
                (
                    "Unexpected breaks",
                    "Unexploited lifetime",
                    "Cost J",
                ),
            )
        ):
            ax.get_legend().remove()
            ax.set_title("")
            ax.set_title(f"({chr(97 + i)}) {title}", loc="left", pad=4)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=3, pad=2)
            ax.grid(axis="y", alpha=0.2, linewidth=0.5)
            ax.set_axisbelow(True)
            for line in ax.lines:
                line.set_linewidth(1.1)
                line.set_markersize(3)
            if i < 2:
                ax.set_xlabel("Maintenance window")
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            if ax.get_yscale() == "linear":
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axes[0].set_ylabel("Breakage risk")
        axes[1].set_ylabel("Unused lifetime")
        axes[2].set_ylabel("J")
        axes[2].set_xlabel(r"$c_{UL}/c_{UB}$")
        # The 1:40 label overlaps 1:120 on a paper-width panel. Keep the
        # original ratio range and values, with fewer labelled reference ticks.
        axes[2].set_xticks(
            [1 / 120, 1 / 20, 1 / 10, 1 / 5], labels=["1:120", "1:20", "1:10", "1:5"]
        )

        destination = Path(plot_path)
        destination.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "png"):
            output = (
                destination / f"{base_filename}_business_metrics_compact.{extension}"
            )
            fig.savefig(output, dpi=300)
            print(f"Plot saved at: {output}")
        return fig
