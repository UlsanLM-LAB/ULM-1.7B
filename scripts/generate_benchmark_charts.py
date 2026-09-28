"""Generate high-quality benchmark comparison charts for UlsanBench v2.

Produces:
- assets/benchmarks/ulsanbench-model-comparison.svg
- assets/benchmarks/ulsanbench-model-comparison.png

Follows strict guidelines:
- Same 0-100 scale
- Honest metric representation (x100 semantic/dialectness, % identification)
- Grouped horizontal bar chart for maximum readability
- Clean modern aesthetics with sans-serif font fallbacks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Metric definitions: (task_key, sub_metric, label_title, multiplier)
METRICS = [
    ("generation", "dialectness_proxy", "Generation Dialectness ×100", 100.0),
    ("grammar", "dialectness_proxy", "Grammar Dialectness ×100", 100.0),
    ("context", "semantic_similarity", "Context Semantic ×100", 100.0),
    ("identification", "accuracy", "Identification Accuracy %", 100.0),
    ("comprehension", "semantic_similarity", "Comprehension Semantic ×100", 100.0),
    ("generation", "semantic_similarity", "Generation Semantic ×100", 100.0),
]

MODEL_SPECS = [
    {
        "id": "ulm-4b-arm-b-neutral",
        "label": "ULM-4B Arm B (Ours)",
        "color": "#1E40AF",  # Deep royal blue
        "edgecolor": "#1E3A8A",
        "hatch": "",
    },
    {
        "id": "qwen3.8-4b-base",
        "label": "Qwen3.8-4B Base",
        "color": "#64748B",  # Slate gray
        "edgecolor": "#475569",
        "hatch": "",
    },
    {
        "id": "qwen2.5-3b-instruct",
        "label": "Qwen2.5-3B-Instruct",
        "color": "#0D9488",  # Teal
        "edgecolor": "#0F766E",
        "hatch": "",
    },
    {
        "id": "qwen2.5-7b-instruct",
        "label": "Qwen2.5-7B-Instruct",
        "color": "#7C3AED",  # Violet
        "edgecolor": "#6D28D9",
        "hatch": "",
    },
    {
        "id": "llama-3.2-korean-bllossom-3b",
        "label": "Llama-3.2-Bllossom-3B",
        "color": "#E11D48",  # Rose
        "edgecolor": "#BE123C",
        "hatch": "",
    },
]


def extract_scores(summary_data: dict, model_ids: list[str]) -> tuple[list[str], dict[str, list[float]]]:
    metric_labels = [m[2] for m in METRICS]
    model_scores = {mid: [] for mid in model_ids}

    for mid in model_ids:
        m_data = summary_data.get(mid, {})
        metrics_dict = m_data.get("metrics", {})
        for task, sub, _, mult in METRICS:
            val = metrics_dict.get(task, {}).get(sub, 0.0)
            if val is None:
                val = 0.0
            model_scores[mid].append(round(float(val) * mult, 2))

    return metric_labels, model_scores


def create_grouped_horizontal_chart(
    metric_labels: list[str],
    model_scores: dict[str, list[float]],
    output_svg: Path,
    output_png: Path,
):
    plt.rcParams["font.sans-serif"] = [
        "DejaVu Sans",
        "Segoe UI",
        "Arial",
        "Liberation Sans",
        "sans-serif",
    ]
    plt.rcParams["axes.edgecolor"] = "#E2E8F0"
    plt.rcParams["axes.linewidth"] = 1.0

    n_metrics = len(metric_labels)
    n_models = len(MODEL_SPECS)
    y_indices = np.arange(n_metrics)

    # Reverse order so the most prominent metric appears on top
    y_indices = y_indices[::-1]

    bar_height = 0.15
    fig_height = 9.5
    fig_width = 13.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    # Draw subtle vertical gridlines from 0 to 100
    ax.grid(axis="x", color="#E2E8F0", linestyle="--", linewidth=0.8, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Offsets centered around y_indices
    offsets = np.linspace(-(n_models - 1) / 2 * bar_height, (n_models - 1) / 2 * bar_height, n_models)

    for i, spec in enumerate(MODEL_SPECS):
        mid = spec["id"]
        scores = model_scores.get(mid, [0.0] * n_metrics)
        y_positions = y_indices + offsets[i]

        bars = ax.barh(
            y_positions,
            scores,
            height=bar_height * 0.90,
            label=spec["label"],
            color=spec["color"],
            edgecolor=spec["edgecolor"],
            linewidth=1.0,
            alpha=0.92,
            zorder=3,
        )

        # Annotate bar values
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            text_x = width + 1.2
            ha = "left"
            text_color = "#1E293B"
            # If bar is very close to 100, place inside
            if width > 93:
                text_x = width - 1.5
                ha = "right"
                text_color = "#FFFFFF"

            ax.text(
                text_x,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}",
                va="center",
                ha=ha,
                fontsize=8.5,
                fontweight="600",
                color=text_color,
                zorder=4,
            )

    ax.set_yticks(y_indices)
    ax.set_yticklabels(metric_labels, fontsize=11, fontweight="600", color="#1E293B")
    ax.set_xlim(0, 105)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xticklabels([f"{x}" for x in range(0, 101, 10)], fontsize=10, color="#475569")

    # Titles & Subtitles
    fig.text(
        0.08,
        0.96,
        "UlsanBench v2 — Model Comparison",
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.08,
        0.935,
        "Same 500-item evaluation pipeline. Deterministic decoding (do_sample=False, temp=0). Higher is better.",
        fontsize=10.5,
        color="#64748B",
    )

    # Legend at the top right
    leg = ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 1.10),
        ncol=3,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        fontsize=9.5,
        handlelength=1.5,
        handleheight=1.0,
    )
    for text in leg.get_texts():
        text.set_color("#1E293B")

    # Spines styling
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    plt.tight_layout(rect=[0.05, 0.04, 0.98, 0.92])

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, format="svg", bbox_inches="tight")
    print(f"Generated SVG: {output_svg}")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, format="png", dpi=300, bbox_inches="tight")
    print(f"Generated PNG: {output_png}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate UlsanBench v2 comparison chart")
    parser.add_argument(
        "--summary-file",
        default="reports/model-comparison/summary.json",
        help="Path to combined summary.json",
    )
    parser.add_argument(
        "--output-svg",
        default="assets/benchmarks/ulsanbench-model-comparison.svg",
        help="Path to output SVG",
    )
    parser.add_argument(
        "--output-png",
        default="assets/benchmarks/ulsanbench-model-comparison.png",
        help="Path to output PNG",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_file)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary_data = json.load(f)

    model_ids = [spec["id"] for spec in MODEL_SPECS]
    metric_labels, model_scores = extract_scores(summary_data, model_ids)

    create_grouped_horizontal_chart(
        metric_labels=metric_labels,
        model_scores=model_scores,
        output_svg=Path(args.output_svg),
        output_png=Path(args.output_png),
    )


if __name__ == "__main__":
    main()
