"""Render the recorded ULM-1.7B Phase 3 training history for a portfolio.

Run: python3 scripts/plot_training_curve.py
The latest checkpoint's trainer_state.json is the sole source of plotted metrics.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs/ulm-1.7b-phase3-chat-l40s"
DEFAULT_OUTPUT = ROOT / "portfolio"


def load_history(run_dir: Path):
    checkpoints = sorted(
        (p for p in run_dir.glob("checkpoint-*") if p.is_dir() and p.name[11:].isdigit()),
        key=lambda p: int(p.name[11:]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    source = checkpoints[-1] / "trainer_state.json"
    state = json.loads(source.read_text(encoding="utf-8"))
    if state.get("global_step") != int(checkpoints[-1].name[11:]):
        raise ValueError(f"Checkpoint and global_step disagree: {source}")

    def points(metric):
        result = []
        for entry in state.get("log_history", []):
            if metric in entry:
                step, value = entry.get("step"), entry[metric]
                if not isinstance(step, int) or not isinstance(value, (int, float)):
                    raise ValueError(f"Invalid {metric} entry in {source}")
                if not math.isfinite(value) or (result and step <= result[-1][0]):
                    raise ValueError(f"Unordered or nonfinite {metric} in {source}")
                result.append((step, value))
        return result

    train = points("loss")
    validation = points("eval_loss")
    if not train:
        raise ValueError(f"No training loss found in {source}")
    if train[-1][0] > state["global_step"] or (
        validation and validation[-1][0] > state["global_step"]
    ):
        raise ValueError(f"Logged step exceeds checkpoint step in {source}")
    return source, state, train, validation


def render(source, state, train, validation, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    bg, white, muted, faint = "#0D0D0D", "#F2F2F0", "#A8AAA9", "#656A68"
    train_color, eval_color = "#76D2C0", "#E6AB83"
    plt.rcParams.update({
        "font.family": "Inter",
        "font.size": 11,
        "svg.fonttype": "none",
        "savefig.facecolor": bg,
    })
    fig = plt.figure(figsize=(10, 5.4), facecolor=bg)

    fig.text(0.075, 0.91, "ULM-1.7B Fine-tuning", color=white,
             size=21, weight="semibold", va="center")
    fig.text(0.075, 0.855, "Qwen3-1.7B · LoRA Fine-tuning", color=muted, size=11)
    fig.text(0.925, 0.917, "PHASE 03  /  CHAT ALIGNMENT", color=faint,
             size=8.5, ha="right", va="center", weight="medium")
    fig.add_artist(Line2D([0.075, 0.925], [0.818, 0.818], transform=fig.transFigure,
                          color="#303332", linewidth=0.8))

    ax = fig.add_axes((0.10, 0.225, 0.615, 0.505), facecolor=bg)
    tx, ty = zip(*train)
    ax.plot(tx, ty, color=train_color, linewidth=2.25, solid_capstyle="round",
            solid_joinstyle="round", zorder=3, label="Training loss")
    ax.scatter([tx[-1]], [ty[-1]], s=28, color=train_color, edgecolors=bg,
               linewidths=1, zorder=5)
    values = list(ty)
    if validation:
        vx, vy = zip(*validation)
        ax.plot(vx, vy, color=eval_color, linewidth=2.15, marker="o",
                markersize=5, markeredgecolor=bg, markeredgewidth=0.8,
                solid_capstyle="round", zorder=4, label="Validation loss")
        values.extend(vy)

    ax.set_xlim(0, state["global_step"] * 1.025)
    low, high = min(values), max(values)
    span = high - low
    ax.set_ylim(max(0, low - span * 0.11), high + span * 0.10)
    ax.xaxis.set_major_locator(MultipleLocator(250 if state["global_step"] > 600 else 50))
    ax.yaxis.set_major_locator(MultipleLocator(0.25 if span > 1 else 0.1))
    ax.grid(axis="y", color="#343838", linewidth=0.65, alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=muted, labelsize=9.5, length=0, pad=9)
    ax.set_xlabel("Training Step", color=muted, size=10, labelpad=12)
    ax.set_ylabel("Loss", color=muted, size=10, labelpad=11)
    ax.legend(loc="upper right", frameon=False, labelcolor=white, fontsize=9.5,
              handlelength=2.0, borderaxespad=0.5)

    fig.add_artist(Line2D([0.758, 0.758], [0.19, 0.745], transform=fig.transFigure,
                          color="#303332", linewidth=0.8))
    x = 0.795
    def stat(y, label, value, detail=None):
        fig.text(x, y, label.upper(), color=faint, size=8.5, weight="medium")
        fig.text(x, y - 0.054, value, color=white, size=18, weight="medium")
        if detail:
            fig.text(x, y - 0.086, detail, color=muted, size=8)

    stat(0.714, "Base model", "Qwen3-1.7B", "via Phase 2 merge")
    stat(0.572, "Method", "LoRA")
    stat(0.457, "Completed steps", f"{state['global_step']:,}")
    stat(0.342, "Final train loss", f"{train[-1][1]:.3f}",
         f"last logged · step {train[-1][0]:,}")
    if validation:
        stat(0.206, "Final eval loss", f"{validation[-1][1]:.3f}",
             f"step {validation[-1][0]:,}")

    fig.text(0.075, 0.055,
             f"SOURCE  {source.relative_to(ROOT)}   ·   recorded Trainer metrics",
             color=faint, size=7.8)
    png = output_dir / "ulm_training_curve.png"
    svg = output_dir / "ulm_training_curve.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source, state, train, validation = load_history(args.run_dir)
    png, svg = render(source, state, train, validation, args.output_dir)
    print(f"Source: {source}")
    print(f"Steps: {state['global_step']}; train: {train[0]} → {train[-1]}; "
          f"final eval: {validation[-1] if validation else 'none'}")
    print(f"Saved: {png}, {svg}")


if __name__ == "__main__":
    main()
