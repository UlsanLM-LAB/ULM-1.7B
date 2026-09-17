"""ULM-1.7B 터미널 모니터링 대시보드 (미니멀/엔지니어링 스타일)."""

from __future__ import annotations

import subprocess
import time
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def get_gpu_info() -> dict[str, Any]:
    """nvidia-smi를 호출하여 GPU 정보를 조회한다."""
    default = {
        "name": "NVIDIA GPU",
        "vram_used": 0.0,
        "vram_total": 8.0,
        "utilization": 0,
        "temperature": 0,
        "available": False,
    }
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
            timeout=1.0,
        ).strip()
        parts = [p.strip() for p in output.splitlines()[0].split(",")]
        return {
            "name": parts[0],
            "vram_total": float(parts[1]) / 1024.0,
            "vram_used": float(parts[2]) / 1024.0,
            "utilization": int(parts[3]),
            "temperature": int(parts[4]),
            "available": True,
        }
    except Exception:
        return default


def make_bar(
    ratio: float, width: int = 24, fill_color: str = "green", empty_color: str = "grey23"
) -> Text:
    """단순화된 바 게이지 생성."""
    ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    empty = width - filled
    text = Text()
    text.append("█" * filled, style=fill_color)
    text.append("░" * empty, style=empty_color)
    return text


def build_dashboard_layout(
    *,
    step: int,
    max_steps: int,
    epoch: float,
    max_epochs: float,
    loss: float | None,
    eval_loss: float | None,
    learning_rate: float | None,
    loss_history: list[float],
    speed_sec_per_step: float | None,
    elapsed_sec: float,
    events: list[str],
    model_name: str = "Qwen/Qwen3-1.7B",
    gpu_info: dict[str, Any] | None = None,
) -> Layout:
    """간결하고 실용적인 개발자 중심 대시보드 레이아웃."""
    if gpu_info is None:
        gpu_info = get_gpu_info()

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="progress", size=4),
        Layout(name="body", size=10),
        Layout(name="logs", size=6),
    )

    # 1. Header (이모지 제거, 미니멀 상단 바)
    header_text = Text.assemble(
        (" ULM-1.7B Training Monitor ", "bold white on blue"),
        ("  model: ", "dim"),
        (model_name, "white"),
        ("  method: ", "dim"),
        ("QLoRA (4-bit NF4)", "white"),
        ("  precision: ", "dim"),
        ("FP16", "white"),
    )
    layout["header"].update(Panel(header_text, style="white", border_style="grey37"))

    # 2. Progress
    progress_ratio = (step / max_steps) if max_steps > 0 else 0.0
    percent = progress_ratio * 100.0

    rem_sec = (
        (max_steps - step) * speed_sec_per_step
        if (speed_sec_per_step and max_steps > step)
        else 0.0
    )
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_sec))
    eta_str = time.strftime("%H:%M:%S", time.gmtime(rem_sec)) if rem_sec > 0 else "--:--:--"

    p_bar = make_bar(progress_ratio, width=36, fill_color="cyan")
    prog_text = Text.assemble(
        (" Progress: ", "dim"),
        p_bar,
        (f" {percent:5.1f}% ", "bold cyan"),
        (f"({step:,} / {max_steps:,} steps)\n", "dim"),
        (" Epoch   : ", "dim"),
        (f"{epoch:.2f} / {max_epochs:.1f}", "white"),
        ("   Elapsed: ", "dim"),
        (f"{elapsed_str}", "white"),
        ("   ETA: ", "dim"),
        (f"{eta_str}", "white"),
        ("   Speed: ", "dim"),
        (f"{speed_sec_per_step:.2f}s/step" if speed_sec_per_step else "--", "white"),
    )
    layout["progress"].update(Panel(prog_text, title="Training Progress", border_style="grey37"))

    # 3. Body: Metrics & Hardware (간결한 테이블)
    layout["body"].split_row(
        Layout(name="metrics", ratio=1),
        Layout(name="system", ratio=1),
    )

    # Left: Metrics
    metrics_table = Table.grid(padding=(0, 2))
    metrics_table.add_column(style="dim", width=14)
    metrics_table.add_column(style="white", width=26)

    loss_str = f"{loss:.4f}" if loss is not None else "--"
    eval_loss_str = f"{eval_loss:.4f}" if eval_loss is not None else "--"
    lr_str = f"{learning_rate:.2e}" if learning_rate is not None else "--"

    metrics_table.add_row("Train Loss", loss_str)
    metrics_table.add_row("Eval Loss", eval_loss_str)
    metrics_table.add_row("Learning Rate", lr_str)

    if loss_history:
        recent = loss_history[-5:]
        trend = " -> ".join(f"{val:.3f}" for val in recent)
        metrics_table.add_row("Loss History", f"[dim]{trend}[/dim]")
    else:
        metrics_table.add_row("Loss History", "--")

    layout["metrics"].update(Panel(metrics_table, title="Metrics", border_style="grey37"))

    # Right: System Info
    sys_table = Table.grid(padding=(0, 2))
    sys_table.add_column(style="dim", width=14)
    sys_table.add_column(style="white", width=28)

    vram_u = gpu_info["vram_used"]
    vram_t = gpu_info["vram_total"]
    vram_ratio = vram_u / vram_t if vram_t > 0 else 0.0
    vram_bar = make_bar(vram_ratio, width=12, fill_color="magenta")

    sys_table.add_row("GPU Device", gpu_info["name"])
    sys_table.add_row(
        "VRAM Usage",
        Text.assemble(f"{vram_u:.1f}/{vram_t:.1f} GB ", vram_bar, f" {vram_ratio * 100:.0f}%"),
    )
    gpu_util = gpu_info["utilization"]
    util_bar = make_bar(gpu_util / 100.0, width=12, fill_color="green")
    sys_table.add_row("GPU Util", Text.assemble(f"{gpu_util:3d}% ", util_bar))
    sys_table.add_row("Temperature", f"{gpu_info['temperature']} C")

    layout["system"].update(Panel(sys_table, title="Hardware", border_style="grey37"))

    # 4. Logs (최근 이벤트, 이모지 없음)
    recent_events = events[-4:] if events else ["Ready"]
    log_text = Text()
    for ev in recent_events:
        log_text.append(f" {ev}\n", style="white")

    layout["logs"].update(Panel(log_text, title="Recent Logs", border_style="grey37"))

    return layout


try:
    from transformers import TrainerCallback
except ImportError:

    class TrainerCallback:
        """Fallback base callback when transformers is not installed."""


class RichDashboardCallback(TrainerCallback):
    """Trainer용 미니멀 터미널 콜백."""

    def __init__(
        self,
        console: Console | None = None,
        max_epochs: float = 2.0,
        model_name: str = "Qwen/Qwen3-1.7B",
    ):
        self.console = console or Console()
        self.live: Any = None
        self.max_epochs = max_epochs
        self.model_name = model_name
        self.loss_history: list[float] = []
        self.events: list[str] = []
        self.start_time = time.time()
        self.last_step_time = time.time()
        self.speed_sec_per_step: float | None = None
        self.last_train_loss: float | None = None
        self.last_eval_loss: float | None = None
        self.last_lr: float | None = None

    def _render(self, state: Any) -> Layout:
        step = getattr(state, "global_step", 0)
        max_steps = getattr(state, "max_steps", 2600)
        epoch = getattr(state, "epoch", 0.0) or 0.0
        elapsed = time.time() - self.start_time
        return build_dashboard_layout(
            step=step,
            max_steps=max_steps,
            epoch=epoch,
            max_epochs=self.max_epochs,
            loss=self.last_train_loss,
            eval_loss=self.last_eval_loss,
            learning_rate=self.last_lr,
            loss_history=self.loss_history,
            speed_sec_per_step=self.speed_sec_per_step,
            elapsed_sec=elapsed,
            events=self.events,
            model_name=self.model_name,
        )

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        from rich.live import Live

        self.start_time = time.time()
        self.last_step_time = time.time()
        self.events.append(
            f"[{time.strftime('%H:%M:%S')}] Training started. Target steps: {state.max_steps}"
        )
        self.live = Live(self._render(state), console=self.console, refresh_per_second=2)
        self.live.start()

    def on_log(
        self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        if logs:
            if "loss" in logs:
                self.last_train_loss = logs["loss"]
                self.loss_history.append(logs["loss"])
            if "learning_rate" in logs:
                self.last_lr = logs["learning_rate"]
            if "eval_loss" in logs:
                self.last_eval_loss = logs["eval_loss"]
                ts = time.strftime("%H:%M:%S")
                ev = f"[{ts}] Eval step {state.global_step}: eval_loss={logs['eval_loss']:.4f}"
                self.events.append(ev)
            now = time.time()
            step_delta = now - self.last_step_time
            log_steps = getattr(args, "logging_steps", 10)
            if log_steps > 0 and step_delta > 0:
                self.speed_sec_per_step = step_delta / log_steps
            self.last_step_time = now
        if self.live:
            self.live.update(self._render(state))

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        loss_str = f"{self.last_train_loss:.4f}" if self.last_train_loss is not None else "N/A"
        ts = time.strftime("%H:%M:%S")
        self.events.append(f"[{ts}] Checkpoint step {state.global_step} saved (loss: {loss_str})")
        if self.live:
            self.live.update(self._render(state))

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.events.append(
            f"[{time.strftime('%H:%M:%S')}] Training completed. Total steps: {state.global_step}"
        )
        if self.live:
            self.live.update(self._render(state))
            self.live.stop()
