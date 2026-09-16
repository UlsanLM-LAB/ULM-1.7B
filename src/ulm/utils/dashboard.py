"""ULM-1.7B 실시간 터미널 모니터링 대시보드 (Rich TUI)."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Any

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def get_gpu_info() -> dict[str, Any]:
    """nvidia-smi를 호출하여 GPU VRAM, 사용률, 온도를 조회한다."""
    default = {
        "name": "GPU",
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
            timeout=1.5,
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


def make_bar(ratio: float, width: int = 20, fill_color: str = "cyan", empty_color: str = "grey30") -> Text:
    """텍스트 기반 게이지 바를 생성한다."""
    ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    empty = width - filled
    text = Text()
    text.append("━" * filled, style=fill_color)
    text.append("━" * empty, style=empty_color)
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
    """Rich TUI 대시보드 레이아웃을 구성한다."""
    if gpu_info is None:
        gpu_info = get_gpu_info()

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="progress", size=4),
        Layout(name="body", size=13),
        Layout(name="footer", size=7),
    )

    # 1. Header
    title = Text.assemble(
        (" ⚡ ULM-1.7B ", "bold bright_cyan"),
        (" 울산 지역어 특화 파인튜닝 모니터 ", "bold bright_white"),
        (f"[{model_name} / 4-bit QLoRA] ⚡ ", "bold bright_yellow"),
    )
    layout["header"].update(Panel(Align.center(title), style="bright_cyan", border_style="cyan"))

    # 2. Progress
    progress_ratio = (step / max_steps) if max_steps > 0 else 0.0
    percent = progress_ratio * 100.0

    rem_sec = (max_steps - step) * speed_sec_per_step if (speed_sec_per_step and max_steps > step) else 0.0
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_sec))
    eta_str = time.strftime("%H:%M:%S", time.gmtime(rem_sec)) if rem_sec > 0 else "계산 중..."

    p_bar = make_bar(progress_ratio, width=40, fill_color="bright_green")
    prog_text = Text.assemble(
        (f" 진행률: ", "bold white"),
        (f"{percent:5.1f}% ", "bold bright_green"),
        p_bar,
        (f"  ({step:,} / {max_steps:,} 스텝)\n", "bold cyan"),
        (f" 에포크: ", "bold white"),
        (f"{epoch:.2f} / {max_epochs:.1f}", "bold bright_yellow"),
        (f"   경과 시간: ", "bold white"),
        (f"{elapsed_str}", "bold bright_cyan"),
        (f"   남은 시간(ETA): ", "bold white"),
        (f"{eta_str}", "bold bright_magenta"),
    )
    layout["progress"].update(Panel(prog_text, title="[bold]🎯 학습 진행 상황[/bold]", border_style="bright_green"))

    # 3. Body: Left (Metrics) + Right (GPU & System)
    layout["body"].split_row(
        Layout(name="metrics", ratio=1),
        Layout(name="system", ratio=1),
    )

    # Left: Metrics
    metrics_table = Table.grid(padding=(0, 1))
    metrics_table.add_column(style="bold white", width=18)
    metrics_table.add_column(style="bold", width=22)

    loss_str = f"{loss:.4f}" if loss is not None else "대기 중..."
    eval_loss_str = f"{eval_loss:.4f}" if eval_loss is not None else "대기 중..."
    lr_str = f"{learning_rate:.2e}" if learning_rate is not None else "대기 중..."

    metrics_table.add_row("🔥 현재 Train Loss:", f"[bright_red]{loss_str}[/bright_red]")
    metrics_table.add_row("📊 최근 Eval Loss:", f"[bright_cyan]{eval_loss_str}[/bright_cyan]")
    metrics_table.add_row("📈 Learning Rate:", f"[bright_yellow]{lr_str}[/bright_yellow]")

    # Loss Sparkline
    if loss_history:
        recent = loss_history[-6:]
        spark = " ➔ ".join(f"{val:.3f}" for val in recent)
        metrics_table.add_row("📉 손실 추이 (최근):", f"[dim green]{spark}[/dim green]")
    else:
        metrics_table.add_row("📉 손실 추이 (최근):", "[dim]수집 중...[/dim]")

    layout["metrics"].update(
        Panel(metrics_table, title="[bold]📊 학습 지표 (Metrics)[/bold]", border_style="bright_blue")
    )

    # Right: System & Hardware
    sys_table = Table.grid(padding=(0, 1))
    sys_table.add_column(style="bold white", width=18)
    sys_table.add_column(style="bold", width=26)

    vram_u = gpu_info["vram_used"]
    vram_t = gpu_info["vram_total"]
    vram_ratio = vram_u / vram_t if vram_t > 0 else 0.0
    vram_bar = make_bar(vram_ratio, width=14, fill_color="bright_magenta")

    sys_table.add_row("🎮 GPU 모델:", f"[bright_white]{gpu_info['name']}[/bright_white]")
    sys_table.add_row(
        "💾 VRAM 사용량:",
        Text.assemble(f"{vram_u:.1f}/{vram_t:.1f}GB ", vram_bar, f" {vram_ratio*100:.0f}%"),
    )
    gpu_util = gpu_info["utilization"]
    util_bar = make_bar(gpu_util / 100.0, width=14, fill_color="bright_cyan")
    sys_table.add_row("⚡ GPU 연산률:", Text.assemble(f"{gpu_util:3d}% ", util_bar))
    
    temp = gpu_info["temperature"]
    temp_color = "bright_green" if temp < 65 else ("bright_yellow" if temp < 75 else "bright_red")
    sys_table.add_row("🌡️ GPU 온도:", f"[{temp_color}]{temp}°C[/{temp_color}]")

    speed_str = f"{speed_sec_per_step:.2f} 초/step" if speed_sec_per_step else "측정 중..."
    sys_table.add_row("⏱️ 처리 속도:", f"[bright_yellow]{speed_str}[/bright_yellow]")

    layout["system"].update(
        Panel(sys_table, title="[bold]💻 하드웨어 및 리소스[/bold]", border_style="bright_magenta")
    )

    # 4. Footer: Event Logs
    recent_events = events[-3:] if events else ["학습 대시보드가 준비되었습니다."]
    event_text = Text()
    for ev in recent_events:
        event_text.append(f"• {ev}\n", style="dim white")

    layout["footer"].update(Panel(event_text, title="[bold]📜 실시간 이벤트 로그[/bold]", border_style="grey50"))

    return layout


class RichDashboardCallback:
    """Hugging Face Trainer용 실시간 Rich TUI 콜백."""

    def __init__(self, console: Console | None = None, max_epochs: float = 2.0, model_name: str = "Qwen/Qwen3-1.7B"):
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
        self.events.append(f"🚀 학습 시작: 총 {state.max_steps} 스텝 예정")
        self.live = Live(self._render(state), console=self.console, refresh_per_second=4)
        self.live.start()

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if logs:
            if "loss" in logs:
                self.last_train_loss = logs["loss"]
                self.loss_history.append(logs["loss"])
            if "learning_rate" in logs:
                self.last_lr = logs["learning_rate"]
            if "eval_loss" in logs:
                self.last_eval_loss = logs["eval_loss"]
                self.events.append(f"📊 Step {state.global_step} 검증: eval_loss={logs['eval_loss']:.4f}")
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
        self.events.append(f"💾 체크포인트 저장: step {state.global_step} (Train Loss: {loss_str})")
        if self.live:
            self.live.update(self._render(state))

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.events.append(f"🎉 학습 완료! 총 {state.global_step} 스텝 완주.")
        if self.live:
            self.live.update(self._render(state))
            self.live.stop()

