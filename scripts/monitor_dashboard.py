#!/usr/bin/env python3
"""실시간 학습 모니터링 대시보드 CLI.

백그라운드에서 학습 중인 모델의 출력 디렉터리를 감시하며
GPU 자원(VRAM/온도/연산률), 실시간 Loss, ETA, 체크포인트를 사이버펑크 TUI 화면으로 렌더링합니다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live

from ulm.utils.dashboard import build_dashboard_layout, get_gpu_info


def read_trainer_state(output_dir: Path) -> dict | None:
    """output_dir 또는 하위 checkpoint 폴더에서 가장 최신 trainer_state.json을 찾는다."""
    candidates = list(output_dir.glob("checkpoint-*/trainer_state.json"))
    direct = output_dir / "trainer_state.json"
    if direct.is_file():
        candidates.append(direct)
    if not candidates:
        return None
    # 가장 최근 수정된 파일 선택
    latest = max(candidates, key=os.path.getmtime)
    try:
        with latest.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="ULM-1.7B 실시간 학습 모니터링 대시보드")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qwen3-1.7b-sft"),
        help="학습 결과가 저장되는 디렉터리 경로 (기본값: outputs/qwen3-1.7b-sft)",
    )
    parser.add_argument("--max-steps", type=int, default=2600, help="목표 스텝 수 (기본값: 2600)")
    parser.add_argument("--refresh-rate", type=float, default=1.0, help="화면 갱신 주기 (초 단위, 기본값: 1.0)")
    args = parser.parse_args()

    console = Console()
    output_dir = args.output_dir

    start_time = time.time()
    events = [f"[{time.strftime('%H:%M:%S')}] Monitoring started: {output_dir}"]
    loss_history: list[float] = []
    last_step = 0
    last_step_time = time.time()
    speed_sec = None

    try:
        with Live(console=console, refresh_per_second=2, screen=True) as live:
            while True:
                state = read_trainer_state(output_dir)
                step = 0
                epoch = 0.0
                max_steps = args.max_steps
                last_loss = None
                last_eval_loss = None
                last_lr = None

                if state:
                    step = state.get("global_step", 0)
                    epoch = state.get("epoch", 0.0) or 0.0
                    if state.get("max_steps"):
                        max_steps = state["max_steps"]
                    history = state.get("log_history", [])
                    loss_history = [h["loss"] for h in history if "loss" in h]
                    eval_losses = [h["eval_loss"] for h in history if "eval_loss" in h]
                    if loss_history:
                        last_loss = loss_history[-1]
                    if eval_losses:
                        last_eval_loss = eval_losses[-1]
                    lrs = [h["learning_rate"] for h in history if "learning_rate" in h]
                    if lrs:
                        last_lr = lrs[-1]

                    if step != last_step:
                        now = time.time()
                        delta_steps = step - last_step
                        if delta_steps > 0:
                            speed_sec = (now - last_step_time) / delta_steps
                        last_step = step
                        last_step_time = now

                # 체크포인트 폴더 감지
                checkpoints = sorted(output_dir.glob("checkpoint-*"), key=os.path.getmtime)
                if checkpoints:
                    latest_cp = checkpoints[-1].name
                    msg = f"[{time.strftime('%H:%M:%S')}] Checkpoint found: {latest_cp}"
                    if not events or events[-1] != msg:
                        events.append(msg)

                gpu = get_gpu_info()
                layout = build_dashboard_layout(
                    step=step,
                    max_steps=max_steps,
                    epoch=epoch,
                    max_epochs=2.0,
                    loss=last_loss,
                    eval_loss=last_eval_loss,
                    learning_rate=last_lr,
                    loss_history=loss_history,
                    speed_sec_per_step=speed_sec,
                    elapsed_sec=time.time() - start_time,
                    events=events,
                    model_name="Qwen/Qwen3-1.7B",
                    gpu_info=gpu,
                )
                live.update(layout)
                time.sleep(args.refresh_rate)
    except KeyboardInterrupt:
        console.print("\n[dim]Monitoring stopped.[/dim]")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
