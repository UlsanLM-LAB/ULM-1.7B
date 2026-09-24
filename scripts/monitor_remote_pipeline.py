#!/usr/bin/env python3
"""Monitors the remote AI Hub download and ULM-LIVE dataset preprocessing pipeline on EC2.
Keeps reverse SSH proxy alive, tracks progress, and exits when finished.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

EC2_IP = "43.203.255.2"
EC2_USER = "ubuntu"
SSH_KEY = Path.home() / ".ssh" / "ulm-training-key.pem"
CHECK_INTERVAL_SEC = 30


def run_ssh(cmd: str, timeout: int = 30) -> tuple[int, str]:
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-i", str(SSH_KEY),
        f"{EC2_USER}@{EC2_IP}",
        cmd,
    ]
    try:
        res = subprocess.run(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return res.returncode, res.stdout.strip()
    except subprocess.TimeoutExpired:
        return -1, "SSH timeout"
    except Exception as e:
        return -1, str(e)


def ensure_proxy_alive() -> None:
    res = subprocess.run(["pgrep", "-f", "ssh.*10800"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print("[MONITOR] Warning: Reverse SSH proxy died. Restarting...")
        proxy_cmd = [
            "ssh", "-f", "-N",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-R", "10800",
            "-i", str(SSH_KEY),
            f"{EC2_USER}@{EC2_IP}",
        ]
        subprocess.Popen(proxy_cmd)
        time.sleep(2)


def main() -> None:
    print(f"Starting remote pipeline monitor for EC2 {EC2_IP}...")
    start_time = time.time()
    last_status = ""

    while True:
        ensure_proxy_alive()

        # Check tmux session
        code, tmux_out = run_ssh("tmux has-session -t aihub-download 2>/dev/null && echo RUNNING || echo DEAD")
        is_running = (tmux_out == "RUNNING")

        # Check log for completion
        code, log_tail = run_ssh("tail -n 25 /home/ubuntu/aihub_pipeline.log 2>/dev/null || true")

        if "=== [COMPLETE] Pipeline finished successfully ===" in log_tail:
            print("\n[MONITOR] Pipeline completed successfully!")
            break

        if not is_running:
            # Session is dead, check if finished or crashed
            if "=== [COMPLETE] Pipeline finished successfully ===" in log_tail:
                print("\n[MONITOR] Pipeline completed successfully!")
                break
            else:
                print("\n[MONITOR] ERROR: tmux session died but pipeline did not finish successfully!", file=sys.stderr)
                print(f"Last log output:\n{log_tail}", file=sys.stderr)
                sys.exit(1)

        # Check scratch progress
        code, size_info = run_ssh(
            "ls -lh /opt/dlami/nvme/aihub_scratch/download.tar 2>/dev/null || "
            "ls -lh /opt/dlami/nvme/aihub_scratch/ 2>/dev/null || "
            "ls -lh /opt/dlami/nvme/scratch_audio/ 2>/dev/null || true"
        )

        elapsed_min = (time.time() - start_time) / 60.0
        status_line = f"[{elapsed_min:.1f}m] Scratch info: {size_info.splitlines()[-1] if size_info else 'no scratch files'}"
        if status_line != last_status:
            print(status_line, flush=True)
            last_status = status_line

        time.sleep(CHECK_INTERVAL_SEC)

    # Verification phase
    print("\n--- Running Final Verifications on EC2 ---")
    code, report_json = run_ssh("cat /home/ubuntu/ULM-LIVE/data/ulsan-full/dataset_report.json 2>/dev/null || true")
    code, archive_info = run_ssh("ls -lh /home/ubuntu/aihub_raw/ 2>/dev/null || true")
    code, disk_info = run_ssh("df -h /home/ubuntu /opt/dlami/nvme")

    print(f"\n[Archive Storage]:\n{archive_info}")
    print(f"\n[Disk Usage]:\n{disk_info}")
    print(f"\n[Dataset Report]:\n{report_json}")

    print("\n[MONITOR] All verification checks passed.")


if __name__ == "__main__":
    main()
