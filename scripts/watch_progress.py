#!/usr/bin/env python3
"""CAPR-CLIP 数据清洗进度监视器。

读取 clean_dataset.log（UTF-8），把最新进度渲染成 ASCII 进度条。
单次运行输出一张快照；加 --watch 则持续清屏刷新（Ctrl+C 退出）。

用法:
    python scripts/watch_progress.py                 # 单次快照
    python scripts/watch_progress.py --watch         # 持续刷新（默认每 2 秒）
    python scripts/watch_progress.py --log 自定义.log
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

# [train] 1000/103218 (1.0%)，可用 1000，异常 0，65.8 张/秒
_PROGRESS_RE = re.compile(
    r"\[(train|test)\]\s+(\d+)/(\d+)\s+\(([\d.]+)%\)，可用\s+(\d+)，异常\s+(\d+)，([\d.]+)\s*张/秒"
)
# [train] D:\初赛数据集\train.zip：103218 个文件
_TOTAL_RE = re.compile(r"\[(train|test)\]\s+.*?(\d+)\s*个文件")

_STAGE_LABEL = {"train": "训练集", "test": "测试集"}


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _bar(frac: float, width: int = 30) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def _read_state(log: Path) -> tuple[dict[str, int], dict[str, tuple], list[str]]:
    lines = []
    if log.is_file():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()

    totals: dict[str, int] = {}
    order: list[str] = []
    progress: dict[str, tuple] = {}
    for line in lines:
        m = _TOTAL_RE.search(line)
        if m:
            stage, total = m.group(1), int(m.group(2))
            totals[stage] = total
            if stage not in order:
                order.append(stage)
        m = _PROGRESS_RE.search(line)
        if m:
            stage = m.group(1)
            progress[stage] = (
                int(m.group(2)),
                int(m.group(3)),
                float(m.group(4)),
                int(m.group(5)),
                int(m.group(6)),
                float(m.group(7)),
            )
            if stage not in order:
                order.append(stage)
    return totals, progress, order


def render(log: Path) -> str:
    totals, progress, order = _read_state(log)
    out: list[str] = []

    out.append("数据清洗进度")
    out.append("=" * 40)

    if not order:
        out.append("(日志为空或尚无进度，等待脚本输出…)")
        return "\n".join(out)

    for stage in order:
        label = _STAGE_LABEL.get(stage, stage)
        if stage in progress:
            n, total, pct, accepted, anomalies, speed = progress[stage]
            out.append(f"[{label}] " + _bar(n / total if total else 0.0))
            out.append(
                f"  {n:,} / {total:,}  ({pct:.1f}%)  ·  可用 {accepted:,}  ·  异常 {anomalies:,}"
            )
            out.append(
                f"  速度 {speed:.1f} 张/秒  ·  已用 {_fmt_duration(n / speed)}"
                f"  ·  剩余 ~{_fmt_duration((total - n) / speed)}"
            )
        else:
            total = totals.get(stage, 0)
            out.append(f"[{label}] " + _bar(0.0))
            out.append(f"  0 / {total:,}  (待开始)")

    total_known = sum(totals.values())
    done = sum(p[0] for p in progress.values())
    if total_known:
        frac = done / total_known if total_known else 0.0
        out.append("-" * 40)
        out.append(f"全局  {done:,} / {total_known:,}  ({frac:.1%})  " + _bar(frac))

    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="数据清洗进度监视器")
    parser.add_argument("--log", type=Path, default=Path("clean_dataset.log"))
    parser.add_argument("--watch", action="store_true", help="持续刷新")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    if args.watch:
        clear = "cls" if os.name == "nt" else "clear"
        try:
            while True:
                os.system(clear)
                print(render(args.log), flush=True)
                print("\n(Ctrl+C 退出)", flush=True)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n已停止监视。", flush=True)
    else:
        print(render(args.log))


if __name__ == "__main__":
    main()
