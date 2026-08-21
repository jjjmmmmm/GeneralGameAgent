# -*- coding: utf-8 -*-
"""M7 可视化工具（D6 骨架，D7 批量完成）

目标（立项书 M7）：批量导出 ≥20 段手柄动作曲线，每段自动标出差异最大的 5 帧。
- 单段模式：--start 640 --frames 20 --out xxx.png（D6）
- 批量模式：--batch，按内置段起点列表渲染全部段 + 生成 index.md（D7）

用法（serve.py 运行中）：
    python scripts/viz_curves.py --start 640 --frames 20 --out results/figures/curves/seq_demo.png
    python scripts/viz_curves.py --batch [--k 1] [--frames 20]
"""
import argparse
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import (
    BUTTONS, FPS, BTN_THRESHOLD, PRED_ROW, BUTTON_TO_MODEL_COL, get_chunk_df, fetch_frame,
    CHUNK_SIZE,
)

sys.path.insert(0, str(ROOT.parent.parent.parent / "NitroGen"))
from nitrogen.inference_client import ModelClient  # type: ignore

CURVES_DIR = ROOT / "results" / "figures" / "curves"

# 批量模式段起点（秒）：统计集 60~600 每 30s 一段 + 测试集 630/660 两段
BATCH_STARTS = [60, 90, 120, 150, 180, 210, 240, 270, 300, 330,
                360, 390, 420, 450, 480, 510, 540, 570, 600, 630, 660]


def render_segment(client, start_s: int, frames: int, k: int, out_p: Path, tmp_dir: Path):
    """渲染单段曲线图，返回 (top5_indices, top5_diff_values)。"""
    fids = [int(start_s * FPS) + i * FPS for i in range(frames)]

    gt_btn = np.zeros((frames, len(BUTTONS)), dtype=int)
    pred_btn = np.zeros((frames, len(BUTTONS)), dtype=int)
    gt_jl = np.zeros((frames, 2), dtype=float)
    pred_jl = np.zeros((frames, 2), dtype=float)

    for i, fid in enumerate(fids):
        img = fetch_frame(fid, tmp_dir)
        cid = f"{fid // CHUNK_SIZE:04d}"
        df = get_chunk_df(cid).slice(fid % CHUNK_SIZE, 1)
        gt_btn[i] = np.array([int(df[b].sum()) for b in BUTTONS])
        gt_jl[i] = np.array(df["j_left"].to_list()[0], dtype=float)
        votes = np.zeros((len(BUTTONS),), dtype=int)
        for _ in range(k):
            client.reset()
            p = client.predict(img)
            votes += np.array([
                int((p["buttons"][PRED_ROW, BUTTON_TO_MODEL_COL[b]] > BTN_THRESHOLD))
                for b in BUTTONS
            ])
            pred_jl[i] = p["j_left"][PRED_ROW]
        pred_btn[i] = (votes >= (k + 1) // 2).astype(int)

    diff = np.abs(pred_btn - gt_btn).sum(axis=1) + np.abs(pred_jl - gt_jl).sum(axis=1)
    top5 = np.argsort(-diff)[:5]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax = axes[0]
    ax.imshow(gt_btn.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(BUTTONS))); ax.set_yticklabels(BUTTONS, fontsize=7)
    ax.set_title(f"Human buttons (1=pressed) start={start_s}s, {frames} frames")
    ax2 = axes[1]
    ax2.imshow(pred_btn.T, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax2.set_yticks(range(len(BUTTONS))); ax2.set_yticklabels(BUTTONS, fontsize=7)
    ax2.set_title("Model buttons (threshold >0.5)")

    ax3 = axes[2]
    ax3.plot(range(frames), gt_jl[:, 0], label="gt x", color="C0", ls="-")
    ax3.plot(range(frames), gt_jl[:, 1], label="gt y", color="C1", ls="-")
    ax3.plot(range(frames), pred_jl[:, 0], label="pred x", color="C0", ls="--")
    ax3.plot(range(frames), pred_jl[:, 1], label="pred y", color="C1", ls="--")
    ax3.set_title("j_left joystick: solid=human, dashed=model")
    ax3.legend(fontsize=8)

    ax3.scatter(top5, gt_jl[top5, 0], color="red", zorder=5, label="top5 diff")
    for i in top5:
        ax3.annotate(f"{diff[i]:.1f}", (i, gt_jl[i, 0]), fontsize=7, color="red")

    fig.suptitle(f"Action curves ({start_s}~{start_s+frames}s) top5 diff frames: {top5.tolist()}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_p, dpi=110)
    plt.close(fig)
    return top5, diff[top5].round(2).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=640)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--out", default="results/figures/curves/seq_demo.png")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--batch", action="store_true", help="批量模式：渲染 BATCH_STARTS 全段 + index.md")
    args = ap.parse_args()

    tmp_dir = ROOT / "results" / "_tmp_curves"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)

    client = ModelClient(host="localhost", port=5555)
    client.reset()

    if args.batch:
        print(f"批量模式：渲染 {len(BATCH_STARTS)} 段（全量，保证 index.md 完整）")
        index_rows = []
        for done, s in enumerate(BATCH_STARTS, 1):
            out_p = CURVES_DIR / f"seq_{s:03d}.png"
            top5, diffs = render_segment(client, s, args.frames, args.k, out_p, tmp_dir)
            index_rows.append(f"| {s} | {s+args.frames} | {out_p.name} | {top5.tolist()} | {diffs} |")
            print(f"  [{done}/{len(BATCH_STARTS)}] seq_{s:03d} done, top5={top5.tolist()}")
        index_p = CURVES_DIR / "index.md"
        with index_p.open("w", encoding="utf-8") as f:
            f.write("# 动作曲线索引（M7）\n\n")
            f.write(f"- 段数：{len(BATCH_STARTS)}，每段 {args.frames} 帧（1fps），K={args.k}\n")
            f.write(f"- diff = 按键 L1 + 摇杆 L1；top5 = 每段差异最大 5 帧\n")
            f.write(f"- 生成时间：{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("| start | end | file | top5 diff frames | diff values |\n|---|---|---|---|---|\n")
            for r in index_rows:
                f.write(r + "\n")
        print(f"批量完成：{len(BATCH_STARTS)} 段 -> {CURVES_DIR} + index.md")
    else:
        out_p = ROOT / args.out
        out_p.parent.mkdir(parents=True, exist_ok=True)
        top5, diffs = render_segment(client, args.start, args.frames, args.k, out_p, tmp_dir)
        print(f"DONE -> {out_p}")
        print(f"top5 diff frames: {top5.tolist()}, diff values: {diffs}")

    client.close()


if __name__ == "__main__":
    main()
