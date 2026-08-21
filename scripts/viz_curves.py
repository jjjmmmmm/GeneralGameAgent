# -*- coding: utf-8 -*-
"""M7 可视化工具骨架（D6 起步，D7 完成批量）

目标（立项书 M7）：批量导出 ≥20 段手柄动作曲线，每段自动标出差异最大的 5 帧。
本文件 D6 版本：单段动作曲线（按键双线横条图 + j_left 双线曲线），参数化段起点/帧数。
D7 版本（G8）：批量遍历段、差异帧标注、≥20 段导出。

用法（serve.py 运行中）：
    python scripts/viz_curves.py --start 640 --frames 20 --out results/figures/curves/seq_demo.png
"""
import argparse
import subprocess
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import (
    BUTTONS, FPS, BTN_THRESHOLD, PRED_ROW, SHARD, BUTTON_TO_MODEL_COL, get_chunk_df,
)

sys.path.insert(0, str(ROOT.parent.parent.parent / "NitroGen"))
from nitrogen.inference_client import ModelClient  # type: ignore

VIDEO = r"D:/2+课产品/TOP 1 IN 2S _ Ranked 2v2 w_ oKhaliD (1).mp4"
CHUNK_SIZE = 1200


def fetch_frame(fid: int, tmp_dir: Path) -> np.ndarray:
    sec = fid / FPS
    p = tmp_dir / f"f{fid}.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{sec:.3f}",
         "-i", VIDEO, "-frames:v", "1", "-q:v", "2", str(p)],
        check=True, capture_output=True,
    )
    img = mpimg.imread(str(p))
    p.unlink()
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=640, help="起始秒（测试集内）")
    ap.add_argument("--frames", type=int, default=20, help="曲线帧数（每秒 1 帧）")
    ap.add_argument("--out", default="results/figures/curves/seq_demo.png")
    ap.add_argument("--k", type=int, default=1, help="每帧推理次数（多数票）")
    args = ap.parse_args()

    tmp_dir = ROOT / "results" / "_tmp_curves"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_p = ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fids = [int(args.start * FPS) + i * FPS for i in range(args.frames)]  # 每秒 1 帧

    client = ModelClient(host="localhost", port=5555)
    client.reset()

    gt_btn = np.zeros((args.frames, len(BUTTONS)), dtype=int)
    pred_btn = np.zeros((args.frames, len(BUTTONS)), dtype=int)
    gt_jl = np.zeros((args.frames, 2), dtype=float)
    pred_jl = np.zeros((args.frames, 2), dtype=float)

    for i, fid in enumerate(fids):
        img = fetch_frame(fid, tmp_dir)
        # 标注
        cid = f"{fid // CHUNK_SIZE:04d}"
        df = get_chunk_df(cid).slice(fid % CHUNK_SIZE, 1)
        gt_btn[i] = np.array([int(df[b].sum()) for b in BUTTONS])
        gt_jl[i] = np.array(df["j_left"].to_list()[0], dtype=float)
        # 推理（多数票）
        votes = np.zeros((len(BUTTONS),), dtype=int)
        for _ in range(args.k):
            client.reset()
            p = client.predict(img)
            votes += np.array([
                int((p["buttons"][PRED_ROW, BUTTON_TO_MODEL_COL[b]] > BTN_THRESHOLD))
                for b in BUTTONS
            ])
            pred_jl[i] = p["j_left"][PRED_ROW]
        pred_btn[i] = (votes >= (args.k + 1) // 2).astype(int)
    client.close()

    # 差异帧：按键差异 L1 距离（每帧 pred vs gt 的按下列差异）
    diff = np.abs(pred_btn - gt_btn).sum(axis=1) + np.abs(pred_jl - gt_jl).sum(axis=1)

    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    y_labels = BUTTONS
    # 上：按键双线（gt 上、pred 下）——数据为 (帧, 键)，imshow 转置为 (键, 帧)
    ax = axes[0]
    ax.imshow(gt_btn.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(BUTTONS))); ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_title(f"Human buttons (1=pressed) start={args.start}s, {args.frames} frames")
    ax2 = axes[1]
    ax2.imshow(pred_btn.T, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax2.set_yticks(range(len(BUTTONS))); ax2.set_yticklabels(y_labels, fontsize=7)
    ax2.set_title("Model buttons (threshold >0.5)")

    # 下：j_left 双线
    ax3 = axes[2]
    ax3.plot(range(args.frames), gt_jl[:, 0], label="gt x", color="C0", ls="-")
    ax3.plot(range(args.frames), gt_jl[:, 1], label="gt y", color="C1", ls="-")
    ax3.plot(range(args.frames), pred_jl[:, 0], label="pred x", color="C0", ls="--")
    ax3.plot(range(args.frames), pred_jl[:, 1], label="pred y", color="C1", ls="--")
    ax3.set_title("j_left joystick: solid=human, dashed=model")
    ax3.legend(fontsize=8)

    # 差异最大 5 帧标注（D7 完整实现；此处先画标记）
    top5 = np.argsort(-diff)[:5]
    ax3.scatter(top5, gt_jl[top5, 0], color="red", zorder=5, label="top5 diff")
    for i in top5:
        ax3.annotate(f"{diff[i]:.1f}", (i, gt_jl[i, 0]), fontsize=7, color="red")

    fig.suptitle(f"Action curves ({args.start}~{args.start+args.frames}s) top5 diff frames: {top5.tolist()}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_p, dpi=120)
    plt.close(fig)
    print(f"DONE -> {out_p}")
    print(f"top5 diff frames: {top5.tolist()}, diff values: {diff[top5].round(2).tolist()}")


if __name__ == "__main__":
    main()
