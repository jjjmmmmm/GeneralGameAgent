# -*- coding: utf-8 -*-
"""M2 后半：摇杆分布图 + 序列可视化（帧画面与动作标注对齐）

抽帧策略（D5 砍单版）：从统计集 chunk_0001 起步（避开 chunk_0000 菜单期）
选 3 段起点 [60s, 180s, 300s]（均在比赛进行中），每段 5 帧连续，共 15 帧

输出：
- _data/frames/Z1r1S--MJS4/seq/seqNN_fN.png   抽出的帧画面
- results/figures/joystick_distribution.png   2x2 摇杆分布直方图
- results/figures/seq_overview.png            3 段序列横排总览图（含动作标注）

D6 扩展：序列起点列表加至 10 段（G4）；M7 批量曲线可视化另建 viz_curves.py（G8）。
"""
import subprocess
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import BUTTONS, SHARD, FPS, get_chunk_df, frame_to_chunk_row

VIDEO = Path(r"D:/2+课产品/TOP 1 IN 2S _ Ranked 2v2 w_ oKhaliD (1).mp4")
FRAME_OUT = Path(r"D:/2+课产品/_data/frames/Z1r1S--MJS4/seq")
FIG_OUT = ROOT / "results" / "figures"
FRAME_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

# D5 用 3 段（D6 补至 ≥10 段，见差距清单 G4）
SEQ_STARTS = [60, 180, 300]
SEQ_LEN = 5

# ---------- 1. 抽帧 ----------
print("=== 抽帧 ===")
for si, start_s in enumerate(SEQ_STARTS):
    for fi in range(SEQ_LEN):
        sec = start_s + fi
        out_png = FRAME_OUT / f"seq{si+1:02d}_f{fi}.png"
        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-ss", f"{sec}.000",
            "-i", str(VIDEO),
            "-frames:v", "1",
            "-q:v", "2",
            str(out_png),
        ]
        subprocess.run(cmd, check=True)
        print(f"  seq{si+1} f{fi} @ {sec}s -> {out_png.name}")

# ---------- 2. 摇杆分布图（全 35 chunks） ----------
print("\n=== 摇杆分布图 ===")
all_jlx, all_jly, all_jrx, all_jry = [], [], [], []
for cid in range(35):
    df = get_chunk_df(f"{cid:04d}")
    all_jlx.extend(df["j_left"].list.get(0).to_list())
    all_jly.extend(df["j_left"].list.get(1).to_list())
    all_jrx.extend(df["j_right"].list.get(0).to_list())
    all_jry.extend(df["j_right"].list.get(1).to_list())

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, data, name in zip(axes.flat,
                          [all_jlx, all_jly, all_jrx, all_jry],
                          ["j_left.x", "j_left.y", "j_right.x", "j_right.y"]):
    ax.hist(data, bins=40, edgecolor="black", alpha=0.7)
    ax.set_title(f"{name}  (n={len(data)})")
    ax.set_xlabel("value")
    ax.set_ylabel("frame count")
fig.suptitle("Joystick distribution (Z1r1S--MJS4, 35 chunks = 42000 frames)")
fig.tight_layout()
fig.savefig(FIG_OUT / "joystick_distribution.png", dpi=120)
plt.close(fig)
print(f"  -> {FIG_OUT / 'joystick_distribution.png'}")

# ---------- 3. 序列总览图（3 段横排，每段 5 帧 + 动作条形） ----------
print("\n=== 序列总览图 ===")
fig = plt.figure(figsize=(16, 9))
gs = GridSpec(3, SEQ_LEN, figure=fig, hspace=0.6, wspace=0.05)

for si, start_s in enumerate(SEQ_STARTS):
    for fi in range(SEQ_LEN):
        sec = start_s + fi
        fid = int(sec * FPS)
        cid, row = frame_to_chunk_row(fid)
        df = get_chunk_df(cid)
        row_df = df.slice(row, 1)
        _ = row_df  # 保留行数据供后续扩展

        ax_img = fig.add_subplot(gs[si, fi])
        ax_img.imshow(plt.imread(FRAME_OUT / f"seq{si+1:02d}_f{fi}.png"))
        ax_img.set_xticks([]); ax_img.set_yticks([])
        ax_img.set_title(f"seq{si+1} f{fi}\n{sec}s", fontsize=8)

    seq_actions = []
    for fi in range(SEQ_LEN):
        sec = start_s + fi
        fid = int(sec * FPS)
        cid, row = frame_to_chunk_row(fid)
        df = get_chunk_df(cid).slice(row, 1)
        pressed = [b for b in BUTTONS if int(df[b].sum()) > 0]
        seq_actions.append(pressed)
    summary = f"buttons: {' '.join(seq_actions[0]) or '(none)'}"
    fig.text(0.02, 0.85 - si*0.30, summary, fontsize=8, family="monospace")

fig.suptitle("Sequence overview (3 seqs × 5 frames, Z1r1S--MJS4)", fontsize=14)
fig.savefig(FIG_OUT / "seq_overview.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  -> {FIG_OUT / 'seq_overview.png'}")

print("\n=== 完成 ===")
