# -*- coding: utf-8 -*-
"""G11 假设 2 验证：PRED_ROW 时序偏移——8 步动作块中哪一步与当前帧标注最相关

对测试集 30 帧逐帧推理（每次 reset 单帧），记录所有 8 步的 j_left；
对每步 row r 计算与当前帧标注 j_left 的相关系数（x 与 y 分开 + 合并），
看是否存在某个 row 显著优于 row 0。

结论判定：
- 若 max |corr| 集中在某 row>0 且明显高于 row0 → "PRED_ROW 应为 X"，M4 可重跑对比
- 若全部接近 0 → 支持"模型摇杆能力上限"，G11 归因收口

用法：serve.py 运行中；python scripts/probe_pred_row.py
"""
import subprocess
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import polars as pl
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import FPS, PRED_ROW, SHARD, get_row

sys.path.insert(0, str(ROOT.parent.parent.parent / "NitroGen"))
from nitrogen.inference_client import ModelClient  # type: ignore

VIDEO = r"D:/2+课产品/TOP 1 IN 2S _ Ranked 2v2 w_ oKhaliD (1).mp4"
CHUNK_SIZE = 1200
N_FRAMES = 30
START_SEC = 640  # 测试集首段

client = ModelClient(host="localhost", port=5555)
client.reset()

# 收集：gt_jl (N,2) + pred_jl (N,N_STEPS,2)（实测模型输出 18 步）
N_STEPS = 18
gt_jl = np.zeros((N_FRAMES, 2), dtype=float)
pred_jl = np.zeros((N_FRAMES, N_STEPS, 2), dtype=float)

tmp_dir = ROOT / "results" / "_tmp_probe"
tmp_dir.mkdir(parents=True, exist_ok=True)

for i in range(N_FRAMES):
    sec = START_SEC + i
    fid = int(sec * FPS)
    p = tmp_dir / f"f{fid}.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{sec:.3f}",
                    "-i", VIDEO, "-frames:v", "1", "-q:v", "2", str(p)],
                   check=True, capture_output=True)
    img = mpimg.imread(str(p))
    p.unlink()
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]

    client.reset()
    pred = client.predict(img)
    pred_jl[i] = pred["j_left"]  # 实测 (18,2)

    gt = get_row(fid)
    gt_jl[i] = np.array(gt["j_left"].to_list()[0], dtype=float)

client.close()

# 每个 row 的相关系数（与标注）
print(f"=== PRED_ROW 相关系数矩阵（30 帧测试集，j_left，共 {N_STEPS} 步）===")
print(f"{'row':>4} | {'corr_x':>7} {'corr_y':>7} {'corr_xy(合并)':>10}")
best = (-1, None)
for r in range(N_STEPS):
    cx = float(np.corrcoef(pred_jl[:, r, 0], gt_jl[:, 0])[0, 1])
    cy = float(np.corrcoef(pred_jl[:, r, 1], gt_jl[:, 1])[0, 1])
    # 合并 x,y 向量（展平为 2N）求相关
    flat_pred = np.concatenate([pred_jl[:, r, 0], pred_jl[:, r, 1]])
    flat_gt = np.concatenate([gt_jl[:, 0], gt_jl[:, 1]])
    cxy = float(np.corrcoef(flat_pred, flat_gt)[0, 1])
    print(f"{r:>4} | {cx:+.3f} {cy:+.3f} {cxy:+.3f}")
    if abs(cxy) > best[0]:
        best = (abs(cxy), r)

print(f"\n最佳 row: {best[1]} (|corr_xy|={best[0]:.3f})，当前使用 PRED_ROW={PRED_ROW}")
if best[0] > 0.3 and best[1] != PRED_ROW:
    print(">>> 结论：存在更优 PRED_ROW，M4 应重跑对比（记录为发现）")
elif best[0] <= 0.3:
    print(">>> 结论：所有 row 与标注相关系数均接近 0 → 支持'模型摇杆能力上限'，G11 归因收口")
else:
    print(f">>> 结论：当前 PRED_ROW={PRED_ROW} 已是最优，维持")
