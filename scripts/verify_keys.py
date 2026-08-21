# -*- coding: utf-8 -*-
"""G1 键序映射相关性法验证

方法：对测试集首 30 帧跑推理，统计模型 21 列各自按下率；
与标注 17 键按下率做顺序比较（秩相关）。若官方 BUTTON_ACTION_TOKENS 映射正确，
模型"高按下率列"应大致对应标注"高按下率键"（至少 right_trigger / dpad_left 两个强信号）。

用法：serve.py 已运行（端口 5555）；python scripts/verify_keys.py
"""
import sys
import time
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import polars as pl
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import (
    BUTTONS, MODEL_BUTTON_DIM, MODEL_BUTTON_ORDER, MODEL_COL_TO_BUTTON,
    BUTTON_TO_MODEL_COL, FPS, BTN_THRESHOLD, PRED_ROW, SHARD,
)

sys.path.insert(0, str(ROOT.parent.parent.parent / "NitroGen"))
from nitrogen.inference_client import ModelClient  # type: ignore

CHUNK_SIZE = 1200
N_FRAMES = 30
START_SEC = 640  # chunk_0032 起（测试集首 chunk，640~660s）

# ---------- 1. 标注按下率（30 帧） ----------
gt_counts = {b: 0 for b in BUTTONS}
for i in range(N_FRAMES):
    sec = START_SEC + i
    fid = int(sec * FPS)
    cid = f"{fid // CHUNK_SIZE:04d}"
    row = fid % CHUNK_SIZE
    df = pl.read_parquet(SHARD / f"Z1r1S--MJS4_chunk_{cid}" / "actions_processed.parquet").slice(row, 1)
    for b in BUTTONS:
        if int(df[b].sum()) > 0:
            gt_counts[b] += 1

# ---------- 2. 模型预测按下率 ----------
client = ModelClient(host="localhost", port=5555)
client.reset()
pred_counts = [0] * MODEL_BUTTON_DIM
for i in range(N_FRAMES):
    sec = START_SEC + i
    fid = int(sec * FPS)
    # 抽帧：ffmpeg -ss 精确到秒（对齐误差 ±1 帧，统计按下率可接受）
    img_path = ROOT / "results" / f"_vkey_{i}.png"
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{sec}.000",
                    "-i", r"D:/2+课产品/TOP 1 IN 2S _ Ranked 2v2 w_ oKhaliD (1).mp4",
                    "-frames:v", "1", "-q:v", "2", str(img_path)], check=True)
    img = mpimg.imread(str(img_path))
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    pred = client.predict(img)
    pred_btn = (pred["buttons"][PRED_ROW] > BTN_THRESHOLD).astype(int)
    for c in range(MODEL_BUTTON_DIM):
        if pred_btn[c] == 1:
            pred_counts[c] += 1
    img_path.unlink()  # 清理临时帧
client.close()

# ---------- 3. 汇总 ----------
print("=== 模型 21 列按下率（30 帧）===")
for c in range(MODEL_BUTTON_DIM):
    lbl = MODEL_COL_TO_BUTTON.get(c)
    print(f"  列{c:2d} {MODEL_BUTTON_ORDER[c]:<15} -> {lbl if lbl else '(无标注)'}  按下 {pred_counts[c]}/{N_FRAMES}")

print("\n=== 标注 17 键按下率（30 帧）===")
gt_rates = {b: gt_counts[b] / N_FRAMES for b in BUTTONS}
for b in sorted(gt_rates, key=lambda x: -gt_rates[x]):
    print(f"  {b:<15} 按下 {gt_counts[b]}/{N_FRAMES} ({gt_rates[b]:.0%})")

# ---------- 4. 秩相关对比 ----------
# 对每个标注键 b，找其模型列按下率
mapped = [(b, pred_counts[BUTTON_TO_MODEL_COL[b]] / N_FRAMES, gt_rates[b]) for b in BUTTONS]
print("\n=== 标注键 → 模型对应列按下率 vs 标注按下率 ===")
for b, p, g in sorted(mapped, key=lambda x: -x[2]):
    print(f"  {b:<15} 模型列按下率 {p:.0%}  标注按下率 {g:.0%}")

# 秩相关（Spearman，手写避免 scipy 依赖）：只比较"标注按下率>0"的键
pairs = [(p, g) for b, p, g in mapped if g > 0]
if len(pairs) >= 3:
    def rank(v):
        order = np.argsort(np.argsort(v))
        return order.astype(float)
    rx, ry = rank([x[0] for x in pairs]), rank([x[1] for x in pairs])
    rho = float(np.corrcoef(rx, ry)[0, 1])
    print(f"\nSpearman 秩相关（{len(pairs)} 个有信号键）: rho={rho:.3f}")
    print(f"结论: {'映射与官方 BUTTON_ACTION_TOKENS 一致（相关性法通过）' if rho > 0.5 else '相关性法未能充分确认映射，需人工复核'}")
else:
    print("\n有信号键不足，无法做秩相关")
