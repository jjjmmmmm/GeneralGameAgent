# -*- coding: utf-8 -*-
"""PRED_ROW 时序偏移探测：对 5 帧取动作块不同行(0~4)，统计与标注的"按下事件命中"数

背景：cfg action_per_chunk=8、action_shift=1，模型输出的 8 步动作块的时序语义未严格验证。
若 row 0 是"当前帧之后第 1 步"，则应对比帧+1 的标注；验证方式：
对每帧 f，取 pred row r，对比标注 f、f+1、f+2... 看哪个时序匹配按下事件最多。
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import polars as pl
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import BUTTON_TO_MODEL_COL, FPS, BTN_THRESHOLD, PRED_ROW, SHARD

sys.path.insert(0, str(ROOT.parent.parent.parent / "NitroGen"))
from nitrogen.inference_client import ModelClient  # type: ignore

CHUNK_SIZE = 1200
N_FRAMES = 5
START_SEC = 60
VIDEO = r"D:/2+课产品/TOP 1 IN 2S _ Ranked 2v2 w_ oKhaliD (1).mp4"

# 抽帧 + 推理
import subprocess
client = ModelClient(host="localhost", port=5555)
client.reset()
preds = []
frames = []
for i in range(N_FRAMES):
    sec = START_SEC + i
    fid = int(sec * FPS)
    img_path = ROOT / "results" / f"_probe_{i}.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{sec}.000",
                    "-i", VIDEO, "-frames:v", "1", "-q:v", "2", str(img_path)], check=True)
    img = mpimg.imread(str(img_path))
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    preds.append((fid, client.predict(img)))
    frames.append(fid)
    img_path.unlink()
client.close()

# 标注读取（缓存 chunk）
cache = {}
def get_label(fid, shift):
    f = fid + shift
    cid = f"{f // CHUNK_SIZE:04d}"
    row = f % CHUNK_SIZE
    if cid not in cache:
        cache[cid] = pl.read_parquet(SHARD / f"Z1r1S--MJS4_chunk_{cid}" / "actions_processed.parquet")
    return {b: int(cache[cid].slice(row, 1)[b].sum()) for b in BUTTON_TO_MODEL_COL}

# 对每个 row r (0..4) 和每个 shift (0..4)，统计按下事件命中（pred 按 & gt 按）
print("行/时序 命中矩阵：行=模型 row r，列=标注偏移 shift")
print("           shift0  shift1  shift2  shift3  shift4")
all_res = {}
for r in range(5):
    row_hits = []
    for shift in range(5):
        hits = 0
        for fid, pred in preds:
            pred_btn = (pred["buttons"][r] > BTN_THRESHOLD).astype(int)
            lbl = get_label(fid, shift)
            for b, col in BUTTON_TO_MODEL_COL.items():
                if pred_btn[col] == 1 and lbl[b] == 1:
                    hits += 1
        row_hits.append(hits)
        all_res[(r, shift)] = hits
    print(f"  row{r}:  {row_hits[0]:5d}  {row_hits[1]:5d}  {row_hits[2]:5d}  {row_hits[3]:5d}  {row_hits[4]:5d}")

best = max(all_res, key=all_res.get)
print(f"\n最佳 (row, shift) = {best}, 命中 {all_res[best]}")
print("若 best shift=0 → row 0 即当前帧（PRED_ROW=0 正确）；shift>0 → 模型预测偏未来")
