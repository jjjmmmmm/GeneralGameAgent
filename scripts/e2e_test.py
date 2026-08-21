# -*- coding: utf-8 -*-
"""T3 端到端贯通：真帧 → ModelClient.predict → 与标注对齐 → 简化指标

M1 推理服务已起，端口 5555
输入：5 帧真图（seq1 5 帧，60~64s）
输出：results/e2e_metrics.md（含实际 shape、每帧指标、合计）
"""
import sys, pickle
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import polars as pl
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent.parent.parent / "NitroGen"))
from nitrogen.inference_client import ModelClient  # type: ignore

SEQ_DIR = Path(r"D:/2+课产品/_data/frames/Z1r1S--MJS4/seq")
SHARD = Path(r"D:/2+课产品/_data/SHARD_0088/Z1r1S--MJS4")
FIG_OUT = ROOT / "results" / "figures"
FIG_OUT.mkdir(exist_ok=True)
OUT_MD = ROOT / "results" / "e2e_metrics.md"

BUTTONS_17 = ["back","dpad_down","dpad_left","dpad_right","dpad_up","east","guide",
              "left_shoulder","left_thumb","left_trigger","north","right_shoulder",
              "right_thumb","right_trigger","south","start","west"]

# 帧号 → parquet 行号
CHUNK_SIZE = 1200
def frame_to_chunk_row(fid):
    return f"{fid // CHUNK_SIZE:04d}", fid % CHUNK_SIZE

chunk_cache = {}
def get_row(fid):
    cid, row = frame_to_chunk_row(fid)
    if cid not in chunk_cache:
        chunk_cache[cid] = pl.read_parquet(SHARD / f"Z1r1S--MJS4_chunk_{cid}" / "actions_processed.parquet")
    return chunk_cache[cid].slice(row, 1)

# ---------- 1. 用 ModelClient.predict 跑 5 帧 ----------
print("=== T3 端到端（真帧）===")
client = ModelClient(host="localhost", port=5555)
client.reset()
preds = []
shapes_printed = False
for fi in range(5):
    sec = 60 + fi
    fid = int(sec * 60)
    img_path = SEQ_DIR / f"seq01_f{fi}.png"
    img = mpimg.imread(str(img_path))
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.shape[-1] == 4:  # RGBA
        img = img[..., :3]
    t0 = __import__("time").time()
    pred = client.predict(img)
    dt = __import__("time").time() - t0
    if not shapes_printed:
        print(f"  pred keys: {list(pred.keys())}")
        for k, v in pred.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        shapes_printed = True
    preds.append((fi, sec, fid, pred, dt))
    print(f"  fi={fi} sec={sec}s predict dt={dt:.3f}s")
client.close()

# ---------- 2. 与标注对齐（用模型行 0 试，索引 0~16 对齐标注 17 键） ----------
print("\n=== 对齐指标 ===")
rows_md = []
correct_per_frame = []
mse_per_frame = []
n_pred_btn_0_pressed_total = 0
n_gt_btn_pressed_total = 0
n_both_pressed_total = 0

# 模型行 0 = 当前帧预测
PRED_ROW = 0

for fi, sec, fid, pred, dt in preds:
    gt = get_row(fid)
    # 模型按钮（行 0，前 17 列假设对应标注 17 键）
    pred_btn = (pred["buttons"][PRED_ROW, :17] > 0.5).astype(int)  # 阈值 0.5
    gt_btn = np.array([int(gt[b].sum()) for b in BUTTONS_17], dtype=int)
    n_pred_pressed = int(pred_btn.sum())
    n_gt_pressed = int(gt_btn.sum())
    n_match = int(((pred_btn == 1) & (gt_btn == 1)).sum())
    n_pred_only = int(((pred_btn == 1) & (gt_btn == 0)).sum())
    n_gt_only = int(((pred_btn == 0) & (gt_btn == 1)).sum())
    n_both_zero = int(((pred_btn == 0) & (gt_btn == 0)).sum())
    # 准确率：所有 17 键全对比例
    n_all_correct = int((pred_btn == gt_btn).sum())
    acc = n_all_correct / len(BUTTONS_17)
    # 摇杆 j_left MSE
    pred_jl = pred["j_left"][PRED_ROW]
    gt_jl = np.array(gt["j_left"].to_list()[0], dtype=float)
    jl_mse = float(np.mean((pred_jl - gt_jl) ** 2))

    correct_per_frame.append(acc)
    mse_per_frame.append(jl_mse)
    n_pred_btn_0_pressed_total += n_pred_pressed
    n_gt_btn_pressed_total += n_gt_pressed
    n_both_pressed_total += n_match

    rows_md.append(
        f"| fi={fi} | {sec}s | {fid} | {n_pred_pressed} | {n_gt_pressed} | {n_match} | {n_pred_only} | {n_gt_only} | {n_all_correct}/17 | {acc:.1%} | {jl_mse:.4f} |"
    )
    print(f"  fi={fi}: acc={acc:.1%} (correct={n_all_correct}/17)  jl_mse={jl_mse:.4f}  pred_press={n_pred_pressed} gt_press={n_gt_pressed}")

# ---------- 3. 写 Markdown ----------
total_acc = sum(correct_per_frame) / len(correct_per_frame)
total_mse = sum(mse_per_frame) / len(mse_per_frame)
precision = n_both_pressed_total / max(1, n_pred_btn_0_pressed_total)
recall = n_both_pressed_total / max(1, n_gt_btn_pressed_total)

with OUT_MD.open("w", encoding="utf-8") as f:
    f.write("# T3 端到端主路径指标（5 真帧 + ModelClient.predict）\n\n")
    f.write("## 输入\n- 5 帧：seq01 f0~f4（视频 60~64s，从统计集 chunk_0001）\n")
    f.write("- 真帧从视频抽帧，与 actions_processed.parquet 按映射 `帧号 = chunk_id×1200 + 行号` 对齐\n")
    f.write("- 模型侧：取 `pred['buttons'][0, :17]`（行 0 = 当前帧预测，前 17 列假设对应标注 17 键——待 D6 验证键序对齐）\n")
    f.write("- 标注侧：按 17 键列名读取，值 0/1\n")
    f.write("- 阈值：按钮预测 > 0.5 视为按下\n\n")
    f.write("## 模型输出 shape（实际）\n")
    f.write("- j_left / j_right: `(N_steps, 2)`\n")
    f.write("- buttons: `(N_steps, 21)`（21 键含 trigger axis 等）\n\n")
    f.write("## 每帧指标\n\n")
    f.write("| frame | sec | 帧号 | pred_press | gt_press | both_press | pred_only | gt_only | 一致按键 | 按键准确率 | j_left MSE |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows_md:
        f.write(r + "\n")
    f.write(f"\n## 合计（5 帧）\n")
    f.write(f"- 平均按键准确率（17 键全对比例）: **{total_acc:.1%}**\n")
    f.write(f"- 平均 j_left MSE: **{total_mse:.4f}**\n")
    f.write(f"- 总 pred_press={n_pred_btn_0_pressed_total}, gt_press={n_gt_btn_pressed_total}, both_press={n_both_pressed_total}\n")
    f.write(f"- 查准率（pred 命中 / pred 全）: {precision:.1%}\n")
    f.write(f"- 查全率（pred 命中 / gt 全）: {recall:.1%}\n\n")
    f.write("## 已知不确定项（D6 块 A 处理）\n")
    f.write("- 17↔21 键位置映射未验证（今日假设前 17 列对应 actions_processed 17 键）\n")
    f.write("- 模型行 0 vs 当前帧标注的对齐方式未严格验证（cfg action_per_chunk=8 暗示未来步预测）\n")
    f.write("- 摇杆 j_right 全 0（按计划 Rocket League 不使用右摇杆），未比对\n")

print(f"\nDONE -> {OUT_MD}")
print(f"avg acc={total_acc:.1%}, jl_mse={total_mse:.4f}")
