# -*- coding: utf-8 -*-
"""T3 端到端贯通：真帧 → ModelClient.predict → 与标注对齐 → 简化指标

M1 推理服务已起，端口 5555
输入：5 帧真图（seq1 5 帧，60~64s）——D5 验证链路；D6 块 A 扩展至 ≥200 帧（G2）
输出：results/e2e_metrics.md（含实际 shape、每帧指标、合计）

已知不确定项（D6 块 A 处理，见 docs/差距清单.md G1~G3）：
- 17↔21 键位置映射（当前假设前 17 列）
- 模型行 0 vs 当前帧标注的对齐方式
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
from common import BUTTONS, FPS, BTN_THRESHOLD, PRED_ROW, get_row

sys.path.insert(0, str(ROOT.parent.parent.parent / "NitroGen"))
from nitrogen.inference_client import ModelClient  # type: ignore

SEQ_DIR = Path(r"D:/2+课产品/_data/frames/Z1r1S--MJS4/seq")
OUT_MD = ROOT / "results" / "e2e_metrics.md"

N_FRAMES = 5
SEQ_START_SEC = 60
GRAY_BASELINE = 0.48  # D4 灰图单帧推理耗时基线（秒）

# ---------- 1. 推理 ----------
print("=== T3 端到端（真帧）===")
client = ModelClient(host="localhost", port=5555)
client.reset()
preds = []
times = []
shapes_printed = False
for fi in range(N_FRAMES):
    sec = SEQ_START_SEC + fi
    fid = int(sec * FPS)
    img_path = SEQ_DIR / f"seq01_f{fi}.png"
    img = mpimg.imread(str(img_path))
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.shape[-1] == 4:  # RGBA
        img = img[..., :3]
    t0 = time.time()
    pred = client.predict(img)
    dt = time.time() - t0
    if not shapes_printed:
        print(f"  pred keys: {list(pred.keys())}")
        for k, v in pred.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        shapes_printed = True
    preds.append((fi, sec, fid, pred, dt))
    times.append(dt)
    print(f"  fi={fi} sec={sec}s predict dt={dt:.3f}s")
client.close()

# ---------- 2. 指标 ----------
print("\n=== 对齐指标 ===")
rows_md = []
correct_per_frame = []
mse_per_frame = []
corr_per_frame = []
n_pred_pressed_total = 0
n_gt_pressed_total = 0
n_both_pressed_total = 0

for fi, sec, fid, pred, dt in preds:
    gt = get_row(fid)
    # 模型按钮（PRED_ROW 行，前 17 列假设对应标注 17 键——G1 待验证）
    pred_btn = (pred["buttons"][PRED_ROW, :len(BUTTONS)] > BTN_THRESHOLD).astype(int)
    gt_btn = np.array([int(gt[b].sum()) for b in BUTTONS], dtype=int)
    n_pred_pressed = int(pred_btn.sum())
    n_gt_pressed = int(gt_btn.sum())
    n_match = int(((pred_btn == 1) & (gt_btn == 1)).sum())
    n_pred_only = int(((pred_btn == 1) & (gt_btn == 0)).sum())
    n_gt_only = int(((pred_btn == 0) & (gt_btn == 1)).sum())
    n_all_correct = int((pred_btn == gt_btn).sum())
    acc = n_all_correct / len(BUTTONS)
    # 摇杆 j_left MSE 与相关系数
    pred_jl = pred["j_left"][PRED_ROW]
    gt_jl = np.array(gt["j_left"].to_list()[0], dtype=float)
    jl_mse = float(np.mean((pred_jl - gt_jl) ** 2))
    jl_corr = float(np.corrcoef(pred_jl, gt_jl)[0, 1]) if np.std(pred_jl) > 0 and np.std(gt_jl) > 0 else float("nan")

    correct_per_frame.append(acc)
    mse_per_frame.append(jl_mse)
    corr_per_frame.append(jl_corr)
    n_pred_pressed_total += n_pred_pressed
    n_gt_pressed_total += n_gt_pressed
    n_both_pressed_total += n_match

    rows_md.append(
        f"| fi={fi} | {sec}s | {fid} | {n_pred_pressed} | {n_gt_pressed} | {n_match} | {n_pred_only} | {n_gt_only} | {n_all_correct}/{len(BUTTONS)} | {acc:.1%} | {jl_mse:.4f} | {jl_corr:+.3f} |"
    )
    print(f"  fi={fi}: acc={acc:.1%} (correct={n_all_correct}/{len(BUTTONS)})  jl_mse={jl_mse:.4f} jl_corr={jl_corr:+.3f}  pred_press={n_pred_pressed} gt_press={n_gt_pressed}")

# ---------- 3. 写 Markdown ----------
total_acc = sum(correct_per_frame) / len(correct_per_frame)
total_mse = sum(mse_per_frame) / len(mse_per_frame)
valid_corr = [c for c in corr_per_frame if not np.isnan(c)]
total_corr = sum(valid_corr) / len(valid_corr) if valid_corr else float("nan")
avg_dt = sum(times) / len(times)
precision = n_both_pressed_total / max(1, n_pred_pressed_total)
recall = n_both_pressed_total / max(1, n_gt_pressed_total)

with OUT_MD.open("w", encoding="utf-8") as f:
    f.write("# T3 端到端主路径指标（5 真帧 + ModelClient.predict）\n\n")
    f.write("## 输入\n")
    f.write(f"- {N_FRAMES} 帧：seq01 f0~f{N_FRAMES-1}（视频 60~64s，从统计集 chunk_0001）\n")
    f.write("- 真帧从视频抽帧，与 actions_processed.parquet 按映射 `帧号 = chunk_id×1200 + 行号` 对齐\n")
    f.write(f"- 模型侧：取 `pred['buttons'][{PRED_ROW}, :{len(BUTTONS)}]`（行 {PRED_ROW} = 当前帧预测，前 {len(BUTTONS)} 列假设对应标注 17 键——G1 待 D6 验证）\n")
    f.write("- 标注侧：按 17 键列名读取，值 0/1\n")
    f.write(f"- 阈值：按钮预测 > {BTN_THRESHOLD} 视为按下\n\n")
    f.write("## 模型输出 shape（实际）\n")
    f.write("- j_left / j_right: `(N_steps, 2)`\n")
    f.write("- buttons: `(N_steps, 21)`（21 键含 trigger axis 等）\n\n")
    f.write("## 每帧指标\n\n")
    f.write("| frame | sec | 帧号 | pred_press | gt_press | both_press | pred_only | gt_only | 一致按键 | 按键准确率 | j_left MSE | j_left 相关系数 |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows_md:
        f.write(r + "\n")
    f.write(f"\n## 合计（{N_FRAMES} 帧）\n")
    f.write(f"- 平均按键准确率（17 键全对比例）: **{total_acc:.1%}**\n")
    f.write(f"- 平均 j_left MSE: **{total_mse:.4f}**\n")
    f.write(f"- 平均 j_left 相关系数: **{total_corr:+.3f}**\n")
    f.write(f"- 单帧推理均时: **{avg_dt:.3f}s**（对照 D4 灰图基线 {GRAY_BASELINE}s）\n")
    f.write(f"- 总 pred_press={n_pred_pressed_total}, gt_press={n_gt_pressed_total}, both_press={n_both_pressed_total}\n")
    f.write(f"- 查准率（pred 命中 / pred 全）: {precision:.1%} —— ⚠️ 非 M4 验收口径，D6 块 A 重算\n")
    f.write(f"- 查全率（pred 命中 / gt 全）: {recall:.1%} —— ⚠️ 非 M4 验收口径，D6 块 A 重算\n\n")
    f.write("## 已知不确定项（D6 块 A 处理，见 docs/差距清单.md）\n")
    f.write(f"- G1 17↔21 键位置映射未验证（假设前 {len(BUTTONS)} 列对应 actions_processed 17 键）\n")
    f.write("- G2 本验证仅 5 帧，非 M3 的 ≥200 帧测试集评测\n")
    f.write("- 摇杆 j_right 全 0（按计划 Rocket League 不使用右摇杆），未比对\n")

print(f"\nDONE -> {OUT_MD}")
print(f"avg acc={total_acc:.1%}, jl_mse={total_mse:.4f}, jl_corr={total_corr:+.3f}, avg_dt={avg_dt:.3f}s")
