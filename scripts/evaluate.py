# -*- coding: utf-8 -*-
"""M3 测试集全量评测（G2+G3）：≥200 帧隔离测试集 → 推理 → 指标

测试集：chunk_0032~0034（640~700s，3600 帧），与统计集（0~640s）隔离
抽样：等间隔取 N=200 帧（默认），可 --n 调整
随机性：flow matching 采样非确定性（D5 发现）。默认 K=3 次多数票；--k 1 时单次。
       serve 侧无 seed 参数（D6 确认），故用客户端多数票方案。

用法：serve.py 运行中；python scripts/evaluate.py [--n 200] [--k 3] [--out results/test_set_metrics.md]
输出：每帧明细 + 合计（M4 判定）
"""
import argparse
import sys
import time
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import (
    BUTTONS, FPS, BTN_THRESHOLD, PRED_ROW, SHARD, BUTTON_TO_MODEL_COL,
    VALID_MODEL_COLS, MODEL_BUTTON_DIM, CHUNK_SIZE, fetch_frame,
)

sys.path.insert(0, str(ROOT.parent.parent.parent / "NitroGen"))
from nitrogen.inference_client import ModelClient  # type: ignore

TEST_CHUNKS = ["0032", "0033", "0034"]
TEST_START_SEC = 640          # chunk_0032 起点
TEST_DURATION_SEC = 60        # 3 chunks × 20s


def pick_frames(n: int) -> list[int]:
    """从测试集 3600 帧中等间隔取 n 帧（全局帧号）。"""
    total = len(TEST_CHUNKS) * CHUNK_SIZE
    step = total / n
    return [TEST_START_SEC * FPS + int(i * step) for i in range(n)]


def get_label_btn(fid: int, cache: dict) -> np.ndarray:
    cid = f"{fid // CHUNK_SIZE:04d}"
    row = fid % CHUNK_SIZE
    if cid not in cache:
        cache[cid] = pl.read_parquet(SHARD / f"Z1r1S--MJS4_chunk_{cid}" / "actions_processed.parquet")
    return np.array([int(cache[cid].slice(row, 1)[b].sum()) for b in BUTTONS], dtype=int)


def predict_vote(client, img: np.ndarray, k: int, pred_row: int):
    """K 次推理按钮多数票（每帧前 reset，保证单帧上下文）；返回 (投票按钮21维, 末次预测)。"""
    votes = np.zeros((MODEL_BUTTON_DIM,), dtype=int)
    last_pred = None
    for _ in range(k):
        client.reset()  # 每次推理前清上下文，保证单帧输入
        last_pred = client.predict(img)
        votes += (last_pred["buttons"][pred_row] > BTN_THRESHOLD).astype(int)
    return (votes >= (k + 1) // 2).astype(int), last_pred  # 简单多数


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=3, help="每帧推理次数（多数票，flow matching 随机性控制）")
    ap.add_argument("--out", default="results/test_set_metrics.md")
    args = ap.parse_args()

    frames = pick_frames(args.n)
    tmp_dir = ROOT / "results" / "_tmp_frames"
    tmp_dir.mkdir(exist_ok=True)

    print(f"测试集评测：{len(frames)} 帧（chunk 0032~0034），K={args.k} 多数票")
    client = ModelClient(host="localhost", port=5555)
    client.reset()

    cache = {}
    rows_md = []
    agg = {  # 汇总
        "n_pred": 0, "n_gt": 0, "n_both": 0,
        "accs": [], "jl_mse": [], "jl_corr": [],
        "times": [],
    }
    t_start = time.time()

    for i, fid in enumerate(frames):
        img = fetch_frame(fid, tmp_dir)
        t0 = time.time()
        pred_btn, last_pred = predict_vote(client, img, args.k, PRED_ROW)
        dt = time.time() - t0
        pred_jl = last_pred["j_left"][PRED_ROW]

        gt_btn = get_label_btn(fid, cache)
        gt_jl = np.array(cache[f"{fid // CHUNK_SIZE:04d}"].slice(fid % CHUNK_SIZE, 1)["j_left"].to_list()[0], dtype=float)

        # 按键：按映射取 17 维
        pred_btn17 = np.array([pred_btn[BUTTON_TO_MODEL_COL[b]] for b in BUTTONS], dtype=int)
        n_pred = int(pred_btn17.sum())
        n_gt = int(gt_btn.sum())
        n_both = int(((pred_btn17 == 1) & (gt_btn == 1)).sum())
        n_all_correct = int((pred_btn17 == gt_btn).sum())
        acc = n_all_correct / len(BUTTONS)
        jl_mse = float(np.mean((pred_jl - gt_jl) ** 2))
        jl_corr = float(np.corrcoef(pred_jl, gt_jl)[0, 1]) if np.std(pred_jl) > 1e-6 and np.std(gt_jl) > 1e-6 else float("nan")

        agg["n_pred"] += n_pred
        agg["n_gt"] += n_gt
        agg["n_both"] += n_both
        agg["accs"].append(acc)
        agg["jl_mse"].append(jl_mse)
        agg["jl_corr"].append(jl_corr)
        agg["times"].append(dt)

        rows_md.append(
            f"| {i} | {fid} | {n_pred} | {n_gt} | {n_both} | {n_all_correct}/17 | {acc:.1%} | {jl_mse:.4f} | {jl_corr:+.3f} | {dt:.3f}s |"
        )
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(frames)} 帧（累计 {time.time()-t_start:.0f}s）")

    client.close()
    total_sec = time.time() - t_start

    # ---------- 汇总 ----------
    n = len(frames)
    avg_acc = sum(agg["accs"]) / n
    avg_mse = sum(agg["jl_mse"]) / n
    valid_corr = [c for c in agg["jl_corr"] if not np.isnan(c)]
    avg_corr = sum(valid_corr) / len(valid_corr) if valid_corr else float("nan")
    precision = agg["n_both"] / max(1, agg["n_pred"])
    recall = agg["n_both"] / max(1, agg["n_gt"])
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    avg_dt = sum(agg["times"]) / n

    out = ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"# 测试集评测指标（M3/M4）\n\n")
        f.write(f"- 测试集：chunk_0032~0034（640~700s），与统计集隔离\n")
        f.write(f"- 采样：等间隔 {n} 帧；键序映射：G1 官方 BUTTON_ACTION_TOKENS\n")
        f.write(f"- 随机性控制：每帧 K={args.k} 次推理按钮多数票 + 摇杆末次（flow matching 无 seed 参数）\n")
        f.write(f"- PRED_ROW={PRED_ROW}（时序语义未严格验证，命中法无信号，见 docs/键序映射表.md）\n\n")
        f.write("## 合计\n\n")
        f.write(f"- 按键准确率（17 键全对比例）: **{avg_acc:.1%}**（M4 要求 ≥50%）\n")
        f.write(f"- 触发精确率（pred 命中/pred 全）: **{precision:.1%}**\n")
        f.write(f"- 触发召回率（pred 命中/gt 全）: **{recall:.1%}**\n")
        f.write(f"- F1: **{f1:.3f}**\n")
        f.write(f"- 摇杆 j_left MSE: **{avg_mse:.4f}**\n")
        f.write(f"- 摇杆 j_left 相关系数: **{avg_corr:+.3f}**（M4 要求 ≥0.4）\n")
        f.write(f"- 单帧推理均时: {avg_dt:.3f}s；{n} 帧总耗时: {total_sec:.0f}s\n")
        f.write(f"- 总按键事件: pred={agg['n_pred']}, gt={agg['n_gt']}, 共同命中={agg['n_both']}\n\n")
        f.write(f"## M4 判定\n\n")
        f.write(f"- 按键准确率 ≥50%: **{'✅ 达标' if avg_acc >= 0.5 else '❌ 未达标'}**\n")
        f.write(f"- 摇杆相关系数 ≥0.4: **{'✅ 达标' if avg_corr >= 0.4 else '❌ 未达标'}**\n\n")
        f.write("## 每帧明细\n\n")
        f.write("| idx | 帧号 | pred_press | gt_press | both | 一致按键 | 准确率 | jl_mse | jl_corr | 耗时 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows_md:
            f.write(r + "\n")

    print(f"\nDONE -> {out}")
    print(f"avg_acc={avg_acc:.1%}, precision={precision:.1%}, recall={recall:.1%}, f1={f1:.3f}, avg_corr={avg_corr:+.3f}, total={total_sec:.0f}s")


if __name__ == "__main__":
    main()
