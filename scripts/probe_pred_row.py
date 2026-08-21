# -*- coding: utf-8 -*-
"""G11 假设 2 验证：PRED_ROW 时序偏移——18 步动作块中哪一步与当前帧标注最相关

对测试集 30 帧逐帧推理（每次 reset 单帧），记录全部 MODEL_STEPS 步的 j_left；
对每步 row r 计算与当前帧标注 j_left 的相关系数（x 与 y 分开 + 合并），
看是否存在某个 row 显著优于 row 0。

结论判定：
- 若 max |corr| 集中在某 row>0 且明显高于 row0 → "PRED_ROW 应为 X"，M4 可重跑对比
- 若全部接近 0 → 支持"模型摇杆能力上限"，G11 归因收口

用法：serve.py 运行中；python scripts/probe_pred_row.py
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import FPS, MODEL_STEPS, SHARD, get_row, fetch_frame

sys.path.insert(0, str(ROOT.parent.parent.parent / "NitroGen"))
from nitrogen.inference_client import ModelClient  # type: ignore

N_FRAMES = 30
START_SEC = 640  # 测试集首段


def main():
    client = ModelClient(host="localhost", port=5555)
    client.reset()

    # 收集：gt_jl (N,2) + pred_jl (N, MODEL_STEPS, 2)
    gt_jl = np.zeros((N_FRAMES, 2), dtype=float)
    pred_jl = np.zeros((N_FRAMES, MODEL_STEPS, 2), dtype=float)

    tmp_dir = ROOT / "results" / "_tmp_probe"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i in range(N_FRAMES):
        sec = START_SEC + i
        fid = int(sec * FPS)
        img = fetch_frame(fid, tmp_dir)

        client.reset()
        pred = client.predict(img)
        pred_jl[i] = pred["j_left"]  # 实测 (MODEL_STEPS, 2)

        gt = get_row(fid)
        gt_jl[i] = np.array(gt["j_left"].to_list()[0], dtype=float)

    client.close()

    # 每个 row 的相关系数（与标注）
    print(f"=== PRED_ROW 相关系数矩阵（{N_FRAMES} 帧测试集，j_left，共 {MODEL_STEPS} 步）===")
    print(f"{'row':>4} | {'corr_x':>7} {'corr_y':>7} {'corr_xy(合并)':>10}")
    best: tuple[float, int] = (-1.0, -1)
    for r in range(MODEL_STEPS):
        cx = float(np.corrcoef(pred_jl[:, r, 0], gt_jl[:, 0])[0, 1])
        cy = float(np.corrcoef(pred_jl[:, r, 1], gt_jl[:, 1])[0, 1])
        # 合并 x,y 向量（展平为 2N）求相关
        flat_pred = np.concatenate([pred_jl[:, r, 0], pred_jl[:, r, 1]])
        flat_gt = np.concatenate([gt_jl[:, 0], gt_jl[:, 1]])
        cxy = float(np.corrcoef(flat_pred, flat_gt)[0, 1])
        print(f"{r:>4} | {cx:+.3f} {cy:+.3f} {cxy:+.3f}")
        if abs(cxy) > best[0]:
            best = (abs(cxy), r)

    print(f"\n最佳 row: {best[1]} (|corr_xy|={best[0]:.3f})")
    if best[0] > 0.3 and best[1] != 0:
        print(">>> 结论：存在更优 PRED_ROW，M4 应重跑对比（记录为发现）")
    elif best[0] <= 0.3:
        print(">>> 结论：所有 row 与标注相关系数均接近 0 → 支持'模型摇杆能力上限'，G11 归因收口")
    else:
        print(">>> 结论：PRED_ROW=0 已是最优，维持")


if __name__ == "__main__":
    main()
