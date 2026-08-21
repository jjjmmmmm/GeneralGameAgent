# -*- coding: utf-8 -*-
"""T2 全量统计批处理：遍历 Z1r1S--MJS4 全部 35 个 chunk 调 stats.button_stats，汇总到 results/stats_summary.md

测试集划分：chunk_0032~0034（尾部 3 个 chunks，60s/3600 帧）作为测试集候选（≥200 帧隔离）
其余 chunk_0000~0031 作为统计集（600s/38400 帧，远超 M2 的 ≥500 帧要求）

输出：仓库 results/stats_summary.md（带每 chunk + 合计 + 测试集/统计集分列）
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

# 让脚本能 import 仓库内 stats.py
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from stats import button_stats, BUTTONS

SHARD = Path(r"D:/2+课产品/_data/SHARD_0088/Z1r1S--MJS4")
TEST_CHUNKS = {f"{i:04d}" for i in range(32, 35)}  # 0032~0034

rows = []  # (chunk_id, frame_count, in_test, counts_dict)
all_pq = sorted(SHARD.glob("Z1r1S--MJS4_chunk_*/actions_processed.parquet"))
for p in all_pq:
    chunk_id = p.parent.name.split("_chunk_")[-1]  # "0000"
    chunk_dir = f"chunk_{chunk_id}"
    in_test = chunk_id in TEST_CHUNKS
    fc, freq, counts = button_stats(p)
    rows.append((chunk_id, fc, in_test, counts))

# 合计
total_counts = {b: 0 for b in BUTTONS}
total_frames = 0
for _, fc, _, counts in rows:
    total_frames += fc
    for b in BUTTONS:
        total_counts[b] += counts[b]

# 写 Markdown
out = ROOT / "results" / "stats_summary.md"
out.parent.mkdir(exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    f.write("# 统计数据汇总（Z1r1S--MJS4 / rocket_league）\n\n")
    f.write(f"- 来源：`_data/SHARD_0088/Z1r1S--MJS4/`\n")
    f.write(f"- 总 chunks: {len(rows)}（每 chunk 1200 帧 × {len(rows)} = {total_frames} 帧）\n")
    f.write(f"- 测试集候选（待 M3 验证后正式划定）：chunk_0032~0034（3 个 chunks = 60s = 3600 帧）\n")
    f.write(f"- 统计集：其余 32 个 chunks = 600s = {total_frames - 3*1200} 帧（远超 M2 的 ≥500 帧要求）\n\n")
    f.write("## 总计（全部 35 chunks）\n\n")
    f.write(f"总帧数: {total_frames}\n\n")
    f.write("| 按键 | 按下次数 | 占比 |\n|---|---|---|\n")
    for b in BUTTONS:
        f.write(f"| {b} | {total_counts[b]} | {total_counts[b]/total_frames:.2%} |\n")
    f.write(f"\n## 分 chunk 明细\n\n")
    f.write("| chunk_id | 帧数 | 归属 | 17 键总激活 |\n|---|---|---|---|\n")
    for cid, fc, in_test, counts in rows:
        total_act = sum(counts.values())
        f.write(f"| chunk_{cid} | {fc} | {'测试' if in_test else '统计'} | {total_act} |\n")
print(f"OK -> {out} (total frames: {total_frames})")
print(f"test chunks: {sorted(TEST_CHUNKS)}, stat chunks: 32")
