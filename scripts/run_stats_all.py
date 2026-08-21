# -*- coding: utf-8 -*-
"""全量统计批处理：遍历 Z1r1S--MJS4 全部 35 个 chunk，输出统计集/测试集分列的频率表

测试集划分（D5 决定）：chunk_0032~0034（尾部 3 个 chunks，60s/3600 帧）作测试集
其余 chunk_0000~0031 作统计集（600s/38400 帧，远超 M2 的 ≥500 帧要求）

输出：仓库 results/stats_summary.md（每 chunk 明细 + 统计集/测试集分列频率表 + 合计）
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import BUTTONS, SHARD
from stats import button_stats

TEST_CHUNKS = {f"{i:04d}" for i in range(32, 35)}  # 0032~0034

rows = []  # (chunk_id, frame_count, in_test, counts_dict)
all_pq = sorted(SHARD.glob("Z1r1S--MJS4_chunk_*/actions_processed.parquet"))
for p in all_pq:
    chunk_id = p.parent.name.split("_chunk_")[-1]  # "0000"
    in_test = chunk_id in TEST_CHUNKS
    fc, freq, counts = button_stats(p)
    rows.append((chunk_id, fc, in_test, counts))


def sum_counts(rs):
    c = {b: 0 for b in BUTTONS}
    for _, fc, _, counts in rs:
        for b in BUTTONS:
            c[b] += counts[b]
    return c, sum(r[1] for r in rs)


stat_rows = [r for r in rows if not r[2]]
test_rows = [r for r in rows if r[2]]
stat_counts, stat_frames = sum_counts(stat_rows)
test_counts, test_frames = sum_counts(test_rows)
all_counts, all_frames = sum_counts(rows)


def freq_table(c, frames, title):
    lines = [f"## {title}\n", f"帧数: {frames}\n", "| 按键 | 按下次数 | 占比 |", "|---|---|---|"]
    for b in BUTTONS:
        lines.append(f"| {b} | {c[b]} | {c[b]/frames:.2%} |")
    return lines


out = ROOT / "results" / "stats_summary.md"
out.parent.mkdir(exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    f.write("# 统计数据汇总（Z1r1S--MJS4 / rocket_league）\n\n")
    f.write(f"- 来源：`_data/SHARD_0088/Z1r1S--MJS4/`\n")
    f.write(f"- 总 chunks: {len(rows)}（每 chunk 1200 帧 × {len(rows)} = {all_frames} 帧）\n")
    f.write(f"- 测试集：chunk_0032~0034（3 个 chunks = 60s = {test_frames} 帧），与统计集隔离\n")
    f.write(f"- 统计集：其余 32 个 chunks = 600s = {stat_frames} 帧（远超 M2 的 ≥500 帧要求）\n\n")
    f.write("\n".join(freq_table(stat_counts, stat_frames, "统计集频率表（32 chunks，M2 口径）")) + "\n\n")
    f.write("\n".join(freq_table(test_counts, test_frames, "测试集频率表（3 chunks，M3 口径）")) + "\n\n")
    f.write("\n".join(freq_table(all_counts, all_frames, "全部 35 chunks 合计（仅供参考，非验收口径）")) + "\n\n")
    f.write("## 分 chunk 明细\n\n")
    f.write("| chunk_id | 帧数 | 归属 | 17 键总激活 |\n|---|---|---|---|\n")
    for cid, fc, in_test, counts in rows:
        total_act = sum(counts.values())
        f.write(f"| chunk_{cid} | {fc} | {'测试' if in_test else '统计'} | {total_act} |\n")
print(f"OK -> {out}")
print(f"stat frames: {stat_frames}, test frames: {test_frames}")
