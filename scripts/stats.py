# -*- coding: utf-8 -*-
"""统计脚本骨架（M2 第一步）：读单视频 parquet 输出 17 键频率表 + 帧数

用法:
    python scripts/stats.py <parquet路径>
    python scripts/stats.py _data/SHARD_0088/Z1r1S--MJS4/Z1r1S--MJS4_chunk_0000/actions_processed.parquet

输出:
    17 键频率表（Markdown 表格，stdout）+ 帧数；不画图（D5 范围）
"""
import sys
from pathlib import Path

import polars as pl

# 标注 17 键（与 actions_processed.parquet 列一致）
BUTTONS = [
    "back", "dpad_down", "dpad_left", "dpad_right", "dpad_up", "east",
    "guide", "left_shoulder", "left_thumb", "left_trigger", "north",
    "right_shoulder", "right_thumb", "right_trigger", "south", "start", "west",
]


def button_frequency(parquet_path: Path) -> dict[str, float]:
    """返回 17 键按下占比 {键名: 0~1}。缺列抛 KeyError。"""
    if not parquet_path.exists():
        raise FileNotFoundError(f"parquet not found: {parquet_path}")
    df = pl.read_parquet(parquet_path)
    n = df.height
    if n == 0:
        return {b: 0.0 for b in BUTTONS}
    freq: dict[str, float] = {}
    for b in BUTTONS:
        if b not in df.columns:
            raise KeyError(f"missing column: {b}")
        freq[b] = float(df[b].sum()) / n
    return freq


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    parquet = Path(sys.argv[1])
    try:
        freq = button_frequency(parquet)
    except (FileNotFoundError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # 帧数
    df = pl.read_parquet(parquet)
    total = df.height

    # Markdown 表格
    print(f"# 按键频率表（{parquet.parent.name}）")
    print(f"帧数: {total}")
    print()
    print("| 按键 | 按下帧数 | 占比 |")
    print("|---|---|---|")
    for b in BUTTONS:
        count = int(df[b].sum())
        print(f"| {b} | {count} | {freq[b]:.1%} |")


if __name__ == "__main__":
    main()
