# -*- coding: utf-8 -*-
"""D5 共享常量与工具：按钮列表、帧号映射、分块读取缓存

供 run_stats_all.py / visualize.py / e2e_test.py 复用，避免重复实现。
stats.py 为既有接口（D4 审查通过），保持独立不依赖本模块。
"""
import polars as pl
from pathlib import Path

# 标注 17 键（与 actions_processed.parquet 列一致）
BUTTONS = [
    "back", "dpad_down", "dpad_left", "dpad_right", "dpad_up", "east",
    "guide", "left_shoulder", "left_thumb", "left_trigger", "north",
    "right_shoulder", "right_thumb", "right_trigger", "south", "start", "west",
]

# 数据布局常量（来自 T1 帧对齐验证）
CHUNK_SIZE = 1200          # 每 chunk 帧数（20s × 60fps）
FPS = 60                   # 视频帧率
BUTTON_DIM = 17            # 标注键数
MODEL_BUTTON_DIM = 21      # 模型输出 buttons 维度（含拨片等）
MODEL_STEPS = 8            # 动作块步数
BTN_THRESHOLD = 0.5        # 按钮预测视为按下的阈值
PRED_ROW = 0               # 取动作块第 0 步作为当前帧预测（D6 需严格验证）

# 选定视频的数据根路径
SHARD = Path(r"D:/2+课产品/_data/SHARD_0088/Z1r1S--MJS4")


def frame_to_chunk_row(fid: int) -> tuple[str, int]:
    """视频帧号 → (chunk_id 四位字符串, parquet 行号)。"""
    return f"{fid // CHUNK_SIZE:04d}", fid % CHUNK_SIZE


_chunk_cache: dict[str, pl.DataFrame] = {}


def get_chunk_df(cid: str) -> pl.DataFrame:
    """读取指定 chunk 的 actions_processed.parquet，带缓存。"""
    if cid not in _chunk_cache:
        p = SHARD / f"Z1r1S--MJS4_chunk_{cid}" / "actions_processed.parquet"
        _chunk_cache[cid] = pl.read_parquet(p)
    return _chunk_cache[cid]


def get_row(fid: int) -> pl.DataFrame:
    """按视频帧号取单行标注（1 行 DataFrame）。"""
    cid, row = frame_to_chunk_row(fid)
    return get_chunk_df(cid).slice(row, 1)
