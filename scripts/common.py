# -*- coding: utf-8 -*-
"""D5 共享常量与工具：按钮列表、帧号映射、分块读取缓存、抽帧

供 run_stats_all.py / visualize.py / e2e_test.py / evaluate.py / viz_curves.py / probe_pred_row.py 复用，
避免重复实现。stats.py 为既有接口（D4 审查通过），保持独立不依赖本模块。
"""
import subprocess
import numpy as np
import polars as pl
import matplotlib.image as mpimg
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
MODEL_BUTTON_DIM = 21      # 模型输出 buttons 维度
MODEL_STEPS = 18           # 动作块步数（D7 实测 pred shape (18,2)/(18,21)，原 8 为错误推断）
BTN_THRESHOLD = 0.5        # 按钮预测视为按下的阈值
PRED_ROW = 0               # 取动作块第 0 步作为当前帧预测（D7 已 18 步全测排除时序偏移）

# 模型 buttons 21 维动作顺序（官方 NitroGen/nitrogen/shared.py BUTTON_ACTION_TOKENS，G1 源码法结论）
MODEL_BUTTON_ORDER = [
    "BACK", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT", "DPAD_UP", "EAST", "GUIDE",
    "LEFT_SHOULDER", "LEFT_THUMB", "LEFT_TRIGGER", "NORTH",
    "RIGHT_BOTTOM", "RIGHT_LEFT", "RIGHT_RIGHT",
    "RIGHT_SHOULDER", "RIGHT_THUMB", "RIGHT_TRIGGER", "RIGHT_UP",
    "SOUTH", "START", "WEST",
]

# 模型列号 → 标注键名（小写）；RIGHT_BOTTOM/LEFT/RIGHT/UP 无对应标注键，映射 None
MODEL_COL_TO_BUTTON: dict[int, str | None] = {
    0: "back", 1: "dpad_down", 2: "dpad_left", 3: "dpad_right", 4: "dpad_up",
    5: "east", 6: "guide", 7: "left_shoulder", 8: "left_thumb", 9: "left_trigger",
    10: "north", 11: None, 12: None, 13: None,
    14: "right_shoulder", 15: "right_thumb", 16: "right_trigger", 17: None,
    18: "south", 19: "start", 20: "west",
}

# 标注键名 → 模型列号（反查）
BUTTON_TO_MODEL_COL: dict[str, int] = {
    v: k for k, v in MODEL_COL_TO_BUTTON.items() if v is not None
}

# 有效比对列（标注 17 键在模型中的列号），用于指标计算
VALID_MODEL_COLS: list[int] = [k for k, v in MODEL_COL_TO_BUTTON.items() if v is not None]

# 选定视频的数据根路径
SHARD = Path(r"D:/2+课产品/_data/SHARD_0088/Z1r1S--MJS4")
VIDEO = Path(r"D:/2+课产品/TOP 1 IN 2S _ Ranked 2v2 w_ oKhaliD (1).mp4")


def frame_to_chunk_row(fid: int) -> tuple[str, int]:
    """视频帧号 → (chunk_id 四位字符串, parquet 行号)。"""
    return f"{fid // CHUNK_SIZE:04d}", fid % CHUNK_SIZE


def fetch_frame(fid: int, tmp_dir: Path) -> np.ndarray:
    """按帧号从视频抽帧（ffmpeg -ss 精确到秒），返回 HxWx3 uint8 图像。

    抽帧失败抛 subprocess 异常（宁可中断不静默），图像自动去 alpha、转 uint8。
    """
    sec = fid / FPS
    p = tmp_dir / f"f{fid}.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{sec:.3f}",
         "-i", str(VIDEO), "-frames:v", "1", "-q:v", "2", str(p)],
        check=True, capture_output=True,
    )
    img = mpimg.imread(str(p))
    p.unlink()
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.shape[-1] == 4:  # RGBA
        img = img[..., :3]
    return img


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
