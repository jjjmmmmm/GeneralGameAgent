# -*- coding: utf-8 -*-
"""通检 C5：坏输入拒绝测试——预期明确报错，不允许静默产出错误结果

用例：
1. 不存在的帧路径（fetch_frame 找不到文件，应抛异常）
2. 全零图像（合法但无意义输入——期望模型正常响应，验证不崩溃）
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import MODEL_BUTTON_DIM, MODEL_STEPS, fetch_frame

sys.path.insert(0, str(ROOT.parent.parent.parent / "NitroGen"))
from nitrogen.inference_client import ModelClient  # type: ignore


def test_missing_file():
    print("=== C5-1: 不存在的帧路径 ===")
    # 用远超视频帧号范围的 fid（chunk 35+ 越界），fetch_frame 对 ffmpeg 失败抛异常
    tmp_dir = ROOT / "results" / "_tmp_badinput"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        fetch_frame(999999, tmp_dir)  # 越界帧号：ffmpeg -ss 超视频时长，-frames:v 1 无输出 → 非 0 退出
    except Exception as e:
        print(f"PASS: 正确报错 -> {type(e).__name__}")
    else:
        print("UNEXPECTED: 越界帧号竟然抽到了帧")


def test_zero_image():
    print("=== C5-2: 全零图像（合法但无意义）===")
    client = ModelClient(host="localhost", port=5555)
    client.reset()
    zero = np.zeros((1080, 1920, 3), dtype=np.uint8)
    try:
        pred = client.predict(zero)
        shape_ok = pred["j_left"].shape == (MODEL_STEPS, 2) and pred["buttons"].shape == (MODEL_STEPS, MODEL_BUTTON_DIM)
        print(f"PASS: 模型正常响应，shape={pred['j_left'].shape}/{pred['buttons'].shape}（符合预期）" if shape_ok
              else f"UNEXPECTED: 返回 shape 异常 {pred['j_left'].shape}")
    except Exception as e:
        print(f"UNEXPECTED: 全零输入触发异常 -> {type(e).__name__}: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    test_missing_file()
    test_zero_image()
    print("\nC5 完成")
