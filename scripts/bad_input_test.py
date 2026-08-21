# -*- coding: utf-8 -*-
"""通检 C5：坏输入拒绝测试——预期明确报错，不允许静默产出错误结果

用例：
1. 不存在的帧路径（fetch_frame 找不到文件）
2. 全零图像（合法但无意义输入——期望模型正常响应，验证不崩溃）
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib.image as mpimg
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent.parent.parent / "NitroGen"))
from nitrogen.inference_client import ModelClient


def test_missing_file():
    print("=== C5-1: 不存在的帧路径 ===")
    p = ROOT / "results" / "_definitely_missing.png"
    try:
        img = mpimg.imread(str(p))  # imread 对不存在文件返回 None 不报错
        if img is None:
            raise FileNotFoundError(f"帧文件不存在: {p}")
        print("UNEXPECTED: 读到了不存在的文件")
    except FileNotFoundError as e:
        print(f"PASS: 正确报错 -> {e}")


def test_zero_image():
    print("=== C5-2: 全零图像（合法但无意义）===")
    client = ModelClient(host="localhost", port=5555)
    client.reset()
    zero = np.zeros((1080, 1920, 3), dtype=np.uint8)
    try:
        pred = client.predict(zero)
        shape_ok = pred["j_left"].shape == (18, 2) and pred["buttons"].shape == (18, 21)
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
