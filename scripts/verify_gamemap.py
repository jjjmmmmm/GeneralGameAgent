# -*- coding: utf-8 -*-
"""验证 ng.pt 的 game mapping 配置（D3 结论可复现脚本）

用法:
    python scripts/verify_gamemap.py <ng.pt路径>

输出:
    game_mapping_cfg 是否为空 → serve.py 走无条件分支可直接启动（结论见 docs/项目备忘.md 第 4 节）
"""
import sys
from pathlib import Path

import torch


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    ng_pt = Path(sys.argv[1])
    if not ng_pt.exists():
        print(f"ERROR: ng.pt not found: {ng_pt}", file=sys.stderr)
        sys.exit(1)

    # 只读 config，不构造模型（weights_only=False 保证 dict 结构可读）
    ckpt = torch.load(ng_pt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("ckpt_config", {})
    tok_cfg = cfg.get("tokenizer_cfg", {})
    gm = tok_cfg.get("game_mapping_cfg")
    model_cfg = cfg.get("model_cfg", {})
    enc = model_cfg.get("vision_encoder_name")

    print(f"ng.pt: {ng_pt}")
    print(f"game_mapping_cfg: {gm!r}")
    if gm is None:
        print("结论: game_mapping_cfg=None → serve.py 走无条件分支，可直接启动，无需适配层")
        print("      （inference_session.from_ckpt 打印 'No game mapping available'）")
    else:
        src = gm.get("src_files", [])
        print(f"src_files: {src}")
        for s in src:
            print(f"  [{'EXISTS' if Path(str(s)).exists() else 'MISSING'}] {s}")
        print("结论: game_mapping_cfg 非空，需按 src_files 检查本地可用性")
    print(f"vision_encoder: {enc}  （首次加载需联网下载）")


if __name__ == "__main__":
    main()
