# GeneralGameAgent · 通用游戏智能体

基于 [NitroGen](https://github.com/MineDojo/NitroGen) 的通用游戏智能体离线推理评测与可视化系统。

> 课程：AI 开发实践 · 课题七（腾讯 IEG） | 独立完成 | 2026-08-19 ~ 2026-08-27

## 背景与目标

NitroGen 是 NVIDIA 开源的通用游戏智能体基础模型（500M 参数 DiT），输入单帧游戏画面，输出 8 步手柄动作块（按键 + 双摇杆），经由互联网游戏视频的行为克隆训练得到。本项目的目标是：**在 9 天课程周期内，跑通该模型的离线推理链路，建立可复现、可核对的评测体系，并开发手柄动作可视化工具**，验证 zero-shot 条件下模型与人类操作标注的一致性。

主路径：

```
下载 ng.pt 权重与单款游戏数据分片 → 启动推理服务 → 对测试帧离线批量推理
→ 与人类标注对齐计算指标（按键准确率 / 摇杆误差）→ 可视化对比 → 归档
```

完整范围、验收标准与风险见 [`docs/立项书.md`](docs/立项书.md)。

## 完成情况

> 随进度更新；完整自测对照表见第 8 天报告。

| MVP | 内容 | 状态 |
|-----|------|------|
| M1 | 跑通官方 ng.pt 推理，README 可复现 | ✅ 已达成（D4） |
| M2 | 单游戏 ≥500 帧统计 + ≥10 条序列可视化 | ⬜ 未开始（计划 D5） |
| M3 | 离线评测脚本（≥200 帧隔离测试集） | ⬜ 未开始（计划 D5） |
| M4 | zero-shot 基线：按键准确率 ≥50%、摇杆相关系数 ≥0.4 | ⬜ 未开始（计划 D6） |
| M5 | 模型输出 vs 人类标注对比演示 | ⬜ 未开始（计划 D5） |
| M6 | 归档：代码 + 指标表 + ≥3000 字实验报告 | ⬜ 未开始（计划 D8） |
| M7 | 扩展：可视化工具（≥20 段动作曲线 + 差异帧标注） | ⬜ 未开始（计划 D6-D7） |

进度日志：

- **D1（08-19）**：立项完成——立项书定稿（MVP 七条、不做清单附原因、六项风险、九天节奏），参考仓库已克隆，Python 3.12 环境就绪。
- **D2（08-20）**：环境就绪（venv + PyTorch cu128，RTX 5060 验证通过）；权重 ng.pt 与数据分片（SHARD_0088）下载完成；选定评测视频 Z1r1S--MJS4（rocket_league，35 chunks，含官方 actions_processed）；模块图 + 职责表 + 技术选型比选完成。
- **D3（08-20）**：数据模型与接口约定定稿（5 实体 + A~F 接口 + 17↔21 键对照）；项目备忘建立；game mapping 验证（ng.pt 的 `game_mapping_cfg=None`，serve.py 无条件分支可直启）；统计脚本 `scripts/stats.py` 起步（M2 骨架）。
- **D4（08-21）**：**M1 达成**——serve.py 首跑成功并返回 8 步动作块；README 可复现路径验证。

**范围外（明确不做）**：实机 Windows 游戏接入、模型微调、数据过滤管线、Open P2P 对照、数据集全库下载、自采数据。原因详见立项书第 4 节。

## 环境要求

| 项 | 要求 | 状态 |
|----|------|------|
| 操作系统 | Windows 11 | ✅ |
| Python | ≥ 3.12（官方要求） | ✅ 已安装（系统级，项目用 venv 隔离） |
| GPU | NVIDIA RTX 5060 8GB（Blackwell，sm_120，需 cu128 及以上 PyTorch） | ⏳ 待 D2 验证 |
| 磁盘 | ≥ 5GB 可用（权重 + 数据分片，存放于仓库外） | ✅ |

## 安装（D4 已验证）

```bash
# 1. 创建并激活虚拟环境
py -3.12 -m venv .venv
.venv\Scripts\activate

# 2. 安装 PyTorch（Blackwell 架构 RTX 5060 需 cu128 系）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"   # 预期 True

# 3. 安装项目依赖（requirements.txt 含 serve 依赖清单）
pip install -r requirements.txt

# 4. 安装参考库 NitroGen（仅 serve 用，--no-deps 避免拉实机注入库）
pip install -e ../NitroGen --no-deps

# 5. 下载模型权重（仓库外 _models/，约 1.97GB）
#    https://huggingface.co/nvidia/NitroGen 的 ng.pt
#    首次启动推理需网络下载视觉编码器 siglip2（建议开启 TUN/全局代理）
```

> 说明：`NitroGen` 默认依赖含实机注入库（vgamepad/dxcam 等），本项目仅做离线推理，故用 `--no-deps` 并手动装 serve 所需依赖。

## 使用方法（D4 已验证：M1 推理首跑）

### 启动推理服务（serve.py，M1）

```powershell
# 方式一：脚本启动（监听 tcp://*:5555）
powershell -File scripts/run_serve.ps1

# 方式二：直接启动（需先开 TUN/全局代理下载 siglip2，之后可离线）
.venv\Scripts\python.exe ..\NitroGen\scripts\serve.py ..\..\_models\ng.pt --port 5555
```

启动后约 8 秒加载完成（torch.load + 模型构建 + 迁移 GPU），监听 5555。
> 判断服务就绪请测端口：`Test-NetConnection 127.0.0.1 -Port 5555`（WMI/重定向下日志缓冲，勿以日志为准）。

### 提交 1 帧推理请求（M1 验收）

```python
from nitrogen.inference_client import ModelClient
import numpy as np
frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)   # 帧假实现：灰图
resp = ModelClient("127.0.0.1", 5555).predict(frame)
# resp['j_left'] shape (18,2) / resp['j_right'] shape (18,2) / resp['buttons'] shape (18,21)
# 即 8 步动作块：双摇杆连续值 + 21 键动作空间
```

### 停止服务

```powershell
# 找到 serve.py 的 python 进程后结束；或按实际 pid 结束
Get-Process python | Where-Object { $_.CommandLine -match 'serve.py' } | Stop-Process
```

### 数据统计（M2 骨架，D3 已实现）

```powershell
# 输出 17 键频率表 + 帧数
.venv\Scripts\python.exe scripts\stats.py <parquet路径>
.venv\Scripts\python.exe scripts\stats.py ..\..\_data\SHARD_0088\Z1r1S--MJS4\Z1r1S--MJS4_chunk_0000\actions_processed.parquet
```

### 后续（随 D5-D8 开发补全）

```bash
# 离线批量评测（待 D5-D6）
python scripts/evaluate.py --test-set <path> --output results/

# 批量动作曲线可视化（待 D6-D7）
python scripts/visualize.py --results results/ --out figures/
```

## 项目结构

```
GeneralGameAgent/
├── docs/                 # 立项书、项目备忘、选型/数据模型/接口约定/serve 适配记录
├── scripts/              # stats.py（统计）、verify_gamemap.py（game mapping 验证）、run_serve.ps1
├── requirements.txt      # serve 依赖清单（显式，非 extras）
├── AGENTS.md             # 工程工作流配置（Matt Pocock skills）
├── .scratch/             # 本地 issue tracker（spec/tickets，不入仓）
└── README.md
```

> 模型权重（`_models/ng.pt`）与数据集分片（`_data/`）**不入仓库**；仓库体积控制在 10MB 以内。

> 模型权重（`ng.pt`）与数据集分片**不入仓库**，通过脚本按需下载至本机指定目录；仓库体积控制在 10MB 以内。

## 数据与模型

| 资源 | 来源 | 许可 | 说明 |
|------|------|------|------|
| NitroGen 权重 `ng.pt` | [huggingface.co/nvidia/NitroGen](https://huggingface.co/nvidia/NitroGen) | CC BY-NC 4.0 | 仅下载单文件，存本机 `_models/` |
| 手柄标注数据集 | [huggingface.co/datasets/nvidia/NitroGen](https://huggingface.co/datasets/nvidia/NitroGen) | 见数据集页面 | Parquet 格式，仅下载单款游戏必要分片，存本机 `_data/` |
| 参考代码 | [github.com/MineDojo/NitroGen](https://github.com/MineDojo/NitroGen) | 见其 LICENSE | 仅作参考；本仓库代码独立编写 |

## 九天节奏

| 天 | 主题 | 里程碑 |
|----|------|--------|
| D1 | 选题立项 | ✅ 立项书 |
| D2 | 组成与技术选型 | 环境、权重、数据就绪；模块图与比选表 |
| D3 | 数据与调用约定 | 数据模型、接口约定、项目备忘 |
| D4 | 工程起步 | M1 推理首跑可复现 |
| D5 | 主路径贯通 | M2/M3/M5 阶段演示 |
| D6 | 核心推进 | M4 指标达标；M7 起步 |
| D7 | 贯通验证 | M7 完成；通检 ≥5 条 |
| D8 | 交付准备 | M6 归档；验收自测表 |
| D9 | 结课 | 大报告 |

## 致谢与引用

本项目基于 NitroGen 开展离线评测与可视化：

```bibtex
@misc{magne2026nitrogen,
      title={NitroGen: An Open Foundation Model for Generalist Gaming Agents},
      author={Loïc Magne and Anas Awadalla and Guanzhi Wang and Yinzhen Xu and Joshua Belofsky and Fengyuan Hu and Joohwan Kim and Ludwig Schmidt and Georgia Gkioxari and Jan Kautz and Yisong Yue and Yejin Choi and Yuke Zhu and Linxi "Jim" Fan},
      year={2026},
      eprint={2601.02427},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.02427},
}
```

## 许可证

本仓库自研代码的许可证待定（见第 8 天定稿）；所用模型权重遵循 CC BY-NC 4.0，仅用于课程研究与教学。
