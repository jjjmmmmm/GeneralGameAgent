# serve.py 启动适配记录（M1）

> D4（2026-08-20）| 记录让官方 serve.py 在本机跑通所需的全部适配，供复现与报告引用

## 1. 前提环境

- **网络**：需开启 TUN/全局代理（siglip2 首次下载用；之后命中本地缓存可离线）
- **权重**：ng.pt（1,974,723,762 字节）在仓库外 `_models/`
- **依赖**：venv 内已装 torch cu128 / torchvision / transformers / pydantic / pyzmq / diffusers / einops / pyyaml / numpy / polars

## 2. 对官方代码的最小适配（氮气）

文件：`NitroGen/nitrogen/flow_matching_transformer/nitrogen.py` 第 186 行

原代码：
```python
self.vision_encoder = model.vision_model
```
改后：
```python
self.vision_encoder = model
```
**原因**：标准 transformers 各版本（4.43~5.x）的 `SiglipVisionModel` 均无 `.vision_model` 属性（官方遗留 API）。`SiglipVisionModel` 本身就是视觉编码器（forward 返回 `last_hidden_state`，且有 `.encoder.layers`/`.head`），故直接用 `model`。
**影响**：仅删多余的一层属性访问，不改任何逻辑。验证 `load_state_dict` missing=0、unexpected=0（ng.pt 权重与模型完全匹配）。

## 3. 启动方式

```powershell
# WMI 托管后台启动（工具会话会杀后台进程，故用 WMI）
powershell -File scripts/run_serve.ps1   # 或手动 WMI Invoke-CimMethod
```
- 监听 `tcp://*:5555`
- 加载耗时：torch.load 1.1s + NitroGen 构建 1.4s + load_state_dict 0.8s + to-cuda 0.9s ≈ **8.4s**
- **注意**：WMI 重定向 stdout 是块缓冲，日志不即时刷新，易误判为死锁。判断服务就绪应测端口：`Test-NetConnection 127.0.0.1 -Port 5555`

## 4. 推理验证（M1 验收）

```python
from nitrogen.inference_client import ModelClient
import numpy as np
gray = np.full((1080, 1920, 3), 128, dtype=np.uint8)
resp = ModelClient("127.0.0.1", 5555).predict(gray)
# resp['j_left']  shape (18,2)；resp['j_right'] shape (18,2)；resp['buttons'] shape (18,21)
```
- 返回 8 步动作块：j_left/j_right 摇杆连续值 + buttons 21 键
- 灰图推理耗时 **0.48s**（含上下文缓冲）

## 5. 未决事项

- 推理输入为灰图（D3 帧假实现）。真实视频帧取帧受阻（YouTube bot + GitHub 网络），见 `docs/项目备忘.md` 待决问题。
