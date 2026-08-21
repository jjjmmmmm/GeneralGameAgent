# M5 演示素材 · 模型预测 vs 人类标注对比表

> 5 帧真帧（视频 60~64s，统计集 chunk_0001），帧缩略图见 `_data/frames/Z1r1S--MJS4/seq/`。
> 详细指标见 `e2e_metrics.md`；T3 端到端跑通，无假实现。
> 键序映射：**G1 已确认**（官方 `BUTTON_ACTION_TOKENS`，见 `docs/键序映射表.md`），本表按 `BUTTON_TO_MODEL_COL` 对齐。

## 对比表

| 帧 | 时间 | 帧缩略图 | 人类标注（按下的键） | 模型预测（按下的键） | 摇杆 j_left 标注→预测 | 按键一致 |
|---|---|---|---|---|---|---|
| f0 | 60s | `seq01_f0.png` | dpad_left, right_trigger | （无） | (-0.80, 0.14) → (-0.44, -0.63) | 15/17 |
| f1 | 61s | `seq01_f1.png` | dpad_left, right_trigger | left_trigger | (0.67, 0.09) → (0.87, 0.28) | 14/17 |
| f2 | 62s | `seq01_f2.png` | dpad_left, right_trigger | **right_trigger** ✅ | (-0.87, 0.06) → (-0.71, 0.09) | 16/17 |
| f3 | 63s | `seq01_f3.png` | dpad_left | right_trigger | (-0.92, 0.08) → (-0.94, 0.01) | 15/17 |
| f4 | 64s | `seq01_f4.png` | dpad_left, right_trigger | （无） | (-0.93, -0.03) → (-0.00, -0.00) | 15/17 |

> 帧缩略图完整路径：`D:/2+课产品/_data/frames/Z1r1S--MJS4/seq/seq01_fX.png`（不入仓）。

## 可逐帧核对点

- **帧画面**：60~64s 处于比赛进行中（玩家持续转向 `dpad_left` + 油门 `right_trigger`）
- **人类标注**：`actions_processed.parquet` 的 17 键 0/1（T1 帧对齐验证通过）
- **模型预测**：`pred['buttons'][PRED_ROW, BUTTON_TO_MODEL_COL[b]] > 0.5`，`BUTTON_TO_MODEL_COL` 为 G1 确认的官方映射
- **按键一致**：17 键逐键比对的全对比例

## 关键观察（D6 更新）

1. **zero-shot 模型倾向"保守不操作"**：5 帧中 2 帧预测全 0、2 帧错按无关键，而玩家实际持续按 2 键——按键准确率 88.2%（正确映射）虚高（大量 `0==0`），**M4 验收必须以触发召回率/精确率为准**（G3，200 帧实测召回率仅 4.2%）
2. **flow matching 预测有随机性**：同一输入多次推理输出不同（采样非确定性），本表为单次运行结果，重跑会有波动（如 f2 时对时错）；正式评测统一用 K=3 多数票
3. **摇杆方向**：f2/f3 的 x 轴预测与标注接近，f0/f4 偏差明显——单帧 MSE 波动大（0.0008~1.2），200 帧实测相关系数 -0.03~+0.06（未达标，见 G11）
