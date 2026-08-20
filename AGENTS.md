# AGENTS.md

工程工作流（Matt Pocock skills）的仓库级配置入口。

## 项目一句话

课程课题七：基于 NitroGen（500M DiT 游戏智能体）做离线推理评测与可视化，独立完成。范围、验收标准（M1~M7）与风险见 `docs/立项书.md`。

## 领域文档（探索前先读）

- `docs/立项书.md` — 范围 / 验收 / 风险（M1~M7，验收句"给定—当—则"）
- `docs/模块设计与职责表.md` — 7 模块设计（A 数据获取 ~ G 归档）+ M1~M7 对照
- `docs/技术选型比选.md` — 选型决策与放弃理由（serve.py ZeroMQ / matplotlib / polars / yt-dlp）
- `docs/项目备忘.md` — 技术栈结论、接口索引、假实现边界、待决问题（建立后维护）

## 工作方式约定

- 每日内容在 `每日内容/第N天内容.md`（工作区外，非本仓库）
- 每日计划在 `每日计划/`、报告在 `每日实践报告/`（均在仓库外）
- 仓库只放代码 / README / docs，≤10MB；`.scratch/`、`.venv/`、权重、数据一律不入仓
- 助手负责 git add/commit（message 用英文），用户用 GitHub Desktop push

## Agent skills

### Issue tracker

Work is tracked in local markdown under `.scratch/` (one dir per feature). See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical labels are used as-is: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo — domain docs live under `docs/`. See `docs/agents/domain.md`.
