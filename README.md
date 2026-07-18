# 🔬 twinScientist — AI Scientist for Environment-Human Health Research

> **自主科研与实验迭代智能体** — 融合多源传感器数据，自动完成从假设生成到因果推断的完整科研闭环。  
> **基于 Qwen 系列模型 + LangGraph 多智能体架构**（挑战杯 AI Scientist 赛题）。

---

## 📋 项目概述

twinScientist 是一个面向**环境—人体关联研究**的自主科研智能体系统。它模拟人类科学家的研究流程，自动执行以下核心任务：

| # | 科研环节 | 实现方式 |
|---|---------|---------|
| 1 | **文献调研与事实提取** | LLM 驱动的知识图谱构建 |
| 2 | **假设生成与进化** | Tournament 淘汰赛 + Bayesian 置信度量化 |
| 3 | **实验方案设计** | 针对胜出假设自动生成可验证实验 |
| 4 | **数据接入与时序对齐** | 多源传感器 (环境/PPG/眼动) 异步合并 |
| 5 | **因果推断分析** | 8 类方法 (CCM/Granger/贝叶斯等) + AI 自动选择 |
| 6 | **五维评审** | Reviewer Agent: 新颖性/可行性/方法论/证据/影响 |
| 7 | **反思-修正闭环** | 失败资产化: 根因分析 → 派生新假设 |
| 8 | **标准化报告输出** | 赛题规范 12+ 字段完整覆盖 |

### 应用场景

- **室内环境影响评估**: 温度/CO₂/PM/VOC 对 HRV、心率变异性、视觉疲劳的影响
- **暴露组学研究**: 多污染物复合暴露的因果效应估计
- **个体化健康研究**: N-of-1 设计追踪个体生理响应
- **科研辅助工具**: 从原始数据到同行评审级报告的自动化管线

---

## 🏗️ 系统架构：九层 28 项设计

```
┌─────────────────────────────────────────────────────────┐
│ Layer 9: 输出与成果规范     [Item 28] Report Generator   │ ← Markdown/PDF/HTML
├─────────────────────────────────────────────────────────┤
│ Layer 8: 假设生成质量       [Items 26-27]                │
│   ├── Tournament Evolution (PK 淘汰)                    │
│   └── Bayesian Confidence Quantification               │
├─────────────────────────────────────────────────────────┤
│ Layer 7: 执行架构            [Items 23-25]               │
│   ├── Agentic Tree Search                               │
│   ├── Dual-thread Inference/Execution                   │
│   └── 3-tier Semantic Termination Evaluation           │
├─────────────────────────────────────────────────────────┤
│ Layer 6: 工具集              [Items 20-22]               │
│   ├── Causal Inference Toolkit (8 methods)             │
│   ├── Literature Monitor                                │
│   └── Knowledge Graph Builder                           │
├─────────────────────────────────────────────────────────┤
│ Layer 5: 数据管道            [Items 16-19]               │
│   ├── Multi-source Time-Series Engine                  │
│   ├── Temporal Alignment & Quality Assessment          │
│   └── Counterfactual Reasoning Engine                  │
├─────────────────────────────────────────────────────────┤
│ Layer 4: 人机协同            [Items 11-15]               │
│   ├── Gradio Web UI (structured decision panel)        │
│   ├── Human Approval Gate (interrupt_before)           │
│   ├── PI Agent (首席研究员)                             │
│   ├── Reviewer Agent (五维审稿)                         │
│   └── Ethics & Safety Watchdog                         │
├─────────────────────────────────────────────────────────┤
│ Layer 3: 记忆管理            [Items 7-10]                │
│   ├── 4D Memory State (Working/Episodic/Semantic/Evidence)│
│   ├── Self-Evolution Manager                            │
│   └── Failure Assetization (教训→修正假设)              │
├─────────────────────────────────────────────────────────┤
│ Layer 2: 认知编排            [Items 3-6]                 │
│   ├── Cognitive Graph (DAG of scientific operations)   │
│   ├── Orchestrator Dynamic Routing                     │
│   ├── Hypothesis Tree Architecture                     │
│   └── Active Learning Experiment Design                │
├─────────────────────────────────────────────────────────┤
│ Layer 1: 基座模型            [Items 1-2]                 │
│   ├── Qwen via Alibaba Cloud Bailian API               │
│   └── Prompt Engineering (instruction following format)│
└─────────────────────────────────────────────────────────┘
```

### 各层详细说明

#### Layer 1: 基座模型与基础设施
- **Item 1**: 强制使用 Qwen 系列模型（通过阿里云百炼平台 API 调用 `qwen-max` 等）
- **Item 2**: 针对 Qwen 优化的 Prompt 模板体系，支持指令遵循与结构化输出

#### Layer 2: 智能体编排与认知架构
- **Item 3**: 将传统 DAG 扩展为完整认知图，包含文献调研→假设生成→实验→解读→反思节点
- **Item 4**: Orchestrator 动态路由引擎，基于证据强度、不确定性和收敛度做出全局决策
- **Item 5**: 假设树数据结构，支持动态生长、分支和修剪
- **Item 6**: 主动学习实验设计框架（高熵区域优先探索）

#### Layer 3: 状态空间与记忆管理
- **Item 7**: 四维记忆体——工作记忆 (当前轮次)、情景记忆 (历史实验)、语义记忆 (领域知识)、证据链 (因果推理记录)
- **Item 8**: 三层持久化记忆（L1 Kernel / L2 SQLite + 向量检索 / L3 知识图谱）
- **Item 9**: Evolution Manager 自我进化机制，从成功/失败模式中提炼 Meta-insights
- **Item 10**: 失败资产化——每次评审未通过都进行根因分析并派生修正假设

#### Layer 4: 人机协同与质量控制
- **Item 11**: Gradio Web UI 提供结构化决策面板
- **Item 12**: `interrupt_before=["human_approval"]` 断点机制支持用户暂停/重定向
- **Item 13**: PI Agent 整合多智能体成果产出最终报告
- **Item 14**: Reviewer Agent 五维评分 (新颖性/可行性/方法论/证据/影响)，<75 分打回修改
- **Item 15**: 伦理与安全看门狗，三级风险判断 (Approved/Human Review/Blocked)

#### Layer 5: 代码执行沙箱与数据管道
- **Item 16**: DVC Lite 数据版本控制
- **Item 17**: LangGraph Checkpoint 时间旅行能力
- **Item 18**: 多源时序引擎支持环境传感器、PPG、血氧、眼动数据异步接入与互相关对齐
- **Item 19**: 反事实推理引擎 (GP 代理模型预测干预效果)

#### Layer 6: 工具调用与外部能力
- **Item 20**: 因果推断工具箱 — 8 类方法:
  - **CCM** (Convergent Cross Mapping): 非线性因果关系检测
  - **Granger Causality**: 线性时间序列因果检验
  - **PC-FCI**: 因果结构学习 (含潜变量)
  - **PSM** (Propensity Score Matching): 倾向得分匹配
  - **Instrumental Variable**: 工具变量法
  - **Bayesian Network**: 贝叶斯网络推断
  - **Counterfactual**: 反事实推演
  - **Auto-Select**: AI 自动选择最优方法

- **Item 21**: 持续文献监控 (Semantic Scholar API 集成)
- **Item 22**: 自动知识图谱构建 (三元组抽取)

#### Layer 7: 执行架构与终止条件
- **Item 23**: 双线程协作 (推理线程 + 执行线程异步解耦)
- **Item 24**: Agentic Tree Search (并行探索 + 动态剪枝)
- **Item 25**: 三层语义终止评估 (收敛度 ≥85% + 证据充分 + 穷尽搜索综合评分 >0.85)

#### Layer 8: 科学假设生成质量
- **Item 26**: Tournament 淘汰赛 — N 个候选假设两两 PK，最终选出 Top-1 进入实验阶段
- **Item 27**: Bayesian 置信度量化 — 先验 P(H) → 后验 P(H|D) via log-odds update

#### Layer 9: 输出与成果规范
- **Item 28**: 赛题规范输出格式 (12+ 标准字段)，包含 Problem Statement、Rationale、Technical Details、Datasets、Methods、Experiments、Results、References、Evidence Chain 等

---

## 🧠 思维框架：从问题到结论的认知流程

```
START
  │
  ▼
┌──────────────┐     blocked     ┌─────────────────┐
│ ethics_check │────────────────▶ termination_eval │
│ (伦理审查)    │     approved    │ (终止评估 + 报告) │
└──────┬───────┘                 └─────────────────┘
       │
       ▼
┌──────────────┐
│literature_   │ ← 提取≥8条真实事实 + DOI/PMID
│ review       │ ← 构建初步知识图谱
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│hypothesis_gen    │ ← 生成 5-10 个候选假设
│ (假设生成)        │    基于文献事实 + 领域知识
└──────┬───────────┘
       │
       ▼
┌──────────────────┐     eliminated      ┌──────────────┐
│tournament_eval   │────────────────────▶ │ refuted list │
│ (淘汰赛)          │     winner→active   └──────────────┘
│ PK 两两比较，选 Top-1
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│experiment_design │ ← 为胜出假设设计可验证实验
│ (实验设计)        │    指定数据源、分析方法、预期输出
└──────┬───────────┘
       │
       ▼
┌──────────────────┐     has_real_data?
│data_analysis     │──────── no ──▶ theoretical_analysis
│ (数据分析)        │
│ • 加载传感器/Csv  │
│ • 自动选择因果方法│
│ • Granger/CCM/    │
│   贝叶斯网络      │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│interpretation    │ ← 更新假设置信度
│ (结果解读)        │    识别反直觉模式
└──────┬───────────┘
       │
       ▼
┌──────────────────┐     score≥75?
│reviewer_agent    │──────── yes ──▶ report_writing
│ (五维评审)        │
│ 新颖性/可行性/    │         no
│ 方法论/证据/影响  │         ▼
└──────┬───────────┘ reflection ──▶ back to hypothesis_generation
       │                                    (修正循环)
       ▼
┌──────────────────┐
│termination_eval  │ ← 检查停止条件
│ (终止评估)        │    1. convergence≥0.85 + stable x2 rounds
│                  │    2. max_iterations reached
│                  │    3. combined_score ≥ 0.85
└──────┬───────────┘
       │ terminate
       ▼
┌──────────────────┐     approve?
│report_writing    │──────── yes ──▶ pi_agent_meeting
│ (报告撰写)        │
│• 12+ 标准字段     │         no
│• 含真实分析结果   │         ▼
└──────────────────┘ human_approval ──▶ ...
```

**关键特征：**
- **确定性 Guardrails**: ethics_check → literature_review 保持确定流转
- **动态路由**: 大部分流程通过 Orchestrator 动态决策
- **Reflection Loop**: 评审未通过时回到假设生成，但会注入反思洞察避免重复错误
- **Termination Safeguards**: 最多 N 轮迭代 (默认 200) + 收敛度检查防止无限循环

---

## 🚀 快速开始

### 前置要求

- Python 3.10+ (推荐 3.12+)
- 阿里云百炼平台 API Key (`BAILIAN_API_KEY`)

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/KevinShin124/TwinScientist.git
cd TwinScientist

# 2. 安装依赖
cd twinScientist
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 BAILIAN_API_KEY
```

### 三种运行模式

#### 模式 1: CLI 交互模式 (推荐新手)
```bash
python main.py
```
进入交互式提示符，输入研究问题即可启动完整科研流程。

#### 模式 2: 命令行单次运行
```bash
python main.py \
  --question "高温高CO2复合暴露对室内工作人员HRV有何因果影响？" \
  --domain "环境—人体关联" \
  --iterations 5
```
自动走完伦理审查→文献→假设→实验→因果推断→评审→报告全流程。

#### 模式 3: Web UI 模式
```bash
python main.py --ui
```
浏览器访问 `http://127.0.0.1:7860` 打开 Gradio 界面。

---

## 📊 多模态数据管道

### 支持的传感器类型

| 数据类型 | 列名示例 | 文件存放位置 | 说明 |
|---------|---------|-------------|------|
| **环境传感器** | T, H, CO₂, VOC, NO₂, PMS1, PMS10, PMS2_5, C₂H₅OH | `data/sensors/` | Dalton IoT 格式兼容 |
| **生物信号** | HR_BPM, SDNN_ms, RMSSD_ms, PPG_amplitude, SpO2_pct, ECG_RR_interval | `data/biometric/` | PPG/ECG/SpO₂/HRV |
| **视觉疲劳** | blink_frequency_per_min, pupil_diameter_mm, gaze_stability_score, drowsiness_index | `data/visual_fatigue/` | 眼动追踪指标 |

### 数据格式要求

CSV 第一行必须为列名，包含至少 timestamp 和一个数值型传感器列。系统会自动检测数据格式并分类到对应目录。

### 合成数据模拟器

项目内置科研级多模态数据生成器 `gen_multimodal_simulator.py`，基于已发表文献的因果系数生成真实感合成数据：

```bash
# 基本用法：6 人 × 14 天 × 4 房间 × ~400 点/天
python gen_multimodal_simulator.py

# 自定义参数
python gen_multimodal_simulator.py \
  --subjects 12 --days 30 --n-points 600 \
  --rooms Study_Desk Kitchen Bedroom Lounge \
  --output ./my_dataset --seed 42
```

**模拟器特性：**
- ✅ **因果 DAG 建模**: 所有生理响应基于 T↑→Sympathetic↑→HR↑ 等路径推导
- ✅ **个体异质性**: Per-subject random effects (年龄、基础 HRV、敏感性差异)
- ✅ **日节律耦合**: Temperature/CO₂ 共享昼夜正弦周期
- ✅ **房间特异性**: Bedroom/Kitchen/Lounge 独立基线水平
- ✅ **文献依据**: 系数源自 Allen et al., Brook et al., Wolkove et al. 等元分析

生成的数据完全兼容 TwinScientist 数据管道，可直接用于因果推断测试。

---

## 🧪 因果推断引擎详解

`tools/causal_inference.py` 实现了 8 类因果推断方法，部分为简化 numpy/scipy 实现，等待安装完整库后可升级为生产级算法：

### 已实现方法

| 方法 | 适用场景 | 核心原理 |
|------|---------|---------|
| **CCM** | 非线性双向因果 | 收敛交叉映射 — 检测系统是否共享同一吸引子流形 |
| **Granger** | 线性时间序列 | 过去值改善预测 → F-test 显著性检验 |
| **Auto-Select** | 未知数据特征 | 基于样本量、时序属性、非线性检测自动选择最优方法 |
| **Counterfactual** | 两组对比实验 | Welch's t-test 估计平均处理效应 ATE |

### 待完善方法（需额外依赖）

| 方法 | 需要包 | 用途 |
|------|--------|------|
| PC-FCI | `causalgraphicalmodels` | 带潜变量的因果结构学习 |
| PSM | `dowhy` | 倾向得分匹配消除混杂偏倚 |
| Instrumental Variable | `linearmodels` | 工具变量法解决内生性 |
| Bayesian Network | `pgmpy` | 概率图模型推断条件独立关系 |

### 使用方法

```python
from tools.causal_inference import CausalInferenceEngine

engine = CausalInferenceEngine(data)
result = await engine.run(
    method="auto_select",  # or "granger", "ccm", "counterfactual"
    feature_info={
        "sample_size": 1000,
        "num_variables": 5,
        "is_time_series": True,
        "nonlinear_relationships": False,
    }
)
# result = {selected_method: "granger", reasoning: [...], parameters: {...}}
```

---

## 📁 项目结构

```
TwinScientist/
├── twinScientist/                    # 主程序
│   ├── main.py                       # CLI/Web UI 入口 (argparse)
│   ├── .env.example                  # 环境变量模板
│   ├── requirements.txt              # Python 依赖
│   │
│   ├── config/                       # ━━ Layer 1: 配置 ━━
│   │   ├── settings.py               # Settings (pydantic-settings, .env 加载)
│   │   └── __init__.py
│   │
│   ├── core/                         # ━━ Layer 1-3: 核心引擎 ━━
│   │   ├── llm_client.py             # QwenClient (OpenAI-compatible API)
│   │   ├── prompts.py                # Prompt 模板库 (LITERATURE_REVIEW, TOURNAMENT_EVAL 等)
│   │   ├── state.py                  # AgentState (LangGraph StateGraph schema)
│   │   ├── graph.py                  # build_cognitive_graph() — LangGraph 编排
│   │   ├── orchestrator.py           # Dynamic routing logic + stop conditions
│   │   └── nodes.py                  # 12+ cognitive node functions (async)
│   │
│   ├── channels/                     # ━━ Layer 5: 数据管道 ━━
│   │   ├── base.py                   # Channel 基类
│   │   ├── time_series.py            # TimeSeriesChannel + SignalQualityEvaluator
│   │   └── metadata_channel.py       # SQLite 元数据存储
│   │
│   ├── tools/                        # ━━ Layer 6: 工具集 ━━
│   │   ├── causal_inference.py       # CausalInferenceEngine (8 methods)
│   │   └── literature_monitor.py     # 持续文献监控 (placeholder)
│   │
│   ├── output/                       # ━━ Layer 9: 输出 ━━
│   │   ├── report_generator.py       # ReportGenerator (Markdown/PDF/HTML)
│   │   └── scientific_hypothesis_report.md  # 样例报告
│   │
│   ├── ui/                           # ━━ Layer 4: 人机交互 ━━
│   │   └── app.py                    # Gradio Web UI
│   │
│   ├── data/                         # 数据目录
│   │   ├── sensors/                  # 环境传感器 CSV 文件
│   │   ├── biometric/                # PPG/血氧/HRV 数据
│   │   └── visual_fatigue/           # 视觉疲劳数据
│   │
│   ├── logs/                         # 运行日志 (twinscientist.log)
│   └── README.md
│
├── gen_multimodal_simulator.py       # 多模态合成数据生成器
├── gen_synthetic_dalton.py           # Dalton 格式合成数据 (旧版)
├── create_issue.ps1                  # GitHub Issue 生成脚本
├── run_with_data.bat                 # Windows 一键启动脚本
└── README.md
```

---

## 🎯 比赛/赛题匹配度

| 赛题要求 | 本项目实现状态 |
|---------|-------------|
| ✅ 基座为 Qwen（百炼平台 API） | `core/llm_client.py` — Qwen via OpenAI-compatible API |
| ✅ 问题理解 / 知识整合 / 关联发现 / 假设生成 | 完整认知图工作流 (ethics→lit→hyp→exp→analysis→review) |
| ✅ 多维度/多模态实测数据接入 | `channels/time_series.py` — 环境+PPG+HRV+眼动 |
| ✅ 多智能体协作 | PI Agent + Reviewer Agent + Ethics Watchdog + Orchestrator |
| ✅ 人机协作审核 | Gradio UI + interrupt_before 断点机制 |
| ✅ 参考文献真实性保障 | ReportGenerator 强调 DOI/PMID 必填，禁止虚构 |
| ✅ 标准化输出格式 | `output/report_generator.py` — 12+ 字段 |
| ✅ 前端界面 (加分项) | Gradio Web UI (port 7860) |
| ✅ SFT 微调支持 | 架构允许替换 llm_client 为微调模型 |
| ✅ 因果推断能力 | 8 类工具 + AI 自动选择框架 |

---

## 🛣️ 开发与维护路线图

### ✅ 已完成

| 编号 | 功能 | 状态 |
|------|------|------|
| Bug Fix 1/2/3 | 终止死循环修复 (max_iterations 硬编码→动态配置) | ✅ |
| Bug Fix 4 | `_parse_daltons_records()` break 导致只读取温度列 | ✅ |
| Item 25 | 三层语义终止评估 | ✅ |
| Item 27 | Bayesian 置信度量化 | ✅ |
| Item 28 | 赛题规范 12+ 字段报告输出 | ✅ |
| Feature | 多模态数据模拟器 | ✅ |

### 🔴 P0 — 下一个优先级

- [ ] 接入真实数据集 (`data/` 目录放入实测 CSV)
- [ ] 端到端流水线集成测试 (CLI → Report Generation)
- [ ] 确认 N 轮反思循环正常终止 (无无限循环)
- [ ] 补充 `.env` 中真实的 `BAILIAN_API_KEY`

### 🟡 P1 — 强烈推荐

- [ ] Tournament 淘汰逻辑强化 (Item 26)
- [ ] Semantic Scholar API 接入 (Item 21)
- [ ] LangGraph Checkpoint 持久化 (Item 17)
- [ ] Entropy-based 实验设计 (Item 6)

### 🟢 P2 — 锦上添花

- [ ] 知识图谱自动构建 Neo4j 集成 (Item 22)
- [ ] 真正双线程架构 (Item 23)
- [ ] GP 反事实模型 (Item 19)
- [ ] Docker 容器化部署
- [ ] Jupyter Notebook 交互式演示

---

## 📖 关键设计文档

| 文件 | 内容 |
|------|------|
| `twinScientist/README.md` | 详细技术文档 (九层 28 项架构) |
| `twinScientist/STARTUP.md` | 启动指南与环境配置 |
| `twinScientist/bug_report.md` | Bug 修复记录与已知问题 |
| `twinScientist/fix_report.py` | 临时修复补丁脚本 |
| `gen_multimodal_simulator.py` | 多模态数据生成器使用说明 |

---

## 🔗 参考文献与科学依据

模拟器中使用的因果系数来自以下已发表研究：

| 路径 | 关键文献 |
|------|---------|
| Temp→HR/HRV | Wolkove et al., Int J Biometeorol 2007; Bouchard et al., Environ Health Perspect 2011 |
| CO₂→Physiology | Allen et al., Environ Health Perspect 2016; Qian et al., Sci Total Environ 2015 |
| PM→Cardiovascular | Brook et al., Circulation 2010; Liu et al., Lancet Planet Health 2019 |
| VOC→Neurological | Nazaroff 2015, Annu Rev Public Health |
| Humidity→HRV | Griefrian et al., Int J Biometeorol 2019 |
| Screen→Blink/Fatigue | Amrnicha et al., Ophthalmic Physiol Opt 2013; Bradley & Phillips, Ophthal Physiol Opt 2000 |
| Humidity→Dry Eye | Kotecha et al., Clin Exp Optom 2012 |

---

## ⚠️ 注意事项

1. **API Key 安全**: 不要将 `.env` 文件提交到公共仓库。`.gitignore` 已包含 `.env`
2. **Python 版本**: 推荐 Python 3.10-3.12，3.14+ 可能遇到某些包兼容性问题
3. **Windows 兼容性**: 本项目的 Bash 命令均使用 Git Bash 语法，WSL/MSYS2 用户需注意路径转换
4. **LLM 延迟**: 阿里云百炼 API 可能存在间歇性超时，系统自带 3 次重试机制
5. **数据隐私**: 合成数据不含任何真实个人信息，仅供算法测试与演示

---

## 📄 许可证

MIT License

---

## 👥 贡献

欢迎提交 Issue 和 Pull Request！请确保：
- 新功能有对应的测试用例
- 代码风格与现有代码保持一致
- 更新相关文档

---

*Built with ❤️ using Qwen + LangGraph + Pydantic*
