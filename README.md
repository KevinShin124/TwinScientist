# 🔬 TwinScientist — AI Scientist for Environment-Human Health Research

> **自主科研智能体** — 基于国产开源大模型 (Qwen) 的 AI Scientist，融合真实传感器数据，自动完成从假设生成到因果推断的完整科研闭环。
>
> 挑战杯 XH-202619 · 阿里云 & 中国科学院 & 中国计算机学会联合主办

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Qwen](https://img.shields.io/badge/Model-Qwen--Max-orange.svg)](https://tongyi.aliyun.com/)
[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-green.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 核心定位

TwinScientist 是一款面向**环境—人体关联研究**的自主科研与实验迭代智能体。系统融合温湿度、CO₂、PPG、血氧、HRV 等多源真实传感器数据，自动完成：

```
问题输入 → 伦理审查 → 文献调研 → 假设生成 → 淘汰赛 → 实验设计 →
数据分析 → 因果推断 → 结果解读 → 五维评审 → 智能体辩论 → 终止评估 →
报告撰写 → 人机审核 → 自我进化
```

**与竞品的本质区别**：所有国际 AI Scientist（Sakana、Agent Laboratory、OpenAI Deep Research）只能跑代码或搜网页。TwinScientist 接入**真实 IoT 传感器数据**，运行**因果推断**（CCM/Granger/Counterfactual），支持 **N-of-1 个体化研究**。

---

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/KevinShin124/TwinScientist.git
cd TwinScientist/twinScientist

# 2. 配置
cp .env.example .env
# 编辑 .env，填入 BAILIAN_API_KEY

# 3. 安装依赖
pip install -r requirements.txt

# 4. CLI 模式
python -m main --question "温度对心率变异性的影响" --iterations 5

# 5. Gradio UI
python -m main --ui
# → 浏览器打开 http://127.0.0.1:7860
```

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                    Orchestrator (LangGraph)                   │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Ethics  │→ │Literature│→ │Hypothesis│→ │  Tournament  │  │
│  │ Check   │  │ Review   │  │   Gen    │  │    Eval      │  │
│  └─────────┘  └──────────┘  └──────────┘  └──────────────┘  │
│                                                  ↓           │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  Data   │← │Experiment│← │ Causal   │← │   Time       │  │
│  │Analysis │  │  Design  │  │Inference │  │  Alignment   │  │
│  └─────────┘  └──────────┘  └──────────┘  └──────────────┘  │
│       ↓                                                      │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │Interpret│→ │ Reviewer │→ │  Debate  │→ │ Termination  │  │
│  │         │  │  (5-Dim) │  │(Pro/Con/ │  │    Eval      │  │
│  └─────────┘  └──────────┘  │ Judge)   │  └──────────────┘  │
│                             └──────────┘        ↓           │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  Self   │← │  Human   │← │   PI     │← │   Report     │  │
│  │Evolution│  │ Approval │  │  Agent   │  │   Writing    │  │
│  └─────────┘  └──────────┘  └──────────┘  └──────────────┘  │
│                                                              │
│  ↻ Reflection Loop: 反思修正 → 新假设生成 → 再验证 (最多N轮) │
└──────────────────────────────────────────────────────────────┘
```

### 14 个认知节点

| # | 节点 | 功能 |
|---|------|------|
| 1 | Ethics Check | 结构化 JSON 安全审查，三段式评估 |
| 2 | Literature Review | Crossref + arXiv 并行搜索，CitationValidator 验证 |
| 3 | Hypothesis Generation | LogicEngine 三路推理 (归纳/演绎/溯因) + LLM 增强 |
| 4 | Tournament Eval | N 选 1 淘汰赛，LLM 裁判打分 |
| 5 | Experiment Design | 12 字段完整实验方案，自动关联数据文件 |
| 6 | Data Analysis | Granger/CCM/Counterfactual 自动选择 + 执行 |
| 7 | Interpretation | LLM 解读 + Bayesian 后验概率更新 |
| 8 | Reviewer Agent | 五维评审 (新颖性/可行性/方法论/证据/影响力) |
| 9 | Reflection | 根因分析 → 派生修正假设 → 失败资产化 |
| 10 | Debate | Pro/Con/Judge 三方对抗辩论 |
| 11 | Termination Eval | 边际收益递减检测，预算优先 + 质量早退 |
| 12 | Report Writing | 13 节标准化报告 + 可视化注入 |
| 13 | PI Agent | 多智能体成果整合 |
| 14 | HITL + Evolution | 人机审核 + 元洞察提取 |

### 6 大升级 (v2.0)

| 升级 | 对标 | 模块 |
|------|------|------|
| **Streaming Dashboard** | OpenAI Deep Research 实时进度 | `core/progress.py` |
| **Multi-Modal Output** | Apple 级可视化 (ASCII图表/Mermaid/时间线) | `core/visualization.py` |
| **Adaptive Iterations** | OpenAI Test-Time Compute Scaling | `core/adaptive.py` |
| **Agent Personas** | Google Co-Scientist 6-Agent 架构 | `core/prompts.py` |
| **Research Memory** | 跨会话知识持久化 | `core/memory.py` |
| **SFT Data Pipeline** | Qwen 微调数据自动收集 | `core/sft_pipeline.py` |

---

## 国际竞品对比

| 系统 | 数据源 | 因果推断 | 辩论 | 人在回路 | N-of-1 |
|------|--------|---------|------|---------|--------|
| Sakana AI-Scientist | 代码执行 | ❌ | ❌ | ❌ | ❌ |
| Agent Laboratory | 代码执行 | ❌ | ❌ | 可选 | ❌ |
| OpenAI Deep Research | 网页搜索 | ❌ | ❌ | ❌ | ❌ |
| Google Co-Scientist | 文献库 | ❌ | ❌ | 可选 | ❌ |
| **TwinScientist** | **真实IoT传感器** | **✅ CCM/Granger/Counterfactual** | **✅ Pro/Con/Judge** | **✅ 全流程** | **✅ 跨场景** |

---

## 项目结构

```
twinScientist/
├── main.py                  # CLI 入口 + Gradio UI
├── core/
│   ├── graph.py             # LangGraph 认知图编排 (14 节点)
│   ├── nodes.py             # 14 个认知节点实现
│   ├── state.py             # AgentState 四维状态空间
│   ├── orchestrator.py      # 路由决策 + 终止条件
│   ├── prompts.py           # System Prompts + Agent Personas
│   ├── llm_client.py        # Qwen HTTP 客户端 (连接复用+重试)
│   ├── logic_engine.py      # 三路推理引擎 (归纳/演绎/溯因)
│   ├── debate.py            # Pro/Con/Judge 辩论编排
│   ├── progress.py          # 流式进度仪表盘
│   ├── visualization.py     # 多模态可视化 (ASCII/Mermaid)
│   ├── adaptive.py          # 自适应迭代预算
│   ├── memory.py            # 跨会话研究记忆
│   ├── sft_pipeline.py      # SFT 微调数据管道
│   ├── chat_agent.py        # 对话 Agent
│   ├── education.py         # 教学注释
│   ├── cross_disciplinary.py # 跨学科迁移分析
│   ├── experience.py        # 经验存储 (默认关闭)
│   └── mc_learning.py       # MC 强化学习 (默认关闭)
├── tools/
│   ├── causal_inference.py  # 8 类因果推断方法
│   └── lit_search.py        # 文献检索 (Crossref/arXiv)
├── output/
│   └── report_generator.py  # 标准化报告生成
├── config/
│   └── settings.py          # 配置管理
├── ui/
│   └── app.py               # Gradio 交互界面
└── data/
    ├── sensors/              # 16 个环境传感器 CSV
    └── biometric/            # 8 个生物特征 CSV
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 大模型 | Qwen-Max (阿里云百炼平台, OpenAI 兼容 API) |
| Agent 编排 | LangGraph (StateGraph + MemorySaver) |
| 因果推断 | CCM 收敛交叉映射 / Granger 因果检验 / 反事实推演 |
| 文献检索 | Crossref REST API / arXiv API / Semantic Scholar |
| 知识图谱 | NetworkX 有向图 + 实体关系自动提取 |
| 数据管道 | Daltons 格式解析 + 时序对齐 + 质量评估 |
| UI | Gradio |
| 输出 | Markdown 标准化报告 + 可视化注入 |

---

## 比赛评分对标

| 维度 | 分值 | TwinScientist |
|------|------|---------------|
| 科学价值 — 创新性 | 20 | 因果推断方法论 + 三路推理 + N-of-1 前沿方向 |
| 科学价值 — 可验证性 | 20 | 可追溯证据链 + Bayesian 置信度 + 统计依据 |
| 技术实现 — Agent 协作 | 15 | 14 节点 LangGraph + 6 Persona + 辩论 |
| 技术实现 — 多模态处理 | 15 | 环境+生物特征融合 + Daltons 解析 + 可视化 |
| 应用潜力 — 实际场景 | 10 | 真实传感器数据 + 室内环境健康 |
| 应用潜力 — 转化潜力 | 10 | 开源 + SFT 管道 + 标准化 benchmark |
| 应用潜力 — 社会价值 | 10 | N-of-1 个性化健康 + 环境风险管理 |
| **总分** | **100** | **全面覆盖** |

---

## License

MIT © 2025 TwinScientist