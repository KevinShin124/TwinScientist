# 🔬 twinScientist — AI Scientist for Environment-Human Research

面向环境—人体关联研究的自主科研与实验迭代智能体。  
**基于 Qwen 系列模型 + LangGraph 多智能体架构**（挑战杯 AI Scientist 赛题）。

---

## 📋 项目简介

twinScientist 融合温湿度、CO₂、PPG、血氧、HRV 及视觉疲劳等多源数据，自动完成：

1. **数据清洗 → 时间对齐 → 质量评估**
2. **科学假设生成**（Tournament 进化 + Bayesian 置信度量化）
3. **实验方案设计**（主动学习：针对高熵区域设计实验）
4. **因果推断分析**（8 类工具 + AI 自主选择）
5. **五维评审**（新颖性/可行性/方法论/证据/影响）
6. **反思-修正闭环**（失败资产化：教训存储 + 派生新假设）
7. **标准化报告输出**（含赛题要求的 12+ 字段）

通过可追溯证据链、N-of-1 个体化研究和人机协同审核，实现从数据采集到结论验证的**完整科研闭环**。

---

## 🏗️ 九层 28 项架构总览

### 第一层：基座模型与基础设施
| # | 特性 | 文件 | 状态 |
|---|------|------|------|
| 1 | 基座模型强制切换 Qwen（阿里云百炼平台 API） | `config/settings.py`, `core/llm_client.py` | ✅ 已配置占位 |
| 2 | Prompt 工程适配 Qwen（指令遵循/工具调用格式定制） | `core/prompts.py` | ✅ 已定制 |

### 第二层：智能体编排与认知架构
| # | 特性 | 文件 | 状态 |
|---|------|------|------|
| 3 | DAG→认知图（文献/假设/实验/解读/反思节点） | `core/graph.py` | ✅ 已实现 |
| 4 | Orchestrator 动态路由（基于证据强度/不确定性决策） | `core/orchestrator.py` | ✅ 已实现 |
| 5 | 假设树架构（动态生长与修剪，支持分支） | `core/state.py` (`hypothesis_tree`) | ✅ 已建模 |
| 6 | 主动学习实验设计（高熵区域优先探索） | `core/nodes.py` (experiment_design) | ⚠️ TODO: 熵计算 |

### 第三层：状态空间与记忆管理
| # | 特性 | 文件 | 状态 |
|---|------|------|------|
| 7 | 双轨→四维记忆体（工作/情景/语义/证据/异常） | `core/state.py` | ✅ 已实现 |
| 8 | 三层持久化记忆（L1 Kernel / L2 SQLite+向量 / L3 KG+向量） | `channels/metadata_channel.py` | ⚠️ L2/L3 待接入 |
| 9 | 自我进化机制（Evolution Manager） | `core/nodes.py` (`node_evolution_manager`) | ✅ 已实现 |
| 10 | 失败资产化（根因分析+教训存储+派生修正假设） | `core/nodes.py` (`node_reflection`) | ✅ 已实现 |

### 第四层：人机协同与质量控制
| # | 特性 | 文件 | 状态 |
|---|------|------|------|
| 11 | 结构化决策面板（Radio/Slider/Checkbox） | `ui/app.py` | ✅ Gradio UI |
| 12 | 主动介入通道（暂停/重定向/修改/回滚） | `ui/app.py`, `core/nodes.py` (`human_approval`) | ✅ 断点机制 |
| 13 | PI Agent 首席研究员 | `core/nodes.py` (`node_pi_agent_meeting`) | ✅ 已实现 |
| 14 | Reviewer Agent 五维审稿 | `core/nodes.py` (`node_reviewer_agent`) | ✅ 已实现 |
| 15 | 伦理与安全看门狗 | `core/nodes.py` (`node_ethics_check`) | ✅ 已实现 |

### 第五层：代码执行沙箱与数据管道
| # | 特性 | 文件 | 状态 |
|---|------|------|------|
| 16 | DVC Lite 版本控制（关键数据快照） | `core/nodes.py` (placeholder) | ⚠️ TODO |
| 17 | 时间旅行能力（LangGraph Checkpoint 历史加载） | `core/graph.py` (checkpoint_store placeholder) | ⚠️ TODO |
| 18 | 多源时序引擎（传感器+PPG+血氧异步接入） | `channels/time_series.py` | ⚠️ 占位 |
| 19 | 反事实推理引擎（GP 代理模型预测） | `tools/causal_inference.py` | ⚠️ 占位 |

### 第六层：工具调用与外部能力
| # | 特性 | 文件 | 状态 |
|---|------|------|------|
| 20 | 因果推断工具箱（8 类方法 + AI 自主选择） | `tools/causal_inference.py` | ✅ 框架已搭 |
| 21 | 持续文献监控（后台定时检索+关联度评估） | `tools/literature_monitor.py` | ⚠️ 需接 API |
| 22 | 自动知识图谱构建（三元组抽取） | `core/state.py` (`knowledge_graph`) | ⚠️ TODO |

### 第七层：执行架构与终止条件
| # | 特性 | 文件 | 状态 |
|---|------|------|------|
| 23 | 双线程协作（推理线程 + 执行线程异步解耦） | `core/graph.py` (stream_mode) | ⚠️ TODO: 真正双线程 |
| 24 | Agentic Tree Search（并行探索 + 动态剪枝） | `core/nodes.py` (`hypothesis_generation`) | ⚠️ TODO: 真实并行 |
| 25 | 三层语义终止评估（收敛+充分+穷尽综合评分>0.85） | `core/nodes.py` (`termination_eval`) | ✅ 已实现 |

### 第八层：科学假设生成质量
| # | 特性 | 文件 | 状态 |
|---|------|------|------|
| 26 | Tournament 进化（候选假设两两 PK） | `core/nodes.py` (`hypothesis_generation`) | ⚠️ TODO: 真实淘汰逻辑 |
| 27 | Bayesian 置信度量化（P(H|D) 后验概率） | `core/state.py` (`confidence_prior/posterior`) | ✅ 已建模 |

### 第九层：输出与成果规范
| # | 特性 | 文件 | 状态 |
|---|------|------|------|
| 28 | 赛题规范输出格式（12+ 标准字段） | `output/report_generator.py` | ✅ 已实现 |

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 BAILIAN_API_KEY（阿里云百炼）
```

### 运行方式

```bash
# CLI 交互模式
python main.py

# 命令行模式
python main.py --question "高温环境对老年人心率变异性有何影响？" --domain "环境健康"

# 启动 Web UI
python main.py --ui
```

---

## 📁 项目结构

```
TwinScientist/
├── .env.example                    # 环境变量模板
├── requirements.txt                # Python 依赖
├── main.py                         # CLI/Web UI 入口
│
├── config/                         # ━━ Layer 1: 配置 ━━
│   ├── __init__.py
│   └── settings.py                 # Settings（从 .env 加载）
│
├── core/                           # ━━ Layer 1-3: 核心 ━━
│   ├── llm_client.py               #     Qwen Client（百炼 API）
│   ├── prompts.py                  #     Qwen 优化 Prompt 模板
│   ├── state.py                    #     AgentState（四维记忆体）
│   ├── graph.py                    #     Cognitive Graph（DAG 编排）
│   ├── orchestrator.py             #     Dynamic Routing 引擎
│   └── nodes.py                    #     12+ 认知操作节点函数
│
├── channels/                       # ━━ Layer 5: 数据管道 ━━
│   ├── base.py                     #     Channel 基类
│   ├── time_series.py              #     多源时序引擎
│   └── metadata_channel.py         #     SQLite 元数据存储
│
├── tools/                          # ━━ Layer 6: 工具集 ━━
│   ├── causal_inference.py         #     8 类因果推断工具
│   └── literature_monitor.py       #     持续文献监控
│
├── output/                         # ━━ Layer 9: 输出 ━━
│   ├── report_generator.py         #     标准化报告生成器
│   └── __init__.py
│
├── ui/                             # ━━ Layer 4: 人机交互 ━━
│   └── app.py                      #     Gradio Web UI
│
├── data/                           # 数据目录（存放真实数据）
│   ├── sensors/                    #     环境传感器数据
│   ├── biometric/                  #     PPG/血氧/HRV 数据
│   └── visual_fatigue/             #     视觉疲劳数据
│
├── logs/                           # 日志
├── output/                         # 生成的报告
└── README.md
```

---

## 🎯 比赛匹配度对照

| 比赛要求 | 本项目覆盖 |
|---------|-----------|
| ✅ 基座必须为 Qwen（百炼平台） | `core/llm_client.py` — Qwen via Bailian API |
| ✅ 问题理解 / 知识整合 / 关联发现 / 假设生成 | 完整认知图工作流覆盖 |
| ✅ 基于多维度/多模态实测数据 | `channels/time_series.py` — 环境+生物信号+视觉 |
| ✅ 多智能体协作 | PI Agent + Reviewer Agent + Ethics Watchdog + Orchestrator |
| ✅ 人机协作流程 | Gradio UI + interrupt_before 断点 |
| ✅ 参考文献真实性 | ReportGenerator 强调 DOI/PMID 必填 |
| ✅ 标准化输出格式 | `output/report_generator.py` — 12+ 字段完整覆盖 |
| ✅ 前端界面（加分项） | Gradio Web UI |
| ✅ SFT 微调支持 | 架构允许替换 llm_client 为微调模型 |
| ✅ 因果推断 | 8 类工具 + AI 自主选择框架 |

---

## 📝 开发里程碑建议

| 阶段 | 内容 | 预估工作量 |
|------|------|-----------|
| Phase 1 | 基础功能跑通（Layer 1-4，约 15 项） | ~2 周 |
| Phase 2 | 数据管道接入（Layer 5-6，约 10 项） | ~3 周 |
| Phase 3 | 高级能力完善（Layer 7-9，约 5 项） | ~2 周 |
| Phase 4 | 前端打磨 + 演示视频 | ~1 周 |
| **总计** | **~8 周**（截止 9 月 5 日足够） | |

---

## 🛣️ TODO 路线图（按优先级排列）

### 🔴 P0 — 必须完成
- [ ] 填入真实的 `BAILIAN_API_KEY`
- [ ] 将 `causal_inference.py` 中的占位替换为真实算法调用
- [ ] 准备至少一组真实数据集放入 `data/` 目录
- [ ] 测试端到端流水线：CLI → Report 生成
- [ ] 确认 5 轮循环硬上限正常工作（每一轮自动回答三个反思问题）

### 🟡 P1 — 强烈推荐
- [ ] 实现 Tournament 淘汰逻辑（Item 26）
- [ ] 接入 Semantic Scholar API（Item 21）
- [ ] 添加 LangGraph Checkpoint 持久化（Item 17）
- [ ] 实现 Entropy-based 实验设计（Item 6）

### 🟢 P2 — 锦上添花
- [ ] 知识图谱自动构建（Item 22，Neo4j）
- [ ] 真正的双线程架构（Item 23）
- [ ] GP 反事实模型（Item 19）
- [ ] Docker 容器化部署
