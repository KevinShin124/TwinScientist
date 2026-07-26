# 🔬 TwinScientist — AI Scientist for Environment-Human Health Research

> **Autonomous Research Agent** — Built on domestic open-source LLMs (Qwen), fusing real IoT sensor data with causal inference to autonomously complete the full scientific discovery pipeline.
>
> Challenge Cup XH-202619 · Hosted by Alibaba Cloud, Chinese Academy of Sciences & CCF

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Qwen](https://img.shields.io/badge/Model-Qwen--Max-orange.svg)](https://tongyi.aliyun.com/)
[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-green.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

TwinScientist is an autonomous AI Scientist for **environment-human health research**. It fuses temperature, humidity, CO₂, PPG, SpO₂, HRV, and visual fatigue data from real IoT sensors, and autonomously completes:

```
Question → Ethics Check → Literature Review → Hypothesis Generation → Tournament →
Experiment Design → Causal Inference → Interpretation → Peer Review → Debate →
Termination → Report Generation → Human Approval → Self-Evolution
```

**What sets it apart**: All major AI Scientists (Sakana, Agent Laboratory, OpenAI Deep Research, Google Co-Scientist) can only run code or search the web. TwinScientist connects to **real IoT sensor data**, runs **causal inference** (CCM/Granger/Counterfactual), and supports **N-of-1 personalized research** — a combination no other system provides.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/KevinShin124/TwinScientist.git
cd TwinScientist/twinScientist

# 2. Configure
cp .env.example .env
# Edit .env and add your BAILIAN_API_KEY

# 3. Install
pip install -r requirements.txt

# 4. CLI mode
python -m main --question "How does temperature affect heart rate variability?" --iterations 5

# 5. Gradio UI
python -m main --ui
# → Open http://127.0.0.1:7860 in your browser
```

---

## Architecture

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
│  ↻ Reflection Loop: critique → revise → re-validate (max N)  │
└──────────────────────────────────────────────────────────────┘
```

### 14 Cognitive Nodes

| # | Node | Description |
|---|------|-------------|
| 1 | Ethics Check | Structured JSON safety review, 3-tier evaluation |
| 2 | Literature Review | Crossref + arXiv parallel search, CitationValidator |
| 3 | Hypothesis Generation | LogicEngine triple-path reasoning + LLM enhancement |
| 4 | Tournament Eval | N-to-1 elimination bracket, LLM judge scoring |
| 5 | Experiment Design | 12-field experiment plan, auto data file linking |
| 6 | Data Analysis | Granger/CCM/Counterfactual auto-selection + execution |
| 7 | Interpretation | LLM analysis + Bayesian posterior update |
| 8 | Reviewer Agent | 5-dimension peer review (novelty/feasibility/methodology/evidence/impact) |
| 9 | Reflection | Root cause analysis → derived hypotheses → failure capitalization |
| 10 | Debate | Pro/Con/Judge adversarial argumentation |
| 11 | Termination Eval | Marginal improvement detection, budget-first + quality early-exit |
| 12 | Report Writing | 13-section standardized report + visualization injection |
| 13 | PI Agent | Multi-agent synthesis |
| 14 | HITL + Evolution | Human-in-the-loop approval + meta-insight extraction |

### 6 Upgrades (v2.0)

| Upgrade | Benchmark | Module |
|---------|-----------|--------|
| **Streaming Dashboard** | OpenAI Deep Research real-time progress | `core/progress.py` |
| **Multi-Modal Output** | Apple-grade visualization (ASCII charts/Mermaid/timeline) | `core/visualization.py` |
| **Adaptive Iterations** | OpenAI Test-Time Compute Scaling | `core/adaptive.py` |
| **Agent Personas** | Google Co-Scientist 6-Agent architecture | `core/prompts.py` |
| **Research Memory** | Cross-session knowledge persistence | `core/memory.py` |
| **SFT Data Pipeline** | Automated Qwen fine-tuning data collection | `core/sft_pipeline.py` |

---

## Competitive Landscape

| System | Data Source | Causal Inference | Debate | HITL | N-of-1 |
|--------|------------|-----------------|--------|------|--------|
| Sakana AI-Scientist | Code execution | ❌ | ❌ | ❌ | ❌ |
| Agent Laboratory | Code execution | ❌ | ❌ | Optional | ❌ |
| OpenAI Deep Research | Web search | ❌ | ❌ | ❌ | ❌ |
| Google Co-Scientist | Literature DB | ❌ | ❌ | Optional | ❌ |
| **TwinScientist** | **Real IoT sensors** | **✅ CCM/Granger/Counterfactual** | **✅ Pro/Con/Judge** | **✅ Full pipeline** | **✅ Cross-scenario** |

---

## Project Structure

```
twinScientist/
├── main.py                  # CLI entry + Gradio UI
├── core/
│   ├── graph.py             # LangGraph cognitive graph (14 nodes)
│   ├── nodes.py             # 14 cognitive node implementations
│   ├── state.py             # AgentState 4-dimensional state space
│   ├── orchestrator.py      # Routing decisions + termination logic
│   ├── prompts.py           # System prompts + Agent personas
│   ├── llm_client.py        # Qwen HTTP client (connection pooling + retry)
│   ├── logic_engine.py      # Triple-path reasoning (inductive/deductive/abductive)
│   ├── debate.py            # Pro/Con/Judge debate orchestration
│   ├── progress.py          # Streaming progress dashboard
│   ├── visualization.py     # Multi-modal output (ASCII/Mermaid)
│   ├── adaptive.py          # Adaptive iteration budget
│   ├── memory.py            # Cross-session research memory
│   ├── sft_pipeline.py      # SFT fine-tuning data pipeline
│   ├── chat_agent.py        # Conversational agent
│   ├── education.py         # Educational annotations
│   ├── cross_disciplinary.py # Cross-disciplinary transfer analysis
│   ├── experience.py        # Experience store (disabled by default)
│   └── mc_learning.py       # MC reinforcement learning (disabled by default)
├── tools/
│   ├── causal_inference.py  # 8 causal inference methods
│   └── lit_search.py        # Literature search (Crossref/arXiv)
├── output/
│   └── report_generator.py  # Standardized report generation
├── config/
│   └── settings.py          # Configuration management
├── ui/
│   └── app.py               # Gradio interactive interface
└── data/
    ├── sensors/              # 16 environmental sensor CSVs
    └── biometric/            # 8 biometric CSVs
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Qwen-Max (Alibaba Cloud Bailian, OpenAI-compatible API) |
| Agent Orchestration | LangGraph (StateGraph + MemorySaver) |
| Causal Inference | CCM / Granger Causality / Counterfactual |
| Literature Search | Crossref REST API / arXiv API / Semantic Scholar |
| Knowledge Graph | NetworkX directed graph + entity-relation extraction |
| Data Pipeline | Daltons format parsing + time alignment + quality assessment |
| UI | Gradio |
| Output | Markdown standardized report + visualization injection |

---

## Competition Scoring

| Dimension | Points | TwinScientist |
|-----------|--------|---------------|
| Scientific Value — Innovation | 20 | Causal inference methodology + triple-path reasoning + N-of-1 frontier |
| Scientific Value — Verifiability | 20 | Traceable evidence chains + Bayesian confidence + statistical basis |
| Technical — Agent Collaboration | 15 | 14-node LangGraph + 6 personas + debate |
| Technical — Multi-modal | 15 | Environment + biometric fusion + Daltons parsing + visualization |
| Application — Real-world | 10 | Real sensor data + indoor environmental health |
| Application — Transferability | 10 | Open source + SFT pipeline + standardized benchmark |
| Application — Social Impact | 10 | N-of-1 personalized health + environmental risk management |
| **Total** | **100** | **Full coverage across all dimensions** |

---

## License

MIT © 2025 TwinScientist