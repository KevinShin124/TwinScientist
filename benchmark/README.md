# TwinScientist Benchmark Suite

因果发现能力的金标准评估框架。

## 快速开始

```bash
# 1. 运行完整 benchmark（10 个场景 + 基线对照）
py benchmark/runner.py --method all --report

# 2. 仅运行单一场景
py benchmark/runner.py --scenario s1_temp_hr

# 3. 强制重新生成数据
py benchmark/runner.py --force --report

# 4. 查看报告
cat benchmark/results/report.md
```

## 目录结构

```
benchmark/
├── scenarios.py      # 10 个金标准场景定义 + ground truth（13 条因果边）
├── metrics.py        # 评估指标：F1 / 方向准确率 / 假阳性率 / SHD
├── baselines.py      # 3 个对照基线（相关 / Granger / 随机）
├── runner.py         # 主执行器：生成数据 → 推理 → 评分
├── report.py         # Markdown 评估报告生成器
├── README.md         # 本文件
├── data/cache/       # 缓存生成的合成数据
└── results/          # 输出：metrics.json + report.md
```

## 金标准数据

所有测试数据由 `gen_multimodal_simulator.py` 生成，该模拟器基于 10 篇同行评审文献的因果系数实现 7 个隐藏因果负载因子。每条环境→生理的因果边均有确切的文献来源和预期效应方向。

### 13 条金标准因果边

| 原因变量 | 结果变量 | 效应方向 | 生物学路径 |
|---|---|---|---|
| T | HR_BPM | ↑→↑ | 温度↑ → 交感神经激活 → 心率↑ |
| T | SDNN_ms | ↑→↓ | 温度↑ → 热负荷 → HRV↓ |
| T | PPG_amplitude | ↑→↑ | 温度↑ → 血管舒张 → PPG振幅↑ |
| CO2 | HR_BPM | ↑→↑ | CO₂↑ → 脑血流增加 → 心率↑ |
| CO2 | RMSSD_ms | ↑→↓ | CO₂↑ → 自主神经偏移 → HRV↓ |
| CO2 | SpO2_pct | ↑→↓ | CO₂↑ → 血氧饱和度↓ |
| PMS2_5 | HR_BPM | ↑→↑ | PM2.5↑ → 氧化应激 → 心率↑ |
| PMS2_5 | SDNN_ms | ↑→↓ | PM2.5↑ → 系统性炎症 → HRV↓ |
| PMS2_5 | RMSSD_ms | ↑→↓ | PM2.5↑ → 系统性炎症 → RMSSD↓ |
| PMS2_5 | SpO2_pct | ↑→↓ | PM2.5↑ → 动脉氧合下降 → SpO₂↓ |
| C2H5OH | RMSSD_ms | ↑→↓ | VOC↑ → 神经毒性 → HRV↓ |
| C2H5OH | HR_BPM | ↑→↑ | VOC↑ → 刺激反应 → 心率↑ |
| H | SDNN_ms | ↑→↓ | 湿度↑ → 热舒适受损 → HRV↓ |

## 10 个 Benchmark 场景

| # | 场景 | 测试目标 |
|---|---|---|
| S1 | 温度→心率 | 单因果链基础测试 |
| S2 | CO₂→SpO₂ | 反向因果方向判断 |
| S3 | PM2.5→HRV(多指标) | 一对多的多维影响 |
| S4 | VOC→HRV(神经毒性) | 特定病理路径检测 |
| S5 | 多因一果(T+CO₂+PM→HR) | 多因素去混淆能力 |
| S6 | 零效应对照 | 假阳性率控制 |
| S7 | 弱效应(短序列) | 小样本鲁棒性 |
| S8 | 湿度→HRV | 间接效应检测 |
| S9 | 温度→多指标 | 多条因果边并发 |
| S10 | 全因果图(13边) | 综合能力终测 |

## 评估指标

| 指标 | 公式 | 含义 |
|---|---|---|
| F1 | 2×P×R/(P+R) | 因果边发现综合准确率 |
| Precision | TP/(TP+FP) | 报告的因果边中真正因果的比例 |
| Recall | TP/(TP+FN) | 真实因果边中被发现的比例 |
| 方向准确率 | 方向正确/TP | 检测到的边中方向正确的比例 |
| 假阳性率 | FP/(FP+TN) | 无因果关系的变量对中被误报的比例 |

## 基线方法

| 基线 | 方法 | 用途 |
|---|---|---|
| Pearson 相关 | |r|>0.3 即判为因果 | 最简单方法的底线 |
| 纯 Granger | 单变量 Granger 检验 | 传统因果检验的上限 |
| 随机基线 | 50% 随机判定 | 统计下界参考 |

## 添加新场景

编辑 `benchmark/scenarios.py`：

```python
BenchmarkScenario(
    id="s11_new",
    name="新场景名称",
    description="场景描述",
    causal_pairs=[_cp("变量A", "变量B", "positive", "说明")],
    null_pairs=[("变量C", "变量D")],
    n_subjects=2, n_days=14,
)
```
