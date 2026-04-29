# GEPA Study - Genetic-Evolutionary Prompt Architecture

> GEPA (Genetic-Evolutionary Prompt Architecture) 深度研究与实验代码仓库
>
> 论文: [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457)
> 官方仓库: [gepa-ai/gepa](https://github.com/gepa-ai/gepa) (4100+ stars)

## 核心结论

**GEPA 的本质：用 LLM 的推理能力来弥补优化过程中的信息损失。**

GEPA 比 RL 快 35 倍、比人工调优高效的**根本原因**：

| 传统方法 | GEPA |
|---------|------|
| `score: 0.3` (标量) | 完整执行轨迹 + LLM 诊断 |
| 丢弃失败原因 | 分析失败原因生成 ASI |
| 只能告诉"错了" | 告诉"哪里错、为什么错、如何改" |

**ASI (Actionable Side Information)** = GEPA 与其他所有优化方法的**关键差异点**

## 目录结构

```
gepa-study/
├── README.md                         # 本文件
├── requirements.txt                  # Python 依赖
├── GEPA深度研究报告.md              # 完整研究报告
└── experiments/
    ├── 01_gepa_core_concepts.py     # 核心概念演示（无需 API key）
    ├── 02_gepa_with_api.py          # API 实验（需要 OpenAI key）
    └── 03_gepa_official_package.py  # 官方包实验（需要 API keys）
```

## 快速开始

### 环境准备

```bash
# 克隆本仓库
git clone https://github.com/floatin/gepa-study.git
cd gepa-study

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 实验 1: 核心概念演示（无需 API key）

```bash
python experiments/01_gepa_core_concepts.py
```

这个脚本演示 GEPA 的三大核心概念：
- **ASI vs 标量反馈**：为什么诊断信息比分数更重要
- **反思性循环**：Select → Execute → Reflect → Mutate → Accept
- **帕累托前沿选择**：如何保持多目标优化的多样性

### 实验 2: API 实验（需要 `OPENAI_API_KEY`）

```bash
export OPENAI_API_KEY="sk-..."
python experiments/02_gepa_with_api.py
```

使用模拟或真实的 OpenAI API 运行 GEPA 流程。

### 实验 3: 官方包实验（需要 API keys）

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
pip install gepa
python experiments/03_gepa_official_package.py
```

使用官方 `gepa` 包进行完整优化。

## 核心概念速览

### 1. Actionable Side Information (ASI)

ASI 是 GEPA 最重要的创新——它解决了"LLM 知道自己错了，但不知道错在哪"的问题：

```python
# 传统优化器只能看到：
{"score": 0.3}  # 信息量: ~2 bits

# GEPA 的 ASI 看到的是：
{
  "input": "解方程: -3x + 7 = 0",
  "output": "x = -7/3",
  "diagnosis": "助手忽略了负号。正确解是 x = 7/3。",
  "fix_suggestion": "在解题前增加一步：'首先检查系数符号'",
  "asi_type": "sign_error"
}
```

ASI 相当于文本优化任务中的"梯度"——这就是 GEPA 比 RL 快 35 倍的根本原因。

### 2. 反思性循环

```
┌─────────────────────────────────────────────────────────────┐
│  1. SELECT   从 Pareto 前沿选择父候选                        │
│      ↓                                                       │
│  2. EXECUTE  在小批量样本上执行，捕获完整轨迹               │
│      ↓                                                       │
│  3. REFLECT  LLM 阅读轨迹，诊断失败原因（ASI）              │
│      ↓                                                       │
│  4. MUTATE   基于 ASI 生成改进的候选项                       │
│      ↓                                                       │
│  5. ACCEPT   验证改进效果，更新 Pareto 前沿                   │
└─────────────────────────────────────────────────────────────┘
```

### 3. 帕累托前沿选择

不同于 RL 把多个目标压缩成单标量，GEPA 维护**帕累托前沿**：

```
                    Pareto Front
                    (非支配解)
    任务A ──────────────────→
         │    ● C1 (A优, B中)
         │         ● C2 (A中, B优)
         │    ◆ C3 (A差, B优)
    ─────┼────────────────────→ 任务B
              ◆ C4 (A优, B差)
```

## 主要实验结果

| 任务 | 基线 | GEPA 优化后 | 提升 |
|------|------|-------------|------|
| AIME 数学 | GPT-4.1 Mini: 46.6% | 56.6% | +10pp |
| ARC-AGI | 32% | 89% | +57pp |
| Databricks Agent | Claude Opus 4.1 | 开源模型+GEPA | 90x 更便宜 |
| 评估效率 | RL: 5000~25000次 | GEPA: 100~500次 | 35x 更快 |

## 与 RL/DSPy 的对比

| 维度 | RL/GRPO | DSPy | **GEPA** |
|------|---------|------|---------|
| 反馈信息 | 标量 reward | 可微分签名 | **完整执行轨迹（ASI）** |
| 失败诊断 | ❌ | ❌ | **✅ LLM 反思诊断** |
| 多目标处理 | 压缩成单标量 | N/A | **Pareto 前沿** |
| 评估次数 | 5000~25000 | 几百 | **100~500** |

## 参考资料

- 论文: https://arxiv.org/abs/2507.19457
- 官方 GitHub: https://github.com/gepa-ai/gepa
- 官方文档: https://gepa-ai.github.io/gepa/
- DSPy 集成: https://dspy.ai/
- MLflow 集成: https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/

## 维护

本仓库由 [@floatin](https://github.com/floatin) 维护，用于技术研究与学习。

---
*"GEPA 的本质是：用 LLM 的推理能力来弥补优化过程中的信息损失。"*
