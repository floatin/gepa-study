# GEPA (Genetic-Evolutionary Prompt Architecture) 深度研究报告

> **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning**
> - Authors: Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, et al. (UC Berkeley, Stanford, CMU, etc.)
> - Published: July 2025 | Updated: February 2026
> - Paper: https://arxiv.org/abs/2507.19457
> - GitHub: https://github.com/gepa-ai/gepa (4100+ stars)
> - License: MIT

---

## 1. 背景与动机

### 1.1 传统提示词优化的困境

在 GEPA 出现之前，提示词优化主要有三条路：

| 方法 | 原理 | 缺点 |
|------|------|------|
| **人工调优** | 人工反复写 prompt、测效果 | 效率低、依赖专家经验、无法规模化 |
| **RL/GRPO** | 强化学习让模型自我改进 | 需要 5000~25000 次评估、算力成本极高 |
| **DSPy** | 程序员式编译优化 | 基于梯度/签名，需要可微分的模块 |

这些方法有一个共同的**根本缺陷**：优化器只知道"候选失败了"，但不知道"为什么失败"。错误信息、推理日志、性能剖析数据这些**丰富的执行轨迹（Execution Traces）**被丢弃了。

### 1.2 GEPA 的核心洞察

> **用 LLM 自己来诊断失败原因，再让 LLM 根据诊断结果提出改进。**

GEPA 的核心理念：
- **Reflective（反思性）**：让 LLM 阅读完整执行轨迹，诊断失败原因——这是其他方法完全忽略的"可操作反馈"（Actionable Side Information, ASI）
- **Pareto-Efficient（多目标帕累托最优）**：不像 RL 把多个目标压缩成一个标量，GEPA 维护一个**帕累托前沿**，保留在不同子任务上各有优势的多样化候选
- **Evolutionary（进化式）**：通过迭代"选择→执行→反思→变异→接受"来进化候选项

---

## 2. 核心算法机制

### 2.1 整体流程（5 步循环）

```
┌─────────────────────────────────────────────────────────────┐
│  1. SELECT   从 Pareto 前沿选择父候选                        │
│      ↓                                                       │
│  2. EXECUTE  在小批量样本上执行，捕获完整轨迹                 │
│      ↓                                                       │
│  3. REFLECT  LLM 阅读轨迹，诊断失败原因（ASI）              │
│      ↓                                                       │
│  4. MUTATE   基于 ASI 生成改进的候选项                       │
│      ↓                                                       │
│  5. ACCEPT   验证改进效果，更新 Pareto 前沿                   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 关键组件详解

#### 2.2.1 Reflective Reflection（反思诊断）

传统优化器只能看到 `score = 0.3` 这样的标量。GEPA 的 LLM 阅读：
- **错误信息**（Error messages）
- **推理日志**（Reasoning logs）
- **性能剖析数据**（Profiling data）
- **执行轨迹**（Trajectories）

然后输出**诊断性反馈**（ASI），例如：
```
"The assistant failed because it assumed all numbers were positive,
but the problem contained negative values. Add a step to check sign first."
```

这就是 ASI——相当于文本优化任务中的"梯度"。

#### 2.2.2 Pareto-Efficient Selection

不像 RL 把所有目标压缩成一个 reward scalar，GEPA 维护一个 **Pareto Front**：

```
                    Pareto Front
                    (非支配解)
    任务A表现 ─────────────────→
         │
         │    ● C1 (A优, B中)
         │         ● C2 (A中, B优)
         │    ◆ C3 (A差, B优)
    ─────┼───────────────────────────────────→ 任务B表现
              ◆ C4 (A优, B差)
                   ■ Seed
```

**多目标的意义**：一个提示词可能在数学题上表现好但在推理题上差，另一个提示词反过来。两者都有价值，GEPA 保留它们并在进化中让它们各自继续进化。

#### 2.2.3 Actionable Side Information (ASI)

ASI 是 GEPA 最重要的创新。它解决了"LLM 知道自己错了，但不知道错在哪"的问题：

| 传统方法 | GEPA |
|---------|------|
| `score: 0.3` | `error: "division by zero in step 3, numbers can be negative"` |
| 只能告诉优化器"不好" | 告诉优化器"哪里不好、为什么、怎么改" |

#### 2.2.4 System-Aware Merge

GEPA 还支持**跨 Pareto 前沿合并**：当两个在不同任务上表现优秀的候选项被合并时，可能产生一个在所有任务上都优秀的"全能"候选。

### 2.3 帕累托前沿更新算法

```python
def update_pareto_front(new_candidate, frontier, candidates_scores):
    """
    如果 new_candidate 在所有维度上都不被现有候选支配，
    则加入前沿；被支配的候选则从前沿移除。
    """
    is_dominated = False
    for existing in frontier:
        if all(existing[dim] >= new_candidate[dim] for dim in dims):
            # existing 在所有维度都不差，new 被支配
            is_dominated = True
            break

    if not is_dominated:
        # 移除被 new_candidate 支配的候选
    frontier.remove_if(lambda c: all(new_candidate[dim] >= c[dim] for dim in dims))
        frontier.add(new_candidate)
```

---

## 3. 核心代码架构

### 3.1 目录结构

```
gepa/
├── core/                      # 核心引擎
│   ├── engine.py              # GEPAEngine - 优化循环主控
│   ├── state.py               # GEPAState - 状态管理 + Pareto前沿
│   ├── adapter.py             # GEPAAdapter - 用户系统的接口
│   ├── data_loader.py         # 数据加载器
│   └── callbacks.py           # 事件回调系统
├── proposer/
│   ├── reflective_mutation/   # 核心变异逻辑
│   │   ├── reflective_mutation.py  # 反思变异
│   │   └── base.py           # LLM调用接口
│   └── merge.py              # 系统感知合并
├── strategies/
│   ├── batch_sampler.py      # 小批量采样策略
│   ├── instruction_proposal.py # 指令生成签名（核心提示模板）
│   └── accept_criterion.py    # 接受标准
└── adapters/                  # 预置适配器
    ├── default_adapter/       # 单轮LLM任务
    ├── dspy_full_program/     # DSPy程序适配器
    ├── generic_rag/           # RAG适配器
    └── confidence_adapter/    # 置信度感知适配器
```

### 3.2 GEPAEngine 核心循环

```python
class GEPAEngine:
    def run(self) -> GEPAState:
        # 初始化种子候选 + Pareto前沿
        state = initialize_gepa_state(self.seed_candidate, ...)

        while not self.stop_callback.should_stop(state):
            # Step 1: SELECT - 从 Pareto 前沿选择父候选
            parent_ctx = self.reflective_proposer.prepare_proposal(state)

            # Step 2: EXECUTE - 评估当前候选 + 捕获轨迹
            eval_result = self.adapter.evaluate(
                parent_ctx.minibatch,
                parent_ctx.curr_prog,
                capture_traces=True  # 关键：捕获完整轨迹
            )

            # Step 3: REFLECT - 构建反思数据集
            # 4: MUTATE - 生成改进候选
            # 5: ACCEPT - 验证并更新 Pareto 前沿
            self._process_proposal_output(output, state)
```

### 3.3 反思变异的核心流程

```python
def execute_proposal(self, ctx, state):
    # 1. 评估当前候选，捕获轨迹
    eval_curr = self.adapter.evaluate(ctx.minibatch, ctx.curr_prog, capture_traces=True)

    # 2. 确定要更新的组件（哪些提示词需要改）
    components_to_update = self.module_selector(state, eval_curr.trajectories, ...)

    # 3. 构建反思数据集（包含 ASI）
    reflective_dataset = self.adapter.make_reflective_dataset(
        ctx.curr_prog, eval_curr, components_to_update
    )

    # 4. LLM 阅读反思数据集，生成改进
    new_texts = self.reflection_lm.run(
        prompt_template=DEFAULT_PROMPT_TEMPLATE,  # 见下方
        current_instruction=ctx.curr_prog[name],
        dataset_with_feedback=reflective_dataset[name]
    )

    # 5. 评估新候选
    eval_after = self.adapter.evaluate(ctx.minibatch, new_candidate, capture_traces=True)

    # 6. 接受标准：改进则加入 Pareto 前沿
    if sum(eval_after.scores) > sum(eval_curr.scores):
        state.update_with_new_program(...)
```

### 3.4 核心提示模板

```python
DEFAULT_PROMPT_TEMPLATE = """I provided an assistant with the following instructions to perform a task for me:
<curr_param>

The following are examples of different task inputs provided to the assistant
along with the assistant's response for each of them, and some feedback on
how the assistant's response could be better:
<side_info>

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed
task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback.
Identify all niche and domain specific factual information about the task
and include it in the instruction, as a lot of it may not be available to
the assistant in the future.

Provide the new instructions within ``` blocks."""
```

---

## 4. 关键实验结果

### 4.1 AIME 数学推理

| 配置 | 准确率 | 成本 |
|------|--------|------|
| GPT-4.1 Mini (基线) | 46.6% | - |
| GPT-4.1 Mini + GEPA | **56.6%** | 150 次评估 |
| Claude Opus 4.1 (对比) | 55.5% | - |

> +10 个百分点的提升，仅用 150 次评估。

### 4.2 ARC-AGI 任务

从 32% → 89%（+57个百分点），通过架构发现（Architecture Discovery）。

### 4.3 成本与速度对比

| 方法 | 评估次数 | 相对成本 |
|------|---------|---------|
| GRPO (RL) | 5000~25000+ | 基准 |
| **GEPA** | **100~500** | **35x 更快** |

### 4.4 实际生产案例

- **Shopify, Databricks, Dropbox, OpenAI, Pydantic, MLflow** 等 50+ 企业使用
- **Databricks**: 开源模型 + GEPA 比 Claude Opus 4.1 便宜 **90 倍**
- **云调度策略**: GEPA 发现比专家启发式规则好 **40.2%**

---

## 5. 与 RL/DSPy 的本质区别

| 维度 | RL/GRPO | DSPy | **GEPA** |
|------|---------|------|---------|
| **反馈信息** | 标量 reward | 可微分签名 | **完整执行轨迹（ASI）** |
| **失败诊断** | ❌ 不知道为什么 | ❌ 不知道为什么 | **✅ LLM 反思诊断** |
| **多目标处理** | 压缩成单标量 | N/A | **Pareto 前沿** |
| **评估次数** | 5000~25000 | 几百 | **100~500** |
| **适用场景** | 封闭可量化任务 | 流水线编译 | **任意文本优化** |
| **执行轨迹利用** | ❌ 丢弃 | ❌ 丢弃 | **✅ 核心输入** |

---

## 6. 工程实现要点

### 6.1 GEPAAdapter 接口

用户必须实现两个方法：

```python
class GEPAAdapter(Generic[DataInst, Trajectory, RolloutOutput]):
    # 评估函数：运行候选，捕获轨迹，返回分数
    def evaluate(
        self,
        batch: list[DataInst],
        program: dict[str, str],
        capture_traces: bool
    ) -> RolloutOutput:
        ...

    # 构建反思数据集：从轨迹中提取 ASI
    def make_reflective_dataset(
        self,
        program: dict[str, str],
        eval_result: RolloutOutput,
        component_names: list[str]
    ) -> dict[str, list[dict]]:
        """
        返回结构：
        {
          "system_prompt": [
            {"input": "...", "output": "...", "feedback": "..."},
            ...
          ],
          ...
        }
        """
        ...
```

### 6.2 轨迹捕获的意义

```python
# 不捕获轨迹：只能得到分数
{"scores": [0.0, 1.0, 0.5]}  # 传统方法

# 捕获轨迹：得到完整的失败诊断
{
  "trajectories": [
    {
      "input": "Solve: -3x + 7 = 0",
      "output": "x = 7/3",  # 错误：没考虑负号
      "error": "Division by zero in step 2",
      "metadata": {"reasoning": "..."}
    },
    ...
  ],
  "scores": [0.0, 1.0, 0.5]
}
```

ASI 才是 GEPA 比 RL 快 35 倍的真正原因——每次评估的"信息量"完全不同。

### 6.3 帕累托前沿的四种模式

```python
FrontierType = Literal[
    "instance",    # 按每个验证样本维护前沿
    "objective",   # 按每个目标指标维护前沿
    "hybrid",      # 两者结合
    "cartesian"    # example × objective 的笛卡尔积
]
```

"instance" 模式最适合提示词优化——不同的 prompt 擅长不同的子任务。

---

## 7. 实际应用场景

### 7.1 提示词优化（最早、最成熟）

```python
result = gepa.optimize(
    seed_candidate={"system_prompt": "You are a helpful assistant."},
    trainset=trainset,
    valset=valset,
    task_lm="openai/gpt-4.1-mini",
    reflection_lm="openai/gpt-5",
    max_metric_calls=150,
)
```

### 7.2 Agent 系统优化

不只是提示词——还可以优化**工具描述、代理架构、工作流配置**：

```python
result = oa.optimize_anything(
    seed_candidate="""Use tool X then tool Y for processing""",
    evaluator=lambda cfg: run_agent(cfg),
    objective="最大化任务完成率"
)
```

### 7.3 DSPy 集成

```python
import dspy

optimizer = dspy.GEPA(
    metric=your_metric,
    max_metric_calls=150,
    reflection_lm="openai/gpt-5"
)
optimized = optimizer.compile(student=MyProgram(), trainset=trainset)
```

---

## 8. 局限性

1. **依赖 LLM 的反思质量**：如果 reflection_lm 本身能力不足，ASI 质量也会受限
2. **评估函数设计是关键**：需要正确捕获轨迹并返回有信息量的 ASI
3. **不适用于需要真正梯度更新的场景**：如神经网络权重训练
4. **超参数敏感性**：`max_metric_calls`、批量大小等需要调优

---

## 9. 核心结论

> **GEPA 的本质是：用 LLM 的推理能力来弥补优化过程中的信息损失。**

RL 把执行轨迹压缩成标量，GEPA 把执行轨迹展开成诊断报告。这是它比 RL 快 35 倍、比人工调优高效的**根本原因**。

ASI（可操作反馈）= GEPA 与其他所有优化方法的**关键差异点**。

---

## 参考资料

- 论文: https://arxiv.org/abs/2507.19457
- 官方仓库: https://github.com/gepa-ai/gepa
- 官方文档: https://gepa-ai.github.io/gepa/
- DSPy 集成: https://dspy.ai/
- 集成 MLflow: https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/
