#!/usr/bin/env python3
"""
GEPA 实际 API 实验

这个脚本使用真实 API 调用实现一个简化版的 GEPA 流程。
需要设置环境变量:
  OPENAI_API_KEY  - 用于 task_lm (优化对象模型)
  ANTHROPIC_API_KEY - 用于 reflection_lm (反思诊断模型，可选，默认用 OpenAI)

演示流程:
1. 用 task_lm 在数学题上生成答案
2. 用 reflection_lm 分析错误轨迹，生成 ASI
3. 生成改进的提示词
4. 评估改进效果
"""

import os
import json
import time
import random
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

# =============================================================================
# 配置
# =============================================================================

# 设置你的 API keys
TASK_LM = os.environ.get("OPENAI_API_KEY", os.environ.get("OPENAI_BASE_URL", ""))
REFLECTION_LM = os.environ.get("ANTHROPIC_API_KEY", "")

# 如果没有 key，使用模拟模式
USE_MOCK = not TASK_LM

# =============================================================================
# 数据集
# =============================================================================

MATH_DATASET = [
    {"id": 1, "question": "小明有12个苹果，给了小红5个，又买了8个，小明现在有多少个苹果？", "answer": "15"},
    {"id": 2, "question": "计算: 25 + 17 - 9 = ?", "answer": "33"},
    {"id": 3, "question": "一个篮子里有45个橙子，卖出去了28个，又放进了15个，现在有多少个？", "answer": "32"},
    {"id": 4, "question": "计算: 100 - 37 + 25 = ?", "answer": "88"},
    {"id": 5, "question": "图书馆有156本书，借出去了43本，又还回来了29本，图书馆现在有多少本？", "answer": "142"},
    {"id": 6, "question": "小明有200元，买书花了78元，又得到了50元找零，小明现在有多少元？", "answer": "172"},
    {"id": 7, "question": "计算: 45 + 67 - 32 = ?", "answer": "80"},
    {"id": 8, "question": "一辆车每小时行驶65公里，行驶了4小时，总共行驶了多少公里？", "answer": "260"},
]

TRAIN_SIZE = 4
VAL_SIZE = 4
TRAINSET = MATH_DATASET[:TRAIN_SIZE]
VALSET = MATH_DATASET[TRAIN_SIZE:]

# =============================================================================
# 模拟 API 调用（当没有真实 API key 时）
# =============================================================================

def mock_openai_call(prompt: str, system: str = "", model: str = "gpt-4") -> str:
    """模拟 OpenAI API 调用"""
    print(f"    [MOCK API] 调用 model={model}")
    time.sleep(0.1)  # 模拟延迟

    # 模拟模型回答
    if "负数" in system or "负号" in system or "减法" in system:
        # 模拟一个理解负数的模型
        if "25 + 17 - 9" in prompt:
            return "25 + 17 - 9 = 33"
        elif "100 - 37 + 25" in prompt:
            return "100 - 37 + 25 = 88"
        elif "45 + 67 - 32" in prompt:
            return "45 + 67 - 32 = 80"
        else:
            return "33"  # 默认正确

    # 模拟一个粗心的模型
    if "苹果" in prompt:
        return "15"
    elif "橙子" in prompt:
        return "32"  # 正确
    elif "100 - 37 + 25" in prompt:
        return "62"  # 错误：100-37=63, 63+25=88
    elif "45 + 67 - 32" in prompt:
        return "80"  # 正确
    elif "156 - 43 + 29" in prompt:
        return "132"  # 错误
    elif "200 - 78 + 50" in prompt:
        return "122"  # 错误
    elif "65 * 4" in prompt:
        return "260"  # 正确
    elif "25 + 17 - 9" in prompt:
        return "33"  # 正确
    else:
        return "0"


def mock_anthropic_call(prompt: str) -> str:
    """模拟 Anthropic API 调用用于反思"""
    print(f"    [MOCK API] 调用 reflection model")
    time.sleep(0.1)

    # 模拟反思输出
    if "156 - 43 + 29" in prompt:
        return """你应该新增一条解题规则：

【新增规则】
- 验算步骤：完成计算后，用逆运算验证答案是否正确
  例如：142 - 29 = 113, 113 + 43 = 156 ✓

【规则说明】
这条规则能帮助你检查多步加减法运算的错误。"""

    if "200 - 78 + 50" in prompt:
        return """你应该新增一条解题规则：

【新增规则】
- 细心计算：做减法时注意被减数和减数的顺序，不要混淆
  正确：被减数 - 减数 = 差
  错误示例：把减法做成加法

【规则说明】
当出现连续加减法运算时，要严格按照从左到右的顺序计算。"""

    if "65 * 4" in prompt:
        return """【规则已足够】

你的解题规则已经能够正确处理这类乘法问题，无需修改。

输出格式保持清晰即可。"""

    return """【反思】

分析发现错误可能来源于粗心大意。建议：
1. 增加验算步骤
2. 明确运算顺序
3. 强调仔细检查每一步"""


# =============================================================================
# 真实 API 调用
# =============================================================================

def real_openai_call(prompt: str, system: str = "", model: str = "gpt-4o-mini") -> str:
    """真实的 OpenAI API 调用"""
    import openai
    client = openai.OpenAI(api_key=TASK_LM)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()


def real_anthropic_call(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """真实的 Anthropic API 调用用于反思"""
    import anthropic
    client = anthropic.Anthropic(api_key=REFLECTION_LM)

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


# =============================================================================
# 核心函数
# =============================================================================

def call_task_lm(system_prompt: str, question: str) -> tuple[str, float]:
    """
    调用 task_lm 回答问题，返回 (答案, 是否正确)
    """
    if USE_MOCK:
        answer = mock_openai_call(question, system_prompt)
    else:
        try:
            answer = real_openai_call(question, system_prompt)
        except Exception as e:
            print(f"    [ERROR] OpenAI API 调用失败: {e}")
            answer = "ERROR"

    # 提取答案（简单匹配数字）
    import re
    answer_digits = re.findall(r'-?\d+', str(answer))
    if answer_digits:
        extracted = answer_digits[-1]  # 取最后一个数字
    else:
        extracted = "0"

    return answer, extracted


def call_reflection_lm(
    system_prompt: str,
    results: list[dict]
) -> str:
    """
    调用 reflection_lm 分析错误，生成 ASI 和改进建议
    """
    # 构建反思 prompt
    reflection_prompt = f"""你是一个提示词优化专家。请分析以下数学解题提示词的表现：

【当前提示词】
{system_prompt}

【评估结果】
"""

    for r in results:
        status = "✓ 正确" if r["correct"] else "✗ 错误"
        reflection_prompt += f"""
题{r['id']}: {r['question']}
正确答案: {r['expected_answer']}
模型回答: {r['model_answer']}
状态: {status}
"""

    reflection_prompt += """
请分析：
1. 模型在哪类题目上犯错？
2. 犯错的原因是什么？（粗心？理解错误？规则缺失？）
3. 如何改进提示词来减少这类错误？

请以【新增规则】格式给出具体的改进建议。"""

    if USE_MOCK:
        return mock_anthropic_call(reflection_prompt)
    else:
        try:
            return real_anthropic_call(reflection_prompt)
        except Exception as e:
            print(f"    [ERROR] Anthropic API 调用失败: {e}")
            return "反思失败"


def evaluate_candidate(system_prompt: str, dataset: list[dict]) -> tuple[float, list[dict]]:
    """评估一个提示词候选在数据集上的表现"""
    results = []
    correct = 0

    for item in dataset:
        _, extracted = call_task_lm(system_prompt, item["question"])

        # 判断是否正确（允许格式差异）
        is_correct = False
        expected_clean = "".join(c for c in item["answer"] if c.isdigit() or c == '-')
        extracted_clean = "".join(c for c in extracted if c.isdigit() or c == '-')

        if expected_clean == extracted_clean:
            is_correct = True
            correct += 1

        results.append({
            "id": item["id"],
            "question": item["question"],
            "expected_answer": item["answer"],
            "model_answer": extracted,
            "correct": is_correct,
        })

    return correct / len(dataset), results


def generate_improved_prompt(
    original: str,
    reflection: str
) -> str:
    """
    基于反思生成改进的提示词
    """
    # 简单实现：追加反思建议
    lines = original.strip().split("\n")

    # 检查是否已有规则列表
    has_rules = any("规则" in line for line in lines)

    improvement = f"""
【反思改进建议】
{reflection}
"""

    if has_rules:
        return original + improvement
    else:
        return original + "\n\n" + improvement


# =============================================================================
# 主实验
# =============================================================================

def run_experiment():
    print("=" * 70)
    print("GEPA 实际 API 实验")
    print("=" * 70)
    print(f"模式: {'模拟 API' if USE_MOCK else '真实 API'}")
    print(f"数据集: {len(MATH_DATASET)} 题 (训练: {TRAIN_SIZE}, 验证: {VAL_SIZE})")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 种子提示词
    seed_prompt = """你是一个数学解题助手。解题并给出答案。"""

    candidates = [seed_prompt]
    pareto_front = [0]  # 帕累托前沿索引
    max_iterations = 3
    max_metric_calls = 20

    print(f"\n【初始提示词】")
    print(f"  {seed_prompt}")

    # 评估基线
    print(f"\n--- 基线评估 ---")
    baseline_score, baseline_results = evaluate_candidate(seed_prompt, VALSET)
    print(f"  验证集准确率: {baseline_score:.2%}")
    for r in baseline_results:
        status = "✓" if r["correct"] else "✗"
        print(f"    {status} 题{r['id']}: {r['question'][:30]}... → 期望:{r['expected_answer']} 得到:{r['model_answer']}")

    total_calls = len(TRAINSET)  # 基线评估

    # GEPA 循环
    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*70}")
        print(f"迭代 {iteration}/{max_iterations}")
        print(f"{'='*70}")

        # Step 1: SELECT - 选择父候选
        parent_idx = pareto_front[0]  # 简化：选第一个
        parent_prompt = candidates[parent_idx]
        print(f"\n[SELECT] 选择父候选 #{parent_idx}")

        # Step 2: EXECUTE - 在训练集上评估
        print(f"[EXECUTE] 在训练集上评估...")
        train_score, train_results = evaluate_candidate(parent_prompt, TRAINSET)
        total_calls += len(TRAINSET)
        print(f"  训练集准确率: {train_score:.2%}")

        failed = [r for r in train_results if not r["correct"]]
        if failed:
            print(f"  失败题目: {len(failed)} 道")
            for f in failed:
                print(f"    题{f['id']}: {f['question'][:40]}")

        # Step 3: REFLECT - 生成 ASI
        print(f"\n[REFLECT] 调用反思模型...")
        reflection = call_reflection_lm(parent_prompt, train_results)
        total_calls += 1  # 反思调用
        print(f"  反思结果:")
        for line in reflection.split("\n")[:5]:
            print(f"    {line}")

        # Step 4: MUTATE - 生成改进
        print(f"\n[MUTATE] 生成改进提示词...")
        improved_prompt = generate_improved_prompt(parent_prompt, reflection)
        print(f"  新提示词预览: {improved_prompt[:100]}...")

        # Step 5: ACCEPT - 在训练集上验证
        print(f"\n[ACCEPT] 验证改进...")
        new_score, new_results = evaluate_candidate(improved_prompt, TRAINSET)
        total_calls += len(TRAINSET)

        print(f"  新候选训练集准确率: {new_score:.2%} (原: {train_score:.2%})")

        if new_score > train_score:
            print(f"  → 接受！加入候选池")
            candidates.append(improved_prompt)
            new_idx = len(candidates) - 1
            pareto_front.append(new_idx)

            # 在验证集上评估
            val_score, val_results = evaluate_candidate(improved_prompt, VALSET)
            print(f"  验证集准确率: {val_score:.2%}")
        else:
            print(f"  → 拒绝（未改进）")

        print(f"\n[进度] 已使用 {total_calls}/{max_metric_calls} 次评估")

        if total_calls >= max_metric_calls:
            print("达到评估上限，停止")
            break

    # 最终结果
    print(f"\n{'='*70}")
    print("实验完成！")
    print(f"{'='*70}")

    print(f"\n【最终帕累托前沿】")
    for idx in pareto_front:
        score, _ = evaluate_candidate(candidates[idx], VALSET)
        print(f"  候选 #{idx}: 验证准确率 = {score:.2%}")
        print(f"    提示词: {candidates[idx][:80]}...")

    print(f"\n【总评估次数】{total_calls}")
    print(f"【基线对比】{baseline_score:.2%} → 最终: {max([evaluate_candidate(c, VALSET)[0] for c in candidates])}")

    return candidates, pareto_front


if __name__ == "__main__":
    if USE_MOCK:
        print("[提示] 未设置 API key，使用模拟模式")
        print("[提示] 设置 OPENAI_API_KEY 和 ANTHROPIC_API_KEY 可使用真实 API")
        print()

    candidates, pareto_front = run_experiment()

    # 保存结果
    results_file = "experiment_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "mode": "mock" if USE_MOCK else "real",
            "candidates_count": len(candidates),
            "pareto_front": pareto_front,
            "candidates": [
                {"id": i, "prompt": c} for i, c in enumerate(candidates)
            ]
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {results_file}")
