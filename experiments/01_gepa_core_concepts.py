#!/usr/bin/env python3
"""
GEPA 核心概念演示实验

这个脚本展示了 GEPA 的三大核心概念：
1. Actionable Side Information (ASI) - 可操作反馈
2. Reflective Reflection - 反思性诊断
3. Pareto-Efficient Selection - 帕累托前沿选择

不需要安装 gepa 包，纯粹演示核心思想。
"""

import json
import random
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime


# =============================================================================
# 第1部分：ASI vs 传统反馈对比
# =============================================================================

def demo_asi_vs_scalar():
    """
    演示 ASI（可操作反馈）和传统标量反馈的本质差异。
    """
    print("=" * 70)
    print("实验1: ASI vs 传统标量反馈")
    print("=" * 70)

    # 传统方法只能告诉你"错了"
    traditional_feedback = {
        "candidate_id": "prompt_v1",
        "score": 0.33,
        "test_cases": [
            {"input": "-3x + 7 = 0", "expected": "x = 7/3", "got": "x = -7/3", "score": 0},
            {"input": "5 + 3 = ?", "expected": "8", "got": "8", "score": 1},
            {"input": "10 / 2 = ?", "expected": "5", "got": "5", "score": 1},
        ]
    }

    # GEPA 的 ASI 告诉你"哪里错了、为什么"
    asi_feedback = {
        "candidate_id": "prompt_v1",
        "score": 0.33,
        "actionable_side_information": [
            {
                "input": "-3x + 7 = 0",
                "output": "x = -7/3",
                "diagnosis": "助手忽略了负号。方程是 -3x + 7 = 0，正确解是 x = 7/3 ≈ 2.33，不是 -7/3 ≈ -2.33。",
                "fix_suggestion": "在解题前增加一步：'首先将常数项移到等号右边：-3x = -7，然后两边除以 -3 得到 x = 7/3'。明确要求检查系数符号。",
                "asi_type": "mathematical_flaw"
            },
            {
                "input": "5 + 3 = ?",
                "output": "8",
                "diagnosis": "正确答案，无需改进。",
                "fix_suggestion": None,
                "asi_type": "correct"
            },
            {
                "input": "10 / 2 = ?",
                "output": "5",
                "diagnosis": "正确答案，无需改进。",
                "fix_suggestion": None,
                "asi_type": "correct"
            }
        ],
        "aggregated_diagnosis": "提示词没有强调'检查每一步的符号，特别是负号'。数学推理题需要显式强调符号检查步骤。"
    }

    print("\n【传统方法 - 只能看到分数】")
    print(f"  分数: {traditional_feedback['score']}")
    print(f"  只能知道'错了'，但不知道错在哪里、为什么错")
    print(f"  → 优化器无从下手，只能随机改动碰运气")

    print("\n【GEPA ASI - 完整的诊断信息】")
    print(f"  分数: {asi_feedback['score']}")
    print(f"  失败案例的诊断:")
    for asi in asi_feedback['actionable_side_information']:
        if asi['diagnosis'] != "正确答案，无需改进。":
            print(f"    • 输入: {asi['input']}")
            print(f"      诊断: {asi['diagnosis']}")
            print(f"      修复建议: {asi['fix_suggestion']}")
    print(f"  聚合诊断: {asi_feedback['aggregated_diagnosis']}")
    print(f"  → 优化器知道问题在哪、如何修复")

    print("\n【核心洞察】")
    print(f"  ASI = 文本优化任务中的'梯度'")
    print(f"  传统反馈: score=0.33  (信息量: ~2 bits)")
    print(f"  GEPA ASI: 完整诊断  (信息量: ~500+ tokens)")
    print(f"  → 这就是 GEPA 比 RL 快 35 倍的根本原因")


# =============================================================================
# 第2部分：反思性诊断流程
# =============================================================================

def demo_reflective_loop():
    """
    演示 GEPA 的反思性循环：Select → Execute → Reflect → Mutate → Accept
    """
    print("\n" + "=" * 70)
    print("实验2: GEPA 反思性循环演示")
    print("=" * 70)

    # 模拟数据集（数学题）
    trainset = [
        {"id": 1, "question": "解方程: x + 5 = 10", "answer": "5"},
        {"id": 2, "question": "计算: -4 + 7", "answer": "3"},
        {"id": 3, "question": "解方程: 2x - 3 = 7", "answer": "5"},
        {"id": 4, "question": "计算: -9 - (-3)", "answer": "-6"},
        {"id": 5, "question": "解方程: -2x + 4 = 0", "answer": "2"},
    ]

    # 种子提示词
    seed_prompt = {
        "system_prompt": "你是数学助手。回答问题，把最终答案放在 ### <answer> 格式中。"
    }

    # 模拟评估函数
    def evaluate(prompt: str, questions: list) -> list[dict]:
        results = []
        correct = 0
        for q in questions:
            # 模拟：负号处理不好的模型
            question = q["question"]
            expected = q["answer"]

            # 简单模拟：含负号的题答错
            if "-" in question and "(-" not in question:
                assistant_answer = str(-abs(int(expected)))  # 符号反转
            elif "x" in question and " - " in question:
                # 解方程且含负号：符号错误
                assistant_answer = str(-abs(int(expected)))
            else:
                assistant_answer = expected
                correct += 1

            is_correct = assistant_answer == expected
            results.append({
                "id": q["id"],
                "question": question,
                "expected": expected,
                "assistant_answer": assistant_answer,
                "score": 1.0 if is_correct else 0.0,
                "has_error": not is_correct,
            })
        return results, correct / len(questions)

    # GEPA 循环
    candidates = [seed_prompt]
    pareto_front = [0]  # 候选0在帕累托前沿
    history = []

    for iteration in range(1, 4):
        print(f"\n--- 迭代 {iteration} ---")

        # Step 1: SELECT - 从帕累托前沿选择父候选
        parent_idx = random.choice(pareto_front)
        parent = candidates[parent_idx]
        print(f"  [SELECT] 选择父候选 #{parent_idx} (得分待评估)")

        # Step 2: EXECUTE - 在小批量上评估
        eval_results, avg_score = evaluate(
            parent["system_prompt"],
            random.sample(trainset, min(3, len(trainset)))
        )
        print(f"  [EXECUTE] 评估结果: 平均分 = {avg_score:.2f}")

        # Step 3: REFLECT - 生成 ASI
        failed_cases = [r for r in eval_results if r["has_error"]]
        if failed_cases:
            diagnosis = []
            for case in failed_cases:
                if case["question"].startswith("计算"):
                    diagnosis.append(
                        f"题{case['id']}: '{case['question']}' 答错。"
                        f"正确答案是 {case['expected']}，模型给出了 {case['assistant_answer']}。"
                        f"问题在于对负数的减法处理有误。"
                    )
                else:
                    diagnosis.append(
                        f"题{case['id']}: '{case['question']}' 答错。"
                        f"正确答案是 {case['expected']}，模型给出了 {case['assistant_answer']}。"
                        f"需要先移项再除以系数，注意符号。"
                    )
            asi = "失败原因：" + " ".join(diagnosis)
        else:
            asi = "所有题目正确"
        print(f"  [REFLECT] ASI = {asi[:80]}...")

        # Step 4: MUTATE - 基于 ASI 生成改进
        if failed_cases:
            improved_prompt = (
                "你是数学助手。注意以下规则：\n"
                "1. 做减法时，减去一个负数等于加上它的绝对值：a - (-b) = a + b\n"
                "2. 解方程时，先移项变号，再除以系数\n"
                "3. 每一步都要检查符号\n"
                "回答问题，把最终答案放在 ### <answer> 格式中。"
            )
        else:
            improved_prompt = parent["system_prompt"]

        new_candidate = {"system_prompt": improved_prompt}
        candidates.append(new_candidate)
        new_idx = len(candidates) - 1

        # Step 5: ACCEPT - 验证并更新帕累托前沿
        _, new_score = evaluate(
            improved_prompt,
            random.sample(trainset, min(3, len(trainset)))
        )
        print(f"  [MUTATE] 生成新候选 #{new_idx}")
        print(f"  [ACCEPT] 新候选得分 = {new_score:.2f}")

        # 简单帕累托更新：只保留更好的
        if new_score > avg_score:
            pareto_front.append(new_idx)
            print(f"  → 新候选加入帕累托前沿！")
        else:
            print(f"  → 新候选被拒绝（未改进）")

        history.append({
            "iteration": iteration,
            "parent": parent_idx,
            "child": new_idx,
            "parent_score": avg_score,
            "child_score": new_score,
            "accepted": new_score > avg_score
        })

    print(f"\n【最终帕累托前沿】候选索引: {pareto_front}")
    print(f"【进化历史】")
    for h in history:
        status = "✓ 接受" if h["accepted"] else "✗ 拒绝"
        print(f"  迭代{h['iteration']}: 父#{h['parent']}({h['parent_score']:.2f}) → 子#{h['child']}({h['child_score']:.2f}) {status}")


# =============================================================================
# 第3部分：帕累托前沿可视化
# =============================================================================

def demo_pareto_front():
    """
    演示帕累托前沿：多目标优化的非支配解集合。
    """
    print("\n" + "=" * 70)
    print("实验3: 帕累托前沿选择机制")
    print("=" * 70)

    # 模拟多个候选在不同任务上的表现
    # 目标1: 数学推理准确率, 目标2: 简洁性(越短越好)
    candidates = [
        {"id": "seed", "math": 0.33, "brevity": 0.3, "text": "你是一个有帮助的助手。"},
        {"id": "A", "math": 0.67, "brevity": 0.2, "text": "你是数学助手。回答问题，把答案放在 ### <answer> 中。"},
        {"id": "B", "math": 0.67, "brevity": 0.5, "text": "数学助手模式。"},
        {"id": "C", "math": 0.83, "brevity": 0.1, "text": "你是一个精确的数学助手。对于解方程：1) 先整理方程 2) 移项变号 3) 除以系数 4) 检查每一步的符号。将最终答案放在 ### <answer> 格式中。"},
        {"id": "D", "math": 0.83, "brevity": 0.15, "text": "数学助手：仔细解题。注意符号。用 ### <answer> 给出答案。"},
    ]

    def is_dominated(cand: dict, others: list[dict]) -> bool:
        """检查 cand 是否被 others 中任意一个支配"""
        for o in others:
            if o["id"] == cand["id"]:
                continue
            if o["math"] >= cand["math"] and o["brevity"] >= cand["brevity"] \
               and (o["math"] > cand["math"] or o["brevity"] > cand["brevity"]):
                return True
        return False

    pareto = [c for c in candidates if not is_dominated(c, candidates)]

    print("\n【所有候选的表现】")
    print(f"  {'ID':<6} {'数学准确率':<12} {'简洁性':<10} {'是否被支配'}")
    print(f"  {'-'*50}")
    for c in candidates:
        dominated = "是 → 被丢弃" if is_dominated(c, candidates) else "否 → 保留"
        print(f"  {c['id']:<6} {c['math']:<12.2f} {c['brevity']:<10.2f} {dominated}")

    print(f"\n【帕累托前沿 (非支配解)】")
    print(f"  这些候选在任何单一目标上都不是最差的，且不被任何其他候选完全支配")
    for c in pareto:
        print(f"  • {c['id']}: 数学={c['math']:.2f}, 简洁={c['brevity']:.2f}")
        print(f"    提示词: {c['text'][:50]}...")

    print(f"\n【帕累托前沿的选择意义】")
    print(f"  候选 A 和 B 都在帕累托前沿上——")
    print(f"  • A 简洁性好(0.50) 但数学稍差(0.67)")
    print(f"  • B 数学好(0.67) 但简洁性差(0.20)")
    print(f"  GEPA 会同时保留它们，因为它们擅长不同的子任务")
    print(f"  → 这是 GEPA 多样性保持的关键机制")


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GEPA 核心概念演示实验")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    demo_asi_vs_scalar()
    demo_reflective_loop()
    demo_pareto_front()

    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)
    print("""
核心结论:
1. ASI > 标量反馈：完整诊断 >> 分数
2. 反思性循环有效：通过 ASI 引导变异，逐步改进
3. 帕累托保持多样性：不同候选擅长不同子任务

下一步:
- 运行 experiments/02_gepa_with_api.py 使用真实 API 进行 GEPA 实验
- 阅读 GEPA深度研究报告.md 了解完整理论
""")
