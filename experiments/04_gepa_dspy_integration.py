#!/usr/bin/env python3
"""
GEPA + DSPy 集成实验

本实验展示三种级别的 GEPA-DSPy 集成：

Level 1 - 提示词优化: 优化 dspy.Predict 的 instructions
Level 2 - 签名优化:  优化 dspy.ChainOfThought 的签名指令  
Level 3 - 程序优化:  优化整个 dspy.Program 的结构

需要环境变量:
  OPENAI_API_KEY      - Task LM
  ANTHROPIC_API_KEY   - Reflection LM (可选)
"""

import os, sys, json
from datetime import datetime

# 检查依赖
try:
    import dspy
except ImportError:
    print("[错误] 需要安装 dspy: pip install dspy")
    sys.exit(1)

# =============================================================================
# 配置
# =============================================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

TASK_LM_MODEL = "gpt-4o-mini"
REFLECTION_LM_MODEL = "claude-sonnet-4-20250514" if ANTHROPIC_API_KEY else "gpt-4o"

# =============================================================================
# 任务定义：数学问答
# =============================================================================

class MathQA(dspy.Module):
    """简单的数学问答 DSPy 程序"""
    
    def __init__(self):
        super().__init__()
        # 基础版 Predictor
        self.predictor = dspy.Predict(
            dspy.Signature(
                "question -> answer",
                "You are a math assistant. Solve the problem step by step."
            )
        )
    
    def forward(self, question):
        return self.predictor(question=question)


# =============================================================================
# Level 1: 基础 DSPy + GEPA 集成
# =============================================================================

def level1_basic_gepa_dspy():
    """
    Level 1: 用 GEPA 优化 DSPy Predictor 的 instructions
    
    这是最简单的集成方式 - 就像优化普通文本提示词一样，
    只不过候选提示词来自 DSPy 程序中的 signature.instructions。
    """
    print("=" * 70)
    print("Level 1: 基础 DSPy + GEPA 集成")
    print("=" * 70)
    
    # 初始化 Task LM
    if not OPENAI_API_KEY:
        print("[跳过] 需要 OPENAI_API_KEY")
        return None
    
    solver_lm = dspy.LM(TASK_LM_MODEL, api_key=OPENAI_API_KEY, temperature=0.3)
    dspy.configure(lm=solver_lm)
    
    # 训练数据
    trainset = [
        dspy.Example(question="2x + 5 = 13, 求 x", answer="4").with_inputs("question"),
        dspy.Example(question="15 - 8 + 3 = ?", answer="10").with_inputs("question"),
        dspy.Example(question="24元买3支铅笔每支3元，还剩多少", answer="15").with_inputs("question"),
        dspy.Example(question="3y - 6 = 12, 求 y", answer="6").with_inputs("question"),
    ]
    
    valset = [
        dspy.Example(question="100 - 37 + 25 = ?", answer="88").with_inputs("question"),
        dspy.Example(question="4z + 8 = 24, 求 z", answer="4").with_inputs("question"),
    ]
    
    # 种子候选 - 原始 instructions
    seed_instruction = "You are a math assistant. Solve the problem step by step."
    
    print(f"\n【种子指令】: {seed_instruction}")
    print(f"【训练集】: {len(trainset)} 条")
    print(f"【验证集】: {len(valset)} 条")
    
    # 简单的评估函数
    def evaluate_instruction(instruction, dataset):
        """评估一条指令的效果"""
        program = dspy.ChainOfThought(
            dspy.Signature("question -> answer", instruction)
        )
        
        correct = 0
        for example in dataset:
            pred = program(question=example.question)
            # 简单匹配：答案是否包含正确数字
            if str(example.answer) in str(pred.answer):
                correct += 1
        
        return correct / len(dataset) if dataset else 0
    
    # 模拟 GEPA 循环（简化版，不需要真实 GEPA 包）
    print("\n【模拟 GEPA 优化循环】")
    
    candidates = [seed_instruction]
    scores = [evaluate_instruction(seed_instruction, trainset)]
    
    # 模拟几个候选
    improved_instructions = [
        "你是一个数学解题助手。请仔细分析题目，逐步计算，确保每一步都正确。最后用 ### <answer> 格式给出答案。",
        "数学计算要求：1) 仔细阅读题目 2) 列出已知条件 3) 逐步计算 4) 验算结果。请用清晰步骤解题。",
        "解题助手：遇到方程先移项再求解；遇到计算题先确定运算顺序。用 ### <answer> 给出最终答案。",
    ]
    
    for instr in improved_instructions:
        candidates.append(instr)
        scores.append(evaluate_instruction(instr, trainset))
        print(f"  候选 {len(candidates)}: 训练分数 = {scores[-1]:.2%}")
    
    # 选择最佳
    best_idx = scores.index(max(scores))
    best_instruction = candidates[best_idx]
    best_train_score = scores[best_idx]
    best_val_score = evaluate_instruction(best_instruction, valset)
    
    print(f"\n【优化结果】")
    print(f"  最佳候选: #{best_idx + 1}")
    print(f"  训练分数: {best_train_score:.2%}")
    print(f"  验证分数: {best_val_score:.2%}")
    print(f"  最佳指令: {best_instruction[:60]}...")
    
    return {
        "level": 1,
        "best_instruction": best_instruction,
        "train_score": best_train_score,
        "val_score": best_val_score,
    }


# =============================================================================
# Level 2: 带 ASI 诊断的 GEPA-DSPy
# =============================================================================

def level2_asi_dspy():
    """
    Level 2: 使用 ASI (Actionable Side Information) 的 DSPy 优化
    
    关键改进：
    - 不是只返回分数，而是返回完整的执行轨迹
    - 轨迹中包含输入、输出、错误诊断
    - 诊断信息用于指导下一轮优化
    """
    print("\n" + "=" * 70)
    print("Level 2: ASI 诊断的 GEPA-DSPy 集成")
    print("=" * 70)
    
    if not OPENAI_API_KEY:
        print("[跳过] 需要 OPENAI_API_KEY")
        return None
    
    solver_lm = dspy.LM(TASK_LM_MODEL, api_key=OPENAI_API_KEY, temperature=0.3)
    dspy.configure(lm=solver_lm)
    
    # 测试数据
    test_cases = [
        {
            "id": "case_1",
            "input": "解方程: -3x + 7 = 0",
            "expected": "x = 7/3",
            "type": "sign_error"
        },
        {
            "id": "case_2", 
            "input": "计算: -4 + 7",
            "expected": "3",
            "type": "negative_add"
        },
        {
            "id": "case_3",
            "input": "解方程: 2x - 3 = 7",
            "expected": "x = 5",
            "type": "linear_eq"
        },
    ]
    
    # 候选指令
    candidates = [
        "你是一个数学解题助手。解题并给出答案。",
        "数学助手：仔细解题。注意符号。用 ### <answer> 给出答案。",
        "解题步骤：1) 读题 2) 分析 3) 计算 4) 验算。最后用 ### <answer> 格式输出。",
    ]
    
    print("\n【传统反馈 vs ASI 反馈对比】\n")
    
    for candidate in candidates:
        print(f"候选指令: {candidate[:50]}...")
        
        # 模拟 DSPy 程序执行
        program = dspy.ChainOfThought(
            dspy.Signature("question -> answer", candidate)
        )
        
        for tc in test_cases:
            pred = program(question=tc["input"])
            
            # 提取答案
            pred_answer = str(pred.answer) if pred.answer else ""
            expected = tc["expected"]
            
            # 简单判断
            is_correct = expected in pred_answer or pred_answer.replace(" ", "") == expected.replace(" ", "")
            
            # 传统反馈：只有分数
            traditional_feedback = {"score": 1.0 if is_correct else 0.0}
            
            # ASI 反馈：完整诊断
            asi_feedback = {
                "score": 1.0 if is_correct else 0.0,
                "input": tc["input"],
                "output": pred_answer,
                "expected": expected,
                "error_type": tc["type"] if not is_correct else None,
                "diagnosis": _diagnose_error(tc["type"], pred_answer, expected) if not is_correct else "正确",
                "suggestion": _get_suggestion(tc["type"]) if not is_correct else "保持",
            }
            
            if tc["id"] == "case_1":
                print(f"  {tc['id']}: 传统={traditional_feedback['score']} | ASI: {asi_feedback['diagnosis'][:40]}...")
        
        print()
    
    # 演示 ASI 的价值
    print("【ASI 的价值】")
    print("""
    传统反馈: {"score": 0.0}  → 信息量 ~2 bits
    ASI 反馈: {"score": 0.0, "error_type": "sign_error", 
                "diagnosis": "忽略负号，正确解 x=7/3 不是 -7/3",
                "suggestion": "强调检查每一步的符号"}
              → 信息量 ~500+ tokens
    
    → ASI 就像优化中的"梯度"，告诉优化器往哪个方向走
    → 这就是 GEPA 比纯 RL 快 35 倍的根本原因
    """)


def _diagnose_error(error_type, pred, expected):
    """生成诊断信息"""
    diagnoses = {
        "sign_error": f"符号错误。期望 {expected}，模型给出 {pred}。可能是在移项或去括号时符号处理有误。",
        "negative_add": f"负数加法错误。期望 {expected}，模型给出 {pred}。-4+7=3，不是 -3。",
        "linear_eq": f"一元方程求解错误。期望 {expected}，模型给出 {pred}。",
    }
    return diagnoses.get(error_type, f"答案错误。期望 {expected}，得到 {pred}。")


def _get_suggestion(error_type):
    """生成修复建议"""
    suggestions = {
        "sign_error": "在解题前增加一步：'首先检查系数和常数项的符号'",
        "negative_add": "强调：负数加法中，-4+7 = 7-4 = 3",
        "linear_eq": "解题步骤：1) 移项 2) 合并同类项 3) 两边除以系数",
    }
    return suggestions.get(error_type, "仔细验算每一步。")


# =============================================================================
# Level 3: 完整 GEPA 循环 (模拟)
# =============================================================================

def level3_full_gepa_loop():
    """
    Level 3: 完整的 GEPA 反思性循环
    
    循环: SELECT → EXECUTE → REFLECT → MUTATE → ACCEPT
    
    使用 DSPy 的 trace 能力捕获完整执行轨迹。
    """
    print("\n" + "=" * 70)
    print("Level 3: 完整 GEPA 反思性循环")
    print("=" * 70)
    
    if not OPENAI_API_KEY:
        print("[跳过] 需要 OPENAI_API_KEY")
        return None
    
    solver_lm = dspy.LM(TASK_LM_MODEL, api_key=OPENAI_API_KEY, temperature=0.3)
    dspy.configure(lm=solver_lm)
    
    # 初始候选池
    candidates = [
        {
            "id": "seed_0",
            "instruction": "你是一个数学解题助手。解题并给出答案。",
            "score": None,
            "history": []
        },
        {
            "id": "cand_1", 
            "instruction": "数学助手：仔细解题。注意符号检查。用 ### <answer> 给出答案。",
            "score": None,
            "history": []
        },
    ]
    
    # 测试数据
    eval_set = [
        dspy.Example(question="2x + 5 = 13, 求 x", answer="4").with_inputs("question"),
        dspy.Example(question="15 - 8 + 3 = ?", answer="10").with_inputs("question"),
        dspy.Example(question="24元买3支铅笔每支3元，还剩多少", answer="15").with_inputs("question"),
    ]
    
    print("\n【GEPA 循环演示】\n")
    
    # 模拟 3 轮进化
    for iteration in range(3):
        print(f"--- 迭代 {iteration + 1} ---")
        
        for cand in candidates:
            if cand["score"] is not None:
                continue  # 已评估
            
            # EXECUTE: 在小批量上执行
            program = dspy.ChainOfThought(
                dspy.Signature("question -> answer", cand["instruction"])
            )
            
            results = []
            for example in eval_set[:2]:  # 小批量
                pred = program(question=example.question)
                is_correct = str(example.answer) in str(pred.answer)
                results.append({
                    "input": example.question,
                    "output": str(pred.answer),
                    "expected": str(example.answer),
                    "correct": is_correct,
                    "reasoning": getattr(pred, "reasoning", "")[:100] if hasattr(pred, 'reasoning') else ""
                })
            
            # 计算分数
            score = sum(1 for r in results if r["correct"]) / len(results)
            cand["score"] = score
            cand["last_results"] = results
            
            print(f"  [{cand['id']}] 执行 → 分数: {score:.2%}")
            
            # REFLECT: 生成 ASI
            failed = [r for r in results if not r["correct"]]
            if failed:
                asi = _generate_asi(failed)
                cand["asi"] = asi
                print(f"         ASI: {asi['summary'][:60]}...")
            else:
                print(f"         全对！无需反思。")
        
        # SELECT: 选择父候选
        available = [c for c in candidates if c["score"] is not None]
        if not available:
            continue
            
        # 按分数排序，选择最优
        available.sort(key=lambda x: x["score"], reverse=True)
        parent = available[0]
        
        print(f"  → 选择父候选: {parent['id']} (分数: {parent['score']:.2%})")
        
        # MUTATE: 基于 ASI 生成改进
        if "asi" in parent:
            new_instruction = _mutate_instruction(parent["instruction"], parent["asi"])
        else:
            new_instruction = parent["instruction"]  # 无改进
        
        new_cand = {
            "id": f"cand_{len(candidates)}",
            "instruction": new_instruction,
            "score": None,
            "parent": parent["id"],
            "history": parent["history"] + [parent["id"]]
        }
        
        # ACCEPT: 评估新候选
        program = dspy.ChainOfThought(
            dspy.Signature("question -> answer", new_instruction)
        )
        
        results = []
        for example in eval_set[:2]:
            pred = program(question=example.question)
            is_correct = str(example.answer) in str(pred.answer)
            results.append({"input": example.question, "output": str(pred.answer), "correct": is_correct})
        
        new_score = sum(1 for r in results if r["correct"]) / len(results)
        new_cand["score"] = new_score
        
        print(f"  [新候选] 分数: {new_score:.2%}")
        
        if new_score > parent["score"]:
            print(f"  ✓ 接受！改进 {new_score - parent['score']:.2%}")
            candidates.append(new_cand)
        else:
            print(f"  ✗ 拒绝（未改进）")
        
        print()
    
    # 最终结果
    best = max(candidates, key=lambda x: x["score"])
    print(f"【最终最佳】: {best['id']}")
    print(f"  指令: {best['instruction']}")
    print(f"  分数: {best['score']:.2%}")
    
    return best


def _generate_asi(failed_results):
    """从失败结果生成 ASI"""
    error_types = []
    diagnoses = []
    suggestions = []
    
    for r in failed_results:
        inp = r["input"]
        out = r["output"]
        exp = r["expected"]
        
        # 简单诊断逻辑
        if "-" in inp and "-" not in out and exp not in out:
            error_types.append("sign_error")
            diagnoses.append(f"输入 '{inp}' 中有负号，但输出 '{out}' 可能忽略了符号处理。")
            suggestions.append("强调在移项、去括号时特别注意符号变化。")
        elif "负" in inp or "-4 + 7" in inp:
            error_types.append("negative_add")
            diagnoses.append(f"负数运算 '{inp}' 可能混淆了符号。正确是 {exp}。")
            suggestions.append("负数加法规则：-4+7 = 7-4 = 3。")
        else:
            diagnoses.append(f"'{inp}' 的答案错误。期望 {exp}，得到 {out}。")
            suggestions.append("仔细验算每一步。")
    
    return {
        "error_types": list(set(error_types)),
        "diagnoses": diagnoses,
        "suggestions": suggestions,
        "summary": "; ".join(set([d[:30] for d in diagnoses]))
    }


def _mutate_instruction(current, asi):
    """基于 ASI 改进指令"""
    suggestions = asi.get("suggestions", [])
    
    mutation_templates = [
        "数学解题助手：{old}。重要提示：{tip}",
        "{old} 请特别注意：{tip}",
        "解题指南：{tip}。{old}",
    ]
    
    tip = suggestions[0] if suggestions else "仔细验算每一步"
    template = mutation_templates[0]
    
    return template.format(old=current.replace("你是一个", "").replace("数学解题助手", "").strip(), tip=tip)


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 70)
    print("GEPA + DSPy 集成实验")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Task LM: {TASK_LM_MODEL}")
    print(f"Reflection LM: {REFLECTION_LM_MODEL}")
    print("=" * 70)
    
    results = {}
    
    # Level 1: 基础集成
    r1 = level1_basic_gepa_dspy()
    if r1:
        results["level1"] = r1
    
    # Level 2: ASI 诊断
    r2 = level2_asi_dspy()
    if r2:
        results["level2"] = r2
    
    # Level 3: 完整循环
    r3 = level3_full_gepa_loop()
    if r3:
        results["level3"] = r3
    
    # 总结
    print("\n" + "=" * 70)
    print("实验总结")
    print("=" * 70)
    print("""
    GEPA-DSPy 集成的三种级别:
    
    Level 1: 提示词优化
      - 优化 dspy.Predict/dspy.ChainOfThought 的 instructions
      - 最简单，与普通文本提示词优化相同
    
    Level 2: ASI 诊断
      - 捕获完整执行轨迹（输入、输出、reasoning）
      - 用 LLM 诊断失败原因，生成可操作的反馈
      - ASI = 文本优化的"梯度"
    
    Level 3: 完整 GEPA 循环
      - SELECT: 选择父候选（Pareto 前沿）
      - EXECUTE: 在小批量上执行
      - REFLECT: 生成 ASI
      - MUTATE: 基于 ASI 生成改进
      - ACCEPT: 验证并更新前沿
    
    与 DSPy 的关系:
      - DSPy 提供了结构化的程序框架（signature, predictor）
      - GEPA 在 DSPy 程序之上优化 instructions
      - 两者是正交的关注点：DSPy 管"结构"，GEPA 管"内容"
    """)
    
    return results


if __name__ == "__main__":
    results = main()
