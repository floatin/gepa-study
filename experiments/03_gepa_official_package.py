#!/usr/bin/env python3
"""
使用官方 GEPA 包进行完整提示词优化实验

前置条件:
  pip install gepa openai anthropic

环境变量:
  OPENAI_API_KEY      - 用于 task_lm
  ANTHROPIC_API_KEY   - 用于 reflection_lm

这个脚本演示如何使用 gepa 包的标准 API 进行提示词优化。
"""

import os
import sys
import json
from datetime import datetime

# 检查依赖
try:
    import gepa
    from gepa import GEPAConfig, EngineConfig
except ImportError:
    print("[错误] 需要安装 gepa: pip install gepa")
    sys.exit(1)

try:
    import openai
    openai_client = openai.OpenAI()
except ImportError:
    print("[错误] 需要安装 openai: pip install openai")
    sys.exit(1)

# =============================================================================
# 配置
# =============================================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

if not OPENAI_API_KEY:
    print("[错误] 请设置 OPENAI_API_KEY 环境变量")
    sys.exit(1)

# =============================================================================
# 数据集定义
# =============================================================================

# 使用 AIME 风格的数学题数据集（简化版）
MATH_TRAINSET = [
    {
        "id": "train_1",
        "question": "求 x 的值: 2x + 5 = 13",
        "expected_answer": "4",
    },
    {
        "id": "train_2",
        "question": "计算: 15 - 8 + 3 = ?",
        "expected_answer": "10",
    },
    {
        "id": "train_3",
        "question": "小明有 24 元，买了 3 支铅笔，每支 3 元，还剩多少元？",
        "expected_answer": "15",
    },
    {
        "id": "train_4",
        "question": "求 y 的值: 3y - 6 = 12",
        "expected_answer": "6",
    },
]

MATH_VALSET = [
    {
        "id": "val_1",
        "question": "计算: 100 - 37 + 25 = ?",
        "expected_answer": "88",
    },
    {
        "id": "val_2",
        "question": "求 z 的值: 4z + 8 = 24",
        "expected_answer": "4",
    },
    {
        "id": "val_3",
        "question": "一个长方形的长是 12 厘米，宽是 5 厘米，面积是多少平方厘米？",
        "expected_answer": "60",
    },
    {
        "id": "val_4",
        "question": "计算: 45 + 67 - 32 = ?",
        "expected_answer": "80",
    },
]

# =============================================================================
# 自定义 Adapter
# =============================================================================

from dataclasses import dataclass
from typing import Any
import re


@dataclass
class MathRolloutOutput:
    """数学任务的输出"""
    output: str
    score: float
    trajectory: dict | None = None


class MathGEPAAdapter:
    """
    数学提示词优化的 GEPA Adapter

    实现两个核心方法:
    - evaluate(): 评估候选提示词，返回分数和轨迹
    - make_reflective_dataset(): 从轨迹中提取 ASI
    """

    def __init__(self, task_lm_model: str = "gpt-4o-mini"):
        self.task_lm_model = task_lm_model
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def evaluate(
        self,
        batch: list[dict],
        program: dict[str, str],
        capture_traces: bool = False
    ) -> MathRolloutOutput:
        """
        评估函数: 用候选提示词运行任务，返回分数和轨迹
        """
        system_prompt = program.get("system_prompt", "")

        outputs = []
        scores = []
        trajectories = []

        for item in batch:
            question = item["question"]
            expected = item["expected_answer"]

            try:
                response = self.client.chat.completions.create(
                    model=self.task_lm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                answer = response.choices[0].message.content.strip()

            except Exception as e:
                answer = f"ERROR: {e}"

            # 提取答案
            extracted = self._extract_answer(answer)
            expected_clean = self._extract_answer(expected)

            is_correct = (extracted == expected_clean)
            score = 1.0 if is_correct else 0.0

            outputs.append(answer)
            scores.append(score)

            if capture_traces:
                trajectories.append({
                    "input": question,
                    "expected": expected,
                    "extracted": extracted,
                    "model_output": answer,
                    "is_correct": is_correct,
                    "error_type": self._classify_error(answer, expected) if not is_correct else None,
                })

        # 聚合输出
        avg_score = sum(scores) / len(scores) if scores else 0.0
        output = {
            "avg_score": avg_score,
            "num_correct": sum(1 for s in scores if s == 1.0),
            "num_total": len(scores)
        }

        return MathRolloutOutput(
            output=json.dumps(output),
            score=avg_score,
            trajectory={"steps": trajectories} if trajectories else None
        )

    def _extract_answer(self, text: str) -> str:
        """从回答中提取最终答案"""
        # 尝试多种格式
        patterns = [
            r'###\s*<answer>\s*([^\n#]+)',  # ### <answer> format
            r'答案[：:]\s*([^\n]+)',         # 答案: format
            r'=\s*([\d-]+)',                 # = format
            r'([\d-]+)\s*$'                  # trailing number
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        # 最后尝试：找所有数字
        digits = re.findall(r'-?\d+', text)
        return digits[-1] if digits else text.strip()

    def _classify_error(self, answer: str, expected: str) -> str:
        """分类错误类型"""
        a_num = re.findall(r'-?\d+', answer)
        e_num = re.findall(r'-?\d+', expected)

        if not a_num or not e_num:
            return "format_error"

        a_val = int(a_num[-1])
        e_val = int(e_num[-1])

        if a_val == -e_val:
            return "sign_error"
        elif abs(a_val - e_val) == 1:
            return "off_by_one"
        elif a_val < 0 and e_val > 0:
            return "negative_sign_error"
        else:
            return "calculation_error"

    def make_reflective_dataset(
        self,
        program: dict[str, str],
        eval_result: MathRolloutOutput,
        predictor_names: list[str]
    ) -> dict[str, list[dict]]:
        """
        构建反思数据集：从轨迹中提取 ASI

        ASI 格式:
        {
          "input": 问题,
          "output": 模型回答,
          "feedback": 诊断反馈,
          "error_type": 错误类型(如果有)
        }
        """
        if not eval_result.trajectory:
            return {name: [] for name in predictor_names}

        steps = eval_result.trajectory.get("steps", [])

        reflective_data = {}
        for name in predictor_names:
            dataset = []
            for step in steps:
                entry = {
                    "input": step["input"],
                    "output": step["model_output"],
                }

                if step["is_correct"]:
                    entry["feedback"] = "回答正确，无需改进。"
                else:
                    error_type = step.get("error_type", "unknown")
                    entry["feedback"] = self._generate_feedback(step, error_type)
                    entry["error_type"] = error_type

                dataset.append(entry)

            reflective_data[name] = dataset

        return reflective_data

    def _generate_feedback(self, step: dict, error_type: str) -> str:
        """生成诊断反馈"""
        expected = step["expected"]
        got = step.get("extracted", "?")

        feedback_map = {
            "sign_error": f"答案符号错误。期望 {expected}，得到 {got}。可能是在减法运算中符号处理有误。",
            "off_by_one": f"答案偏差1。期望 {expected}，得到 {got}。可能是简单的计算失误。",
            "negative_sign_error": f"负号处理错误。期望 {expected}，得到 {got}。需要强调负数的加减规则。",
            "calculation_error": f"计算错误。期望 {expected}，得到 {got}。需要更仔细的验算步骤。",
            "format_error": f"输出格式不正确。期望 {expected}，得到 {got}。需要按指定格式输出答案。",
        }

        return feedback_map.get(
            error_type,
            f"答案错误。期望 {expected}，得到 {got}。"
        )


# =============================================================================
# 运行 GEPA 优化
# =============================================================================

def run_gepa_optimization():
    print("=" * 70)
    print("GEPA 官方包实验")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Task LM: gpt-4o-mini")
    print(f"Reflection LM: claude-sonnet-4-20250514 (如果可用)")

    # 种子提示词
    seed_candidate = {
        "system_prompt": "你是一个数学解题助手。解题并给出答案。"
    }

    print(f"\n【种子提示词】")
    print(f"  {seed_candidate['system_prompt']}")

    # 初始化 adapter
    adapter = MathGEPAAdapter(task_lm_model="gpt-4o-mini")

    # 配置
    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=30,  # 限制评估次数
            perfect_score=1.0,
        ),
        trainset=MATH_TRAINSET,
        valset=MATH_VALSET,
        seed_candidate=seed_candidate,
        reflection_lm="anthropic/claude-sonnet-4-20250514" if ANTHROPIC_API_KEY else None,
        task_lm="openai/gpt-4o-mini",
        run_dir="./gepa_run_output",
    )

    print(f"\n【配置】")
    print(f"  max_metric_calls: {config.engine.max_metric_calls}")
    print(f"  训练集大小: {len(MATH_TRAINSET)}")
    print(f"  验证集大小: {len(MATH_VALSET)}")

    # 运行优化
    print(f"\n开始 GEPA 优化...")
    result = gepa.optimize(config)

    print(f"\n【优化结果】")
    print(f"  最佳候选索引: {result.best_candidate_idx}")
    print(f"  最佳验证集分数: {result.best_score:.4f}")
    print(f"  总评估次数: {result.total_metric_calls}")

    if result.best_candidate:
        print(f"\n【最佳提示词】")
        print(f"  {result.best_candidate.get('system_prompt', 'N/A')}")

    # 保存结果
    output_file = "gepa_optimization_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "best_candidate_idx": result.best_candidate_idx,
            "best_score": result.best_score,
            "total_metric_calls": result.total_metric_calls,
            "best_candidate": result.best_candidate,
            "all_candidates": result.all_candidates if hasattr(result, 'all_candidates') else [],
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_file}")

    return result


if __name__ == "__main__":
    result = run_gepa_optimization()
