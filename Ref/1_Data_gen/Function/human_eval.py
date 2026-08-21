import os
import json
from typing import List, Dict, Any, Iterable

# HumanEval 数据集路径（按你的说明）
DATASET_PATH = "../data_set/human-eval/human-eval-v2-20210705.jsonl"

# 由主脚本注入（与 CLTS 生成器保持一致的接口）
# 在 Gen_human_eval_cases.py 中会做：
#   human_eval.create_test = test_utils.create_test
#   human_eval.process_prompt = test_utils.process_prompt
create_test = None
process_prompt = None

# 你的固定 HumanEval 聊天模板（保持原样，不自动闭合 assistant 代码块）
humaneval_prompt_template = """<|im_start|>system
You are an intelligent programming assistant to produce Python algorithmic solutions<|im_end|>
<|im_start|>user
Can you complete the following Python function?
```python
{prompt}
```
<|im_end|>
<|im_start|>assistant
```python
"""

def _iter_humaneval_items(dataset_path: str = DATASET_PATH) -> List[Dict[str, Any]]:
    """
    逐行读取 HumanEval JSONL，返回字典列表。
    每条包含：task_id, prompt, entry_point, canonical_solution, test
    """
    items: List[Dict[str, Any]] = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(obj)
    return items

def count_humaneval_items(dataset_path: str = DATASET_PATH) -> int:
    """统计 HumanEval 样本总数（空行不计）。"""
    cnt = 0
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cnt += 1
    return cnt

def generate_humaneval_tests(
    project_name: str,
    template: str,
    indices: Iterable[int],
    max_output_token_length: int = 5000,
    is_gpu_base_model: int = 1
) -> List[Dict[str, Any]]:
    """
    根据指定索引集合生成测试用例列表。
    - prompt：把数据集里的 prompt 套入给定的 template
    - expect：空字符串
    - testCaseName：{project_name}-code-test-humaneval-test-val-code-{i}
    - is_gpu_base_model：固定传 1（输入已含 chat 模板，不叠加模型自带模板）
    """
    assert create_test is not None, "create_test 尚未绑定（应由主脚本设置为 test_utils.create_test）"
    assert process_prompt is not None, "process_prompt 尚未绑定（应由主脚本设置为 test_utils.process_prompt）"

    all_items = _iter_humaneval_items()
    total = len(all_items)
    tests: List[Dict[str, Any]] = []

    for i in indices:
        if i < 0 or i >= total:
            raise IndexError(f"索引越界 i={i}, total={total}")

        item = all_items[i]
        raw_prompt: str = item["prompt"]  # HumanEval 提供的函数签名/代码片段

        # 将原始 prompt 套入你提供的聊天模板
        filled_prompt = template.format(prompt=raw_prompt)

        # 因为 is_gpu_base_model=1，所以 process_prompt 不会再叠加任何模型模板
        final_prompt = process_prompt(filled_prompt, is_gpu_base_model)

        tests.append(
            create_test(
                testCaseName=f"{project_name}-code-test-humaneval-test-val-code-{i}",
                prompt=final_prompt,
                expect="",  # 置空
                maxGenTokens=max_output_token_length
            )
        )

    return tests
