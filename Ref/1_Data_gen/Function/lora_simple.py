# lora_simple.py
import os
import json
from typing import List, Dict, Any

LORA_DATA_PATH = "../data_set/Lora_test/commonsense_test.json"

def generate_lora_tests(
    project_name: str,
    lora_data_path: str = LORA_DATA_PATH,
    max_output_token_length: int = 5000,
    is_gpu_base_model: int = 1,
) -> List[Dict[str, Any]]:
    """
    从 Lora 测试集 JSON 生成测试用例。
    - instruction 直接作为 prompt
    - answer 作为 expect
    - testCaseName = f"{project_name}-Lora-test-origin-exam-val-lora-<id或索引>"
    - 不分 vertical，整文件全量使用

    依赖：环境中已存在 process_prompt(prompt, is_gpu_base_model) 与 create_test(...)
    """

    if not os.path.exists(lora_data_path):
        raise FileNotFoundError(f"Lora test file not found: {lora_data_path}")

    with open(lora_data_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {lora_data_path}: {e}")

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {lora_data_path}, got {type(data)}")

    test_cases: List[Dict[str, Any]] = []

    for idx, entry in enumerate(data):
        instr = (entry.get("instruction") or "").strip()
        expect = (entry.get("answer") or "").strip()

        # 关键字段缺失则跳过
        if not instr or not expect:
            continue

        case_id = entry.get("id", idx)
        test_case_name = f"{project_name}-Lora-test-origin-exam-val-lora-{case_id}"

        test_cases.append(
            create_test(
                testCaseName=test_case_name,
                prompt=process_prompt(instr, is_gpu_base_model),
                expect=expect,
                maxGenTokens=max_output_token_length,
            )
        )

    return test_cases