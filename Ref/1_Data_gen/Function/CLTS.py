import os
import random

CLTS_DIR = "../data_set/CLTS"
CLTS_SOURCE = os.path.join(CLTS_DIR, "test.src")
CLTS_REFERENCE = os.path.join(CLTS_DIR, "test.tgt")

def load_CLTS_dataset(source_path, reference_path):
    def clean_line(line):
        # 去除空格和首尾换行符
        return line.replace(" ", "").strip()

    CLTS_data = []

    with open(source_path, 'r', encoding='utf-8') as source_file, open(reference_path, 'r', encoding='utf-8') as reference_file:
        source_lines = source_file.readlines()
        reference_lines = reference_file.readlines()

        assert len(source_lines) == len(reference_lines), f"原文和参考摘要的行数不匹配: {len(source_lines)} != {len(reference_lines)}"

        for source, summary in zip(source_lines, reference_lines):
            entry = {
                "source": clean_line(source),
                "summary": clean_line(summary)
            }
            CLTS_data.append(entry)

    return CLTS_data

summary_prompt = """以下是新闻原文，请根据原文生成一句简洁、连贯的摘要，保持信息的完整性和准确性，避免冗长或复杂。请确保生成的摘要不超过 80 字。

原文：
{source}

摘要：
"""

def generate_CLTS_tests(project_name, template, n_cases=1200, max_output_token_length=5000, is_gpu_base_model=1, seed=42):
    test_cases = []
    CLTS_data = load_CLTS_dataset(CLTS_SOURCE, CLTS_REFERENCE)

    # 设置随机种子，确保每次随机选择的测试集顺序一致
    random.seed(seed)
    
    # 随机选择 n_cases 个索引
    selected_indices = random.sample(range(len(CLTS_data)), n_cases)
    selected_indices.sort()

    for i in selected_indices:  # 使用选中的索引生成测试用例
        entry = CLTS_data[i]
        prompt_temp = template.format(source=entry['source'])
        test_cases.append(create_test(
            testCaseName=f"{project_name}-CLTS-test-clts-test-val-summary-{i}",
            prompt=process_prompt(prompt_temp, is_gpu_base_model),
            expect=entry["summary"],
            maxGenTokens=max_output_token_length
        ))

    return test_cases