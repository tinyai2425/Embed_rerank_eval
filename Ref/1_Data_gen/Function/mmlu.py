import pandas as pd

# ===== 固定数据路径 =====
MMLU_PARQUET_PATH = "../data_set/MMLU/mmlu-test-00000-of-00001.parquet"

# ===== Prompts (English) =====
fast_prompt = """Here is a single-choice question from the {subject} exam. Choose the correct option. Do not explain your reasoning. The last line must be exactly in the format "Answer: X".

{question}
A. {A}
B. {B}
C. {C}
D. {D}
Answer:"""

slow_prompt = """Here is a single-choice question from the {subject} exam. Choose the correct option. Explain your reasoning step by step. The last line must be exactly in the format "Answer: X".

{question}
A. {A}
B. {B}
C. {C}
D. {D}
Answer:"""

def _normalize_choices(x):
    """
    将 choices 统一转为 list[str]，且长度必须为 4。
    兼容 numpy.ndarray / list / tuple；若异常返回 None。
    """
    if x is None:
        return None
    # 某些实现里 choices 可能是 numpy.ndarray；也可能是 list/tuple
    try:
        # 避免把字符串当作可迭代拆开
        if isinstance(x, str):
            return None
        lst = list(x)
    except Exception:
        return None

    if len(lst) != 4:
        return None

    # 统一转成字符串，空值转为空串，避免后续 format 报错
    return ["" if v is None else str(v) for v in lst]

def _answer_idx_to_letter(a):
    """
    将答案索引(0/1/2/3，可能是字符串或 numpy 标量)安全转换为 A/B/C/D。
    异常返回 None。
    """
    try:
        idx = int(a)
    except Exception:
        return None
    if 0 <= idx <= 3:
        return "ABCD"[idx]
    return None

def generate_mmlu_tests(project_name, template, n_per_vertical, 
                        max_output_token_length=5000, verticals=None, is_gpu_base_model=1):
    # 读取 parquet
    df = pd.read_parquet(MMLU_PARQUET_PATH)

    # 规范化 choices 到一个新列，过滤掉异常行（非 4 选项）
    df = df.copy()
    df["choices4"] = df["choices"].apply(_normalize_choices)
    df = df[df["choices4"].notna()].reset_index(drop=True)

    # verticals (subjects) 选择
    if not verticals:
        verticals = sorted(df["subject"].dropna().astype(str).unique().tolist())

    test_cases = []

    for vertical in verticals:
        sub = df[df["subject"] == vertical].reset_index(drop=True)
        if sub.empty:
            continue

        take = min(len(sub), n_per_vertical)

        for i in range(take):
            row = sub.iloc[i]

            choices = row["choices4"]
            if not choices or len(choices) != 4:
                continue
            A, B, C, D = choices

            question = "" if pd.isna(row["question"]) else str(row["question"])
            subject_str = str(vertical).replace("_", " ")

            # 渲染模板
            prompt_temp = template.format(
                subject=subject_str,
                question=question,
                A=A, B=B, C=C, D=D
            )

            # 答案映射（0/1/2/3 -> A/B/C/D），异常则跳过
            ans_letter = _answer_idx_to_letter(row["answer"])
            if ans_letter is None:
                continue

            # testCaseName：与 CEVAL 对齐，只是换成 MMLU；无 id 时用 subject 内部顺序 i
            test_cases.append(create_test(
                testCaseName=f"{project_name}-MMLU-test-mmlu-exam-val-{vertical}-{i}",
                prompt=process_prompt(prompt_temp, is_gpu_base_model),
                expect=ans_letter,
                maxGenTokens=max_output_token_length
            ))

    return test_cases
