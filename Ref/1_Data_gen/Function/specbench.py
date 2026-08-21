import json

# ===== 固定数据路径 =====
SPEC_BENCH_JSONL_PATH = "../data_set/spec_bench/question.jsonl"

# ===== 固定 System Prompt =====
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def _safe_str(x):
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""

def _read_jsonl(path):
    """按行读取 JSONL，遇到坏行跳过。"""
    with open(path, "r", encoding="utf-8") as f:
        for _, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def generate_specbench_tests(
    project_name,
    n_per_vertical,
    max_output_token_length=5000,
    verticals=None,
    is_gpu_base_model=1,
):
    """
    SpecBench：只取 turns[0]，expect 置空，统一 system prompt
    - is_gpu_base_model=0 => OMC: process_prompt 返回字符串（chat_template）
    - is_gpu_base_model=1 => API/GPU: process_prompt 返回 messages(list[dict])
    """
    groups = {}
    order = []

    for obj in _read_jsonl(SPEC_BENCH_JSONL_PATH):
        cat = _safe_str(obj.get("category", "")).strip() or "unknown"

        turns = obj.get("turns", None)
        if not isinstance(turns, list) or len(turns) < 1:
            continue

        first_turn = _safe_str(turns[0]).strip()
        if not first_turn:
            continue

        if cat not in groups:
            groups[cat] = []
            order.append(cat)

        groups[cat].append(first_turn)

    if not order:
        return []

    if verticals:
        selected_verticals = [v for v in verticals if v in groups]
    else:
        selected_verticals = order

    test_cases = []
    for vertical in selected_verticals:
        prompts = groups.get(vertical, [])
        if not prompts:
            continue

        take = min(len(prompts), n_per_vertical)
        for i in range(take):
            user_prompt = prompts[i]

            processed = process_prompt(
                user_prompt,
                is_gpu_base_model,
                system=SYSTEM_PROMPT,
            )

            testCaseName = f"{project_name}-spec-test-bench-exam-val-{vertical}-{i}"

            if is_gpu_base_model:
                test_cases.append(
                    create_API_test(
                        testCaseName=testCaseName,
                        messages=processed,
                        expect="",
                        maxGenTokens=max_output_token_length,
                    )
                )
            else:
                test_cases.append(
                    create_test(
                        testCaseName=testCaseName,
                        prompt=processed,
                        expect="",
                        maxGenTokens=max_output_token_length,
                    )
                )

    return test_cases
