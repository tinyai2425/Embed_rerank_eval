"""答案正确性判断与测试用例名解析。

判断逻辑：优先用 "答案：X" 模式从回答中提取预测字母（取最后
一次匹配），与期望答案比对；若无该模式则回退到末行子串匹配
（最多取末尾 20 个字符）。
"""
import re

# 测试用例名形如：
#   project-1-CEval-test-ceval-exam-val-accountant-0      （精度用例）
#   project-28-perf-test-prefill-1024-output-50-8         （性能用例）
# 分组：project_name / flavor / vertical
TEST_CASE_NAME_PATTERN = re.compile(
    r"^(?P<project_name>[a-zA-Z0-9_]+(?:-[a-zA-Z0-9_]+)*-[0-9]+)"
    r"-(?P<flavor>[a-zA-Z0-9_]+)-test-"
    r"(?P<vertical>[a-zA-Z0-9_]+(?:-[a-zA-Z0-9_]+)*)-[0-9]+$"
)

# 性能测试用例的 flavor / vertical 特征。这些用例要求模型复述长字符串，
# 其“答案”几乎必然判错，故不应计入 CEval 精度。
PERF_FLAVORS = {"perf"}


# 答案提取模式：匹配 "答案：A" / "答案:B" 等，取 A-D 单字母
ANSWER_PATTERN = re.compile(r"答案[：:]\s*([A-D])\b")


def verify_answer(expect, answer) -> bool:
    """从回答中提取预测答案，与期望答案比对，返回是否正确。

    优先用 "答案：X" 模式提取（取最后一次匹配，避免末行截断和
    子串误匹配问题）；无该模式时回退到末行子串匹配。
    """
    expect = expect.strip()
    if not expect:
        return False
    # 优先用 "答案：X" 模式提取预测字母
    ans_matches = ANSWER_PATTERN.findall(answer or "")
    if ans_matches:
        return ans_matches[-1] == expect
    # 回退：在回答末尾查找期望答案（兼容无 "答案：" 前缀的输出格式）
    expect_re = re.escape(expect)
    non_empty_splits = [s for s in (answer or "").split("\n") if s.strip() != ""]
    if not non_empty_splits:
        return False
    last_line = non_empty_splits[-1]
    if len(last_line) > 20:
        last_line = last_line[-20:]
    return bool(re.search(expect_re, last_line, re.IGNORECASE))


def parse_test_case_name(case_name: str):
    """解析测试用例名，返回 (project_name, flavor, vertical)；无法解析时返回 (None, None, None)。"""
    matched = TEST_CASE_NAME_PATTERN.match(case_name or "")
    if matched:
        return matched.group("project_name"), matched.group("flavor"), matched.group("vertical")
    return None, None, None


def is_perf_case(flavor, vertical) -> bool:
    """判断是否为性能测试用例（不计入精度）。"""
    if flavor and flavor.lower() in PERF_FLAVORS:
        return True
    if vertical and vertical.lower().startswith("prefill"):
        return True
    return False
