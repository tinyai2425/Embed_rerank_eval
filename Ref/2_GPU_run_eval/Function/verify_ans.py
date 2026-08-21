import re

def verify_answer(expect, answer):
    expect_re = expect.strip()
    non_empty_splits = [split for split in answer.split("\n") if split.strip() != ""]
    if len(non_empty_splits) > 0:
        last_line = non_empty_splits[-1]
        if len(last_line) > 20:
            last_line = last_line[-20:]
        return bool(re.search(expect_re, last_line, re.IGNORECASE))
    else:
        return False
    
# Examples of test case name
# project-28-fast-test-ceval-exam-val-accountant-0
# project-28-perf-test-prefill-1024-output-50-8
# Please make sure the pattern matches the actual test config.
TEST_CASE_NAME_PATTERN = re.compile(r"^(?P<project_name>[a-zA-Z0-9_]+(?:-[a-zA-Z0-9_]+)*-[0-9]+)-(?P<flavor>[a-zA-Z0-9_]+)-test-(?P<vertical>[a-zA-Z0-9_]+(?:-[a-zA-Z0-9_]+)*)-[0-9]+$")