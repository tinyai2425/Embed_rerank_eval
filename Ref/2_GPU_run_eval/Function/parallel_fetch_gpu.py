import os
import openai
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
import json

# Fetching using (5 x CPU cores) threads
def parallel_fetch_reference_model(test_config_path, server_ip, server_port, model_id):

    client = openai.OpenAI(
        base_url=f"http://{server_ip}:{server_port}/v1",  # vLLM default port
        api_key="no-api-key-needed"  # vLLM doesn't require a key
    )

    with open(test_config_path, 'r') as f:
         test_cases = json.load(f)

    # Verify test cases before fetching
    assert all(len(test["sentences"])==1 for test in test_cases)
    assert all(re.match(verify_ans.TEST_CASE_NAME_PATTERN, test["testCaseName"]) for test in test_cases)
    assert len({test["testCaseName"] for test in test_cases}) == len(test_cases)

    start_time = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 5) as executor:
        futures = []
        for test in test_cases:
            future = executor.submit(
                measure_perf_gpu.measure_performance, 
                client,
                model_id,
                test["testCaseName"],
                test["sentences"][0])

            futures.append(future)

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(future.result())
            remaining_time = (time.time() - start_time)/(i+1)*(len(test_cases) - i -1)
            print(f"\r{i+1}/{len(test_cases)} test cases processed (TTFT:{result["first_token_time"] :.2f}s/TTC:{result["total_time"] :.2f}s), {remaining_time :.2f}s remaining ...", end="", flush=True)

    # sort all test results by their order in the test case config
    result_map = {result["testCaseName"]: result for result in results}
    sorted_results = [result_map[test["testCaseName"]] for test in test_cases]
    df_results = pd.DataFrame(sorted_results)
    print(f"\r{len(test_cases)} test cases processed（{time.time() - start_time :.2f}s spent）in {test_config_path}") # insert a new line
    return df_results