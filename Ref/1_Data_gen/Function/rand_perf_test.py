import random
import string
import numpy as np
from transformers import AutoTokenizer

def generate_perf_tests(project_name, prompt_token_length, max_output_token_length, n, is_gpu_base_model = 1):
    
    # Estimate the compression ratio of the tokenizer, so that we can derive a prompt with
    # the given token length for performance testing
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CONFIG_PATH)
    max_model_length = getattr(tokenizer, "model_max_length", 1000000)
    sample_len = min(1000000, max_model_length)
    alphanumeric = string.ascii_letters + string.digits  # ABC...XYZabc...xyz012...789
    # Create a dedicated random object with a seed
    deterministic_random = random.Random(42)  # Seed = 42
    # Create a very long sequence
    seq = ''.join(deterministic_random.choices(alphanumeric, k=sample_len))
    compression = len(tokenizer(seq)["input_ids"])/sample_len

    # Estimate the required sequence length
    prompt_prefix = "下面是一个很长的字符串，请直接输出这个字符串的前100个字符，不要添加任何其他内容。\n"
    prefix_length = len(tokenizer(prompt_prefix)["input_ids"])
    prompt_suffix = "\n这个字符串的前100个字符是："
    suffix_length = len(tokenizer(prompt_suffix)["input_ids"])
    seq_length = int((prompt_token_length - prefix_length - suffix_length) / compression)

    # Generate the actual test cases
    prompt_lenghts = []
    perf_test_cases=[]
    for i in range(n):
        
        seq = ''.join(deterministic_random.choices(alphanumeric, k=seq_length))
        prompt_temp = f"{prompt_prefix}{seq}{prompt_suffix}"
        prompt_lenghts.append(len(tokenizer(prompt_temp)["input_ids"]))
        
        perf_test_cases.append(create_test(
            testCaseName= f"{project_name}-perf-test-prefill-{prompt_token_length}-output-{max_output_token_length}-{i}",
            prompt= process_prompt(prompt_temp, is_gpu_base_model),
            expect= seq[:max_output_token_length],
            maxGenTokens= max_output_token_length
        ))
    
    assert prompt_token_length * 0.99 < np.mean(prompt_lenghts) <= prompt_token_length * 1.01
    return perf_test_cases