import time
import re
from transformers import AutoTokenizer

# 全局变量（延迟初始化）
TOKENIZER_CONFIG_PATH = None
tokenizer = None

def init_tokenizer(path):
    global TOKENIZER_CONFIG_PATH, tokenizer
    TOKENIZER_CONFIG_PATH = path
    tokenizer = AutoTokenizer.from_pretrained(path)

def measure_performance(client, model_id, testCaseName, sentence):
    global tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenizer not initialized. Please call init_tokenizer(path) before using measure_performance.")

    start_time = time.time()
    first_token_time = None
    full_response = ""

    # Stream with deterministic parameters
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": sentence["prompt"]}],
        stream=True,
        seed=sentence["seed"],
        temperature=sentence.get("temperature"),
        top_p=sentence.get("topP"),
        max_tokens=sentence["maxGenTokens"],
        extra_body={
            "top_k": sentence.get("topK"),
            "repetition_penalty": sentence.get("repetitionPenalty")
        }
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time() - start_time
            full_response += chunk.choices[0].delta.content

    total_time = time.time() - start_time

    matched = re.match(verify_ans.TEST_CASE_NAME_PATTERN, testCaseName)
    if matched is None:
        raise ValueError(f"testCaseName `{testCaseName}` does not match pattern.")

    result = dict(sentence,
        testCaseName=testCaseName,
        project_name=matched.group("project_name"),
        flavor=matched.group("flavor"),
        vertical=matched.group("vertical"),
        prompt_token_len=len(tokenizer(sentence["prompt"])["input_ids"]),
        response=full_response,
        response_token_len=len(tokenizer(full_response)["input_ids"]),
        first_token_time=first_token_time,
        total_time=total_time,
        decode_time=total_time - first_token_time if first_token_time is not None else None,
        correct=verify_ans.verify_answer(sentence["expect"], full_response),
        get_ans=extend_metrics.GET_answer(full_response),
        repeat=extend_metrics.calculate_repetition_rate(full_response),
        entropy=extend_metrics.calculate_token_entropy(full_response)
    )

    return result
