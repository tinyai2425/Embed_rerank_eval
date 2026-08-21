import sys
import os
import json
import openai
import time
from concurrent.futures import ThreadPoolExecutor

def parse_args():
    if len(sys.argv) != 5:
        print("Usage: python GPU_run_rerank.py <input_json_path> <ip> <port> <model_id>")
        sys.exit(1)
    return sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

def convert_logprobs_to_serializable(logprobs):
    """将TopLogprob对象转换为可JSON序列化的格式"""
    serializable_logprobs = []
    for logprob in logprobs:
        serializable_logprobs.append({
            "token": logprob.token,
            "logprob": logprob.logprob,
            "bytes": logprob.bytes
        })
    return serializable_logprobs

def call_rerank_model(client, model_id, messages, test_case):
    """调用rerank模型并返回logprobs结果"""
    try:
        # 从测试用例中获取参数
        temperature = test_case.get("temperature", 0)
        max_tokens = test_case.get("maxGenTokens", 1)
        
        resp = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=10,
            extra_body={
                "chat_template_kwargs": {
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                },
            },
        )
        
        # 获取logprobs并转换为可序列化格式
        raw_logprobs = resp.choices[0].logprobs.content[0].top_logprobs
        serializable_logprobs = convert_logprobs_to_serializable(raw_logprobs)
        return serializable_logprobs
    
    except Exception as e:
        print(f"Error in API call: {e}")
        return None

def parallel_fetch_rerank_results(test_cases, client, model_id):
    """并行处理所有测试用例"""
    results = []
    start_time = time.time()
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 5) as executor:
        futures = []
        
        for test in test_cases:
            # 直接从测试用例中获取messages
            messages = test["messages"]
            future = executor.submit(
                call_rerank_model, 
                client, 
                model_id, 
                messages,
                test  # 传入整个test_case以获取参数
            )
            futures.append((future, test))
        
        # 收集结果
        for i, (future, test) in enumerate(futures):
            lp = future.result()
            
            if lp is not None:
                results.append({
                    "testCaseName": test.get("testCaseName", ""),
                    "queryName": test.get("queryName", ""),
                    "corpusName": test.get("corpusName", ""),
                    "expect": test.get("expect", 0),
                    "response": lp,
                })
            
            # 打印进度
            print(f"\rProcessed {i+1}/{len(test_cases)} test cases", end="", flush=True)
    
    print(f"\n[INFO] All completed in {time.time() - start_time:.2f}s")
    return results

def main():
    # 解析命令行参数
    input_json_path, ip, port, model_id = parse_args()
    print(f"Input file: {input_json_path}")
    print(f"Server: {ip}:{port}")
    print(f"Model ID: {model_id}")
    print("Notice: Proxy settings should be disabled before proceeding.")
    
    # 解析路径并创建输出目录
    input_dir = os.path.dirname(input_json_path)
    base_name = os.path.splitext(os.path.basename(input_json_path))[0]
    gpu_dir = os.path.join(input_dir, "GPU")
    os.makedirs(gpu_dir, exist_ok=True)
    output_path = os.path.join(gpu_dir, f"{base_name}_Rerank_Results.jsonl")
    
    # 初始化OpenAI客户端
    client = openai.OpenAI(
        base_url=f"http://{ip}:{port}/v1",
        api_key="no-api-key-needed"
    )
    
    # 加载测试数据
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
        print(f"Loaded {len(test_cases)} test cases")
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)
    
    # 执行推理并收集结果
    results = parallel_fetch_rerank_results(test_cases, client, model_id)
    
    # 写入结果到jsonl文件
    try:
        with open(output_path, "w", encoding="utf-8") as f_out:
            for item in results:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[DONE] Results saved to {output_path}")
        print(f"Total successful results: {len(results)}/{len(test_cases)}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()