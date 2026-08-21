import os, json
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

CEVAL_DIR = "../data_set/CEVAL/ceval"      # C-Eval Github项目文件目录
CEVAL_EXAM_DIR = "../data_set/CEVAL/ceval-exam" # C-Eval HF数据文件目录

def generate_speculative_decoding_perf_tests(project_name, template, n_per_vertical_perf, prompt_token_length, max_output_token_length, verticals = None, is_gpu_base_model = 1):

    perf_test_cases = []
    prompt_lenghts = []
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CONFIG_PATH)
    # key/value example 'computer_network': ['Computer Network', '计算机网络', 'STEM']
    with open(f"{CEVAL_DIR}/subject_mapping.json", 'r', encoding='utf-8') as file:  
        # Step 2: Load JSON into a Python object (dict/list)
        subject_mapping = json.load(file)
        
    if not verticals:
        verticals = [ fn for fn in os.listdir(CEVAL_EXAM_DIR)  if not fn.startswith(".") and fn!="README.md"]
    assert len(verticals) == 52
    
    for vertical in verticals:
        dataset=load_dataset(CEVAL_EXAM_DIR, vertical)["val"]
        for i in range(min(len(dataset), n_per_vertical_perf)):
            # {'id': 0, 'question': '下列关于税法基本原则的表述中，不正确的是____。', 'A': '税收法定原则包括税收要件法定原则和税务合法性原则', 'B': '税收公平原则源于法律上的平等性原则', 'C': '税收效率原则包含经济效率和行政效率两个方面', 'D': '税务机关按法定程序依法征税，可以自由做出减征、停征或免征税款的决定', 'answer': 'D', 'explanation': ''}
            entry = dataset[i]
            prompt_orign = template.format(subject=subject_mapping[vertical][1], question=entry["question"], A=entry["A"], B=entry["B"], C=entry["C"], D=entry["D"])
            seq_length = prompt_token_length - len(tokenizer(prompt_orign)["input_ids"]) - 1
            prompt_temp = prompt_orign + ' \t\n'*seq_length
            prompt_lenghts.append(len(tokenizer(prompt_temp)["input_ids"]))
            
            perf_test_cases.append(create_test(
                testCaseName= f"{project_name}-perf-test-ceval-exam-val-{vertical}-{entry["id"]}",
                prompt= process_prompt(prompt_temp, is_gpu_base_model),
                expect= entry["answer"],
                maxGenTokens = max_output_token_length
            ))

    assert prompt_token_length * 0.99 < np.mean(prompt_lenghts) <= prompt_token_length * 1.01
    return perf_test_cases