from datasets import load_dataset
import os
import json

CEVAL_DIR = "../data_set/CEVAL/ceval"      # C-Eval Github项目文件目录
CEVAL_EXAM_DIR = "../data_set/CEVAL/ceval-exam" # C-Eval HF数据文件目录

fast_prompt = """以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。不要解释你的思考过程。答案最后一行必须以“答案：X”的格式给出最终答案。

{question}
A. {A}
B. {B}
C. {C}
D. {D}
答案："""


slow_prompt = """以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。详细解释你的分析思考过程。答案最后一行必须以“答案：X”的格式给出最终答案。

{question}
A. {A}
B. {B}
C. {C}
D. {D}
答案："""


def generate_ceval_tests(project_name, template, n_per_vertical, max_output_token_length = 5000, verticals = None, is_gpu_base_model = 1):

    test_cases = []
    # key/value example 'computer_network': ['Computer Network', '计算机网络', 'STEM']
    with open(f"{CEVAL_DIR}/subject_mapping.json", 'r', encoding='utf-8') as file:  
        # Step 2: Load JSON into a Python object (dict/list)
        subject_mapping = json.load(file)
        
    if not verticals:
        verticals = [ fn for fn in os.listdir(CEVAL_EXAM_DIR)  if not fn.startswith(".") and fn!="README.md"]
    assert len(verticals) == 52
    
    for vertical in verticals:
        dataset=load_dataset(CEVAL_EXAM_DIR, vertical)["val"]
        for i in range(min(len(dataset), n_per_vertical)):
            # {'id': 0, 'question': '下列关于税法基本原则的表述中，不正确的是____。', 'A': '税收法定原则包括税收要件法定原则和税务合法性原则', 'B': '税收公平原则源于法律上的平等性原则', 'C': '税收效率原则包含经济效率和行政效率两个方面', 'D': '税务机关按法定程序依法征税，可以自由做出减征、停征或免征税款的决定', 'answer': 'D', 'explanation': ''}
            entry = dataset[i]
            prompt_temp = template.format(subject=subject_mapping[vertical][1], question=entry["question"], A=entry["A"], B=entry["B"], C=entry["C"], D=entry["D"])
            test_cases.append(create_test(
                testCaseName= f"{project_name}-CEval-test-ceval-exam-val-{vertical}-{entry["id"]}",
                prompt= process_prompt(prompt_temp, is_gpu_base_model),
                expect= entry["answer"],
                maxGenTokens = max_output_token_length
            ))
            
    return test_cases