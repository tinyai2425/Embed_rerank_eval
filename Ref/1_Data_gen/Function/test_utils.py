# test_utils.py

from transformers import AutoTokenizer

# 默认 mp（可在外部注入替换）
mp = None

def set_model_config(model_module):
    global mp
    mp = model_module

def create_test(testCaseName, prompt, expect="", maxGenTokens=2000):
    if mp is None:
        raise ValueError("You must call `set_model_config(mp)` before using create_test.")
    
    return {
        "testCaseName": testCaseName,
        "inferType": mp.INFER_TYPE,
        "tokenizerType": mp.TOKENIZER_TYPE,
        "tokenizerPath": mp.TOKENIZER_PATH,
        "modelType": mp.MODEL_TYPE,
        "modelPath": mp.MODEL_PATH,
        "weightDir": mp.WEIGHT_DIR,
        "prefixPrompt": mp.PREFIX_PROMPT,
        "pmtCacheOperation": mp.PMT_CACHE_OP,
        "pfxInitTokenLen": mp.PFX_INIT_TOKEN_LEN,
        "loraCfgPath": mp.LORA_CFG_PATH,
        "sentences": [
            {
                "prompt": prompt,
                "expect": expect,
                "callbackFreq": mp.CALLBACK_FREQ,
                "sampleFlag": mp.SAMPLE_FLAG,
                "seed": mp.SEED,
                "topK": mp.TOPK,
                "topP": mp.TOPP,
                "temperature": mp.TEMPERATURE,
                "maxGenTokens": maxGenTokens,
                "repetitionPenalty": mp.REPETITIONPENALTY,
                "initTokenLen": mp.INIT_TOKEN_LEN,
                "isAsync": mp.IS_ASYNC,
                "stopSeq": mp.STOP_SEQ,
            }
        ]
    }

def process_prompt(prompt, is_gpu_base_model):
    if mp is None:
        raise ValueError("You must call `set_model_config(mp)` before using process_prompt.")

    if is_gpu_base_model:
        return prompt
    else:
        messages = [{"role": "user", "content": prompt}]
        tokenizer = AutoTokenizer.from_pretrained(mp.TOKENIZER_CONFIG_PATH)
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

def create_API_test(testCaseName, prompt, expect="", maxGenTokens=2000):
    if mp is None:
        raise ValueError("You must call `set_model_config(mp)` before using create_test.")
    
    return {
        "model": mp.model_name,
        "testCaseName": testCaseName,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "expect": expect,
        "stream": False,
        "options": {
            "seed": 99,
            "num_predict": maxGenTokens,
            "temperature": mp.TEMPERATURE,
            "top_k": mp.TOPK,
            "top_p": mp.TOPP,
            "repeat_penalty": mp.REPETITIONPENALTY,
            "stop": mp.STOP_SEQ
        }
    }