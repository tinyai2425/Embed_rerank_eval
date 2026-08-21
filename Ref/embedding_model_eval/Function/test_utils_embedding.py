from typing import Dict, Any, List
mp = None  # 将通过 set_model_config 注入配置（如 mp.model_name）

def set_model_config(model_params):
    global mp
    mp = model_params

def create_gpu_embedding_test(testCaseName: str, input_list: List[str], expect: Any = "") -> Dict[str, Any]:
    return {
        "testCaseName": testCaseName,
        "input": input_list,
        "expect": expect,
    }


def create_api_embedding_test(testCaseName: str, input_list: List[str], expect: Any = "") -> Dict[str, Any]:
    if mp is None or not getattr(mp, "model_name", None):
        raise ValueError("You must call `set_model_config(mp)` and ensure `mp.model_name` is set.")
    return {
        "model": mp.model_name,
        "testCaseName": testCaseName,
        "input": input_list,
        "expect": expect
    }