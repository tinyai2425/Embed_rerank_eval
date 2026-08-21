# Function/Bayesian_search.py
# -*- coding: utf-8 -*-
"""
基于精度的贝叶斯参数搜索工具集（与 vLLM OpenAI 接口兼容）

功能：
- parallel_fetch_with_overrides: 并行拉取，在采样参数覆盖下执行测试，仅计算 correct
- build_search_space: 将命令行/字符串规格转为 skopt 维度与固定值
- run_one_eval: 拉取一次 + 评估精度
- bayesian_search: 主流程（forest_minimize）
- save_search_artifacts: 保存试验轨迹与最优结果

依赖你现有的：
- verify_ans: 提供 TEST_CASE_NAME_PATTERN 与 verify_answer(...)
  >>> Function/verify_ans.py 中已有
"""

import os
import re
import json
import time
import traceback
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# skopt（需安装：pip install scikit-optimize tenacity）
try:
    from skopt import forest_minimize
    from skopt.space import Integer, Categorical
    from skopt.utils import use_named_args
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("缺少依赖：请先安装 scikit-optimize：\n  pip install scikit-optimize") from e

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

import openai  # openai>=1.x

# === 引入你的 verify_ans ===
try:
    import verify_ans
except Exception as e:
    raise RuntimeError("请确认 Function/verify_ans.py 可被导入（sys.path 已包含 Function）。") from e


# ===============================
# 工具：将 "a:b:step" 解析为离散浮点列表；"m:n:step" 对 int 也可
# ===============================
def _float_range(start: float, end: float, step: float, precision: int = 4) -> List[float]:
    n = int(round((end - start) / step))  # 包含端点
    vals = [start + i * step for i in range(n + 1)]
    return [float(f"{v:.{precision}f}") for v in vals]

def _int_range(start: int, end: int, step: int = 1) -> List[int]:
    return list(range(start, end + 1, step))


# ===============================
# 构建搜索空间（混合固定值与离散集合）
# 传入 specs 形如：
#   {"temperature":"0.6:1.2:0.05", "top_p":"0.7:1.0:0.02", "top_k":"8", "repetition_penalty":"0.8:1.4:0.02"}
# 或者固定值：{"temperature":"1.0"}（将被视作固定）
# ===============================
def build_search_space(specs: Dict[str, str]):
    """
    返回：
      dimensions: List[skopt.space.Dimension]
      fixed_values: dict（固定参数会放这里，优化时不参与）
      dim_names: List[str]（与 dimensions 对应的参数名顺序）
    """
    dims = []
    fixed = {}
    names = []

    def parse_numeric_list(s: str) -> List[float]:
        # 支持 "a:b:step" 或 "v1,v2,v3"
        if ":" in s:
            a, b, c = s.split(":")
            a, b, c = float(a), float(b), float(c)
            return _float_range(a, b, c)
        if "," in s:
            return [float(x) for x in s.split(",")]
        return [float(s)]  # 单值

    def parse_int_list(s: str) -> List[int]:
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 3:
                a, b, step = int(parts[0]), int(parts[1]), int(parts[2])
            else:
                a, b = int(parts[0]), int(parts[1])
                step = 1
            return _int_range(a, b, step)
        if "," in s:
            return [int(x) for x in s.split(",")]
        return [int(s)]

    # 统一小写键
    specs_l = {k.lower(): v for k, v in specs.items() if v is not None}

    # temperature / top_p / repetition_penalty: 浮点
    for key in ["temperature", "top_p", "repetition_penalty"]:
        if key in specs_l:
            vals = parse_numeric_list(specs_l[key])
            if len(vals) == 1:
                fixed[key] = vals[0]
            else:
                dims.append(Categorical(vals, name=key))
                names.append(key)

    # top_k: 整数
    if "top_k" in specs_l:
        ivals = parse_int_list(specs_l["top_k"])
        if len(ivals) == 1:
            fixed["top_k"] = ivals[0]
        else:
            # 使用 Categorical，避免整数空间过大且确保离散步长
            dims.append(Categorical(ivals, name="top_k"))
            names.append("top_k")

    return dims, fixed, names


# ===============================
# 最小化执行单次评估：给定参数 -> 计算正确率
# ===============================
def evaluate_parameters_once(
    input_json_path: str,
    server_ip: str,
    server_port: int,
    model_id: str,
    params: Dict[str, Any],
    max_workers: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[float, pd.DataFrame]:
    """
    返回 (accuracy, df_results)
    df_results 仅包含 minimal 字段：testCaseName, response, correct
    """
    df_results = parallel_fetch_with_overrides(
        input_json_path=input_json_path,
        server_ip=server_ip,
        server_port=server_port,
        model_id=model_id,
        overrides=params,
        max_workers=max_workers,
        verbose=verbose,
    )

    # 计算精度
    if "correct" not in df_results.columns or len(df_results) == 0:
        accuracy = 0.0
    else:
        accuracy = float(np.mean(df_results["correct"].astype(float)))
    return accuracy, df_results


# ===============================
# 并行拉取（仅 correct）
# ===============================
def parallel_fetch_with_overrides(
    input_json_path: str,
    server_ip: str,
    server_port: int,
    model_id: str,
    overrides: Dict[str, Any],
    max_workers: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:

    client = openai.OpenAI(
        base_url=f"http://{server_ip}:{server_port}/v1",
        api_key="not-required",
        timeout=120,
    )

    with open(input_json_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    assert all(len(t["sentences"]) == 1 for t in test_cases), "每个测试用例必须仅含 1 个句子"
    assert all(re.match(verify_ans.TEST_CASE_NAME_PATTERN, t["testCaseName"]) for t in test_cases), "testCaseName 格式不匹配"
    assert len({t["testCaseName"] for t in test_cases}) == len(test_cases), "testCaseName 存在重复"

    if max_workers is None:
        max_workers = os.cpu_count() * 5 if os.cpu_count() else 8

    def _worker(test: Dict[str, Any]) -> Dict[str, Any]:
        sent = test["sentences"][0]
        prompt = sent["prompt"]
        seed = sent.get("seed", None)
        max_new_tokens = sent.get("maxGenTokens", 512)

        # 构造参数（固定覆盖）
        temperature = float(overrides.get("temperature", 1.0))
        top_p = float(overrides.get("top_p", 1.0))
        top_k = int(overrides.get("top_k", 0))  # 0/不传 由服务端默认
        repetition_penalty = float(overrides.get("repetition_penalty", 1.0))

        extra_body = {"repetition_penalty": repetition_penalty}
        if top_k and top_k > 0:
            extra_body["top_k"] = top_k

        # 非流式，简单可靠
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stream=False,
            extra_body=extra_body,
        )
        text = resp.choices[0].message.content or ""
        ok = bool(verify_ans.verify_answer(sent["expect"], text))

        return {
            "testCaseName": test["testCaseName"],
            "response": text,
            "correct": int(ok),
        }

    start = time.time()
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, t) for t in test_cases]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            if verbose:
                elapsed = time.time() - start
                remaining = elapsed / i * (len(futures) - i)
                print(f"\r{i}/{len(futures)} 进度，剩余约 {remaining:.1f}s ...", end="", flush=True)
    if verbose:
        print(f"\n完成：共 {len(results)} 个用例，用时 {time.time() - start:.1f}s")

    # 保持与输入顺序一致
    m = {r["testCaseName"]: r for r in results}
    ordered = [m[t["testCaseName"]] for t in test_cases]
    return pd.DataFrame(ordered)


# ===============================
# 主流程：贝叶斯（随机森林）优化
# ===============================
def bayesian_search(
    input_json_path: str,
    server_ip: str,
    server_port: int,
    model_id: str,
    param_specs: Dict[str, str],
    n_calls: int = 50,
    n_init: int = 10,
    seed: int = 42,
    retry_times: int = 2,
    retry_wait: int = 3,
    save_each_eval_dir: Optional[str] = None,
    verbose: bool = True,
):
    """
    param_specs: 例如 {"temperature":"0.6:1.2:0.05","top_p":"0.7:1.0:0.02","top_k":"8","repetition_penalty":"0.8:1.4:0.02"}
    save_each_eval_dir: 若指定，则每轮将 df_results 保存为 parquet/csv，以便排查
    返回： (result_obj, trials_df)
    """
    dimensions, fixed_values, dim_names = build_search_space(param_specs)

    trial_records: List[Dict[str, Any]] = []

    @retry(
        stop=stop_after_attempt(retry_times),
        wait=wait_fixed(retry_wait),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _safe_eval(overrides: Dict[str, Any]) -> Tuple[float, pd.DataFrame]:
        acc, df = evaluate_parameters_once(
            input_json_path=input_json_path,
            server_ip=server_ip,
            server_port=server_port,
            model_id=model_id,
            params=overrides,
            verbose=verbose,
        )
        return acc, df

    # skopt 目标：最小化（所以取 -acc）
    def _objective(x):
        # x 顺序与 dim_names 对齐
        overrides = dict(zip(dim_names, x))
        overrides.update(fixed_values)  # 固定值覆盖
        try:
            acc, df = _safe_eval(overrides)
        except Exception:
            if verbose:
                print("\n评估失败，进入重试/或计为差结果 ...")
            # 若重试后仍失败，这里给一个很差的分数
            acc, df = 0.0, pd.DataFrame()

        # 记录 trial
        rec = {"accuracy": acc}
        rec.update({k: overrides.get(k) for k in ["temperature", "top_p", "top_k", "repetition_penalty"]})
        trial_records.append(rec)

        # 可选保存
        if save_each_eval_dir:
            os.makedirs(save_each_eval_dir, exist_ok=True)
            idx = len(trial_records)
            # 仅当 df 非空才保存
            if not df.empty:
                df.to_parquet(os.path.join(save_each_eval_dir, f"trial_{idx:03d}.parquet"))
                df.to_csv(os.path.join(save_each_eval_dir, f"trial_{idx:03d}.csv"), index=False)
            with open(os.path.join(save_each_eval_dir, "trials_running.csv"), "w", encoding="utf-8") as f:
                pd.DataFrame(trial_records).to_csv(f, index=False)

        if verbose:
            print(f"\n[Trial #{len(trial_records)}] acc={acc:.4f}  params={overrides}")

        return -acc

    # 包装 skopt
    @use_named_args(dimensions)
    def wrapped_objective(**kwargs):
        # kwargs: 只有参与搜索的维度
        x = [kwargs[n] for n in dim_names]
        return _objective(x)

    result = forest_minimize(
        func=wrapped_objective if len(dimensions) > 0 else (lambda **_: _objective([])),
        dimensions=dimensions if len(dimensions) > 0 else [Categorical([0], name="dummy")],
        n_calls=n_calls,
        n_initial_points=n_init,
        random_state=seed,
        verbose=bool(verbose),
    )

    trials_df = pd.DataFrame(trial_records)
    return result, trials_df


def save_search_artifacts(
    out_dir: str,
    result,
    trials_df: pd.DataFrame,
    param_names_order: List[str] = ("temperature", "top_p", "top_k", "repetition_penalty"),
):
    os.makedirs(out_dir, exist_ok=True)

    # 保存 trial 轨迹
    trials_csv = os.path.join(out_dir, "bayes_trials.csv")
    trials_df.to_csv(trials_csv, index=False)

    # 最优
    best_params = dict(zip(param_names_order, [None]*len(param_names_order)))
    # result.x 按参与优化维度顺序；trials_df 里也有最后一行可参考
    # 为简洁，这里从 trials_df 中取最优
    if "accuracy" in trials_df.columns and not trials_df.empty:
        best_row = trials_df.iloc[trials_df["accuracy"].idxmax()]
        for k in param_names_order:
            if k in best_row and pd.notna(best_row[k]):
                best_params[k] = best_row[k]
        best_acc = float(best_row["accuracy"])
    else:
        best_acc = 0.0

    best_json = os.path.join(out_dir, "best_result.json")
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump({"best_accuracy": best_acc, "best_params": best_params}, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] Trials -> {trials_csv}")
    print(f"[SAVE] Best   -> {best_json}")
