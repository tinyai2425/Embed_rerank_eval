# Embedding / Rerank 评估

从 `Ref/` 精简并重组后的 embedding / rerank 评测代码。只保留这两类模型，不再包含 ceval 等任务。

数据集请放到本仓库的 `data_set/`（本机数据未随仓库移动，使用时自行补全）。用例和评估结果写到上一级目录的 `../model-eval-storage/<model_name>/project-N/`。

```
data_set/C-MTEB/test-C-METB-STSB.parquet
data_set/WIKI_perf/wiki_CN_ENG_merged.parquet
data_set/Rerank/RerankerEval_Small_merged.pkl
```

```bash
pip install -r requirements.txt
```

## 目录

| 目录 | 作用 |
|------|------|
| `1_Data_gen/` | 生成 GPU JSON 与 API JSONL 用例 |
| `2_GPU/` | GPU（vLLM OpenAI 接口）推理与评估 |
| `3_API/` | 本机 API 采集与评估 |
| `Ref/` | 原始代码，仅作对照 |

文件名带 `embed` 或 `rerank`，用来区分两类模型。Rerank 当前 GPU 实现是 Qwen3-0.6B；通用评估与 API 接口和模型无关，以后加模型只需新增对应 `.py`。

## Embedding

**1. 造数**（在 `1_Data_gen/`）

```bash
python Gen_embed_STSB_cases.py configs/embed.example.json [MAX_CASES]
python Gen_embed_wiki_cases.py configs/embed.example.json [MAX_CASES]
```

wiki 造数需要配置里的 `TOKENIZER_CONFIG_PATH`。每个 project 会同时写出 GPU `.json` 和 API `.jsonl`。

**2. GPU**（在 `2_GPU/`）

```bash
python Eval_GPU_embed_STSB.py <gpu.json> <ip> <port> <model_id> <version_flag>
python Eval_GPU_embed_wiki.py <gpu.json> <ip> <port> <model_id>
```

STSB 会写 Spearman/Pearson；wiki 只落原始向量（`GPU/GPU-Results-1.jsonl`）。

**3. API**（在 `3_API/`）

```bash
python Collect_API_embed.py <api.jsonl> <out.txt> [http://localhost:11434/v1/embeddings]
python Eval_API_embed_STSB.py <out.txt>
python Eval_GPU_API_embed_wiki.py <gpu_jsonl> <api_txt>
```

采集脚本只向服务端发送 `model` + `input`。wiki 对比会在 GPU 结果目录旁创建 `GPU-API/`，把 cosine 明细和 Excel 放进去。

API 请求兼容 `/v1/embeddings`；响应同时支持 `embeddings: [[...]]` 和 OpenAI 的 `data[].embedding`。

## Rerank

GPU 侧按 Qwen3 的 chat + yes/no logprobs 造数和打分；API 侧走通用接口 `{model, query, documents}`。同一套 system prompt / instruct 在 API 服务端写死，不通过接口下发。

**1. 造数**（在 `1_Data_gen/`）

```bash
python Gen_rerank_cases.py configs/rerank_qwen3.example.json <MAX_query>
```

一次生成 GPU JSON（含 messages）和 API JSONL（仅 query/documents）。`rerank_family` 默认 `qwen3`。

**2. GPU**（在 `2_GPU/`）

```bash
python GPU_run_rerank_qwen3.py <gpu.json> <ip> <port> <model_id> [k_values]
```

推理结束后直接在 `GPU/` 写出 `*_Rerank_Results.jsonl` 和 `evaluation_results.md`，不用再跑一遍评估。

**3. API**（在 `3_API/`）

```bash
python Collect_API_rerank.py <api.jsonl> <out.txt> [http://localhost:11434/v1/reranks]
python Eval_API_rerank.py <out.txt> [k_values]
```

采集脚本只发送 `model` / `query` / `documents`。评估结果在日志同级的 `API/evaluation_results.md`。

### 以后加新的 rerank 模型

1. `1_Data_gen/Function/rerank_<family>.py`：GPU 用例格式（messages 等）
2. 在 `Gen_rerank_cases.py` 的 `GPU_GENERATORS` 里注册
3. `2_GPU/GPU_run_rerank_<family>.py` + 对应打分文件
4. API 造数、采集、评估不用改（通用 `/v1/reranks`）

配置示例见 `1_Data_gen/configs/`。各目录下的 `run_*.sh` 是路径模板，按本机 IP/端口改即可。
