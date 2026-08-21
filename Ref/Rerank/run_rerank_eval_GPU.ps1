# 运行评估脚本 powershell中运行
python evaluate_rerank_results.py --results_file "../../model-eval-storage/Qwen3-Reranker-0.6B/project-2/GPU/project-2-RERANK-GPU-MAXQ100_Rerank_Results.jsonl" --k_values "[5,10]"

# 暂停一下，让您能看到结果（可选）
Write-Host ""
Write-Host "按任意键退出..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")