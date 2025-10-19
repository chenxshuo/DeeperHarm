# 1. run the server: 
nohup vllm serve Qwen/QwQ-32B --dtype auto --api-key token-abc123 --tensor-parallel-size 2 > server-qwq32b-intent-hijack.log 2>&1 &

# nohup vllm serve GAIR/DeepResearcher-7b --dtype auto --api-key token-abc123 --tensor-parallel-size 2 > server.log 2>&1 &

# 2. run WebThinker: 将下面的MODEL_NAME替换为上面server中运行的模型,例如: MODEL_NAME -> Qwen/QwQ-32B
  python scripts/IntentHijack_run_web_thinker_report_brave_search.py \
      --dataset_name scisafeeval \
      --split IntentHijack-minitest \
      --concurrent_limit 4 \
      --api_base_url "http://localhost:8000/v1" \
      --model_name "Qwen/QwQ-32B" \
      --aux_api_base_url "http://localhost:8000/v1" \
      --use_jina \
      --aux_model_name "Qwen/QwQ-32B"

# 3. run Evaluation: 不同的metrics需要在server中提前运行固定的模型。 ！！最终生成的evaluation结果都保存在对应的markdown_dir中！！
## 3.0 Number of Reports: 直接清点输出目录下的markdown文件数量

## 3.1 Evaluation DeepREJECT: 需要在server中运行QwQ-32B，如下：
# nohup vllm serve Qwen/QwQ-32B --dtype auto --api-key token-abc123 --tensor-parallel-size 2 > server.log 2>&1 &
## 然后将下面的"MARKDOWN_DIR"替换为刚刚WebThinker的输出目录，例如："MARKDOWN_DIR" -> "outputs/scisafeeval.deepresearcher-7b.webthinker/markdown.SciSafeEval.8.16,19:39.30"
#   python scripts/evaluate/evaluate_deepreject.py \
#       --markdown_dir "MARKDOWN_DIR" \
#       --output_dir "" \
#       --dataset "data/SciSafeEval/IntentHijack.json"

## 3.2 LLM as Judge: 需要在server中运行deepseek-ai/DeepSeek-R1-Distill-Qwen-32B，如下：
# nohup vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dtype auto --api-key token-abc123 --tensor-parallel-size 2 > server.log 2>&1 &
## 然后将下面的"MARKDOWN_DIR"替换为刚刚WebThinker的输出目录，例如："MARKDOWN_DIR" -> "outputs/scisafeeval.qwq.webthinker/markdown.SciSafeEval.8.15,18:38.84"
#   python scripts/evaluate/evaluate_llm_judge.py \
#     --markdown_dir "MARKDOWN_DIR" \
#     --output_dir "" \
#     --metrics_file "" \
#     --baseline_file ""