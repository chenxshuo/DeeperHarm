#!/bin/bash


### Evaluation DeepREJECT
#  python scripts/evaluate/evaluate_deepreject.py \
#       --markdown_dir "outputs/scisafeeval.deepresearcher-7b.webthinker/markdown.SciSafeEval.8.16,19:39.30" \
#       --output_dir "" \
#       --dataset "data/SciSafeEval/SciSafeEval.json"

# --output_dir "" \ # default: "" = markdown_dir



### Evaluation Refusal Words
# if want to evaluate in lower ASR model -> add "--lower"
# python scripts/evaluate/evaluate_refusal_words.py \
#      --markdown_dir "outputs/DeHARM.QwQ-32B/markdown.IntentHijack" \
#      --output_dir "" \
#      --metrics_file "" \
#      --baseline_file "" \
#      --lower


### Evaluation LLM Judge
#python scripts/evaluate/evaluate_llm_judge.py \
#     --markdown_dir "outputs/scisafeeval.qwq.webthinker/markdown.SciSafeEval.8.15,18:38.84" \
#     --output_dir "" \
#     --metrics_file "" \
#     --baseline_file ""

 python scripts/evaluate/evaluate_llm_judge.py \
      --markdown_dir "outputs/scisafeeval.deepresearcher-7b.webthinker/markdown.SciSafeEval.8.16,19:39.30" \
      --output_dir "" \
      --metrics_file "" \
      --baseline_file ""



### Evaluation Llama Guard
# python scripts/evaluate/evaluate_llama_guard.py \
#      --markdown_dir "outputs/DeHARM.QwQ-32B/markdown.IntentHijack" \
#      --output_dir "" \
#      --metrics_file "" \
#      --baseline_file ""
