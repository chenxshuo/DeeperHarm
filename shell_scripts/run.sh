#!/bin/bash



### WebThinker Report Generation
# single question
python scripts/DirectlyDR.py \
   --single_question "What are the LLMs from OpenAI?" \
   --api_base_url "http://localhost:8000/v1" \
   --model_name "Qwen/QwQ-32B" \
   --aux_api_base_url "http://localhost:8000/v1" \
   --use_jina \
   --aux_model_name "Qwen/QwQ-32B"

# "What are the LLMs from OpenAI?" \
# "What is the topic of the 2. Talk on webseit 'https://kidlanli.github.io/Record-of-Talk/2'? Can you answer it for me?" \


# multiple questions:  Qwen/Qwen3-32B, Qwen/QwQ-32B, deepseek-ai/DeepSeek-R1-Distill-Qwen-32B, deepseek-ai/DeepSeek-R1-Distill-Llama-70B, Qwen/Qwen2.5-72B-Instruct, GAIR/DeepResearcher-7b, Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen3-32B
  python scripts/DirectlyDR.py \
      --dataset_name scisafeeval \
      --split SciSafeEval \
      --concurrent_limit 32 \
      --api_base_url "http://localhost:8000/v1" \
      --model_name "GAIR/DeepResearcher-7b" \
      --aux_api_base_url "http://localhost:8000/v1" \
      --use_jina \
      --aux_model_name "GAIR/DeepResearcher-7b"


### SciSafeEval Pipeline
## 1. Collect Responses
# only process the first 10 samples
# python data/SciSafeEval_json/collect_scisafeeval_responses.py
#  --max-samples 500

## spilt to inference_reasoning and knowledge_retrieval
#python data/SciSafeEval_json/collect_scisafeeval_responses_inference_reasoning.py
# python data/SciSafeEval_json/collect_scisafeeval_responses_knowledge_retrieval.py


## 2. Filter Rejected Responses
# # use default threshold 0.8
# python data/SciSafeEval_json/filter_rejected_responses.py


## 3. Convert to DeepREJECT format
# python data/SciSafeEval_json/convert_to_deepreject_format.py