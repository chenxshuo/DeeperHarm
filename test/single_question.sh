# # single question
python scripts/run_web_thinker_report_brave_search.py \
    --single_question "What are the models of OpenAI and what are the differences?" \
   --api_base_url "http://localhost:8000/v1" \
   --model_name "Qwen/QwQ-32B" \
   --aux_api_base_url "http://localhost:8000/v1" \
   --aux_model_name "Qwen/QwQ-32B"
