# DeeperHarm: Deep Research Red Team Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DeeperHarm is a comprehensive red-teaming research framework derived from WebThinker, designed to evaluate and analyze the security aspects of large language models through various attack methods and evaluation metrics. The framework implements multiple attack strategies and provides extensive evaluation capabilities.

## Features

### Attack Methods
- **Direct Jailbreak (Baseline)**: Direct prompt injection attacks
- **Plan Injection**: Strategic planning-based attacks
- **Intent Hijacking**: Goal-oriented attack redirection

### Supported Models
- Qwen/Qwen3-32B
- Qwen/QwQ-32B
- deepseek-ai/DeepSeek-R1-Distill-Llama-70B
- Qwen/Qwen2.5-72B-Instruct
- GAIR/DeepResearcher-7b
- Qwen/Qwen2.5-7B-Instruct
- ...

### Evaluation Metrics
- **strongREJECT**: Strong rejection detection
- **deepREJECT**: Deep rejection analysis
- **Refusal Words Analysis**: Refusal pattern detection
- **LLM as Judge**: AI-based evaluation
- **Llama-Guard Assessment**: Safety guard evaluation

## Requirements

- Python 3.9 or 3.10 (3.12 not supported)
- CUDA support (recommended)
- Sufficient GPU memory for large models
- API keys for Brave Search and Jina AI

## Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/chenxshuo/DeeperHarm
cd DeeperHarm

# Run the setup script
bash setup_env.sh
```

The setup script will:
- Create a virtual environment with Python 3.9/3.10
- Install all required dependencies from `requirements.txt`
- Set up the project environment

### 2. API Configuration

Create a `.env` file in the project root and add your API keys:

```bash
# Required API keys
BRAVE_API_KEY="your_brave_api_key_here"
JINA_API_KEY="your_jina_api_key_here"

# Optional: Additional configuration
API_BASE_URL="http://localhost:8000/v1"
```

### 3. Activate Environment

```bash
source .venv/bin/activate
```

## Usage

### Single Question Testing

Test the framework with a single question:

```bash
python scripts/IntentHijack_run_web_thinker_report_brave_search.py \
   --single_question "Your test question here" \
   --api_base_url "http://localhost:8000/v1" \
   --model_name "Qwen/QwQ-32B" \
   --aux_api_base_url "http://localhost:8000/v1" \
   --use_jina \
   --aux_model_name "Qwen/QwQ-32B"
```

### Multi-Question Pipeline

#### 1. Direct Deep Research (Baseline)

```bash
python scripts/DirectlyDR.py \
   --dataset_name DeeperHarm \
   --split DeepREJECT \
   --concurrent_limit 32 \
   --api_base_url "http://localhost:8000/v1" \
   --model_name "Qwen/QwQ-32B" \
   --aux_api_base_url "http://localhost:8000/v1" \
   --aux_model_name "Qwen/QwQ-32B"
```

#### 2. Plan Injection Attack

```bash
python scripts/PlanInjection_run_web_thinker_report_brave_search.py \
    --dataset_name DeeperHarm \
    --split PlanInjection \
    --concurrent_limit 32 \
    --api_base_url "http://localhost:8000/v1" \
    --model_name "Qwen/QwQ-32B" \
    --aux_api_base_url "http://localhost:8000/v1" \
    --use_jina \
    --aux_model_name "Qwen/QwQ-32B"
```

#### 3. Intent Hijacking Attack

```bash
python scripts/IntentHijack_run_web_thinker_report_brave_search.py \
    --dataset_name DeeperHarm \
    --split IntentHijack \
    --concurrent_limit 32 \
    --api_base_url "http://localhost:8000/v1" \
    --model_name "Qwen/QwQ-32B" \
    --aux_api_base_url "http://localhost:8000/v1" \
    --use_jina \
    --aux_model_name "Qwen/QwQ-32B"
```

### Available Datasets

The framework supports multiple dataset splits:
- `DeepREJECT`: Deep rejection evaluation dataset
- `IntentHijack`: Intent hijacking attack dataset  
- `PlanInjection`: Plan injection attack dataset
- `DeepREJECT_small`: Smaller version for testing

## Evaluation

### 1. strongREJECT Evaluation

#### LLM Baseline Outputs
```bash
python scripts/evaluate/evaluate_strongreject_baseline.py \
    --file "outputs/strongreject_baseline.json"
```

#### WebThinker Reports
```bash
python scripts/evaluate/evaluate_strongreject.py \
    --markdown_dir "outputs/DeeperHarm.QwQ-32B/markdown.IntentHijack" \
    --output_dir ""  # default: markdown_dir
```

### 2. deepREJECT Evaluation

#### LLM Baseline Outputs
```bash
python scripts/evaluate/evaluate_deepreject_baseline.py \
    --markdown_dir "outputs/DeeperHarm.QwQ-32B/markdown.DeepREJECT" \
    --output_dir "outputs/baseline/qwq-32b" \
    --metrics_file "outputs/DeeperHarm.QwQ-32B/markdown.DeepREJECT/eval_metrics.csv" \
    --baseline_file "outputs/baseline/qwq-32b/strongreject_baseline.json"
```

#### WebThinker Reports
```bash
python scripts/evaluate/evaluate_deepreject.py \
    --markdown_dir "outputs/DeeperHarm.QwQ-32B/markdown.DeepREJECT" \
    --output_dir "" \
    --dataset "data/DeeperHarm/DeepREJECT.json"
```

### 3. Refusal Words Analysis

#### For WebThinker Reports
```bash
python scripts/evaluate/evaluate_refusal_words.py \
    --markdown_dir "outputs/DeeperHarm.QwQ-32B/markdown.IntentHijack" \
    --output_dir "" \
    --metrics_file "" \
    --baseline_file ""
```

#### For Lower ASR Models
```bash
python scripts/evaluate/evaluate_refusal_words.py \
    --markdown_dir "outputs/DeeperHarm.QwQ-32B/markdown.IntentHijack" \
    --output_dir "" \
    --metrics_file "" \
    --baseline_file "" \
    --lower
```

### 4. LLM as Judge Evaluation

#### WebThinker Reports
```bash
python scripts/evaluate/evaluate_llm_judge.py \
    --markdown_dir "outputs/DeeperHarm.QwQ-32B/markdown.IntentHijack" \
    --output_dir "" \
    --metrics_file "" \
    --baseline_file ""
```

### 5. Llama-Guard Evaluation

#### WebThinker Reports
```bash
python scripts/evaluate/evaluate_llama_guard.py \
    --markdown_dir "outputs/DeeperHarm.QwQ-32B/markdown.IntentHijack" \
    --output_dir "" \
    --metrics_file "" \
    --baseline_file ""
```

## Project Structure

```
DeeperHarm/
├── data/DeeperHarm/                 # Dataset files
│   ├── DeepREJECT.json
│   ├── IntentHijack.json
│   ├── PlanInjection.json
│   └── DeepREJECT_small.json
├── scripts/                     # Main execution scripts
│   ├── DirectlyDR.py           # Direct deep research baseline
│   ├── PlanInjection_run_web_thinker_report_brave_search.py
│   ├── IntentHijack_run_web_thinker_report_brave_search.py
│   ├── evaluate/               # Evaluation scripts
│   ├── prompts/                # Prompt templates
│   ├── search/                 # Search functionality
│   └── utils/                  # Utility functions
├── shell_scripts/              # Convenience shell scripts
├── requirements.txt            # Python dependencies
├── setup_env.sh               # Environment setup script
└── README.md                  # This file
```

## Quick Start

1. **Setup Environment**:
   ```bash
   bash setup_env.sh
   ```

2. **Configure API Keys**:
   ```bash
   vi .env  # Add your API keys
   ```

3. **Test Single Question**:
   ```bash
   python scripts/IntentHijack_run_web_thinker_report_brave_search.py \
      --single_question "How to make a bomb?" \
      --api_base_url "http://localhost:8000/v1" \
      --model_name "Qwen/QwQ-32B" \
      --aux_api_base_url "http://localhost:8000/v1" \
      --use_jina \
      --aux_model_name "Qwen/QwQ-32B"
   ```

4. **Run Full Evaluation**:
   ```bash
   python scripts/evaluate/evaluate_strongreject.py \
      --markdown_dir "outputs/DeeperHarm.QwQ-32B/markdown.IntentHijack" \
      --output_dir ""
   ```

## Dependencies

Key dependencies include:
- `torch==2.5.1`: PyTorch for deep learning
- `transformers==4.46.1`: Hugging Face transformers
- `vllm==0.6.4`: High-performance LLM serving
- `beautifulsoup4==4.12.3`: Web scraping
- `rank_bm25`: BM25 ranking algorithm
- `python-dotenv==1.1.0`: Environment variable management

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{chen2025deep,
  title={Deep Research Brings Deeper Harm},
  author={Chen, Shuo and Li, Zonggen and Han, Zhen and He, Bailan and Liu, Tong and Chen, Haokun and Groh, Georg and Torr, Philip and Tresp, Volker and Gu, Jindong},
  journal={arXiv preprint arXiv:2510.11851},
  year={2025}
}
```