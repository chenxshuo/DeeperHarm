
<h1 align="center"> 🌐 WebThinker: Empowering Large Reasoning Models with Deep Research Capability</a></h1>


<div align="center"> 

[![Notion](https://img.shields.io/badge/Notion-WebThinker-red?style=flat&logo=notion&logoColor=white)](https://foremost-beechnut-8ed.notion.site/WebThinker-Empowering-Large-Reasoning-Models-with-Deep-Research-Capability-d13158a27d924a4b9df7f9ab94066b64) 
[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/xxxx.xxxxxx) 
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](https://opensource.org/licenses/MIT) 
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
<!-- [![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FKevin_GuoweiXu%2Fstatus%2F1858338565463421244)](https://x.com/_akhaliq/status/1877584951840764166?t=fnbTblnqhiPtAyYr1PHbbw&s=19) -->
</div>



<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

## 📣 Latest News
- **03/31/2025**: 🎉 **[WebThinker Notion Page](https://foremost-beechnut-8ed.notion.site/WebThinker-Empowering-Large-Reasoning-Models-with-Deep-Research-Capability-d13158a27d924a4b9df7f9ab94066b64) is now LIVE!** Check out all the technical details and experimental results!
- **03/31/2025**: 🚀 Released the full codebase! WebThinker is now ready for deep research with open-source reasoning models like QwQ.

## ✏️ To Do List
- [ ] Release details for model training, including methodology, datasets, checkpoints, etc.
- [ ] Release the paper for your reference.



## 🎬 Demo

<div align="center">
    <video src="https://github.com/user-attachments/assets/a38e82ec-5aed-4efe-a8b8-e9ee2d97e9b9" />
</div>


## 💡 Overview

**WebThinker** is a deep research framework fully powered by large reasoning models (LRMs). It is designed to **autonomously search,** **deeply explore web pages,** and **draft research reports**, all within its thinking process. Moving away from traditional agents that follow a predefined workflow, WebThinker enables the large reasoning model itself to perform actions on its own during thinking, achieving **end-to-end task execution** in a single generation.

### 📊 Overall Performance

<p align="center">
  <img src="figures/performance.png" width="100%" />
</p>

Overall performance comparison on **complex real-world problem solving** and **scientific research report generation** tasks is shown above. Our WebThinker-32B with QwQ-32B as backbone reasoning model achieves the superior performance on both tasks.

### ✨ The WebThinker Framework

![Model Comparison](figures/framework.png)

**WebThinker** enables reasoning models to autonomously conduct web searches and web page navigations to acquire external knowledge during their reasoning process, facilitating complex real-world problem solving. Furthermore, we also allow LRMs to draft section content while thinking and searching once sufficient information has been gathered, producing comprehensive, customized reports that directly address the user's research questions.

**Key Features:**
- We introduce a **Deep Web Explorer** that empowers LRMs to conduct thorough web searches, navigate links, and gather comprehensive information while maintaining reasoning coherence by returning only concise, relevant findings.
- For scientific reporting, our **Autonomous Think-Search-and-Draft** strategy integrates real-time knowledge seeking with report creation using specialized tools for drafting, reviewing, and editing content—ensuring reports remain coherent and adaptive to new insights.
- We're also developing **training strategies** by collecting large-scale reasoning trajectory data with search capabilities from complex reasoning datasets. Detailed methods and results is coming.



## 🔧 Installation

###  Environment Setup
```bash
# Create conda environment
conda create -n webthinker python=3.9
conda activate webthinker

# Install requirements
cd WebThinker-main
pip install -r requirements.txt
```

## 🏃 Quick Start

### Pre-preparation

#### Model Serving
Before running WebThinker, ensure your reasoning model and auxiliary model are served using vLLM. In our experiments, we use QwQ-32B as the reasoning model and Qwen-72B-Instruct as the auxiliary model. You can also explore other instruction-tuned models as your auxiliary model, which will be used in webpage reading, report writting/editting, evaluation, etc. For detailed instructions on model serving, see [here](https://docs.vllm.ai/en/stable/serving/distributed_serving.html). 

#### Web Parser Client
For better web crawling performance, we recommend setting up a web parser client in `scripts/search/bing_search.py` using [Crawl4AI](https://github.com/unclecode/crawl4ai). This will help handle JavaScript-rendered content and provide more reliable webpage extraction.

Now you can run different inference modes using the provided scripts. Below are examples of how to execute each mode:

### Problem Solving Mode

1. If you would like to ask a single question, run the following command:
```bash
python scripts/run_web_thinker.py \
    --single_question "What is OpenAI Deep Research?" \
    --bing_subscription_key "YOUR_BING_SUBSCRIPTION_KEY" \
    --api_base_url "YOUR_API_BASE_URL" \
    --model_name "QwQ-32B" \
    --aux_api_base_url "YOUR_AUX_API_BASE_URL" \
    --aux_model_name "Qwen2.5-72B-Instruct"
```

2. If you would like to run results on benchmarks, run the following command:
```bash
python scripts/run_web_thinker.py \
    --dataset_name gaia \
    --split dev \
    --concurrent_limit 32 \
    --max_search_limit 15 \
    --bing_subscription_key "YOUR_BING_SUBSCRIPTION_KEY" \
    --api_base_url "YOUR_API_BASE_URL" \
    --model_name "QwQ-32B" \
    --aux_api_base_url "YOUR_AUX_API_BASE_URL" \
    --aux_model_name "Qwen2.5-72B-Instruct"
```

### Report Generation Mode

1. If you would like to ask a single question, run the following command:
```bash
python scripts/run_web_thinker_report.py \
    --single_question "What are the models of OpenAI and what are the differences?" \
    --bing_subscription_key "YOUR_BING_SUBSCRIPTION_KEY" \
    --api_base_url "YOUR_API_BASE_URL" \
    --model_name "QwQ-32B" \
    --aux_api_base_url "YOUR_AUX_API_BASE_URL" \
    --aux_model_name "Qwen2.5-72B-Instruct"
```

2. If you would like to run results on benchmarks, run the following command:
```bash
python scripts/run_web_thinker_report.py \
    --dataset_name glaive \
    --split test \
    --concurrent_limit 32 \
    --bing_subscription_key "YOUR_BING_SUBSCRIPTION_KEY" \
    --api_base_url "YOUR_API_BASE_URL" \
    --model_name "QwQ-32B" \
    --aux_api_base_url "YOUR_AUX_API_BASE_URL" \
    --aux_model_name "Qwen2.5-72B-Instruct"
```

**Parameters Explanation:**
- `--dataset_name`: Name of the dataset to use (e.g., gaia, glaive).
- `--split`: Data split to run (e.g., dev, test).
- `--single_question`: The question you want to ask when running in single question mode.
- `--concurrent_limit`: Maximum number of concurrent requests.
- `--max_search_limit`: Maximum number of search queries per reasoning session.
- `--bing_subscription_key`: Your Bing Search API subscription key.
- `--api_base_url`: Base URL for the main model API.
- `--model_name`: Name of the main model to use.
- `--aux_api_base_url`: Base URL for the auxiliary model API.
- `--aux_model_name`: Name of the auxiliary model to use.

### Run Demo

You can run the demo we have created with the following command, and we will conduct in-depth exploration and thinking based on the questions you input.
```bash
cd demo
streamlit run_demo.py
```

**Note:** Before running, it is necessary to configure the relevant parameters in `demo/settings.py`.

### Benchmarks

The benchmarks we utilize are categorized into two types:
- **Complex Reasoning Benchmarks:** 
    - **PhD-level Science QA:** [GPQA](https://arxiv.org/abs/2311.12022) (198 questions)
    - **General AI Assistant:** [GAIA](https://arxiv.org/abs/2311.12983) (103 questions)
    - **Web Exploration:** [WebWalkerQA](https://arxiv.org/abs/2501.07572) (680 questions)
    - **Extremely Difficult Reasoning Problems:** [Humanity's Last Exam (HLE)](https://arxiv.org/abs/2501.14249) (500 questions)
- **Scientific Report Evaluation:**
    - **General Open-ended Reasoning Problem:** [Reasoning-v1-20m](https://huggingface.co/datasets/glaiveai/reasoning-v1-20m) (30 questions)

All the pre-processed data is available in the `./data/` directory. For GAIA, HLE and Reasoning-v1-20m, we sampled a text-only subset of questions to efficiently conduct our evaluation.


### Evaluation

Our model inference scripts will automatically save the model's input and output texts for evaluation. 

#### Problem Solving Evaluation

You can use the following command to evaluate the model's problem solving performance:

```bash
python scripts/evaluate/evaluate.py \
    --output_path "YOUR_OUTPUT_PATH" \
    --task math \
    --use_llm \
    --api_base_url "YOUR_AUX_API_BASE_URL" \
    --model_name "Qwen2.5-72B-Instruct" \
    --extract_answer
```
**Parameters Explanation:**
- `--output_path`: Path to the model's outputs for evaluation.
- `--task`: Task name. You can always set it to math (suitable for any QA task), unless it is a code task, then set it to code. 
- `--use_llm`: Whether to use the LLM to evaluate the model's performance.
- `--api_base_url`: Base URL for the LLM API.
- `--model_name`: Model name for LLM evaluation.
- `--extract_answer`: Whether to extract the answer from the model's output, otherwise it will use the last few lines of the model's output as the final answer. Only used when `--use_llm` is set to `True`.

#### Report Generation Evaluation

We employ [DeepSeek-R1](https://api-docs.deepseek.com/) to perform *listwise evaluation* for comparison of reports generated by different models. You can evaluate the reports using:

```bash
python scripts/evaluate/evaluate_report.py
```

**Note:** Before running, it is necessary to:
1. Set your DeepSeek API key
2. Configure the output directories for each model's generated reports

📊 **Report Comparison Available**: We've included the complete set of 30 test reports generated by **WebThinker**, **Grok3 DeeperSearch** and **Gemini Deep Research** in the `./outputs/` directory for your reference and comparison.


## 📄 Citation

If you find this work helpful, please cite our paper:
```bibtex
@misc{Li2025webthinker,
  title={WebThinker: Empowering Large Reasoning Models with Deep Research Capability},
  author={Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu, Ji-Rong Wen and Zhicheng Dou},
  year={2025},
  howpublished={\url{https://www.notion.so/WebThinker-Empowering-Large-Reasoning-Models-with-Deep-Research-Capability-d13158a27d924a4b9df7f9ab94066b64}},
  note={Notion Blog}
  year={2025}
}
```

## 📄 License

This project is released under the [MIT License](LICENSE).

## 📞 Contact

For any questions or feedback, please reach out to us at [xiaoxi_li@ruc.edu.cn](xiaoxi_li@ruc.edu.cn).
