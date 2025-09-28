# Functional Tool Calling Evaluation of Large Language Models

This repository contains the source code and evaluation framework for the research paper, **"Functional Tool Calling Evaluation of Large Language Models."**

This project provides a structured and automated environment for benchmarking the functional tool-calling capabilities of various state-of-the-art Large Language Models (LLMs) on a suite of mathematical problems.

---

## 📝 Abstract

This paper presents a comprehensive benchmarking study of functional tool calling in state-of-the-art large language models (LLMs) using a customized math agent environment. We systematically evaluate five prominent models: Gemini 2.5 Flash, Mistral Nemo, GPT OBSS, Nvidia/OpenRouter, and Google Gemini 2.0 Flash Lite. These models are tested against a carefully curated suite of 60 diverse mathematical problems, which span basic arithmetic, advanced statistics, trigonometry, linear algebra, and various corner cases. Our evaluation metrics include accuracy, tool selection, latency, and a critical analysis of estimated financial costs. The results reveal significant variation in model capabilities. While OpenRouter achieves top accuracy, the cost analysis shows that models like Mistral Nemo and GPT OBSS are substantially more economical, highlighting a crucial trade-off between performance and operational expense.

---

## 🤖 Models Evaluated

This study benchmarks the following models and providers:

| Provider   | Model Name         |
| :--------- | :----------------- |
| OpenRouter | Nvidia             |
| GPT        | OBSS               |
| Gemini     | 2.5 Flash          |
| Mistral    | Nemo               |
| Gemini     | 2.0 Flash Lite     |

---

## 🔧 Setup and Installation

To run this evaluation framework, you will need to set up the necessary environment and install the required dependencies.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Configure API Keys:**
    To run the evaluation, you will need API keys for the different LLM providers. Create a `.env` file in the root directory and add your keys:
    ```
    OPENROUTER_API_KEY="your-openrouter-key"
    GEMINI_API_KEY="your-gemini-key"
    # Add other keys as needed
    ```

---

## 🚀 Usage: How to Reproduce Results

The main script for running the evaluation is `evaluate_math_agent.py`.

To run the full evaluation suite across all models as described in the paper, execute the following command:

```bash
python evaluate_math_agent.py
```

The script will run the 60 mathematical test queries against each configured model and save the detailed results, including performance metrics and cost estimates, to an output file.

---

## 📂 Repository Structure

```
.
├── evaluate_math_agent.py   # Main script to run the evaluation
├── math_agent/              # Directory for the agent logic and tool definitions
│   ├── __init__.py
│   ├── agent.py             # Core ReAct agent logic using LangGraph
│   └── tools.py             # Mathematical tools with the @tool decorator
├── test_suite/              # Directory containing the test queries
│   └── math_problems.json   # JSON file with the 60 test problems
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 📜 Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{emad2025functional,
  title={Functional Tool Calling Evaluation of Large Language Models},
  author={Emad, Youssef},
  journal={Journal of AI Research (Preprint)},
  year={2025}
}
```

---

## 🙏 Acknowledgments

We gratefully acknowledge the open-source community and the platform providers who made this research possible. Special thanks to Eng Ahmed Khaled for their support.

---

## ⚖️ License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
