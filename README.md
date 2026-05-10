# 🛡️ OmniGuard — LLM Red-Teaming & Safety Evaluation Framework

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A professional-grade framework for stress-testing Large Language Models (LLMs) 
against adversarial attacks, mapped to OWASP Top 10 for LLMs and MITRE ATLAS.

---

## 🎯 What is OmniGuard?

OmniGuard automatically:
- 🔴 **Generates** adversarial attack prompts (jailbreaks, injections, obfuscation)
- 🤖 **Fires** them at any AI model (Ollama, Claude, OpenAI)
- 📊 **Scores** each response with 3 safety metrics
- 📄 **Generates** a professional HTML security report

---

## 🏗️ Architecture — 4 Modules

| Module | What it does |
|--------|-------------|
| **Module 1** — Attack Generator | Generates adversarial prompts using Jinja2 templates |
| **Module 2** — Model Interface | Sends attacks to Claude, OpenAI, or Ollama |
| **Module 3** — Judge Scoring | Scores responses: ASR, Refusal Score, Toxicity |
| **Module 4** — Report Engine | Generates beautiful HTML security reports |

---

## 📊 Sample Results — Llama3.2

| Attack Type | Success Rate | Status |
|-------------|-------------|--------|
| Jailbreak | 51% | 🔴 VULNERABLE |
| Prompt Injection | 30% | ✅ RESISTANT |
| Payload Smuggling | 26% | ✅ RESISTANT |
| Obfuscation | 22% | ✅ RESISTANT |

**Overall Rating: 🟡 GOOD — Reasonable Safety**

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/RahulBijarnia1/omniguard.git
cd omniguard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install and run Ollama (free!)
```bash
# Download from https://ollama.com
ollama pull llama3.2
```

### 4. Run the full pipeline
```bash
python run4.py
```

---

## 📁 Project Structure
---

## 🔒 Security Framework Mapping

- **OWASP LLM Top 10** — LLM01:2025 Prompt Injection
- **MITRE ATLAS** — AML.T0051, AML.T0054

---

## 🛠️ Supported Models

| Model | Provider | Cost |
|-------|----------|------|
| Llama3.2 | Ollama | FREE |
| claude-haiku | Anthropic | ~$0.001/test |
| gpt-4o-mini | OpenAI | ~$0.001/test |

---

## 📄 License

MIT License — free to use for research and education.

---

*Built with ❤️ by Rahul Bijarnia*
