# 🏛 Immigrant Tax Filing Assistant
### AI-Powered Tax Guidance for International Students & Workers in the United States

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/pshah8011/immigrant-tax-assistant)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-orange)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.1-purple)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 Overview

The **Immigrant Tax Filing Assistant** is a production-grade Retrieval-Augmented Generation (RAG) system that provides accurate, cited tax guidance for non-resident aliens in the United States. Built for F-1, OPT, H-1B, and J-1 visa holders, the system answers tax questions by retrieving relevant IRS documentation and generating grounded, citation-mandatory responses.

The system combines **two core generative AI components**:
1. **Retrieval-Augmented Generation (RAG)** — dual-source knowledge base: 23 IRS PDFs + 4 IRS.gov HTML pages, 3,726 chunks, FAISS vector store
2. **Prompt Engineering** — citation-mandatory system prompt, CPA escalation layer, profile-aware context injection

**Live Demo:** [huggingface.co/spaces/pshah8011/immigrant-tax-assistant](https://huggingface.co/spaces/pshah8011/immigrant-tax-assistant)

---

## 🎯 Problem Statement

International students and workers face unique U.S. tax obligations — nonresident alien status, treaty benefits, FICA exemptions, Form 8843 requirements, and more. Generic tax tools don't address visa-specific rules, and professional CPAs are expensive. This system bridges that gap by providing free, accurate, IRS-grounded tax guidance tailored to each user's visa type and home country.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 💬 **Profile-Aware Chat** | Answers tailored to visa type, home country, years in US |
| 📚 **Citation-Mandatory** | Every answer cites specific IRS publications and page numbers |
| ⚠️ **CPA Escalation** | 24 complex topics auto-detected and redirected to professionals |
| 📅 **SPT Calculator** | Deterministic Substantial Presence Test — zero hallucination |
| 📋 **Filing Checklist** | Personalized checklist generated from your session |
| 🌍 **Multi-Country** | Treaty support: India, China, South Korea, Germany, Mexico, Canada, Japan |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────┐
│              USER INTERFACE (Gradio)                 │
│    Tab 1: Chat  │  Tab 2: SPT Calc  │  Tab 3: List  │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│              CPA ESCALATION LAYER                    │
│  24 keywords → refuses and redirects to CPA         │
│  100% precision — never answers out-of-scope        │
└──────────────────────────┬──────────────────────────┘
                           │ (safe to answer)
                           ▼
┌─────────────────────────────────────────────────────┐
│         PROFILE-AWARE RETRIEVAL ENGINE               │
│  Metadata filter (country + visa) → FAISS search    │
│  Smart re-ranking: 9 signal categories, 1.1x–1.6x  │
└──────────────────────────┬──────────────────────────┘
                           │ Top-5 chunks
                           ▼
┌─────────────────────────────────────────────────────┐
│            PROMPT ENGINEERING LAYER                  │
│  Profile + IRS context + citation-mandatory rules   │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│          Groq LLaMA 3.1 8B Instant                  │
│          temperature=0.0 · max_tokens=1024          │
└─────────────────────────────────────────────────────┘
```

---

## 📚 Knowledge Base — Dual Source

**3,726 total chunks from 2 source types**

### IRS PDFs (3,699 chunks)
| Category | Documents |
|---|---|
| Core Publications | Pub 519, 901, 515, 970, 525, 54, 596 |
| Forms | 1040-NR, 1042-S, 8843, W-8BEN, 4868, 8233, 8840 |
| Tax Treaties | India, China, South Korea, Germany, Mexico, Canada, Japan |

### IRS HTML Pages (27 chunks)
| Page | Chunks |
|---|---|
| IRS Web — Substantial Presence Test | 4 |
| IRS Web — Taxation of Nonresident Aliens | 8 |
| IRS Web — Tax Treaties Overview | 4 |
| IRS Web — NRA Figuring Your Tax | 11 |

> **Why both?** PDFs provide detailed legal text. HTML pages use simpler conversational language that matches how users ask questions — bridging the vocabulary gap between colloquial queries and formal IRS legal text.

---

## 📊 Evaluation Results

| Metric | Score | Target | Method |
|---|---|---|---|
| Retrieval Accuracy | **80%** (8/10) | ≥80% | Manual 10-question battery |
| Escalation Precision | **100%** (5/5) | 100% | CPA keyword detection tests |
| SPT Calculator | **100%** (3/3) | 100% | Deterministic Python unit tests |
| Faithfulness | **0.92** | ≥0.85 | LLM-as-judge (Groq) |
| Answer Relevancy | **0.89** | ≥0.85 | LLM-as-judge (Groq) |

---

## 🗂 Repository Structure

```
immigrant-tax-assistant/
│
├── app.py                        # Gradio app (3 tabs: Chat, SPT Calc, Checklist)
├── knowledge_base.py             # FAISS retrieval + metadata filtering + re-ranking
├── spt_calculator.py             # Deterministic SPT calculator (no LLM)
├── prompts.py                    # System prompt + CPA escalation (24 keywords)
├── requirements.txt              # Python dependencies
├── tax_assistant_pipeline.ipynb  # Complete Kaggle pipeline (Steps 1A+1B+1C+2+3)
└── README.md
```

> **Note:** Large binary files (`embeddings.npy`, `faiss_index.bin`, `metadata.json`, `chunks.json`) are excluded via `.gitignore`. Download from the HuggingFace Space files tab.

---

## 🚀 Local Setup

### Prerequisites
- Python 3.10+
- Free Groq API key from [console.groq.com](https://console.groq.com)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/shahparamk/immigrant-tax-assistant
cd immigrant-tax-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set Groq API key
export GROQ_API_KEY=your_groq_api_key_here

# 4. Download data files from HuggingFace
# Go to: https://huggingface.co/spaces/pshah8011/immigrant-tax-assistant/tree/main
# Download: embeddings.npy, faiss_index.bin, metadata.json,
#           chunks.json, filter_maps.json, pipeline_config.json
# Place all in same directory as app.py

# 5. Run
python app.py
```

### Rebuild Knowledge Base (Optional)

```bash
# Run tax_assistant_pipeline.ipynb on Kaggle
# Settings: GPU OFF, Internet ON
# Runtime: ~12 minutes
# Steps: 1A (download PDFs) + 1B (chunk PDFs) + 1C (scrape HTML) + 2 (embed) + 3 (RAG)
```

---

## 🔧 Key Design Decisions

**Why RAG over Fine-Tuning?**
Tax law changes yearly. RAG allows updating the knowledge base by replacing PDF/HTML sources without retraining. Fine-tuning would permanently bake in 2024 rules.

**Why dual-source (PDF + HTML)?**
PDFs provide detailed legal text. IRS.gov HTML pages use simpler language closer to how users ask questions — improving retrieval accuracy for colloquial queries.

**Why Groq LLaMA 3.1 8B?**
Free API tier, ~1-2 second inference, no OpenAI dependency, sufficient accuracy for citation-heavy structured responses.

**Why Temperature = 0.0?**
Tax answers must be deterministic. The same question must give the same answer every time. Creativity is a liability in financial guidance.

**Why CPA Escalation?**
FBAR, FATCA, crypto, state taxes — these require professional judgment. The system refuses rather than hallucinate, protecting users from incorrect advice.

---

## 🌍 Supported Profiles

**Visa Types:** F-1 · OPT · H-1B · J-1

**Countries with Treaty Support:** India · China · South Korea · Germany · Mexico · Canada · Japan

**Income Sources:** TA/RA Stipend · Fellowship · W-2 Employment · CPT/OPT Income · Scholarship

---

## 🔧 Technology Stack

| Component | Technology |
|---|---|
| LLM | Groq LLaMA 3.1 8B Instant |
| Embeddings | all-MiniLM-L6-v2 (384-dim) |
| Vector Store | FAISS IndexFlatIP |
| Orchestration | LangChain 0.3+ |
| PDF Parsing | PyMuPDF (fitz) |
| HTML Scraping | BeautifulSoup4 |
| Frontend | Gradio 5.25.0 |
| Deployment | HuggingFace Spaces |
| Training | Kaggle Notebooks |

---

## ⚠️ Ethical Considerations

- **Hallucination prevention** — temperature=0.0, citation-mandatory prompts, context-only rules
- **Scope limitation** — 24 escalation keywords redirect complex topics to licensed CPAs
- **Privacy** — no user data stored, sessions are ephemeral
- **Disclaimer** — every response reminds users this is not professional tax advice
- **Geographic bias** — 7 countries covered; users from other countries noted in UI
- **Copyright** — all IRS documents are U.S. government public domain publications

---

## 🔮 Future Improvements

- Treaty coverage for 50+ countries (currently 7)
- Real-time IRS publication updates via automated annual scraping
- Fix fellowship income retrieval gap (1042-S accuracy)
- Form auto-fill: pre-populate 1040-NR from conversation context
- Multi-language support (Hindi, Mandarin, Korean, Spanish)
- Voice interface using Whisper STT


## 📄 License

MIT License

---

<div align="center">
Built with LangChain · Groq · FAISS · Gradio · HuggingFace
<br>
<a href="https://huggingface.co/spaces/pshah8011/immigrant-tax-assistant">Live Demo</a> ·
<a href="https://shahparamk.github.io/immigrant-tax-assistant">Web Page</a>
</div>
