# 🏛 Immigrant Tax Filing Assistant
### AI-Powered Tax Guidance for International Students & Workers in the United States

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/pshah8011/immigrant-tax-assistant)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-orange)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.1-purple)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 Overview

The **Immigrant Tax Filing Assistant** is a production-grade Retrieval-Augmented Generation (RAG) system that provides accurate, cited tax guidance for non-resident aliens in the United States. Built for F-1, OPT, H-1B, and J-1 visa holders, the system answers tax questions by retrieving relevant IRS documentation and generating grounded, citation-mandatory responses using a large language model.

The system combines **two core generative AI components** as required by the assignment:
1. **Retrieval-Augmented Generation (RAG)** — 23 IRS documents, 3,699 chunks, FAISS vector store
2. **Prompt Engineering** — Citation-mandatory system prompt, CPA escalation layer, profile-aware context injection

**Live Demo:** [huggingface.co/spaces/pshah8011/immigrant-tax-assistant](https://huggingface.co/spaces/pshah8011/immigrant-tax-assistant)

---

## 🎯 Problem Statement

International students and workers face unique and complex U.S. tax obligations — nonresident alien status, treaty benefits, FICA exemptions, Form 8843 requirements, and more. Generic tax tools don't address visa-specific rules, and professional CPAs are expensive. This system bridges that gap by providing free, accurate, IRS-grounded tax guidance tailored to each user's visa type and home country.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 💬 **Profile-Aware Chat** | Answers tailored to visa type, home country, years in US |
| 📚 **Citation-Mandatory** | Every answer cites specific IRS publications and page numbers |
| ⚠️ **CPA Escalation** | 24 complex topics auto-detected and redirected to professionals |
| 📅 **SPT Calculator** | Deterministic Substantial Presence Test — zero hallucination |
| 📋 **Filing Checklist** | Personalized checklist generated from your conversation session |
| 🌍 **Multi-Country** | Specific treaty support for India, China, South Korea, Germany, Mexico, Canada, Japan |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Gradio)                   │
│          Tab 1: Chat  │  Tab 2: SPT Calc  │  Tab 3: Checklist│
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   CPA ESCALATION LAYER                       │
│   24 keywords (FBAR, crypto, state tax, audit, FATCA...)    │
│   100% precision — system refuses to answer out-of-scope    │
└──────────────────────────┬──────────────────────────────────┘
                           │ (if safe to answer)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PROFILE-AWARE RETRIEVAL ENGINE                  │
│                                                             │
│  ┌─────────────────┐    ┌──────────────────────────────┐   │
│  │ Metadata Filter  │    │    Query Enrichment           │   │
│  │ Country + Visa   │    │    visa_type + country        │   │
│  │ Filtering        │    │    + years_in_us              │   │
│  └────────┬────────┘    └──────────────┬───────────────┘   │
│           │                            │                     │
│           ▼                            ▼                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              FAISS IndexFlatIP                        │   │
│  │         3,699 vectors · 384 dimensions               │   │
│  │         all-MiniLM-L6-v2 embeddings                  │   │
│  └──────────────────────────────────────────────────────┘   │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           SMART RE-RANKING LAYER                      │   │
│  │   9 signal categories · Source specificity boosts    │   │
│  │   Treaty boost · Form boost · Fellowship boost       │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │ Top-5 chunks
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  PROMPT ENGINEERING LAYER                    │
│                                                             │
│  System Prompt: Citation-mandatory, context-only rules      │
│  User Profile injection: visa + country + income            │
│  Retrieved IRS context: up to 5 chunks (600 chars each)     │
│  Session history: last 3 conversation turns                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              GROQ LLaMA 3.1 8B Instant                      │
│         Temperature: 0.0 · Max tokens: 1,024                │
│         Fast inference (~1-2 seconds)                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                  Cited Answer + Sources
```

---

## 📚 Knowledge Base

**23 IRS Documents — 3,699 Chunks — Avg 1,142 chars/chunk**

### Core Publications
| Document | Description | Topics |
|---|---|---|
| IRS Publication 519 | U.S. Tax Guide for Aliens | SPT, FICA, residency, filing |
| IRS Publication 901 | U.S. Tax Treaties | Treaty benefits, student exemptions |
| IRS Publication 515 | Withholding on Nonresidents | Withholding rates, Chapter 3 |
| IRS Publication 970 | Tax Benefits for Education | Fellowship, scholarship taxation |
| IRS Publication 525 | Taxable and Nontaxable Income | Income classification |
| IRS Publication 54 | Tax Guide for U.S. Citizens Abroad | Foreign income |
| IRS Publication 596 | Earned Income Credit | EIC eligibility |

### Forms & Instructions
| Form | Description |
|---|---|
| Form 1040-NR Instructions | Nonresident Alien Tax Return |
| Form 1042-S Instructions | Foreign Person U.S. Source Income |
| Form 8843 | Statement for Exempt Individuals |
| Form W-8BEN Instructions | Certificate of Foreign Status |
| Form 4868 | Extension of Time to File |
| Form 8233 | Exemption from Withholding |
| Form 8840 | Closer Connection Exception |

### Tax Treaties
India · China · South Korea · Germany · Mexico · Canada · Japan

---

## 📊 Evaluation Results

| Metric | Score | Target | Method |
|---|---|---|---|
| Retrieval Accuracy | **80%** (8/10) | ≥80% | Manual 10-question battery |
| Escalation Precision | **100%** (5/5) | 100% | CPA keyword detection |
| SPT Calculator | **100%** (3/3) | 100% | Deterministic Python |
| Faithfulness | **0.92** | ≥0.85 | LLM-as-judge (Groq) |
| Answer Relevancy | **0.89** | ≥0.85 | LLM-as-judge (Groq) |

### Retrieval Accuracy Detail (10-test battery)
```
✓ IRS Publication 519   — Substantial Presence Test query
✓ IRS Publication 519   — FICA exemption query
✓ Form 1040-NR          — Filing forms query
✓ US-India Tax Treaty   — Article 21 exemption query
✗ Form 1042-S           — Fellowship taxation (Pub 970 wins — knowledge gap)
✓ Form W-8BEN           — Certificate of foreign status
✓ Form 4868             — Filing extension
✓ Form 8840             — Closer connection exception
✓ IRS Publication 970   — Scholarship income
✓ IRS Publication 515   — Withholding rates
```

---

## 🗂 Repository Structure

```
immigrant-tax-assistant/
│
├── app.py                       # Main Gradio application (3 tabs)
│   ├── Tab 1: Tax Q&A Chat
│   ├── Tab 2: SPT Calculator
│   └── Tab 3: Filing Checklist
│
├── knowledge_base.py            # TaxKnowledgeBase class
│   ├── FAISS vector retrieval
│   ├── Metadata filtering (country/visa)
│   └── Smart re-ranking with signal boosts
│
├── spt_calculator.py            # Deterministic SPT calculator
│   └── Pure Python — no LLM, no hallucination
│
├── prompts.py                   # Prompt engineering layer
│   ├── Citation-mandatory system prompt
│   ├── CPA escalation keywords (24)
│   └── RAG prompt builder
│
├── requirements.txt             # Python dependencies
├── tax_assistant_pipeline.ipynb # Complete Kaggle pipeline notebook
└── README.md                    # This file
```

> **Note:** Large binary files (`embeddings.npy`, `faiss_index.bin`, `metadata.json`, `chunks.json`) are excluded via `.gitignore`. Download from the HuggingFace Space files tab.

---

## 🚀 Local Setup

### Prerequisites
- Python 3.10+
- Free Groq API key from [console.groq.com](https://console.groq.com)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/pshah8011/immigrant-tax-assistant
cd immigrant-tax-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set Groq API key
export GROQ_API_KEY=your_groq_api_key_here

# 4. Download data files from HuggingFace
# Go to: https://huggingface.co/spaces/pshah8011/immigrant-tax-assistant/tree/main
# Download: embeddings.npy, faiss_index.bin, metadata.json,
#           chunks.json, filter_maps.json, pipeline_config.json
# Place all files in the same directory as app.py

# 5. Run the app
python app.py
```

### Rebuild the Knowledge Base (Optional)

```bash
# Run the complete pipeline notebook on Kaggle
# File: tax_assistant_pipeline.ipynb
# Settings: GPU OFF, Internet ON
# Runtime: ~10 minutes
```

---

## 🔧 Key Design Decisions

### Why RAG over Fine-Tuning?
Tax law changes yearly. RAG allows updating the knowledge base by replacing PDF files without retraining any model. Fine-tuning would permanently bake in 2024 rules — RAG stays current with new IRS publications.

### Why Groq LLaMA 3.1 8B?
Free API tier, ~1-2 second inference, sufficient accuracy for citation-heavy structured responses. No OpenAI dependency means zero cost for the end user.

### Why Temperature = 0.0?
Tax answers must be deterministic and reproducible. The same question must give the same answer every time. Creativity is a liability in financial guidance.

### Why CPA Escalation Architecture?
FBAR, FATCA, crypto income, state taxes, dual-status returns — these require professional judgment beyond what any AI system should attempt. The system refuses rather than hallucinate, protecting users from potentially costly incorrect advice.

### Why Topic Enrichment on Chunks?
Raw IRS text uses formal legal language. By prepending semantic keywords to each chunk before embedding ("FICA social security medicare tax exempt nonresident alien student"), retrieval accuracy improved significantly for colloquial user queries.

---

## 🌍 Supported Profiles

**Visa Types:** F-1 · OPT · H-1B · J-1

**Countries with Treaty Support:** India · China · South Korea · Germany · Mexico · Canada · Japan

**Income Sources:** TA/RA Stipend · Fellowship · W-2 Employment · CPT/OPT Income · Scholarship

---

## ⚠️ Ethical Considerations

**Accuracy & Hallucination Prevention**
The system uses temperature=0.0, citation-mandatory prompts, and context-only rules to minimize hallucination. All answers must cite specific IRS sources.

**Scope Limitation**
24 escalation keywords ensure complex topics (FBAR, crypto, state taxes, audits) are redirected to licensed CPAs. The system knows its limits.

**Disclaimer**
Every response includes: *"This tool provides general tax information. It is NOT a substitute for professional tax advice."*

**Privacy**
No user data is stored or logged. Conversations are session-only and cleared on page refresh.

**Bias Considerations**
The knowledge base covers 7 countries representing the largest international student populations in the US. Users from other countries receive general guidance based on IRS Publication 519.

---

## 🔮 Future Improvements

- Add IRS Publication 901 treaty coverage for 50+ countries
- Integrate real-time IRS publication updates via web scraping
- Add Form auto-fill capability (pre-fill 1040-NR from conversation)
- Support for dual-status return guidance (with CPA oversight)
- Multi-language support for non-English speakers
- Voice interface for accessibility

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ using LangChain · Groq · FAISS · Gradio · HuggingFace
<br>
<a href="https://huggingface.co/spaces/pshah8011/immigrant-tax-assistant">Live Demo</a> ·
<a href="https://github.com/pshah8011/immigrant-tax-assistant">GitHub</a>
</div>
