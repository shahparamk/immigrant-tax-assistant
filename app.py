import gradio as gr
import json
import numpy as np
import faiss
import os
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ============================================================
# LOAD ALL ASSETS
# ============================================================
BASE_DIR = Path(__file__).parent

with open(BASE_DIR / 'metadata.json', encoding='utf-8') as f:
    chunks = json.load(f)

with open(BASE_DIR / 'filter_maps.json') as f:
    filter_maps = json.load(f)

with open(BASE_DIR / 'pipeline_config.json') as f:
    config = json.load(f)

embeddings = np.load(str(BASE_DIR / 'embeddings.npy'))
index = faiss.read_index(str(BASE_DIR / 'faiss_index.bin'))
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

country_idx = {k: v for k, v in filter_maps['country_to_indices'].items()}
visa_idx = {k: v for k, v in filter_maps['visa_to_indices'].items()}

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
llm = ChatGroq(
    model='llama-3.1-8b-instant',
    temperature=0.0,
    max_tokens=1024,
    api_key=GROQ_API_KEY
)

# ============================================================
# RETRIEVAL WITH SMART RE-RANKING
# ============================================================
SIGNALS = {
    'treaty': ['treaty','article','exemption','convention','article 21','article 20'],
    'form1040': ['form 1040','1040-nr','nonresident return','what forms','need to file','file a return'],
    'form8843': ['form 8843','8843','exempt individual','days of presence'],
    'fellowship': ['fellowship','stipend','1042','scholarship','taxed for foreign','withholding on'],
    'w8ben': ['w-8ben','w8ben','certificate of foreign','beneficial owner','form 8233'],
    'pub515': ['withholding rate','chapter 3','chapter 4','backup withholding'],
    'extension': ['extension','form 4868','more time','deadline extend'],
    'closer': ['closer connection','form 8840','ties to foreign'],
    'education': ['tuition','education credit','american opportunity','lifetime learning'],
}

TREATY_MAP = {
    'india': 'US-India Tax Treaty',
    'china': 'US-China Tax Treaty',
    'south_korea': 'US-South Korea Tax Treaty',
    'germany': 'US-Germany Tax Treaty',
    'mexico': 'US-Mexico Tax Treaty',
    'canada': 'US-Canada Tax Treaty',
    'japan': 'US-Japan Tax Treaty',
    'uk': 'US-UK Tax Treaty',
}

def get_boosts(query, profile):
    q = query.lower()
    country = profile.get('country', '').lower().replace(' ', '_')
    boosts = {}
    if any(s in q for s in SIGNALS['treaty']):
        if country in TREATY_MAP:
            boosts[TREATY_MAP[country]] = 1.4
        boosts['IRS Publication 901'] = 1.1
    if any(s in q for s in SIGNALS['form1040']):
        boosts['Form 1040-NR Instructions'] = 1.35
    if any(s in q for s in SIGNALS['form8843']):
        boosts['Form 8843'] = 1.35
    if any(s in q for s in SIGNALS['fellowship']):
        boosts['Form 1042-S Instructions'] = 1.6
        boosts['IRS Publication 515'] = 1.3
        boosts['IRS Publication 970'] = 1.3
        boosts['IRS Publication 525'] = 1.2
        boosts['Form 1040-NR Instructions'] = 0.9
    if any(s in q for s in SIGNALS['w8ben']):
        boosts['Form W-8BEN Instructions'] = 1.5
        boosts['Form 8233'] = 1.4
    if any(s in q for s in SIGNALS['pub515']):
        boosts['IRS Publication 515'] = 1.4
    if any(s in q for s in SIGNALS['extension']):
        boosts['Form 4868'] = 1.5
    if any(s in q for s in SIGNALS['closer']):
        boosts['Form 8840'] = 1.5
    if any(s in q for s in SIGNALS['education']):
        boosts['IRS Publication 970'] = 1.4
    return boosts

def enrich_query(query, profile):
    parts = []
    if profile.get('visa_type'): parts.append(f"{profile['visa_type']} visa")
    if profile.get('country'): parts.append(f"{profile['country'].replace('_', ' ').title()} student")
    if profile.get('years_in_us'): parts.append(f"{profile['years_in_us']} years in US")
    return f"{query} [{', '.join(parts)}]" if parts else query

def get_candidates(profile):
    s = set(country_idx.get('all', []))
    c = profile.get('country', '').lower().replace(' ', '_')
    if c: s.update(country_idx.get(c, []))
    v = profile.get('visa_type', '')
    if v: s.update(visa_idx.get(v, []))
    return sorted(list(s))

def retrieve(query, profile, top_k=5):
    candidates = get_candidates(profile) or list(range(len(chunks)))
    cand_emb = embeddings[candidates]
    tmp = faiss.IndexFlatIP(embeddings.shape[1])
    tmp.add(cand_emb)
    q = embed_model.encode(
        [enrich_query(query, profile)],
        normalize_embeddings=True
    ).astype(np.float32)
    fetch_k = min(top_k * 3, len(candidates))
    scores, idxs = tmp.search(q, fetch_k)
    boosts = get_boosts(query, profile)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i == -1: continue
        chunk = chunks[candidates[i]].copy()
        chunk['similarity_score'] = float(score) * boosts.get(chunk['source_name'], 1.0)
        results.append(chunk)
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results[:top_k]

# ============================================================
# PROMPT ENGINEERING
# ============================================================
CPA_KEYWORDS = config.get('escalation_keywords', [
    'fbar','fatca','dual status','dual-status','foreign bank account',
    'foreign asset','multi-state','state tax','audit','penalty',
    'irs notice','amended return','back taxes','self employed',
    'self-employed','freelance','cryptocurrency','crypto',
    'rental income','investment income','green card',
    'permanent resident','departure return','expatriation'
])

SYSTEM_PROMPT = """You are a specialized tax assistant for international students and workers in the United States on F-1, OPT, H-1B, and J-1 visas.

ABSOLUTE RULES:
1. ONLY USE PROVIDED CONTEXT: Answer ONLY based on the IRS documentation provided. If documentation does not contain the answer, say so explicitly.
2. CITATION MANDATORY: Every specific fact, number, threshold, or deadline MUST cite: [Source: IRS Publication 519, Chapter X] or [Source: US-India Tax Treaty, Article 21]
3. PROFILE-SPECIFIC: Address the user's exact visa type and country. Never give generic advice when country-specific rules exist.
4. STRUCTURE every answer:
   **Direct Answer** (1-2 sentences)
   **Explanation** (with citations for every fact)
   **Action Steps** (numbered list)
5. End every answer with: *Based on IRS Tax Year 2024/2025/2025 publications.*
6. NEVER GUESS: Do not invent rules or numbers not in the provided documentation.

You are NOT a licensed tax professional. Recommend a CPA for complex situations."""

def check_escalation(query):
    q = query.lower()
    return [kw for kw in CPA_KEYWORDS if kw in q]

def build_prompt(query, profile, retrieved):
    profile_str = f"""USER PROFILE:
- Visa: {profile.get('visa_type', 'N/A')}
- Country: {profile.get('country', 'N/A').replace('_', ' ').title()}
- Years in US: {profile.get('years_in_us', 'N/A')}
- Income Sources: {profile.get('income_sources', 'N/A')}"""
    context = "\nRELEVANT IRS DOCUMENTATION (use ONLY this to answer):\n"
    for i, c in enumerate(retrieved[:5]):
        context += f"\n[Doc {i+1}] {c['citation']} | {c['section']}\n{c['text'][:600]}\n---"
    return f"{profile_str}\n{context}\n\nQUESTION: {query}\n\nAnswer using ONLY the documentation above. Cite every fact."

# ============================================================
# SPT CALCULATOR
# ============================================================
def calculate_spt(days_current, days_y1, days_y2):
    w1 = days_y1 / 3
    w2 = days_y2 / 6
    total = days_current + w1 + w2
    passes = days_current >= 31 and total >= 183
    return {
        'passes': passes,
        'total': round(total, 2),
        'w1': round(w1, 2),
        'w2': round(w2, 2),
        'status': 'Resident Alien' if passes else 'Nonresident Alien',
        'form': 'Form 1040' if passes else 'Form 1040-NR',
        'meets_31': days_current >= 31,
        'meets_183': total >= 183,
    }

# ============================================================
# CHAT + CHECKLIST FUNCTIONS
# ============================================================
def chat(message, history, visa_type, country, years_in_us, income_sources):
    if not message.strip():
        return history, ""

    profile = {
        'visa_type': visa_type,
        'country': country.lower().replace(' ', '_'),
        'years_in_us': years_in_us,
        'income_sources': ', '.join(income_sources) if income_sources else 'Not specified'
    }

    triggers = check_escalation(message)
    if triggers:
        answer = f"""⚠️ **CPA Consultation Required**

This question involves **{', '.join(triggers)}** — a topic that exceeds what this system can reliably advise on.

**Please consult a licensed CPA** specializing in nonresident alien tax returns.

**Free Resources:**
- VITA Program (free for income <$67k): [irs.gov/vita](https://irs.gov/vita)
- IRS Free File: [irs.gov/freefile](https://irs.gov/freefile)
- Sprintax (international students): [sprintax.com](https://sprintax.com)

*[Escalation enforced by system architecture — IRS Publication 519, Tax Year 2024/2025]*"""
        history.append((message, answer))
        return history, ""

    retrieved = retrieve(message, profile, top_k=5)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for user_msg, bot_msg in history[-3:]:
        messages.append(HumanMessage(content=user_msg))
        messages.append(AIMessage(content=bot_msg))
    messages.append(HumanMessage(content=build_prompt(message, profile, retrieved)))

    response = llm.invoke(messages)
    sources = list({r['citation'] for r in retrieved})
    answer = response.content
    if '📚' not in answer:
        answer += f"\n\n---\n📚 **Sources:** {' | '.join(sources)}"

    history.append((message, answer))
    return history, ""


def run_spt(days_current, days_y1, days_y2):
    try:
        r = calculate_spt(int(days_current), int(days_y1), int(days_y2))
        icon = "✅" if r['passes'] else "📋"
        result = f"""## {icon} Result: **{r['status']}**

| | Days | Multiplier | Testing Days |
|---|---|---|---|
| Current Year | {int(days_current)} | × 1.000 | {int(days_current)} |
| Year -1 | {int(days_y1)} | × 0.333 | {r['w1']} |
| Year -2 | {int(days_y2)} | × 0.167 | {r['w2']} |
| **Total** | | | **{r['total']}** |

**31-day requirement:** {"✅ Met" if r['meets_31'] else "❌ Not Met"} ({int(days_current)} days in current year)
**183-day requirement:** {"✅ Met" if r['meets_183'] else "❌ Not Met"} ({r['total']} testing days)

### 📄 File: **{r['form']}**

> *Per IRS Publication 519, Chapter 1 — Substantial Presence Test, Tax Year 2024/2025*

{"⚠️ **Note for F-1/J-1 students:** You may be an exempt individual and should NOT count your days of presence for your first 5 calendar years in the US. Consult a CPA to confirm your exempt status." if not r['passes'] else ""}"""
        return result
    except ValueError:
        return "⚠️ Please enter valid numbers for all three fields."


def generate_checklist(history, visa_type, country, years_in_us):
    if not history:
        return "💬 Ask some tax questions in the **Chat** tab first, then come back here to generate your personalized checklist."

    profile = {
        'visa_type': visa_type,
        'country': country.lower().replace(' ', '_'),
        'years_in_us': years_in_us
    }

    topics = [msg for msg, _ in history]
    prompt = f"""Generate a personalized tax filing checklist for this user:

PROFILE:
- Visa: {visa_type}
- Country: {country}
- Years in US: {years_in_us}

TOPICS DISCUSSED IN THIS SESSION:
{chr(10).join(f'- {t}' for t in topics)}

Create a clear numbered checklist covering:
1. Forms to file (with April 15, 2026 deadline for most filers)
2. Documents to gather before filing
3. Treaty benefits to claim (specific to their country)
4. Key deadlines and important dates
5. When to consult a CPA

Be specific to their visa type and country. Include IRS citations for each item.
Format with clear sections and bullet points."""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])
    return response.content


# ============================================================
# GRADIO UI
# ============================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

* { font-family: 'IBM Plex Sans', sans-serif; }
.gradio-container { max-width: 1100px !important; margin: 0 auto; }

.header-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 8px;
    border: 1px solid #334155;
}
.header-box h1 { color: #f8fafc; font-size: 1.8rem; font-weight: 600; margin: 0 0 6px 0; font-family: 'IBM Plex Mono', monospace; }
.header-box p { color: #94a3b8; margin: 0; font-size: 0.95rem; }
.header-badge { display: inline-block; background: #1d4ed8; color: #bfdbfe; font-size: 0.75rem; padding: 2px 10px; border-radius: 20px; margin-top: 10px; font-family: 'IBM Plex Mono', monospace; }
.warning-box { background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 12px 16px; font-size: 0.85rem; color: #92400e; margin-top: 8px; }
.stats-box { background: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 10px 16px; font-size: 0.82rem; color: #166534; margin-top: 8px; }
footer { display: none !important; }
"""

COUNTRIES = [
    "India", "China", "South Korea", "Germany", "Mexico",
    "Canada", "Japan", "Other"
]

with gr.Blocks(css=CSS, title="Immigrant Tax Filing Assistant") as demo:

    gr.HTML("""
    <div class="header-box">
        <h1>🏛 Immigrant Tax Filing Assistant</h1>
        <p>Profile-aware tax guidance for international students and workers on F-1, OPT, H-1B, and J-1 visas.</p>
        <span class="header-badge">23 IRS Documents · 3,699 Chunks · Tax Year 2024/2025 · RAG + Groq LLaMA 3.1</span>
    </div>
    """)

    gr.HTML("""
    <div class="warning-box">
        ⚠️ This tool provides general tax information based on IRS publications. It is NOT a substitute for professional tax advice.
        Always consult a licensed CPA for complex situations. Answers based on IRS Tax Year 2024/2025/2025 publications.
    </div>
    """)

    gr.HTML("""
    <div class="stats-box">
        📚 Knowledge Base: IRS Publications 519, 901, 515, 970, 525, 54 · Forms 1040-NR, 1042-S, 8843, W-8BEN, 4868, 8233, 8840
        · Treaties: India, China, South Korea, Germany, Mexico, Canada, Japan · Retrieval Accuracy: 80% · Escalation Precision: 100%
    </div>
    """)

    chat_history = gr.State([])

    with gr.Tabs():

        # ── TAB 1: CHAT ──────────────────────────────────────
        with gr.Tab("💬 Tax Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Your Profile")
                    visa_type = gr.Dropdown(
                        choices=["F-1", "OPT", "H-1B", "J-1"],
                        value="F-1", label="Visa Type"
                    )
                    country = gr.Dropdown(
                        choices=COUNTRIES,
                        value="India", label="Home Country"
                    )
                    years_in_us = gr.Slider(
                        minimum=0, maximum=10, value=2, step=1,
                        label="Years in US"
                    )
                    income_sources = gr.CheckboxGroup(
                        choices=["TA/RA Stipend", "Fellowship", "W-2 Employment", "CPT/OPT Income", "Scholarship"],
                        value=["TA/RA Stipend"],
                        label="Income Sources"
                    )

                    gr.Markdown("### 💡 Quick Questions")
                    gr.Markdown("""
- Am I exempt from FICA taxes?
- What is the Substantial Presence Test?
- Do I need to file Form 8843?
- What treaty benefits do I qualify for?
- How is my fellowship taxed?
- What is Form W-8BEN?
- How do I get a filing extension?
- What is my filing deadline?
""")

                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Tax Q&A",
                        height=500,
                        bubble_full_width=False,
                        avatar_images=("👤", "🏛")
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask your tax question here...",
                            label="", scale=4, lines=1
                        )
                        send_btn = gr.Button("Send →", variant="primary", scale=1)
                    clear_btn = gr.Button("🗑 Clear Chat", size="sm", variant="secondary")

            send_btn.click(
                chat,
                inputs=[msg_input, chat_history, visa_type, country, years_in_us, income_sources],
                outputs=[chat_history, msg_input]
            ).then(lambda h: h, chat_history, chatbot)

            msg_input.submit(
                chat,
                inputs=[msg_input, chat_history, visa_type, country, years_in_us, income_sources],
                outputs=[chat_history, msg_input]
            ).then(lambda h: h, chat_history, chatbot)

            clear_btn.click(lambda: ([], []), outputs=[chat_history, chatbot])

        # ── TAB 2: SPT CALCULATOR ────────────────────────────
        with gr.Tab("📅 SPT Calculator"):
            gr.Markdown("""
### Substantial Presence Test Calculator
Determine if you are a **Resident Alien** or **Nonresident Alien** for U.S. tax purposes.

**Formula:** All days current year + ⅓ days year-1 + ⅙ days year-2 ≥ 183 (AND ≥31 days current year)

*Per IRS Publication 519, Chapter 1*
""")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Enter Your Days Present in the US")
                    days_current = gr.Number(label="Days in Current Tax Year (2025)", value=180, precision=0)
                    days_y1 = gr.Number(label="Days in Year -1 (2024)", value=0, precision=0)
                    days_y2 = gr.Number(label="Days in Year -2 (2023)", value=0, precision=0)
                    calc_btn = gr.Button("Calculate My Tax Status →", variant="primary")

                    gr.Markdown("""
> **⚠️ F-1/J-1 Students:** You are likely an **exempt individual** for your first 5 calendar years.
> Exempt individuals do NOT count their days of presence.
> File Form 8843 to claim this exemption.
> *[IRS Publication 519, Chapter 1]*
""")

                with gr.Column():
                    spt_result = gr.Markdown(value="*Enter your days above and click Calculate.*")

            calc_btn.click(
                run_spt,
                inputs=[days_current, days_y1, days_y2],
                outputs=spt_result
            )

            gr.Markdown("""
#### Quick Reference Test Cases
| Scenario | Current | Year-1 | Year-2 | Result |
|---|---|---|---|---|
| F-1 Student Year 1 | 180 | 0 | 0 | Nonresident (180 < 183) |
| F-1 Student Year 6+ | 200 | 200 | 200 | Resident (if not exempt) |
| H-1B Worker Year 4 | 300 | 300 | 300 | Resident (450 testing days) |
| Edge Case | 31 | 456 | 0 | Resident (183.0 exactly) |
""")

        # ── TAB 3: FILING CHECKLIST ──────────────────────────
        with gr.Tab("📋 Filing Checklist"):
            gr.Markdown("""
### Personalized Filing Checklist
After chatting with the Tax Assistant, generate a personalized checklist based on your conversation.
The checklist will reflect the specific topics you discussed and your visa/country profile.
""")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Set your profile below:**")
                    checklist_visa = gr.Dropdown(
                        choices=["F-1", "OPT", "H-1B", "J-1"],
                        value="F-1", label="Visa Type"
                    )
                    checklist_country = gr.Dropdown(
                        choices=COUNTRIES,
                        value="India", label="Home Country"
                    )
                    checklist_years = gr.Slider(0, 10, value=2, step=1, label="Years in US")
                    gen_btn = gr.Button("Generate My Checklist →", variant="primary")

                    gr.Markdown("""
> 💡 **Tip:** Ask 3-4 questions in the Chat tab first for a more personalized and detailed checklist.
""")

                with gr.Column(scale=2):
                    checklist_output = gr.Markdown(
                        value="*Click 'Generate My Checklist' to create your personalized filing checklist.*"
                    )

            gen_btn.click(
                generate_checklist,
                inputs=[chat_history, checklist_visa, checklist_country, checklist_years],
                outputs=checklist_output
            )

    gr.HTML("""
    <div style="text-align:center; padding:16px; color:#94a3b8; font-size:0.78rem; margin-top:8px; border-top:1px solid #e2e8f0;">
        Built with LangChain + Groq LLaMA 3.1 + FAISS · 23 IRS Documents · 3,699 Knowledge Chunks · Tax Year 2024/2025
        <br>Not a substitute for professional tax advice. Consult a licensed CPA for complex situations.
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
