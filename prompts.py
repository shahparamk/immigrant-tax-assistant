"""
Prompt Engineering Layer
Citation-mandatory system prompt + CPA escalation keywords
"""

CPA_ESCALATION_KEYWORDS = [
    'fbar', 'fatca', 'dual status', 'dual-status',
    'foreign bank account', 'foreign asset',
    'multi-state', 'state tax', 'audit', 'penalty',
    'irs notice', 'amended return', 'back taxes',
    'self employed', 'self-employed', 'freelance',
    'cryptocurrency', 'crypto', 'rental income',
    'investment income', 'green card',
    'permanent resident', 'departure return', 'expatriation'
]

SYSTEM_PROMPT = """You are a specialized tax assistant for international students and workers in the United States on F-1, OPT, H-1B, and J-1 visas.

ABSOLUTE RULES:
1. ONLY USE PROVIDED CONTEXT: Answer ONLY based on the IRS documentation provided. If documentation does not contain the answer, say so explicitly.
2. CITATION MANDATORY: Every specific fact, number, threshold, or deadline MUST cite: [Source: IRS Publication 519, Chapter X] or [Source: US-India Tax Treaty, Article 21]
3. PROFILE-SPECIFIC: Address the user's exact visa type and country. Never give generic advice when country-specific rules exist.
4. STRUCTURE every answer:
   **Direct Answer** (1-2 sentences)
   **Explanation** (with citations for every fact)
   **Action Steps** (numbered list)
5. End every answer with: *Based on IRS Tax Year 2024/2025 publications.*
6. NEVER GUESS: Do not invent rules or numbers not in the provided documentation.

You are NOT a licensed tax professional. Recommend a CPA for complex situations."""

ESCALATION_MESSAGE = """⚠️ **CPA Consultation Required**

This question involves **{triggers}** — a topic that exceeds what this system can reliably advise on.

**Please consult a licensed CPA** specializing in nonresident alien tax returns.

**Free Resources:**
- VITA Program (free for income <$67k): irs.gov/vita
- IRS Free File: irs.gov/freefile
- Sprintax (international students): sprintax.com

*[Escalation enforced by system architecture — IRS Publication 519, Tax Year 2024]*"""


def check_escalation(query: str) -> list:
    """Return list of triggered escalation keywords."""
    q = query.lower()
    return [kw for kw in CPA_ESCALATION_KEYWORDS if kw in q]


def build_rag_prompt(query: str, profile: dict, retrieved_chunks: list) -> str:
    """Build the full RAG prompt with user profile + retrieved IRS context."""
    profile_str = f"""USER PROFILE:
- Visa: {profile.get('visa_type', 'N/A')}
- Country: {profile.get('country', 'N/A').replace('_', ' ').title()}
- Years in US: {profile.get('years_in_us', 'N/A')}
- Income Sources: {profile.get('income_sources', 'N/A')}"""

    context = "\nRELEVANT IRS DOCUMENTATION (use ONLY this to answer):\n"
    for i, chunk in enumerate(retrieved_chunks[:5]):
        context += f"\n[Doc {i+1}] {chunk['citation']} | {chunk['section']}\n{chunk['text'][:600]}\n---"

    return f"{profile_str}\n{context}\n\nQUESTION: {query}\n\nAnswer using ONLY the documentation above. Cite every fact."
