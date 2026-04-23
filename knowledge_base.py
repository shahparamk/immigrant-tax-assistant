"""
TaxKnowledgeBase — Profile-aware RAG retrieval engine
Uses FAISS for semantic search + metadata filtering for country/visa specificity
"""
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer


SIGNALS = {
    'treaty': ['treaty', 'article', 'exemption', 'convention', 'article 21', 'article 20'],
    'form1040': ['form 1040', '1040-nr', 'nonresident return', 'what forms', 'need to file'],
    'form8843': ['form 8843', '8843', 'exempt individual', 'days of presence'],
    'fellowship': ['fellowship', 'stipend', '1042', 'scholarship', 'taxed for foreign'],
    'w8ben': ['w-8ben', 'w8ben', 'certificate of foreign', 'form 8233'],
    'pub515': ['withholding rate', 'chapter 3', 'chapter 4', 'backup withholding'],
    'extension': ['extension', 'form 4868', 'more time', 'deadline extend'],
    'closer': ['closer connection', 'form 8840', 'ties to foreign'],
    'education': ['tuition', 'education credit', 'american opportunity'],
}

TREATY_MAP = {
    'india': 'US-India Tax Treaty',
    'china': 'US-China Tax Treaty',
    'south_korea': 'US-South Korea Tax Treaty',
    'germany': 'US-Germany Tax Treaty',
    'mexico': 'US-Mexico Tax Treaty',
    'canada': 'US-Canada Tax Treaty',
    'japan': 'US-Japan Tax Treaty',
}


class TaxKnowledgeBase:
    def __init__(self, base_dir: str = '.'):
        base = Path(base_dir)
        with open(base / 'metadata.json', encoding='utf-8') as f:
            self.chunks = json.load(f)
        with open(base / 'filter_maps.json') as f:
            maps = json.load(f)
        self.country_idx = maps['country_to_indices']
        self.visa_idx = maps['visa_to_indices']
        self.embeddings = np.load(str(base / 'embeddings.npy'))
        self.index = faiss.read_index(str(base / 'faiss_index.bin'))
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"Knowledge base loaded: {len(self.chunks)} chunks, {self.index.ntotal} vectors")

    def get_boosts(self, query: str, profile: dict) -> dict:
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
            boosts['Form 1040-NR Instructions'] = 0.9
        if any(s in q for s in SIGNALS['w8ben']):
            boosts['Form W-8BEN Instructions'] = 1.5
        if any(s in q for s in SIGNALS['extension']):
            boosts['Form 4868'] = 1.5
        if any(s in q for s in SIGNALS['closer']):
            boosts['Form 8840'] = 1.5
        if any(s in q for s in SIGNALS['education']):
            boosts['IRS Publication 970'] = 1.4
        return boosts

    def enrich_query(self, query: str, profile: dict) -> str:
        parts = []
        if profile.get('visa_type'): parts.append(f"{profile['visa_type']} visa")
        if profile.get('country'): parts.append(f"{profile['country'].replace('_', ' ').title()} student")
        return f"{query} [{', '.join(parts)}]" if parts else query

    def get_candidates(self, profile: dict) -> list:
        s = set(self.country_idx.get('all', []))
        c = profile.get('country', '').lower().replace(' ', '_')
        if c: s.update(self.country_idx.get(c, []))
        v = profile.get('visa_type', '')
        if v: s.update(self.visa_idx.get(v, []))
        return sorted(list(s))

    def retrieve(self, query: str, profile: dict, top_k: int = 5) -> list:
        candidates = self.get_candidates(profile) or list(range(len(self.chunks)))
        tmp = faiss.IndexFlatIP(self.embeddings.shape[1])
        tmp.add(self.embeddings[candidates])
        q = self.model.encode(
            [self.enrich_query(query, profile)],
            normalize_embeddings=True
        ).astype(np.float32)
        fetch_k = min(top_k * 3, len(candidates))
        scores, idxs = tmp.search(q, fetch_k)
        boosts = self.get_boosts(query, profile)
        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1: continue
            chunk = self.chunks[candidates[i]].copy()
            chunk['similarity_score'] = float(score) * boosts.get(chunk['source_name'], 1.0)
            results.append(chunk)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
