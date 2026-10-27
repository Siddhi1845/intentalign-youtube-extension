"""
Keyword and Keyphrase Extraction Service for IntentAlign
Extracts contextual keywords and n-gram keyphrases from user goals and video metadata.
Primary: KeyBERT
Fallback: Scikit-learn TF-IDF n-gram phrase scoring + frequency filtering
"""
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

STOP_WORDS = set(
    "a about above after again against all am an and any are aren as at be because "
    "been before being below between both but by can cannot could couldn did didn "
    "do does doesn doing don down during each few for from further had hadn has hasn "
    "have haven having he her here hers herself him himself his how if in into is "
    "isn it its itself let me more most my myself no nor not of off on once only "
    "or other ought our ours ourselves out over own same she should shouldn so some "
    "such than that the their theirs them themselves then there these they this those "
    "through to too under until up very was wasn we were weren what when where which "
    "while who whom why with won would wouldn you your yours yourself yourselves "
    "complete guide tutorial latest today new official review best top course learn "
    "video videos watch online full 2024 2025 2026 beginner beginners advanced".split()
)

class KeywordService:
    def __init__(self, embedding_service=None):
        self.embedding_service = embedding_service
        self._keybert = None
        self.is_keybert = False
        self._init_keybert()

    def _init_keybert(self):
        try:
            from keybert import KeyBERT
            model = getattr(self.embedding_service, "_model", None)
            self._keybert = KeyBERT(model=model if model is not None else "all-MiniLM-L6-v2")
            self.is_keybert = True
            print("[KeywordService] KeyBERT initialized successfully.")
        except Exception as e:
            self._keybert = None
            self.is_keybert = False
            print(f"[KeywordService] KeyBERT unavailable ({e}). Using scikit-learn TF-IDF n-gram extractor.")

    def extract_keywords(self, text, top_n=8):
        clean_text = (text or "").strip()
        if not clean_text or len(clean_text) < 3:
            return []

        if self.is_keybert and self._keybert is not None:
            try:
                extracted = self._keybert.extract_keywords(
                    clean_text,
                    keyphrase_ngram_range=(1, 3),
                    stop_words="english",
                    top_n=top_n,
                    use_mmr=True,
                    diversity=0.4
                )
                return [{"keyword": k.strip(), "score": round(float(s), 4)} for k, s in extracted if k.strip()]
            except Exception as e:
                pass  # Fall through to TF-IDF fallback

        # Robust TF-IDF n-gram fallback
        return self._tfidf_extract(clean_text, top_n)

    def _tfidf_extract(self, text, top_n=8):
        # Extract meaningful tokens
        raw_tokens = [
            t.lower() for t in re.findall(r"[a-zA-Z0-9+#.-]+", text)
            if len(t) > 2 and t.lower() not in STOP_WORDS
        ]
        if not raw_tokens:
            return []

        try:
            # Build 1-to-3 gram candidates
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                stop_words=list(STOP_WORDS),
                token_pattern=r"(?u)\b[a-zA-Z0-9+#.-]{2,}\b"
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            pairs = []
            for name, score in zip(feature_names, scores):
                if score > 0:
                    # Give slight preference to 2-3 word keyphrases if present
                    words = name.split()
                    boost = 1.15 if len(words) == 2 else (1.10 if len(words) == 3 else 1.0)
                    pairs.append((name, float(score * boost)))

            pairs.sort(key=lambda x: x[1], reverse=True)
            # Normalize scores to 0..1
            max_s = pairs[0][1] if pairs else 1.0
            results = []
            seen = set()
            for kw, sc in pairs:
                if kw in seen:
                    continue
                seen.add(kw)
                results.append({"keyword": kw, "score": round(min(1.0, sc / max_s), 4)})
                if len(results) >= top_n:
                    break
            return results
        except Exception:
            # Simple token frequency fallback if Tfidf fails on short string
            counts = Counter(raw_tokens)
            total = sum(counts.values()) or 1
            return [{"keyword": k, "score": round(v / total, 4)} for k, v in counts.most_common(top_n)]
