"""
Embedding Service for IntentAlign
Provides semantic embeddings, cosine similarity calculation, and LRU vector caching.
Primary: SentenceTransformer (all-MiniLM-L6-v2)
Fallback: Scikit-learn TF-IDF + TruncatedSVD Latent Semantic Analysis (LSA)
"""
import hashlib
import numpy as np
from collections import OrderedDict
import os

SEMANTIC_CORPUS = [
    # 0: DevOps
    "devops continuous integration continuous deployment ci cd automation infrastructure deployment monitoring reliability operations terraform ansible pipeline gitlab jenkins",
    # 1: Kubernetes
    "kubernetes k8s pods deployments services helm cluster orchestration ingress containers docker kubectl microservices cloud native devops infrastructure container management node",
    # 2: Docker
    "docker containers containerization images dockerfile compose registry microservices build ship run devops kubernetes linux virtualization container management",
    # 3: Linux
    "linux ubuntu shell bash terminal commands operating system server administration permissions processes kernel systemd devops sysadmin redhat debian centos",
    # 4: Cloud
    "cloud aws azure google cloud gcp cloud computing serverless infrastructure virtual machines storage networking lambda ec2 s3 devops docker kubernetes iam vpc cloudflare",
    # 5: Programming
    "programming coding software development java python javascript typescript c c++ algorithms data structures debugging git github developer backend frontend web development fullstack api rest graphql",
    # 6: AI / ML
    "artificial intelligence machine learning deep learning generative ai llm neural networks natural language processing nlp computer vision cv pytorch tensorflow data science python algorithms models prompt engineering transformer",
    # 7: Cybersecurity
    "cybersecurity security hacking vulnerabilities authentication network security phishing malware privacy defense penetration testing ethical hacking infosec cryptography firewalls zero day exploit soc siem cyber defense",
    # 8: Education
    "education tutorial course lecture study learning exam academic lesson training full course beginner to advanced roadmap guide crash course bootcamp masterclass learn step by step",
    # 9: Career
    "career jobs interview placement resume hiring internship professional skills workplace preparation salary questions technical interview coding round hr interview questions mock interview linkedin",
    # 10: Aptitude
    "quantitative aptitude logical reasoning verbal ability placement exam preparation questions solved examples tricks shortcuts math puzzles reasoning aptitude test number series percentage profit loss coding questions",
    # 11: News (independent of politics)
    "news breaking news headlines current affairs latest updates journalism world events live reporting daily digest press media broadcast tv news bulletin daily news update morning news",
    # 12: Politics (independent of general news)
    "politics political election government parliament political party minister policy democracy campaign prime minister president debate congress voting senate legislation republic politician rally political rally",
    # 13: Sports
    "sports cricket football basketball tennis match tournament player league athletics highlights goals wickets fifa ipl world cup score commentary nba stadium referee wicket bat ball goals",
    # 14: Gaming
    "gaming games gameplay esports playthrough xbox playstation pc game gamer walkthrough trailer review twitch stream gaming highlights rpg fps nintendo multiplayer steam lets play",
    # 15: Entertainment
    "entertainment movies music celebrity comedy drama tv show film web series song trailer gossip bollywood hollywood reaction vlog comedy sketch cinema actor actress songs dance teaser review film review",
    # 16: Finance
    "finance stock market investing trading banking cryptocurrency money economy personal finance crypto bitcoin mutual funds trading options shares wealth inflation wall street forex dividend portfolio",
    # 17: Travel
    "travel tourism vacation destination hotel flights itinerary trip tourism guide explore vlog adventure beach mountains resort road trip wanderlust backpacking travel guide flights booking",
    # 18: Health
    "health fitness workout nutrition wellness medicine exercise mental wellbeing healthcare gym diet protein yoga cardio body transformation weight loss bodybuilding strength mental health",
    # 19: Science
    "science physics chemistry biology research experiment astronomy space technology quantum mechanics nasa astrophysics genetics scientific discovery cosmos telescope rocket isro spacex scientific research",
    # 20: Blockchain & Web3
    "blockchain smart contracts solidity ethereum web3 crypto decentralized dapps evm tokens cryptography consensus proof of stake defi audit reentrancy web3 security transaction nft",
    # 21: Data Science
    "data science python pandas numpy dataframe matplotlib seaborn analytics statistics machine learning visualization jupyter data analysis big data business intelligence regression clustering",
    # 22: Food & Cooking
    "cooking culinary recipe food chef kitchen baking dinner cuisine ingredients restaurant meal tasty dishes cooking guide delicious healthy eating cooking show"
]

class EmbeddingService:
    def __init__(self, model_name="all-MiniLM-L6-v2", cache_capacity=4096):
        self.model_name = os.getenv("INTENTALIGN_MODEL", model_name)
        self.cache_capacity = cache_capacity
        self._cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self._model = None
        self.is_neural = False
        self._tfidf = None
        self._svd = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self.is_neural = True
            print(f"[EmbeddingService] Loaded neural SentenceTransformer model: {self.model_name}")
        except Exception as e:
            self._model = None
            self.is_neural = False
            print(f"[EmbeddingService] SentenceTransformer unavailable ({e}). Initializing scikit-learn TF-IDF + LSA semantic space.")
            self._init_lsa()

    def _init_lsa(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        X = self._tfidf.fit_transform(SEMANTIC_CORPUS)
        n_components = min(19, X.shape[0] - 1, X.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(X)

    def _hash_key(self, text):
        return hashlib.sha256((text or "").strip().lower().encode("utf-8")).hexdigest()

    def get_embedding(self, text):
        clean_text = (text or "").strip()
        if not clean_text:
            dim = 384 if self.is_neural else self._svd.n_components
            return np.zeros(dim, dtype=np.float32)

        key = self._hash_key(clean_text)
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        self.cache_misses += 1
        if self.is_neural and self._model is not None:
            vec = self._model.encode(clean_text, normalize_embeddings=True)
            vec = np.asarray(vec, dtype=np.float32)
        else:
            sparse_x = self._tfidf.transform([clean_text])
            dense_v = self._svd.transform(sparse_x)[0]
            norm = np.linalg.norm(dense_v)
            if norm > 0:
                vec = (dense_v / norm).astype(np.float32)
            else:
                vec = np.zeros(self._svd.n_components, dtype=np.float32)

        if len(self._cache) >= self.cache_capacity:
            self._cache.popitem(last=False)
        self._cache[key] = vec
        return vec

    def cosine_similarity(self, vec1, vec2):
        if vec1 is None or vec2 is None:
            return 0.0
        l1, l2 = len(vec1), len(vec2)
        if l1 != l2:
            maxlen = max(l1, l2)
            v1 = np.pad(vec1, (0, maxlen - l1))
            v2 = np.pad(vec2, (0, maxlen - l2))
        else:
            v1, v2 = vec1, vec2
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        dot = float(np.dot(v1, v2) / (norm1 * norm2))
        return max(0.0, min(1.0, dot))

    def similarity(self, text1, text2):
        v1 = self.get_embedding(text1)
        v2 = self.get_embedding(text2)
        return self.cosine_similarity(v1, v2)

    def status(self):
        return {
            "model": self.model_name if self.is_neural else "scikit-learn-tfidf-lsa",
            "is_neural": self.is_neural,
            "cache_size": len(self._cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses
        }
