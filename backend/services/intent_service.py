"""
Intent Profile Engine for IntentAlign
V1.3 — Dynamic Intent + Session-Aware Closed Loop

Responsibilities:
    1. Maintain stable user profile.
    2. Maintain dynamic current intent.
    3. Process explicit current goal.
    4. Derive semantic goal concepts.
    5. Maintain user-defined preferred/unwanted topics.
    6. Preserve learned feedback weights across rebuilds.
    7. Maintain hard/soft negative configuration.
    8. Generate embeddings for the active semantic topic space.
    9. Keep intent sessions isolated.

Architecture:

    Stable User Profile
        profession
        interests
        hobbies

                 +

    Current Intent Session
        current goal
        preferred topics
        unwanted topics
        derived goal topics
        learned weights
        strictness

                 ↓

          Intent Profile
"""

import re
from typing import Dict, List, Any, Optional


# ================================================================
# Supporting Canonical Taxonomy
# ================================================================

TAXONOMY = {
    "DevOps": (
        "devops continuous integration continuous deployment "
        "ci cd automation infrastructure deployment monitoring "
        "reliability operations terraform ansible pipeline"
    ),

    "Linux": (
        "linux ubuntu shell bash terminal commands operating "
        "system server administration permissions processes kernel systemd"
    ),

    "Docker": (
        "docker containers containerization images dockerfile "
        "compose registry microservices build ship run"
    ),

    "Kubernetes": (
        "kubernetes k8s pods deployments services helm cluster "
        "orchestration ingress containers kubectl microservices"
    ),

    "Cloud": (
        "cloud aws azure google cloud gcp cloud computing "
        "serverless infrastructure virtual machines storage "
        "networking lambda ec2 s3"
    ),

    "Programming": (
        "programming coding software development java python "
        "javascript typescript c c++ algorithms data structures "
        "debugging git github web development"
    ),

    "AI": (
        "artificial intelligence machine learning deep learning "
        "generative ai llm neural networks natural language "
        "processing computer vision data science"
    ),

    "Cybersecurity": (
        "cybersecurity security hacking vulnerabilities authentication "
        "network security phishing malware privacy defense "
        "penetration testing ethical hacking"
    ),

    "Education": (
        "education tutorial course lecture study learning exam "
        "academic lesson training full course beginner guide roadmap"
    ),

    "Career": (
        "career jobs interview placement resume hiring internship "
        "professional skills workplace preparation salary questions "
        "technical interview"
    ),

    "Aptitude": (
        "quantitative aptitude logical reasoning verbal ability "
        "placement exam preparation questions solved examples "
        "tricks shortcuts math puzzles"
    ),

    "Blockchain": (
        "blockchain web3 smart contracts solidity ethereum crypto "
        "decentralized dapps evm tokens bitcoin cryptography"
    ),

    "News": (
        "news breaking news headlines current affairs latest "
        "updates journalism world events live reporting daily "
        "digest bulletin"
    ),

    "Politics": (
        "politics political election government parliament political "
        "party minister policy democracy campaign prime minister "
        "president debate congress voting"
    ),

    "Sports": (
        "sports cricket football basketball tennis match tournament "
        "player league athletics highlights goals wickets fifa "
        "ipl world cup"
    ),

    "Gaming": (
        "gaming games gameplay esports playthrough xbox "
        "playstation pc game gamer walkthrough trailer review "
        "twitch stream"
    ),

    "Entertainment": (
        "entertainment movies music celebrity comedy drama tv show "
        "film web series song trailer gossip bollywood hollywood "
        "reaction vlog sketch"
    ),

    "Finance": (
        "finance stock market investing trading banking "
        "cryptocurrency money economy personal finance crypto "
        "bitcoin mutual funds options"
    ),

    "Travel": (
        "travel tourism vacation destination hotel flights "
        "itinerary trip tourism guide explore vlog adventure "
        "beach mountains road trip"
    ),

    "Health": (
        "health fitness workout nutrition wellness medicine "
        "exercise mental wellbeing healthcare gym diet protein "
        "yoga cardio"
    ),

    "Science": (
        "science physics chemistry biology research experiment "
        "astronomy space technology quantum mechanics nasa "
        "astrophysics genetics cosmos"
    )
}


# ================================================================
# Normalization Mapping
# ================================================================

NORMALIZATION_MAP = {
    "k8s": "Kubernetes",
    "kube": "Kubernetes",

    "cicd": "CI/CD",
    "ci cd": "CI/CD",
    "ci/cd": "CI/CD",

    "ml": "Machine Learning",
    "ai": "Artificial Intelligence",
    "genai": "Artificial Intelligence",
    "dl": "Deep Learning",
    "nlp": "Natural Language Processing",
    "cv": "Computer Vision",

    "js": "JavaScript",
    "ts": "TypeScript",
    "py": "Python",

    "infosec": "Cybersecurity",
    "cyber security": "Cybersecurity",

    "devops": "DevOps",
    "devsecops": "DevSecOps",

    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "nosql": "NoSQL",
    "graphql": "GraphQL",

    "breaking news": "News",
    "celebrity gossip": "Celebrity Gossip"
}


# ================================================================
# Goal Expansion Ontology
# ================================================================

GOAL_EXPANSIONS = {

    "devops": [
        "DevOps",
        "Linux",
        "Docker",
        "Kubernetes",
        "Cloud",
        "CI/CD",
        "Git",
        "Jenkins",
        "Career"
    ],

    "placement": [
        "Career",
        "Programming",
        "Aptitude",
        "Education"
    ],

    "interview": [
        "Career",
        "Programming",
        "Education",
        "Interview Preparation"
    ],

    "docker": [
        "Docker",
        "Kubernetes",
        "DevOps",
        "Cloud",
        "Linux",
        "Containers"
    ],

    "kubernetes": [
        "Kubernetes",
        "Docker",
        "DevOps",
        "Cloud",
        "Linux",
        "Microservices"
    ],

    "linux": [
        "Linux",
        "DevOps",
        "Cybersecurity",
        "Programming",
        "Bash"
    ],

    "cloud": [
        "Cloud",
        "DevOps",
        "Kubernetes",
        "Docker",
        "AWS",
        "Azure"
    ],

    "blockchain": [
        "Blockchain",
        "Solidity",
        "Smart Contracts",
        "Ethereum",
        "Web3",
        "Cryptography"
    ],

    "solidity": [
        "Solidity",
        "Smart Contracts",
        "Ethereum",
        "Blockchain",
        "Web3"
    ],

    "ethereum": [
        "Ethereum",
        "Smart Contracts",
        "Solidity",
        "Blockchain",
        "Web3"
    ],

    "smart contract": [
        "Smart Contracts",
        "Solidity",
        "Ethereum",
        "Blockchain",
        "Web3"
    ],

    "web3": [
        "Web3",
        "Blockchain",
        "Solidity",
        "Smart Contracts",
        "Ethereum"
    ],

    "machine learning": [
        "Machine Learning",
        "AI",
        "Python",
        "Data Science",
        "Deep Learning"
    ],

    "artificial intelligence": [
        "Artificial Intelligence",
        "AI",
        "Machine Learning",
        "Deep Learning"
    ],

    "cybersecurity": [
        "Cybersecurity",
        "Ethical Hacking",
        "Network Security",
        "Linux"
    ],

    "aptitude": [
        "Quantitative Aptitude",
        "Logical Reasoning",
        "Verbal Ability",
        "Career"
    ],

    "python": [
        "Python",
        "Programming",
        "Data Science",
        "Backend"
    ]
}


# ================================================================
# Default Negative Configuration
# ================================================================

DEFAULT_HARD_NEGATIVES = {
    "Politics",
    "Celebrity Gossip",
    "Adult Content"
}

DEFAULT_SOFT_NEGATIVES = {
    "News",
    "Entertainment",
    "Gaming",
    "Sports"
}


class IntentService:

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(
        self,
        embedding_service,
        keyword_service
    ):
        self.embedding_service = embedding_service
        self.keyword_service = keyword_service

    # ============================================================
    # Basic Helpers
    # ============================================================

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0
    ) -> float:

        try:
            value = float(value)
        except (TypeError, ValueError):
            value = minimum

        return max(
            minimum,
            min(maximum, value)
        )

    @staticmethod
    def _clean_text(value: Any) -> str:

        if value is None:
            return ""

        return str(value).strip()

    # ============================================================
    # Topic Normalization
    # ============================================================

    def normalize_topic(
        self,
        topic: str
    ) -> str:
        """
        Normalize a topic while preserving user-defined topics.

        The normalization layer should NOT collapse unrelated
        technologies into broad categories.
        """

        clean = re.sub(
            r"[^\w\s/+\-#.]",
            "",
            self._clean_text(topic)
        ).strip()

        if not clean:
            return ""

        lower = clean.lower()

        if lower in NORMALIZATION_MAP:
            return NORMALIZATION_MAP[lower]

        words = clean.split()

        normalized_words = []

        for word in words:

            word_lower = word.lower()

            if word_lower in {
                "ci/cd",
                "cicd"
            }:
                normalized_words.append("CI/CD")

            elif word_lower in {
                "ai",
                "ml",
                "ui",
                "ux",
                "aws",
                "gcp",
                "api",
                "evm",
                "nft",
                "eth"
            }:
                normalized_words.append(
                    word.upper()
                )

            elif word_lower == "web3":
                normalized_words.append(
                    "Web3"
                )

            else:
                normalized_words.append(
                    word.capitalize()
                )

        return " ".join(
            normalized_words
        )

    def normalize_topic_list(
        self,
        topics: List[str]
    ) -> List[str]:
        """
        Normalize and deduplicate topic lists.
        """

        seen = set()
        normalized = []

        for topic in topics or []:

            norm = self.normalize_topic(
                topic
            )

            if not norm:
                continue

            key = norm.lower()

            if key in seen:
                continue

            seen.add(key)
            normalized.append(norm)

        return normalized

    # ============================================================
    # Weight Normalization
    # ============================================================

    def _normalize_weight_dict(
        self,
        weights: Any
    ) -> Dict[str, float]:
        """
        Preserve learned weights while normalizing topic names.
        """

        result = {}

        if not isinstance(
            weights,
            dict
        ):
            return result

        for topic, weight in weights.items():

            normalized_topic = self.normalize_topic(
                topic
            )

            if not normalized_topic:
                continue

            try:
                weight = float(weight)
            except (TypeError, ValueError):
                weight = 1.0

            weight = self._clamp(
                weight,
                0.10,
                2.00
            )

            # Merge duplicate topic variants by retaining
            # the strongest learned weight.
            existing_key = None

            for existing in result:

                if existing.lower() == normalized_topic.lower():
                    existing_key = existing
                    break

            if existing_key is None:

                result[
                    normalized_topic
                ] = round(
                    weight,
                    4
                )

            else:

                result[
                    existing_key
                ] = max(
                    result[existing_key],
                    round(weight, 4)
                )

        return result

    # ============================================================
    # Goal Concept Derivation
    # ============================================================

    def derive_goal_concepts(
        self,
        goal: str
    ) -> List[Dict[str, Any]]:
        """
        Derive semantic concepts from the explicit current goal.

        Sources:

            1. Rule-based domain expansion.
            2. Semantic similarity against supporting taxonomy.

        The result is user-editable.
        """

        clean_goal = self._clean_text(
            goal
        )

        if not clean_goal:
            return []

        lower_goal = clean_goal.lower()

        concept_weights: Dict[
            str,
            float
        ] = {}

        # --------------------------------------------------------
        # 1. Rule-based goal expansion
        # --------------------------------------------------------

        for key, topics in GOAL_EXPANSIONS.items():

            # Word/phrase boundary matching prevents accidental
            # substring matches such as "ai" inside unrelated words.
            if re.search(
                r"(?<!\w)"
                + re.escape(key)
                + r"(?!\w)",
                lower_goal
            ):

                for topic in topics:

                    normalized_topic = (
                        self.normalize_topic(topic)
                    )

                    if not normalized_topic:
                        continue

                    concept_weights[
                        normalized_topic
                    ] = max(
                        concept_weights.get(
                            normalized_topic,
                            0.0
                        ),
                        0.78
                    )

        # --------------------------------------------------------
        # 2. Semantic projection against taxonomy
        # --------------------------------------------------------

        try:

            goal_vec = (
                self.embedding_service
                .get_embedding(clean_goal)
            )

        except Exception:
            goal_vec = None

        if goal_vec is not None:

            for tax_topic, tax_description in (
                TAXONOMY.items()
            ):

                try:

                    tax_vec = (
                        self.embedding_service
                        .get_embedding(
                            tax_description
                        )
                    )

                    similarity = (
                        self.embedding_service
                        .cosine_similarity(
                            goal_vec,
                            tax_vec
                        )
                    )

                except Exception:
                    continue

                similarity = self._clamp(
                    similarity,
                    -1.0,
                    1.0
                )

                if similarity >= 0.22:

                    normalized_topic = (
                        self.normalize_topic(
                            tax_topic
                        )
                    )

                    scaled_similarity = self._clamp(
                        similarity * 1.20
                    )

                    concept_weights[
                        normalized_topic
                    ] = max(
                        concept_weights.get(
                            normalized_topic,
                            0.0
                        ),
                        scaled_similarity
                    )

        # --------------------------------------------------------
        # 3. Structured editable output
        # --------------------------------------------------------

        results = [

            {
                "topic": topic,
                "weight": round(
                    self._clamp(weight),
                    4
                ),
                "editable": True,
                "source": "goal_derivation"
            }

            for topic, weight
            in sorted(
                concept_weights.items(),
                key=lambda item: item[1],
                reverse=True
            )[:12]
        ]

        return results

    # ============================================================
    # Merge Derived Topics
    # ============================================================

    def _merge_preferred_into_derived(
        self,
        derived_topics: List[Dict[str, Any]],
        preferred_topics: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Ensure explicit preferred topics are represented in the
        derived goal space.

        Explicit preferences receive strong representation but
        do not erase semantic goal concepts.
        """

        result = list(
            derived_topics or []
        )

        index = {
            str(item.get("topic", "")).lower(): item
            for item in result
            if isinstance(item, dict)
        }

        for preferred in preferred_topics:

            key = preferred.lower()

            if key in index:

                existing = index[key]

                existing_weight = float(
                    existing.get(
                        "weight",
                        0.0
                    )
                )

                existing[
                    "weight"
                ] = round(
                    min(
                        1.0,
                        existing_weight + 0.20
                    ),
                    4
                )

                existing[
                    "source"
                ] = "goal_and_user_preference"

            else:

                item = {
                    "topic": preferred,
                    "weight": 0.88,
                    "editable": True,
                    "source": "user_preference"
                }

                result.append(
                    item
                )

                index[key] = item

        result.sort(
            key=lambda item: float(
                item.get(
                    "weight",
                    0.0
                )
            ),
            reverse=True
        )

        return result[:16]

    # ============================================================
    # Negative Configuration
    # ============================================================

    def _build_negative_configuration(
        self,
        unwanted_topics: List[str],
        hard_negatives: Optional[List[str]],
        soft_negatives: Optional[List[str]]
    ):
        """
        Build hard/soft negative lists.

        Explicit hard/soft configuration is respected.
        Unwanted topics are then added to the appropriate layer.
        """

        hard = self.normalize_topic_list(
            hard_negatives or []
        )

        soft = self.normalize_topic_list(
            soft_negatives or []
        )

        # Add explicit unwanted topics.
        for unwanted in unwanted_topics:

            unwanted_lower = unwanted.lower()

            is_hard = any(
                hard_topic.lower() == unwanted_lower
                or hard_topic.lower() in unwanted_lower
                or unwanted_lower in hard_topic.lower()
                for hard_topic in DEFAULT_HARD_NEGATIVES
            )

            if is_hard:

                if not any(
                    unwanted.lower() == item.lower()
                    for item in hard
                ):
                    hard.append(
                        unwanted
                    )

            else:

                if not any(
                    unwanted.lower() == item.lower()
                    for item in soft
                ):

                    soft.append(
                        unwanted
                    )

        # A topic explicitly marked hard should not remain soft.
        hard_keys = {
            item.lower()
            for item in hard
        }

        soft = [
            item
            for item in soft
            if item.lower() not in hard_keys
        ]

        return hard, soft

    # ============================================================
    # Build Intent Profile
    # ============================================================

    def build_intent_profile(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build the complete Intent Profile.

        Important closed-loop behavior:

        If data contains an existing intent profile or learned
        weights, those weights are preserved.

        Therefore:

            feedback
                ↓
            learned weights
                ↓
            build_intent_profile()
                ↓
            learned weights remain
        """

        if not isinstance(
            data,
            dict
        ):
            data = {}

        # ========================================================
        # Existing profile / previous intent
        # ========================================================

        existing_profile = (
            data.get(
                "existing_intent_profile"
            )
            or data.get(
                "intent_profile"
            )
            or {}
        )

        if not isinstance(
            existing_profile,
            dict
        ):
            existing_profile = {}

        existing_current_intent = (
            existing_profile.get(
                "current_intent"
            )
            or {}
        )

        if not isinstance(
            existing_current_intent,
            dict
        ):
            existing_current_intent = {}

        # ========================================================
        # Stable User Profile
        # ========================================================

        raw_profile = (
            data.get(
                "profile"
            )
            or data.get(
                "user_profile"
            )
            or existing_profile.get(
                "user_profile"
            )
            or existing_profile.get(
                "profile"
            )
            or {}
        )

        if not isinstance(
            raw_profile,
            dict
        ):
            raw_profile = {}

        profession = self._clean_text(
            raw_profile.get(
                "profession"
            )
            or data.get(
                "profession"
            )
        )

        interests = self.normalize_topic_list(
            raw_profile.get(
                "interests"
            )
            or data.get(
                "interests"
            )
            or []
        )

        hobbies = self.normalize_topic_list(
            raw_profile.get(
                "hobbies"
            )
            or data.get(
                "hobbies"
            )
            or []
        )

        user_profile = {
            "profession": profession,
            "interests": interests,
            "hobbies": hobbies
        }

        # ========================================================
        # Current Goal
        # ========================================================

        current_goal = self._clean_text(
            data.get(
                "goal"
            )
            or data.get(
                "current_goal"
            )
            or existing_current_intent.get(
                "current_goal"
            )
            or existing_profile.get(
                "goal"
            )
        )

        # ========================================================
        # Explicit Preferences
        # ========================================================

        preferred_topics = self.normalize_topic_list(
            data.get(
                "preferredTopics"
            )
            or data.get(
                "preferred_topics"
            )
            or existing_current_intent.get(
                "preferred_topics"
            )
            or existing_profile.get(
                "preferredTopics"
            )
            or []
        )

        unwanted_topics = self.normalize_topic_list(
            data.get(
                "unwantedTopics"
            )
            or data.get(
                "unwanted_topics"
            )
            or existing_current_intent.get(
                "unwanted_topics"
            )
            or existing_profile.get(
                "unwantedTopics"
            )
            or []
        )

        # ========================================================
        # Goal Keywords
        # ========================================================

        if current_goal:

            try:

                goal_keywords = (
                    self.keyword_service
                    .extract_keywords(
                        current_goal,
                        top_n=8
                    )
                )

            except Exception:

                goal_keywords = []

        else:

            goal_keywords = []

        # ========================================================
        # Derived Goal Concepts
        # ========================================================

        derived_goal_topics = (
            self.derive_goal_concepts(
                current_goal
            )
        )

        derived_goal_topics = (
            self._merge_preferred_into_derived(
                derived_goal_topics,
                preferred_topics
            )
        )

        # ========================================================
        # Preserve learned weights
        # ========================================================

        existing_preferred_weights = (
            existing_current_intent.get(
                "preferred_weights"
            )
            or existing_profile.get(
                "preferred_weights"
            )
            or {}
        )

        existing_unwanted_weights = (
            existing_current_intent.get(
                "unwanted_weights"
            )
            or existing_profile.get(
                "unwanted_weights"
            )
            or {}
        )

        preferred_weights = (
            self._normalize_weight_dict(
                existing_preferred_weights
            )
        )

        unwanted_weights = (
            self._normalize_weight_dict(
                existing_unwanted_weights
            )
        )

        # Initialize explicit topics without destroying learned
        # values.
        for topic in preferred_topics:

            existing = next(
                (
                    item
                    for item in preferred_weights
                    if item.lower() == topic.lower()
                ),
                None
            )

            if existing is None:

                preferred_weights[
                    topic
                ] = 1.0

        for topic in unwanted_topics:

            existing = next(
                (
                    item
                    for item in unwanted_weights
                    if item.lower() == topic.lower()
                ),
                None
            )

            if existing is None:

                unwanted_weights[
                    topic
                ] = 1.0

        # ========================================================
        # Hard / Soft Negatives
        # ========================================================

        hard_negatives = data.get(
            "hard_negatives"
        )

        if hard_negatives is None:

            hard_negatives = (
                existing_current_intent.get(
                    "hard_negatives"
                )
                or existing_profile.get(
                    "settings",
                    {}
                ).get(
                    "hard_negatives",
                    []
                )
            )

        soft_negatives = data.get(
            "soft_negatives"
        )

        if soft_negatives is None:

            soft_negatives = (
                existing_current_intent.get(
                    "soft_negatives"
                )
                or existing_profile.get(
                    "settings",
                    {}
                ).get(
                    "soft_negatives",
                    []
                )
            )

        hard_negatives, soft_negatives = (
            self._build_negative_configuration(
                unwanted_topics,
                hard_negatives,
                soft_negatives
            )
        )

        # ========================================================
        # Strictness
        # ========================================================

        strictness_value = (
            data.get(
                "strictness"
            )
        )

        if strictness_value is None:

            strictness_value = (
                existing_current_intent.get(
                    "strictness"
                )
                or existing_profile.get(
                    "settings",
                    {}
                ).get(
                    "strictness",
                    0.65
                )
            )

        strictness = self._clamp(
            strictness_value,
            0.0,
            1.0
        )

        # ========================================================
        # Session ID
        # ========================================================

        session_id = (
            data.get(
                "session_id"
            )
            or data.get(
                "intent_session_id"
            )
            or existing_current_intent.get(
                "session_id"
            )
            or existing_profile.get(
                "session_id"
            )
            or "session_default"
        )

        session_id = str(
            session_id
        ).strip() or "session_default"

        # ========================================================
        # Observation Window
        # ========================================================

        observation_window = data.get(
            "observation_window"
        )

        if observation_window is None:

            observation_window = (
                existing_profile.get(
                    "settings",
                    {}
                ).get(
                    "observation_window",
                    50
                )
            )

        try:

            observation_window = int(
                observation_window
            )

        except (
            TypeError,
            ValueError
        ):

            observation_window = 50

        observation_window = max(
            10,
            min(
                500,
                observation_window
            )
        )

        # ========================================================
        # Feedback Learning
        # ========================================================

        feedback_learning_enabled = data.get(
            "feedback_learning_enabled"
        )

        if feedback_learning_enabled is None:

            feedback_learning_enabled = (
                existing_profile.get(
                    "settings",
                    {}
                ).get(
                    "feedback_learning_enabled",
                    True
                )
            )

        learning_rate = data.get(
            "learning_rate"
        )

        if learning_rate is None:

            learning_rate = (
                existing_profile.get(
                    "settings",
                    {}
                ).get(
                    "learning_rate",
                    0.05
                )
            )

        try:

            learning_rate = float(
                learning_rate
            )

        except (
            TypeError,
            ValueError
        ):

            learning_rate = 0.05

        learning_rate = self._clamp(
            learning_rate,
            0.001,
            0.50
        )

        # ========================================================
        # Dynamic Topic Space
        # ========================================================
        #
        # The user's topics are primary.
        # Taxonomy is only a supporting semantic reference.
        # ========================================================

        dynamic_topics = []

        dynamic_topics.extend(
            preferred_topics
        )

        dynamic_topics.extend(
            unwanted_topics
        )

        dynamic_topics.extend(
            [
                item.get(
                    "topic"
                )
                for item in derived_goal_topics
                if isinstance(item, dict)
            ]
        )

        dynamic_topics.extend(
            goal_keywords
            if isinstance(
                goal_keywords,
                list
            )
            else []
        )

        dynamic_topics = self.normalize_topic_list(
            dynamic_topics
        )

        # ========================================================
        # Topic Embeddings
        # ========================================================

        topic_embeddings = {}

        for topic in dynamic_topics:

            try:

                embedding = (
                    self.embedding_service
                    .get_embedding(
                        topic
                    )
                )

                if hasattr(
                    embedding,
                    "tolist"
                ):
                    embedding = embedding.tolist()

                topic_embeddings[
                    topic
                ] = embedding

            except Exception:

                # Do not fail the complete profile if one topic
                # embedding cannot be generated.
                continue

        # ========================================================
        # Goal Embedding
        # ========================================================

        goal_embedding = None

        if current_goal:

            try:

                goal_embedding = (
                    self.embedding_service
                    .get_embedding(
                        current_goal
                    )
                )

                if hasattr(
                    goal_embedding,
                    "tolist"
                ):
                    goal_embedding = (
                        goal_embedding.tolist()
                    )

            except Exception:

                goal_embedding = None

        # ========================================================
        # Settings
        # ========================================================

        settings = {
            "enabled": True,

            "strictness": round(
                strictness,
                4
            ),

            "observation_window": (
                observation_window
            ),

            "feedback_learning_enabled": bool(
                feedback_learning_enabled
            ),

            "hard_negatives": hard_negatives,

            "soft_negatives": soft_negatives,

            "min_alignment_threshold": 0.35,

            "learning_rate": round(
                learning_rate,
                4
            )
        }

        # ========================================================
        # Current Intent
        # ========================================================

        current_intent = {
            "session_id": session_id,

            "current_goal": current_goal,

            "preferred_topics": preferred_topics,

            "unwanted_topics": unwanted_topics,

            # IMPORTANT:
            # Preserve learned weights.
            "preferred_weights": preferred_weights,

            "unwanted_weights": unwanted_weights,

            "derived_goal_topics": derived_goal_topics,

            "goal_keywords": goal_keywords,

            "strictness": round(
                strictness,
                4
            ),

            "topic_embeddings": topic_embeddings,

            "goal_embedding": goal_embedding,

            "hard_negatives": hard_negatives,

            "soft_negatives": soft_negatives,

            "topic_space": dynamic_topics
        }

        # ========================================================
        # Learning Metadata
        # ========================================================

        learning_metadata = (
            existing_profile.get(
                "learning_metadata"
            )
            or {}
        )

        if not isinstance(
            learning_metadata,
            dict
        ):
            learning_metadata = {}

        # Rebuild clears the "recalculation required" signal
        # because this method has now rebuilt the profile.
        learning_metadata[
            "recalculation_required"
        ] = False

        # ========================================================
        # Final Intent Profile
        # ========================================================

        return {

            "session_id": session_id,

            "user_profile": user_profile,

            "current_intent": current_intent,

            # ----------------------------------------------------
            # Flat compatibility properties
            # ----------------------------------------------------

            "profile": user_profile,

            "profession": profession,

            "interests": interests,

            "hobbies": hobbies,

            "goal": current_goal,

            "preferredTopics": preferred_topics,

            "unwantedTopics": unwanted_topics,

            "derivedTopics": derived_goal_topics,

            "negativeTopics": unwanted_topics,

            "goalKeywords": goal_keywords,

            "preferred_weights": preferred_weights,

            "unwanted_weights": unwanted_weights,

            "settings": settings,

            "learning_metadata": learning_metadata,

            "topic_space": dynamic_topics,

            "version": "1.3.0"
        }