"""
IntentAlign - Video Content Analysis Service
V1.3

Purpose:
    Analyze YouTube video metadata and create a semantic representation
    that can be compared with the user's CURRENT INTENT.

Main pipeline:

    Video metadata
        ↓
    Multi-field embeddings
        ↓
    Content-type detection
        ↓
    Entity/topic extraction
        ↓
    Homonym disambiguation
        ↓
    Dynamic user-topic scoring
        ↓
    Goal relevance
        ↓
    Positive relevance
        ↓
    Negative relevance
        ↓
    Confidence
        ↓
    Uncertain-state detection

Important:
    This service does NOT modify YouTube's internal recommendation
    algorithm. It only analyzes metadata supplied by the extension.
"""

from typing import Dict, List, Any, Tuple
import re

import numpy as np

from backend.services.intent_service import TAXONOMY


# ============================================================
# CONTENT TYPES
# ============================================================

CONTENT_TYPES = {
    "Education":
        "education academic lecture university coursework lesson "
        "study research academic theory conceptual learning school college",

    "Technology":
        "technology tech gadgets engineering hardware systems "
        "computing infrastructure devices innovation",

    "Programming":
        "programming coding software development algorithms backend "
        "frontend api code developer framework debugging",

    "Interview Preparation":
        "interview preparation placement questions technical interview "
        "mock interview coding round interview questions answers",

    "Tutorial":
        "tutorial step by step complete course crash course "
        "beginner guide walkthrough masterclass how to",

    "News":
        "news breaking news current affairs headlines reporting "
        "journalism bulletin press world events",

    "Sports":
        "sports match athletics tournament league championship "
        "highlights score wickets goals cricket football basketball",

    "Gaming":
        "gaming gameplay playthrough esports walkthrough streamer "
        "lets play video game pc console xbox playstation",

    "Music":
        "music song official audio album lyrics concert singer "
        "artist acoustic cover soundtrack",

    "Entertainment":
        "entertainment movies celebrity trailer drama reaction "
        "vlog web series film actor cinema show",

    "Cartoon":
        "cartoon animated animation episode anime animated series "
        "kids cartoon character",

    "Kids":
        "kids children nursery rhymes preschool baby cartoon "
        "toys playground family storytelling",

    "Comedy":
        "comedy standup funny sketch humor jokes parody prank "
        "hilarious laughter",

    "Travel":
        "travel tourism trip itinerary vacation destination "
        "city tour backpacking hotel guide",

    "Fitness":
        "fitness workout exercise bodybuilding gym training "
        "health wellness cardio muscle yoga",

    "Finance":
        "finance stock market investing trading cryptocurrency "
        "banking portfolio mutual funds money",

    "Cooking":
        "cooking recipe food culinary kitchen dish chef baking "
        "meal dinner dessert ingredients",

    "Other":
        "general miscellaneous unclassified vlog"
}


# ============================================================
# STRONG CONTENT MARKERS
# ============================================================

CARTOON_ENTITIES = [
    "oggy and the cockroaches",
    "oggy",
    "tom and jerry",
    "tom & jerry",
    "peppa pig",
    "cartoon network",
    "nickelodeon",
    "chhota bheem",
    "ben 10",
    "doraemon",
    "shinchan",
    "looney tunes",
    "mickey mouse",
    "donald duck",
    "bugs bunny",
    "popeye",
    "scooby doo",
    "spongebob"
]


GAMING_MARKERS = [
    "gameplay",
    "walkthrough",
    "playthrough",
    "lets play",
    "let's play",
    "gaming",
    "game trailer",
    "gameplay trailer",
    "video game",
    "ps5",
    "ps4",
    "xbox",
    "steam",
    "nintendo",
    "esports",
    "game review"
]


SPORTS_MARKERS = [
    "match highlights",
    "match analysis",
    "cricket match",
    "football match",
    "football highlights",
    "cricket highlights",
    "ipl",
    "fifa world cup",
    "premier league",
    "nba final",
    "wimbledon",
    "olympics"
]


INTERVIEW_MARKERS = [
    "interview questions",
    "interview question",
    "interview preparation",
    "placement interview",
    "mock interview",
    "coding round",
    "technical interview",
    "hr interview",
    "interview questions and answers"
]


TUTORIAL_MARKERS = [
    "full course",
    "complete course",
    "crash course",
    "complete tutorial",
    "tutorial",
    "for beginners",
    "step by step",
    "masterclass",
    "learn",
    "how to"
]


COOKING_MARKERS = [
    "recipe",
    "how to cook",
    "how to bake",
    "ingredients",
    "cooking",
    "chef special",
    "dinner recipe",
    "cake recipe",
    "food recipe"
]


# ============================================================
# HOMONYM RULES
# ============================================================

HOMONYM_DISAMBIGUATION = {

    "python": {
        "incidental_cues": [
            "snake",
            "reptile",
            "wildlife",
            "safari",
            "animal",
            "zoo",
            "forest",
            "venom",
            "cobra",
            "species",
            "predator",
            "constrictor",
            "national geographic",
            "jungle",
            "rainforest"
        ],
        "technical_cues": [
            "code",
            "coding",
            "developer",
            "programming",
            "software",
            "script",
            "library",
            "framework",
            "pandas",
            "numpy",
            "django",
            "flask",
            "syntax",
            "data science",
            "machine learning",
            "pip",
            "tensorflow",
            "scikit"
        ],
        "target_topic": "Programming",
        "related_topics": [
            "Programming",
            "Python",
            "Data Science",
            "Machine Learning"
        ],
        "alt_content_type": "Other",
        "incidental_label": "Wildlife / Snake"
    },

    "java": {
        "incidental_cues": [
            "coffee",
            "roast",
            "beans",
            "espresso",
            "latte",
            "brewing",
            "barista",
            "recipe",
            "cafe",
            "drink",
            "cup of java",
            "starbucks"
        ],
        "technical_cues": [
            "programming",
            "developer",
            "interview",
            "spring",
            "spring boot",
            "jvm",
            "jdk",
            "object oriented",
            "class",
            "code",
            "backend",
            "concurrency"
        ],
        "target_topic": "Programming",
        "related_topics": [
            "Programming",
            "Java",
            "Backend",
            "Software Engineering"
        ],
        "alt_content_type": "Cooking",
        "incidental_label": "Coffee / Beverage"
    },

    "cloud": {
        "incidental_cues": [
            "weather",
            "storm",
            "rain",
            "meteorology",
            "sky",
            "cumulus",
            "atmosphere",
            "climate",
            "forecast",
            "lightning",
            "overcast",
            "weather forecast"
        ],
        "technical_cues": [
            "aws",
            "azure",
            "gcp",
            "devops",
            "kubernetes",
            "docker",
            "serverless",
            "computing",
            "infrastructure",
            "cloud engineer",
            "cloud computing",
            "ec2",
            "s3",
            "lambda"
        ],
        "target_topic": "Cloud",
        "related_topics": [
            "Cloud",
            "AWS",
            "Azure",
            "GCP",
            "DevOps",
            "Infrastructure"
        ],
        "alt_content_type": "Other",
        "incidental_label": "Weather / Meteorology"
    },

    "react": {
        "incidental_cues": [
            "reaction",
            "reacts to",
            "reacting to",
            "try not to laugh",
            "tiktoks",
            "cringe",
            "fails",
            "reaction video",
            "shocked",
            "youtubers react"
        ],
        "technical_cues": [
            "javascript",
            "js",
            "frontend",
            "component",
            "hooks",
            "redux",
            "web development",
            "framework",
            "props",
            "nextjs",
            "virtual dom",
            "react 18",
            "react 19"
        ],
        "target_topic": "Programming",
        "related_topics": [
            "Programming",
            "React",
            "Frontend",
            "Web Development",
            "JavaScript"
        ],
        "alt_content_type": "Entertainment",
        "incidental_label": "Entertainment Reaction"
    },

    "docker": {
        "incidental_cues": [
            "boat",
            "harbor",
            "ship",
            "marina",
            "port",
            "fishing",
            "yacht",
            "cruise",
            "sail",
            "docking",
            "seaport"
        ],
        "technical_cues": [
            "container",
            "dockerfile",
            "kubernetes",
            "devops",
            "compose",
            "containers",
            "deployment",
            "daemon",
            "registry"
        ],
        "target_topic": "DevOps",
        "related_topics": [
            "DevOps",
            "Docker",
            "Containers",
            "Kubernetes",
            "Cloud"
        ],
        "alt_content_type": "Travel",
        "incidental_label": "Maritime / Harbor"
    },

    "pandas": {
        "incidental_cues": [
            "giant panda",
            "bear",
            "zoo",
            "bamboo",
            "cubs",
            "wildlife",
            "animal",
            "sanctuary",
            "eating bamboo",
            "national park"
        ],
        "technical_cues": [
            "dataframe",
            "python",
            "data science",
            "csv",
            "series",
            "data analysis",
            "groupby",
            "merge",
            "dataset",
            "numpy"
        ],
        "target_topic": "Data Science",
        "related_topics": [
            "Data Science",
            "Python",
            "Programming"
        ],
        "alt_content_type": "Other",
        "incidental_label": "Wildlife / Panda"
    },

    "apple": {
        "incidental_cues": [
            "pie",
            "recipe",
            "fruit",
            "orchard",
            "cider",
            "harvest",
            "cake",
            "baking",
            "cinnamon",
            "dessert",
            "apple tree"
        ],
        "technical_cues": [
            "ios",
            "iphone",
            "macbook",
            "swift",
            "swiftui",
            "app store",
            "tech",
            "gadget",
            "developer",
            "wwdc",
            "apple silicon",
            "ipad"
        ],
        "target_topic": "Technology",
        "related_topics": [
            "Technology",
            "Apple",
            "iOS",
            "Mobile Development"
        ],
        "alt_content_type": "Cooking",
        "incidental_label": "Fruit / Food"
    },

    "cricket": {
        "incidental_cues": [
            "insect",
            "chirping",
            "pest",
            "grasshopper",
            "biology",
            "night sounds"
        ],
        "technical_cues": [
            "match",
            "ipl",
            "batting",
            "bowling",
            "icc",
            "t20",
            "world cup",
            "wicket",
            "runs",
            "highlights"
        ],
        "target_topic": "Sports",
        "related_topics": [
            "Sports",
            "Athletics"
        ],
        "alt_content_type": "Other",
        "incidental_label": "Insect / Biology"
    }
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _normalise(text: str) -> str:
    """
    Lowercase and normalize whitespace.
    """
    return re.sub(
        r"\s+",
        " ",
        str(text or "").lower()
    ).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    """
    Safer phrase matching.

    Prevents:
        "react" matching "reaction"
        "java" matching unrelated longer strings

    Multi-word phrases are still allowed.
    """
    text = _normalise(text)
    phrase = _normalise(phrase)

    if not text or not phrase:
        return False

    if " " in phrase:
        return phrase in text

    return bool(
        re.search(
            r"\b" + re.escape(phrase) + r"\b",
            text
        )
    )


def _contains_any(
    text: str,
    phrases: List[str]
) -> bool:

    return any(
        _contains_phrase(text, phrase)
        for phrase in phrases
    )


def _tokenize(text: str) -> List[str]:
    return re.findall(
        r"\b[a-zA-Z0-9+#.-]+\b",
        _normalise(text)
    )


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


# ============================================================
# CONTENT SERVICE
# ============================================================

class ContentService:

    def __init__(
        self,
        embedding_service,
        keyword_service
    ):

        self.embedding_service = (
            embedding_service
        )

        self.keyword_service = (
            keyword_service
        )

        self._taxonomy_embeddings = {}
        self._content_type_embeddings = {}

        self._precompute_embeddings()

    # ========================================================
    # EMBEDDING CACHE
    # ========================================================

    def _precompute_embeddings(self):

        for topic, description in TAXONOMY.items():

            self._taxonomy_embeddings[
                topic
            ] = self.embedding_service.get_embedding(
                f"{topic}: {description}"
            )

        for content_type, description in CONTENT_TYPES.items():

            self._content_type_embeddings[
                content_type
            ] = self.embedding_service.get_embedding(
                f"{content_type}: {description}"
            )

    # ========================================================
    # CONTENT TYPE DETECTION
    # ========================================================

    def detect_content_type(
        self,
        title: str,
        channel: str,
        description: str,
        v_title: np.ndarray,
        v_video: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:

        title_l = _normalise(title)
        channel_l = _normalise(channel)
        desc_l = _normalise(description)

        full_text = (
            f"{title_l} "
            f"{channel_l} "
            f"{desc_l}"
        ).strip()

        # ----------------------------------------------------
        # 1. Strong cartoon entities
        # ----------------------------------------------------

        if _contains_any(
            full_text,
            CARTOON_ENTITIES
        ):

            return (
                "Cartoon",
                0.98,
                {
                    "Cartoon": 0.98,
                    "Kids": 0.90,
                    "Entertainment": 0.82
                }
            )

        # ----------------------------------------------------
        # 2. Gaming
        #
        # IMPORTANT:
        # This comes BEFORE generic Technology.
        #
        # Therefore:
        # "GTA 6 Gameplay"
        # becomes Gaming instead of Technology.
        # ----------------------------------------------------

        if _contains_any(
            title_l,
            GAMING_MARKERS
        ):

            return (
                "Gaming",
                0.96,
                {
                    "Gaming": 0.96,
                    "Entertainment": 0.70
                }
            )

        # ----------------------------------------------------
        # 3. Interview preparation
        # ----------------------------------------------------

        if _contains_any(
            title_l,
            INTERVIEW_MARKERS
        ):

            return (
                "Interview Preparation",
                0.96,
                {
                    "Interview Preparation": 0.96,
                    "Education": 0.82
                }
            )

        # ----------------------------------------------------
        # 4. Cooking
        # ----------------------------------------------------

        if _contains_any(
            title_l,
            COOKING_MARKERS
        ):

            return (
                "Cooking",
                0.95,
                {
                    "Cooking": 0.95
                }
            )

        # ----------------------------------------------------
        # 5. Sports
        # ----------------------------------------------------

        if _contains_any(
            title_l,
            SPORTS_MARKERS
        ):

            return (
                "Sports",
                0.95,
                {
                    "Sports": 0.95
                }
            )

        # ----------------------------------------------------
        # 6. Tutorial
        # ----------------------------------------------------

        if _contains_any(
            title_l,
            TUTORIAL_MARKERS
        ):

            # Do not call every "how to" video Tutorial.
            #
            # Require some educational/technical evidence
            # when possible.

            tutorial_context = [
                "python",
                "java",
                "javascript",
                "programming",
                "coding",
                "kubernetes",
                "docker",
                "devops",
                "data science",
                "machine learning",
                "sql",
                "aws",
                "azure",
                "cloud",
                "react",
                "flutter",
                "android",
                "linux"
            ]

            if _contains_any(
                full_text,
                tutorial_context
            ):

                return (
                    "Tutorial",
                    0.93,
                    {
                        "Tutorial": 0.93,
                        "Education": 0.88
                    }
                )

        # ----------------------------------------------------
        # 7. Homonym checks
        #
        # This is intentionally BEFORE broad semantic
        # classification.
        # ----------------------------------------------------

        for key, rule in HOMONYM_DISAMBIGUATION.items():

            if not _contains_phrase(
                title_l,
                key
            ) and not _contains_phrase(
                channel_l,
                key
            ):

                continue

            has_incidental = _contains_any(
                full_text,
                rule["incidental_cues"]
            )

            has_technical = _contains_any(
                full_text,
                rule["technical_cues"]
            )

            # Strong incidental context wins.
            if has_incidental and not has_technical:

                alt_type = rule.get(
                    "alt_content_type",
                    "Other"
                )

                return (
                    alt_type,
                    0.95,
                    {
                        alt_type: 0.95
                    }
                )

        # ----------------------------------------------------
        # 8. Strong domain-specific technical content
        #
        # We use "Programming" for clearly coding-related
        # videos rather than the broad "Technology" class.
        # ----------------------------------------------------

        programming_markers = [
            "programming",
            "coding",
            "code",
            "developer",
            "software development",
            "javascript",
            "python",
            "java programming",
            "react js",
            "reactjs",
            "node js",
            "nodejs",
            "flutter",
            "android development",
            "sql tutorial",
            "algorithm",
            "data structure",
            "debugging",
            "api development",
            "backend development",
            "frontend development"
        ]

        if _contains_any(
            full_text,
            programming_markers
        ):

            return (
                "Programming",
                0.93,
                {
                    "Programming": 0.93,
                    "Technology": 0.72
                }
            )

        # ----------------------------------------------------
        # 9. DevOps / Cloud / Cybersecurity
        # ----------------------------------------------------

        devops_markers = [
            "devops",
            "kubernetes",
            "docker",
            "jenkins",
            "terraform",
            "ansible",
            "ci/cd",
            "continuous integration",
            "continuous deployment"
        ]

        if _contains_any(
            full_text,
            devops_markers
        ):

            return (
                "Technology",
                0.94,
                {
                    "Technology": 0.94
                }
            )

        # ----------------------------------------------------
        # 10. Generic technology
        # ----------------------------------------------------

        technology_markers = [
            "smartphone",
            "laptop",
            "computer hardware",
            "technology review",
            "tech review",
            "gadget review",
            "processor",
            "gpu",
            "hardware",
            "cybersecurity",
            "network security",
            "firewall",
            "malware",
            "artificial intelligence",
            "machine learning",
            "data science",
            "cloud computing",
            "aws",
            "azure",
            "gcp"
        ]

        if _contains_any(
            full_text,
            technology_markers
        ):

            return (
                "Technology",
                0.92,
                {
                    "Technology": 0.92
                }
            )

        # ----------------------------------------------------
        # 11. Travel
        # ----------------------------------------------------

        travel_markers = [
            "travel",
            "tourism",
            "city tour",
            "vacation",
            "trip to",
            "destination guide",
            "travel vlog",
            "metro tour",
            "subway tour"
        ]

        if _contains_any(
            full_text,
            travel_markers
        ):

            return (
                "Travel",
                0.93,
                {
                    "Travel": 0.93
                }
            )

        # ----------------------------------------------------
        # 12. Semantic fallback
        # ----------------------------------------------------

        type_scores = {}

        for content_type, c_vec in (
            self._content_type_embeddings.items()
        ):

            title_sim = (
                self.embedding_service.cosine_similarity(
                    v_title,
                    c_vec
                )
            )

            video_sim = (
                self.embedding_service.cosine_similarity(
                    v_video,
                    c_vec
                )
            )

            score = (
                0.65 * title_sim
                + 0.35 * video_sim
            )

            type_scores[
                content_type
            ] = round(
                float(score),
                4
            )

        sorted_types = sorted(
            type_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if not sorted_types:
            return (
                "Other",
                0.20,
                {"Other": 0.20}
            )

        top_type, top_score = (
            sorted_types[0]
        )

        return (
            top_type,
            float(top_score),
            dict(sorted_types[:5])
        )

    # ========================================================
    # ENTITY / TOPIC EXTRACTION
    # ========================================================

    def extract_entities_and_topics(
        self,
        title: str,
        channel: str,
        content_type: str
    ) -> Tuple[
        List[str],
        str,
        List[str],
        List[str]
    ]:

        entities = []
        primary_topics = []
        secondary_topics = []

        title = title.strip()

        # ----------------------------------------------------
        # Split title into meaningful parts
        # ----------------------------------------------------

        chunks = re.split(
            r"[-:\|•–~]",
            title
        )

        chunks = [
            c.strip()
            for c in chunks
            if c.strip()
        ]

        if chunks:

            primary_entity = chunks[0]

            entities.append(
                primary_entity
            )

            if len(chunks) > 1:

                secondary_topics.extend(
                    chunks[1:3]
                )

        else:

            primary_entity = ""

        if channel:

            entities.append(
                channel
            )

        # ----------------------------------------------------
        # Content-type topics
        # ----------------------------------------------------

        if content_type in [
            "Cartoon",
            "Kids"
        ]:

            primary_topics.extend([
                primary_entity,
                "Cartoon",
                "Animation",
                "Kids Entertainment"
            ])

        elif content_type == "Programming":

            primary_topics.extend([
                primary_entity,
                "Programming",
                "Software Development"
            ])

        elif content_type == "Interview Preparation":

            primary_topics.extend([
                primary_entity,
                "Interview Preparation",
                "Career"
            ])

        elif content_type == "Sports":

            primary_topics.extend([
                primary_entity,
                "Sports"
            ])

        elif content_type == "Gaming":

            primary_topics.extend([
                primary_entity,
                "Gaming",
                "Video Games"
            ])

        elif content_type == "Cooking":

            primary_topics.extend([
                primary_entity,
                "Cooking",
                "Food"
            ])

        elif content_type == "Travel":

            primary_topics.extend([
                primary_entity,
                "Travel"
            ])

        else:

            primary_topics.extend([
                primary_entity,
                content_type
            ])

        # Remove empty / duplicate values.
        primary_topics = list(dict.fromkeys(
            x for x in primary_topics
            if x
        ))

        secondary_topics = list(
            dict.fromkeys(
                x for x in secondary_topics
                if x
            )
        )

        entities = list(
            dict.fromkeys(
                x for x in entities
                if x
            )
        )

        return (
            primary_topics,
            primary_entity,
            secondary_topics,
            entities
        )

    # ========================================================
    # SEMANTIC CONTRADICTIONS
    # ========================================================

    def check_semantic_contradictions(
        self,
        title: str,
        description: str,
        content_type: str,
        primary_entity: str,
        active_topics: List[str]
    ) -> Dict[str, Any]:

        title_l = _normalise(title)
        desc_l = _normalise(description)

        text = (
            f"{title_l} {desc_l}"
        ).strip()

        incidental_flags = {}

        # ----------------------------------------------------
        # CARTOON / KIDS CONTRADICTION
        # ----------------------------------------------------

        if content_type in [
            "Cartoon",
            "Kids"
        ]:

            serious_topics = [
                "Sports",
                "Programming",
                "Technology",
                "Cybersecurity",
                "DevOps",
                "Cloud",
                "Finance",
                "Career",
                "News",
                "Interview Preparation",
                "Data Science",
                "Machine Learning",
                "Artificial Intelligence",
                "Web Development",
                "Mobile Development",
                "Algorithms",
                "Cooking",
                "Fitness"
            ]

            for topic in serious_topics:

                if topic in active_topics:

                    incidental_flags[
                        topic
                    ] = {
                        "is_incidental": True,
                        "dampening_factor": 0.08,
                        "reason":
                            "Primary content is Cartoon/Kids; "
                            f"{topic} may only be incidental."
                    }

            # Specific props.
            props = {
                "computer": "Programming",
                "laptop": "Programming",
                "code": "Programming",
                "robot": "Technology",
                "sports": "Sports",
                "football": "Sports",
                "cricket": "Sports",
                "game": "Gaming"
            }

            for prop, topic in props.items():

                if _contains_phrase(
                    text,
                    prop
                ):

                    incidental_flags[
                        topic
                    ] = {
                        "is_incidental": True,
                        "dampening_factor": 0.08,
                        "trigger_prop": prop,
                        "reason":
                            "The term appears as an incidental "
                            "element of a Cartoon/Kids video."
                    }

        # ----------------------------------------------------
        # HOMONYMS
        # ----------------------------------------------------

        for key, rule in (
            HOMONYM_DISAMBIGUATION.items()
        ):

            if not _contains_phrase(
                text,
                key
            ):

                continue

            has_incidental = _contains_any(
                text,
                rule["incidental_cues"]
            )

            has_technical = _contains_any(
                text,
                rule["technical_cues"]
            )

            if (
                has_incidental
                and not has_technical
            ):

                target_topic = rule[
                    "target_topic"
                ]

                reason = (
                    f"Semantic context indicates "
                    f"{rule['incidental_label']}."
                )

                incidental_flags[
                    target_topic
                ] = {
                    "is_incidental": True,
                    "dampening_factor": 0.05,
                    "reason": reason
                }

                incidental_flags[
                    key
                ] = {
                    "is_incidental": True,
                    "dampening_factor": 0.05,
                    "reason": reason
                }

                for related in rule.get(
                    "related_topics",
                    []
                ):

                    incidental_flags[
                        related
                    ] = {
                        "is_incidental": True,
                        "dampening_factor": 0.05,
                        "reason": reason
                    }

        return incidental_flags

    # ========================================================
    # DYNAMIC TOPIC SPACE
    # ========================================================

    def _build_active_topic_space(
        self,
        intent: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:

        active_topics = dict(
            self._taxonomy_embeddings
        )

        current_intent = (
            intent.get(
                "current_intent"
            )
            or {}
        )

        preferred = (
            current_intent.get(
                "preferred_topics"
            )
            or intent.get(
                "preferredTopics"
            )
            or []
        )

        unwanted = (
            current_intent.get(
                "unwanted_topics"
            )
            or intent.get(
                "unwantedTopics"
            )
            or []
        )

        derived = (
            current_intent.get(
                "derived_goal_topics"
            )
            or intent.get(
                "derivedTopics"
            )
            or []
        )

        topic_embeddings = (
            current_intent.get(
                "topic_embeddings"
            )
            or {}
        )

        user_topics = []

        user_topics.extend(
            preferred
        )

        user_topics.extend(
            unwanted
        )

        for item in derived:

            if isinstance(item, dict):

                topic = item.get(
                    "topic"
                )

                if topic:
                    user_topics.append(
                        topic
                    )

            elif item:

                user_topics.append(
                    str(item)
                )

        for topic in user_topics:

            if not topic:
                continue

            topic = str(topic).strip()

            if not topic:
                continue

            if topic in topic_embeddings:

                try:

                    active_topics[
                        topic
                    ] = np.asarray(
                        topic_embeddings[
                            topic
                        ],
                        dtype=np.float32
                    )

                    continue

                except Exception:
                    pass

            if topic not in active_topics:

                active_topics[
                    topic
                ] = (
                    self.embedding_service
                    .get_embedding(topic)
                )

        return active_topics

    # ========================================================
    # GOAL KEYWORD EXTRACTION
    # ========================================================

    def _get_goal_tokens(
        self,
        goal: str
    ) -> List[str]:

        stop_words = {
            "a", "an", "the",
            "for", "to", "in",
            "and", "or", "of",
            "with", "is", "on",
            "at", "by", "from",
            "my", "your", "our",
            "learn", "learning",
            "master", "prepare",
            "preparing", "study",
            "studying",
            "understand",
            "understanding"
        }

        tokens = _tokenize(
            goal
        )

        return [
            token
            for token in tokens
            if len(token) > 2
            and token not in stop_words
        ]

    # ========================================================
    # TOKEN MATCH
    # ========================================================

    def _tokens_match(
        self,
        a: str,
        b: str
    ) -> bool:

        a = a.lower()
        b = b.lower()

        if a == b:
            return True

        if len(a) >= 5 and len(b) >= 5:

            if (
                a.startswith(b)
                or b.startswith(a)
            ):
                return True

        # Simple plural normalization.
        if (
            len(a) >= 5
            and len(b) >= 5
            and a.rstrip("s") == b.rstrip("s")
        ):

            return True

        return False

    # ========================================================
    # MAIN ANALYSIS
    # ========================================================

    def analyze_content(
        self,
        video: Dict[str, Any],
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:

        # ----------------------------------------------------
        # BASIC VIDEO DATA
        # ----------------------------------------------------

        video_id = str(
            video.get("video_id")
            or video.get("id")
            or ""
        ).strip()

        title = str(
            video.get("title")
            or ""
        ).strip()

        channel = str(
            video.get("channel")
            or ""
        ).strip()

        description = str(
            video.get("description")
            or ""
        ).strip()

        title_l = _normalise(title)
        channel_l = _normalise(channel)
        desc_l = _normalise(description)

        full_text = (
            f"{title_l} "
            f"{channel_l} "
            f"{desc_l}"
        ).strip()

        # ----------------------------------------------------
        # KEYWORDS
        # ----------------------------------------------------

        try:

            keywords = (
                self.keyword_service
                .extract_keywords(
                    f"{title} {description}".strip(),
                    top_n=8
                )
            )

        except Exception:

            keywords = []

        keyword_strings = []

        for item in keywords:

            if isinstance(item, dict):

                value = item.get(
                    "keyword",
                    ""
                )

            else:

                value = str(item)

            if value:
                keyword_strings.append(
                    value
                )

        keyword_text = " ".join(
            keyword_strings
        )

        # ----------------------------------------------------
        # MULTI-FIELD REPRESENTATION
        #
        # Title       50%
        # Description 25%
        # Channel     10%
        # Keywords    15%
        # ----------------------------------------------------

        v_title = (
            self.embedding_service
            .get_embedding(title)
        )

        v_desc = (
            self.embedding_service
            .get_embedding(
                description
            )
            if description
            else v_title
        )

        v_channel = (
            self.embedding_service
            .get_embedding(channel)
            if channel
            else v_title
        )

        v_keywords = (
            self.embedding_service
            .get_embedding(
                keyword_text
            )
            if keyword_text
            else v_title
        )

        combined_vector = (
            0.50 * v_title
            + 0.25 * v_desc
            + 0.10 * v_channel
            + 0.15 * v_keywords
        )

        norm = np.linalg.norm(
            combined_vector
        )

        if norm > 0:

            video_vector = (
                combined_vector / norm
            )

        else:

            video_vector = combined_vector

        # ----------------------------------------------------
        # CONTENT TYPE
        # ----------------------------------------------------

        (
            content_type,
            type_confidence,
            type_scores
        ) = self.detect_content_type(
            title,
            channel,
            description,
            v_title,
            video_vector
        )

        # ----------------------------------------------------
        # TOPICS / ENTITIES
        # ----------------------------------------------------

        (
            primary_topics,
            primary_entity,
            secondary_candidates,
            entities
        ) = self.extract_entities_and_topics(
            title,
            channel,
            content_type
        )

        # ----------------------------------------------------
        # CURRENT INTENT
        # ----------------------------------------------------

        current_intent = (
            intent.get(
                "current_intent"
            )
            or {}
        )

        preferred_topics = (
            current_intent.get(
                "preferred_topics"
            )
            or intent.get(
                "preferredTopics"
            )
            or []
        )

        unwanted_topics = (
            current_intent.get(
                "unwanted_topics"
            )
            or intent.get(
                "unwantedTopics"
            )
            or []
        )

        derived_topics = (
            current_intent.get(
                "derived_goal_topics"
            )
            or intent.get(
                "derivedTopics"
            )
            or []
        )

        # Make sure everything is string based.
        preferred_topics = [
            str(x).strip()
            for x in preferred_topics
            if str(x).strip()
        ]

        unwanted_topics = [
            str(x).strip()
            for x in unwanted_topics
            if str(x).strip()
        ]

        # ----------------------------------------------------
        # DYNAMIC TOPIC SPACE
        # ----------------------------------------------------

        active_topics = (
            self._build_active_topic_space(
                intent
            )
        )

        # ----------------------------------------------------
        # CONTRADICTIONS
        # ----------------------------------------------------

        incidental_flags = (
            self.check_semantic_contradictions(
                title,
                description,
                content_type,
                primary_entity,
                list(active_topics.keys())
            )
        )

        # ----------------------------------------------------
        # TOPIC SCORES
        # ----------------------------------------------------

        topic_scores = {}

        for topic, topic_vector in (
            active_topics.items()
        ):

            try:

                base_similarity = float(
                    self.embedding_service
                    .cosine_similarity(
                        video_vector,
                        topic_vector
                    )
                )

            except Exception:

                base_similarity = 0.0

            score = base_similarity

            topic_lower = (
                topic.lower().strip()
            )

            # ------------------------------------------------
            # Incidental topic
            # ------------------------------------------------

            flag = (
                incidental_flags.get(
                    topic
                )
                or incidental_flags.get(
                    topic_lower
                )
            )

            if flag:

                score *= float(
                    flag.get(
                        "dampening_factor",
                        0.08
                    )
                )

            else:

                # ------------------------------------------------
                # Exact content type relation
                # ------------------------------------------------

                if (
                    content_type.lower()
                    == topic_lower
                ):

                    score = max(
                        score,
                        0.82
                    )

                # ------------------------------------------------
                # Exact phrase in title.
                #
                # This is intentionally smaller than before.
                # A single word must NOT create a 1.0 match.
                # ------------------------------------------------

                if _contains_phrase(
                    title_l,
                    topic_lower
                ):

                    score = min(
                        1.0,
                        score + 0.18
                    )

                # ------------------------------------------------
                # Exact phrase in channel
                # ------------------------------------------------

                elif _contains_phrase(
                    channel_l,
                    topic_lower
                ):

                    score = min(
                        1.0,
                        score + 0.08
                    )

                # ------------------------------------------------
                # Description evidence
                # ------------------------------------------------

                elif _contains_phrase(
                    desc_l,
                    topic_lower
                ):

                    score = min(
                        1.0,
                        score + 0.06
                    )

            topic_scores[
                topic
            ] = round(
                max(
                    0.0,
                    min(
                        1.0,
                        float(score)
                    )
                ),
                4
            )

        sorted_topics = dict(
            sorted(
                topic_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )

        # ====================================================
        # GOAL RELEVANCE - V1.4
        # ====================================================
        #
        # IMPORTANT:
        # Generic words such as:
        #   interview, questions, preparation, tutorial,
        #   course, learning, etc.
        # must NOT dominate the current goal.
        #
        # Example:
        #   Current goal = "Prepare for Kubernetes interview"
        #
        # Correct:
        #   Kubernetes video        -> HIGH
        #   Kubernetes + DevOps     -> HIGH
        #   Docker + Kubernetes     -> HIGH
        #   Linux for Kubernetes    -> MEDIUM/HIGH
        #
        # Incorrect:
        #   Java interview           -> LOW
        #   Data Analyst interview   -> LOW
        #   HR interview             -> LOW
        #
        # The subject of the goal is more important than the
        # format of the video.
        # ====================================================

        goal = str(
            current_intent.get("current_goal")
            or intent.get("goal")
            or ""
        ).strip()

        goal_lower = _normalise(goal)

        # ----------------------------------------------------
        # 1. Remove generic "format" words from the goal.
        # ----------------------------------------------------

        GENERIC_GOAL_WORDS = {
            "a", "an", "the",
            "for", "to", "in", "on", "at",
            "and", "or", "of", "with",
            "my", "your", "our",

            # Goal/action words
            "prepare",
            "preparing",
            "preparation",
            "learn",
            "learning",
            "study",
            "studying",
            "understand",
            "understanding",
            "master",
            "mastering",
            "practice",
            "practicing",

            # Generic video/interview words
            "interview",
            "interviews",
            "question",
            "questions",
            "answer",
            "answers",
            "mock",
            "round",
            "rounds",
            "technical",
            "hr",
            "placement",

            # Generic learning/video format
            "tutorial",
            "tutorials",
            "course",
            "courses",
            "lesson",
            "lessons",
            "guide",
            "guides",
            "beginner",
            "beginners",
            "complete",
            "crash",
            "full",
            "step",
            "steps",
            "how",
            "what",
            "why",
            "basics",
            "basic"
        }


        def extract_specific_goal_tokens(text):
            tokens = _tokenize(text)

            return [
                token
                for token in tokens
                if len(token) > 2
                and token not in GENERIC_GOAL_WORDS
            ]


        # ----------------------------------------------------
        # 2. Specific subject tokens from CURRENT GOAL.
        # ----------------------------------------------------

        goal_tokens = extract_specific_goal_tokens(goal)


        # ----------------------------------------------------
        # 3. Add goal keywords generated by IntentService.
        # ----------------------------------------------------

        goal_keywords = (
            current_intent.get("goal_keywords")
            or intent.get("goalKeywords")
            or []
        )

        for item in goal_keywords:

            if isinstance(item, dict):

                keyword = str(
                    item.get("keyword", "")
                ).strip()

            else:

                keyword = str(item).strip()

            if not keyword:
                continue

            for token in extract_specific_goal_tokens(keyword):

                if token not in goal_tokens:

                    goal_tokens.append(token)


        # ----------------------------------------------------
        # 4. Add DERIVED GOAL TOPICS.
        #
        # Example:
        #
        # Prepare for Kubernetes interview
        #
        # may become:
        # Kubernetes
        # DevOps
        # Docker
        # Containers
        # Cloud
        # Linux
        # ----------------------------------------------------

        derived_goal_topics = (
            current_intent.get("derived_goal_topics")
            or intent.get("derivedTopics")
            or []
        )

        goal_concepts = []

        for token in goal_tokens:

            if token:
                goal_concepts.append(token)


        for item in derived_goal_topics:

            if isinstance(item, dict):

                topic = str(
                    item.get("topic", "")
                ).strip()

            else:

                topic = str(item).strip()

            if not topic:
                continue

            topic_lower = _normalise(topic)

            # Do not treat generic categories as goal subjects.
            if topic_lower in {
                "interview",
                "interview preparation",
                "preparation",
                "questions",
                "answers",
                "tutorial",
                "education",
                "learning",
                "course",
                "career"
            }:
                continue

            if topic not in goal_concepts:

                goal_concepts.append(topic)


        goal_concepts = list(
            dict.fromkeys(goal_concepts)
        )


        # ----------------------------------------------------
        # 5. Calculate SUBJECT MATCH.
        # ----------------------------------------------------

        subject_scores = []

        matched_goal_concepts = []

        for concept in goal_concepts:

            concept_lower = _normalise(
                concept
            )

            if not concept_lower:
                continue

            score = 0.0
            evidence = "none"

            # -----------------------------------------------
            # Exact title match = strongest evidence
            # -----------------------------------------------

            if _contains_phrase(
                title_l,
                concept_lower
            ):

                score = 1.0
                evidence = "title"

            # -----------------------------------------------
            # Channel match
            # -----------------------------------------------

            elif _contains_phrase(
                channel_l,
                concept_lower
            ):

                score = 0.90
                evidence = "channel"

            # -----------------------------------------------
            # Description match
            # -----------------------------------------------

            elif _contains_phrase(
                desc_l,
                concept_lower
            ):

                score = 0.75
                evidence = "description"

            # -----------------------------------------------
            # Existing semantic topic score
            # -----------------------------------------------

            else:

                for topic, topic_score in sorted_topics.items():

                    if (
                        _normalise(topic)
                        == concept_lower
                    ):

                        score = float(
                            topic_score
                        )

                        evidence = "topic"

                        break

            # -----------------------------------------------
            # Token-level matching
            # -----------------------------------------------

            if score == 0.0:

                concept_tokens = extract_specific_goal_tokens(
                    concept
                )

                video_tokens = set(
                    _tokenize(
                        f"{title} {channel}"
                    )
                )

                matched_tokens = 0

                for ct in concept_tokens:

                    for vt in video_tokens:

                        if self._tokens_match(
                            ct,
                            vt
                        ):

                            matched_tokens += 1
                            break

                if matched_tokens > 0:

                    score = min(
                        1.0,
                        0.45
                        + (
                            0.20
                            * matched_tokens
                        )
                    )

                    evidence = "token"

            if score > 0:

                subject_scores.append(
                    score
                )

                if score >= 0.35:

                    matched_goal_concepts.append({
                        "concept": concept,
                        "score": round(
                            score,
                            4
                        ),
                        "evidence": evidence
                    })


        # Strongest subject match.
        specific_subject_score = max(
            subject_scores,
            default=0.0
        )


        # ----------------------------------------------------
        # 6. Whole-goal semantic similarity.
        #
        # This is intentionally LOWER weighted.
        #
        # Otherwise:
        #
        # "Java interview"
        #
        # can look too similar to:
        #
        # "Kubernetes interview"
        #
        # simply because both contain "interview".
        # ----------------------------------------------------

        semantic_goal = 0.0

        if goal:

            try:

                cached_goal_embedding = (
                    current_intent.get(
                        "goal_embedding"
                    )
                )

                if cached_goal_embedding:

                    goal_vector = np.asarray(
                        cached_goal_embedding,
                        dtype=np.float32
                    )

                else:

                    goal_vector = (
                        self.embedding_service
                        .get_embedding(
                            goal
                        )
                    )

                semantic_goal = float(
                    self.embedding_service
                    .cosine_similarity(
                        video_vector,
                        goal_vector
                    )
                )

            except Exception:

                semantic_goal = 0.0


        # ----------------------------------------------------
        # 7. Exact subject evidence.
        # ----------------------------------------------------

        exact_subject_score = 0.0

        for concept in goal_concepts:

            concept_lower = _normalise(
                concept
            )

            if _contains_phrase(
                title_l,
                concept_lower
            ):

                exact_subject_score = max(
                    exact_subject_score,
                    1.0
                )

            elif _contains_phrase(
                channel_l,
                concept_lower
            ):

                exact_subject_score = max(
                    exact_subject_score,
                    0.90
                )

            elif _contains_phrase(
                desc_l,
                concept_lower
            ):

                exact_subject_score = max(
                    exact_subject_score,
                    0.75
                )


        # ----------------------------------------------------
        # 8. Goal token overlap.
        # ----------------------------------------------------

        video_tokens = set(
            _tokenize(
                f"{title} {channel}"
            )
        )

        matched_goal_tokens = set()

        for goal_token in goal_tokens:

            for video_token in video_tokens:

                if self._tokens_match(
                    goal_token,
                    video_token
                ):

                    matched_goal_tokens.add(
                        goal_token
                    )

                    break


        if goal_tokens:

            goal_token_overlap = (
                len(matched_goal_tokens)
                / len(goal_tokens)
            )

        else:

            goal_token_overlap = 0.0


        # ----------------------------------------------------
        # 9. Entertainment contradiction.
        # ----------------------------------------------------

        entertainment_types = {
            "Cartoon",
            "Kids",
            "Comedy",
            "Entertainment"
        }

        entertainment_goal = any(
            word in goal_lower
            for word in [
                "cartoon",
                "animation",
                "anime",
                "comedy",
                "entertainment"
            ]
        )


        # ----------------------------------------------------
        # 10. FINAL GOAL RELEVANCE.
        #
        # SUBJECT MATCH = 50%
        # EXACT SUBJECT = 20%
        # TOKEN OVERLAP = 15%
        # SEMANTIC GOAL = 15%
        # ----------------------------------------------------

        if (
            content_type in entertainment_types
            and goal
            and not entertainment_goal
        ):

            goal_relevance = min(
                0.08,
                (
                    0.10
                    * semantic_goal
                )
                +
                (
                    0.10
                    * specific_subject_score
                )
            )

        else:

            goal_relevance = (

                0.50
                * specific_subject_score

                +

                0.20
                * exact_subject_score

                +

                0.15
                * goal_token_overlap

                +

                0.15
                * min(
                    semantic_goal,
                    0.75
                )
            )


        # ----------------------------------------------------
        # 11. Strong exact subject match should guarantee
        # a meaningful score.
        # ----------------------------------------------------

        if (
            exact_subject_score >= 0.90
            or goal_token_overlap >= 0.50
        ):

            goal_relevance = max(
                goal_relevance,
                0.70
            )


        # ----------------------------------------------------
        # 12. Incidental/homonym contradiction.
        # ----------------------------------------------------

        if incidental_flags:

            goal_relevance *= 0.25


        goal_relevance = round(
            max(
                0.0,
                min(
                    1.0,
                    goal_relevance
                )
            ),
            4
        )
        
        # ====================================================
        # POSITIVE RELEVANCE
        # ====================================================

        preferred_weights = (
            current_intent.get(
                "preferred_weights"
            )
            or {
                topic: 1.0
                for topic in preferred_topics
            }
        )

        positive_candidates = []

        for preferred in preferred_topics:

            score = float(
                sorted_topics.get(
                    preferred,
                    0.0
                )
            )

            # Incidental preferred topic
            if (
                preferred in incidental_flags
                or preferred.lower()
                in incidental_flags
            ):

                score *= 0.08

            weight = _safe_float(
                preferred_weights.get(
                    preferred,
                    1.0
                ),
                1.0
            )

            positive_candidates.append(
                score * weight
            )

        # Derived goal topics
        for item in derived_topics:

            if isinstance(item, dict):

                topic = str(
                    item.get(
                        "topic",
                        ""
                    )
                ).strip()

                weight = _safe_float(
                    item.get(
                        "weight",
                        0.5
                    ),
                    0.5
                )

            else:

                topic = str(
                    item
                ).strip()

                weight = 0.5

            if not topic:
                continue

            score = float(
                sorted_topics.get(
                    topic,
                    0.0
                )
            )

            if (
                topic in incidental_flags
                or topic.lower()
                in incidental_flags
            ):

                score *= 0.08

            positive_candidates.append(
                score * weight
            )

        positive_relevance = round(
            max(
                positive_candidates,
                default=0.0
            ),
            4
        )

        positive_relevance = min(
            1.0,
            positive_relevance
        )

        # ====================================================
        # NEGATIVE RELEVANCE
        # ====================================================

        unwanted_weights = (
            current_intent.get(
                "unwanted_weights"
            )
            or {
                topic: 1.0
                for topic in unwanted_topics
            }
        )

        negative_candidates = []
        matched_negative = []

        for unwanted in unwanted_topics:

            score = float(
                sorted_topics.get(
                    unwanted,
                    0.0
                )
            )

            # Exact unwanted phrase in title/description.
            if _contains_phrase(
                full_text,
                unwanted.lower()
            ):

                score = max(
                    score,
                    0.88
                )

            weight = _safe_float(
                unwanted_weights.get(
                    unwanted,
                    1.0
                ),
                1.0
            )

            weighted = (
                score * weight
            )

            negative_candidates.append(
                weighted
            )

            if score >= 0.35:

                matched_negative.append(
                    unwanted
                )

        # Content type negative evidence.
        for unwanted in unwanted_topics:

            unwanted_l = (
                unwanted.lower()
            )

            content_type_l = (
                content_type.lower()
            )

            if (
                unwanted_l
                == content_type_l
                or unwanted_l
                in content_type_l
                or content_type_l
                in unwanted_l
            ):

                negative_candidates.append(
                    0.85
                )

                if unwanted not in matched_negative:

                    matched_negative.append(
                        unwanted
                    )

        negative_relevance = round(
            min(
                1.0,
                max(
                    negative_candidates,
                    default=0.0
                )
            ),
            4
        )

        # ====================================================
        # CONFIDENCE
        # ====================================================

        topic_values = list(
            sorted_topics.values()
        )

        top1 = (
            topic_values[0]
            if topic_values
            else 0.0
        )

        top2 = (
            topic_values[1]
            if len(topic_values) > 1
            else 0.0
        )

        # Top-vs-second separation.
        margin = max(
            0.0,
            top1 - top2
        )

        # Metadata quality.
        title_quality = min(
            1.0,
            len(title) / 40.0
        )

        description_quality = (
            min(
                1.0,
                len(description) / 120.0
            )
            if description
            else 0.0
        )

        metadata_quality = (
            0.60 * title_quality
            + 0.40 * description_quality
        )

        keyword_quality = min(
            1.0,
            len(keywords) / 5.0
        )

        contradiction_penalty = (
            0.20
            if incidental_flags
            else 0.0
        )

        confidence = (
            0.30 * top1
            + 0.20 * margin
            + 0.20 * type_confidence
            + 0.20 * metadata_quality
            + 0.10 * keyword_quality
            - contradiction_penalty
        )

        confidence = round(
            max(
                0.10,
                min(
                    1.0,
                    confidence
                )
            ),
            4
        )

        # ====================================================
        # UNCERTAIN STATE
        # ====================================================

        is_uncertain = False

        if (
            confidence < 0.35
            and goal_relevance < 0.30
            and positive_relevance < 0.30
            and negative_relevance < 0.30
        ):

            is_uncertain = True

        elif (
            top1 < 0.25
            and goal_relevance < 0.20
            and positive_relevance < 0.20
            and negative_relevance < 0.25
        ):

            is_uncertain = True

        # ====================================================
        # ACTIVE INCIDENTAL FLAGS
        # ====================================================

        user_intent_words = set(
            self._get_goal_tokens(
                goal
            )
        )

        for topic in preferred_topics:

            user_intent_words.update(
                self._get_goal_tokens(
                    topic
                )
            )

        for item in derived_topics:

            if isinstance(item, dict):

                topic = str(
                    item.get(
                        "topic",
                        ""
                    )
                )

            else:

                topic = str(item)

            user_intent_words.update(
                self._get_goal_tokens(
                    topic
                )
            )

        active_incidental_flags = {}

        for flag_topic, flag_data in (
            incidental_flags.items()
        ):

            flag_words = set(
                self._get_goal_tokens(
                    str(flag_topic)
                )
            )

            trigger = str(
                flag_data.get(
                    "trigger_prop",
                    ""
                )
            ).lower()

            present = (
                _contains_phrase(
                    title_l,
                    str(flag_topic).lower()
                )
                or _contains_phrase(
                    full_text,
                    trigger
                )
            )

            if (
                present
                and (
                    not user_intent_words
                    or flag_words
                    & user_intent_words
                )
            ):

                active_incidental_flags[
                    flag_topic
                ] = flag_data

        # ====================================================
        # SECONDARY TOPICS
        # ====================================================

        secondary_topics = list(
            secondary_candidates
        )

        for flag_topic in (
            incidental_flags.keys()
        ):

            if (
                flag_topic
                not in secondary_topics
            ):

                secondary_topics.append(
                    flag_topic
                )

        secondary_topics = list(
            dict.fromkeys(
                x for x in secondary_topics
                if x
            )
        )

        # ====================================================
        # DEBUG ANALYSIS
        # ====================================================

        debug_analysis = {

            "title":
                title,

            "channel":
                channel,

            "primary_topics":
                primary_topics[:6],

            "secondary_topics":
                secondary_topics[:6],

            "content_type":
                content_type,

            "content_type_scores":
                type_scores,

            "entities":
                entities[:6],

            "is_incidental":
                bool(
                    active_incidental_flags
                ),

            "incidental_details":
                active_incidental_flags,

            "goal_relevance":
                goal_relevance,

            "positive_relevance":
                positive_relevance,

            "negative_relevance":
                negative_relevance,

            "confidence":
                confidence,

            "is_uncertain":
                is_uncertain,

            "goal":
                goal,

            "matched_goal_tokens":
                sorted(
                    matched_goal_tokens
                ),

            "matched_negative":
                matched_negative,

            "top_topics":
                list(
                    sorted_topics.items()
                )[:10]
        }

        # ====================================================
        # FINAL RESULT
        # ====================================================

        return {

            "video_id":
                video_id,

            "title":
                title,

            "channel":
                channel,

            "description":
                description,

            "keywords":
                keywords,

            "topics":
                sorted_topics,

            "primary_topics":
                primary_topics[:6],

            "secondary_topics":
                secondary_topics[:6],

            "content_type":
                content_type,

            "entities":
                entities[:6],

            "is_incidental":
                bool(
                    active_incidental_flags
                ),

            "positive_relevance":
                positive_relevance,

            "negative_relevance":
                negative_relevance,

            "goal_relevance":
                goal_relevance,

            "matched_negative":
                matched_negative,

            "confidence":
                confidence,

            "is_uncertain":
                is_uncertain,

            "debug_analysis":
                debug_analysis
        }