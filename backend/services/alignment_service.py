"""
IntentAlign - Alignment Service
V1.4 - Current Goal Subject Gate

Purpose
-------
Combines the video analysis produced by ContentService with the
USER'S CURRENT INTENT.

Priority:

1. Current Goal
2. Derived Goal Topics
3. Preferred Topics
4. Stable Interests / Profession
5. Hobbies / Context

V1.4 additions
---------------
- Current Goal Subject Gate
- Goal-specific subject evidence
- Derived-topic subject evidence
- Prevents generic interview/technology videos from receiving
  high alignment when they do not match the actual goal subject.
- Generic goals such as "prepare for interview" remain unrestricted.
- Keeps alignment separate from intervention.
- Preserves hard-negative / soft-negative behavior.
- Preserves feedback-adjusted preferred topic weights.
"""

from typing import Dict, Any, List, Set, Tuple
import re


class AlignmentService:

    # ============================================================
    # DEFAULT CONFIGURATION
    # ============================================================

    DEFAULT_HARD_NEGATIVES = {
        "Politics",
        "Celebrity Gossip",
        "Adult Content",
    }

    DEFAULT_SOFT_NEGATIVES = {
        "News",
        "Entertainment",
        "Gaming",
        "Sports",
    }

    # Words that describe the PURPOSE of the goal rather than
    # the SUBJECT of the goal.
    #
    # Example:
    # "Prepare for Kubernetes interview"
    #
    # Subject:
    #     Kubernetes
    #
    # Generic purpose:
    #     Prepare / Interview
    GENERIC_GOAL_TERMS = {
        "prepare",
        "preparation",
        "prep",
        "study",
        "learn",
        "learning",
        "practice",
        "practise",
        "understand",
        "master",
        "improve",
        "revise",
        "revision",
        "review",
        "reviewing",
        "explore",
        "exploring",
        "watch",
        "watching",
        "course",
        "courses",
        "tutorial",
        "tutorials",
        "guide",
        "guides",
        "roadmap",
        "roadmaps",
        "interview",
        "interviews",
        "interviewing",
        "exam",
        "exams",
        "test",
        "tests",
        "assessment",
        "assessments",
        "placement",
        "placements",
        "job",
        "jobs",
        "career",
        "careers",
        "question",
        "questions",
        "questionnaire",
        "skills",
        "skill",
        "knowledge",
        "basics",
        "basic",
        "advanced",
        "beginner",
        "beginners",
        "complete",
        "full",
        "best",
        "top",
        "important",
        "important",
        "tips",
        "tip",
        "how",
        "what",
        "why",
        "using",
        "use",
        "with",
        "for",
        "the",
        "a",
        "an",
        "to",
        "of",
        "in",
        "on",
        "and",
        "or",
        "my",
        "your",
        "our",
        "from",
        "by",
        "about",
        "into",
        "towards",
        "toward",
    }

    # Generic technical words that should not themselves define
    # a specific subject.
    GENERIC_TECHNICAL_TERMS = {
        "technology",
        "technologies",
        "technical",
        "tech",
        "software",
        "engineering",
        "engineer",
        "developer",
        "developers",
        "development",
        "programming",
        "coding",
        "code",
        "computer",
        "computing",
        "it",
        "information",
        "information technology",
        "system",
        "systems",
        "application",
        "applications",
        "development",
        "software development",
        "technical interview",
        "technical interviews",
    }

    # Common subject aliases.
    SUBJECT_ALIASES = {
        "k8s": "kubernetes",
        "kube": "kubernetes",
        "js": "javascript",
        "ts": "typescript",
        "nodejs": "node",
        "node js": "node",
        "reactjs": "react",
        "react js": "react",
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "dl": "deep learning",
        "nlp": "natural language processing",
        "db": "database",
        "dbms": "database",
        "sql": "sql",
        "aws": "aws",
        "azure": "azure",
        "gcp": "gcp",
        "ci cd": "ci/cd",
        "cicd": "ci/cd",
        "cyber security": "cybersecurity",
        "cyber security": "cybersecurity",
    }

    def __init__(
        self,
        weight_goal: float = 0.45,
        weight_derived: float = 0.20,
        weight_preferred: float = 0.20,
        weight_interests: float = 0.10,
        weight_hobbies: float = 0.05,
        weight_negative: float = 0.25,
        weight_positive: float = None,
        subject_gate_cap: float = 0.38,
        subject_gate_soft_cap: float = 0.50,
        **kwargs
    ):
        self.w_goal = float(weight_goal)
        self.w_derived = float(weight_derived)
        self.w_preferred = float(weight_preferred)
        self.w_interests = float(weight_interests)
        self.w_hobbies = float(weight_hobbies)
        self.w_neg = float(weight_negative)

        # If a specific subject exists in the current goal but the
        # video does not contain meaningful evidence for that subject,
        # the alignment cannot remain high.
        self.subject_gate_cap = float(subject_gate_cap)

        # Used when weak evidence exists but the subject match is not
        # strong enough to treat as a full match.
        self.subject_gate_soft_cap = float(subject_gate_soft_cap)

    # ============================================================
    # BASIC HELPERS
    # ============================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0
    ) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _normalise(
        value: Any
    ) -> str:
        text = str(value or "").strip().lower()

        # Normalize common punctuation.
        text = text.replace("_", " ")
        text = text.replace("-", " ")
        text = text.replace("/", " / ")

        # Collapse whitespace.
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    @staticmethod
    def _contains(
        text: str,
        phrase: str
    ) -> bool:

        text = str(text or "").lower()
        phrase = str(phrase or "").lower().strip()

        if not text or not phrase:
            return False

        return phrase in text

    @staticmethod
    def _tokenize(
        text: str
    ) -> List[str]:

        text = str(text or "").lower()

        return re.findall(
            r"[a-zA-Z0-9]+(?:[+#./-][a-zA-Z0-9]+)*",
            text
        )

    @staticmethod
    def _canonical_topic(
        topic: Any
    ) -> str:

        topic = str(topic or "").strip().lower()

        topic = re.sub(
            r"\s+",
            " ",
            topic
        )

        if topic in AlignmentService.SUBJECT_ALIASES:
            return AlignmentService.SUBJECT_ALIASES[topic]

        return topic

    # ============================================================
    # TOPIC LOOKUP
    # ============================================================

    def _topic_score(
        self,
        topic_dict: Dict[str, float],
        topic: str
    ) -> float:

        if not topic:
            return 0.0

        topic_norm = self._canonical_topic(topic)

        # Exact match first.
        for key, value in topic_dict.items():

            key_norm = self._canonical_topic(key)

            if key_norm == topic_norm:

                return max(
                    0.0,
                    min(
                        1.0,
                        self._safe_float(value)
                    )
                )

        # Controlled partial matching.
        for key, value in topic_dict.items():

            key_norm = self._canonical_topic(key)

            if len(topic_norm) < 4 or len(key_norm) < 4:
                continue

            if (
                topic_norm in key_norm
                or key_norm in topic_norm
            ):

                return max(
                    0.0,
                    min(
                        1.0,
                        self._safe_float(value)
                    )
                )

        return 0.0

    # ============================================================
    # GOAL SUBJECT EXTRACTION
    # ============================================================

    def _extract_goal_subject_terms(
        self,
        goal: str
    ) -> List[str]:
        """
        Extracts meaningful subject terms from the current goal.

        Example:

        "Prepare for Kubernetes interview"

        returns approximately:

        ["kubernetes"]

        while:

        "Prepare for technical interview"

        returns:

        []

        This distinction is critical because a generic interview
        goal should not be restricted to a particular technology.
        """

        goal = self._normalise(goal)

        if not goal:
            return []

        # First check known multi-word subjects.
        known_subject_phrases = [
            "machine learning",
            "deep learning",
            "artificial intelligence",
            "natural language processing",
            "data science",
            "data analytics",
            "data structures",
            "system design",
            "web development",
            "software engineering",
            "software development",
            "network security",
            "cybersecurity",
            "cloud computing",
            "computer networks",
            "operating systems",
            "database management",
            "database management systems",
            "full stack",
            "full stack development",
            "front end",
            "frontend development",
            "back end",
            "backend development",
            "mobile development",
            "android development",
            "azure devops",
            "aws devops",
            "ci/cd",
            "ci cd",
        ]

        subjects = []

        for phrase in known_subject_phrases:

            if phrase in goal:

                canonical = self._canonical_topic(
                    phrase
                )

                if canonical not in subjects:
                    subjects.append(canonical)

        # Token-level extraction.
        tokens = self._tokenize(goal)

        for token in tokens:

            token_norm = self._canonical_topic(
                token
            )

            if not token_norm:
                continue

            if token_norm in self.GENERIC_GOAL_TERMS:
                continue

            if token_norm in self.GENERIC_TECHNICAL_TERMS:
                continue

            # Very short generic tokens are ignored.
            if len(token_norm) < 3:
                continue

            # Numbers alone are not useful subject evidence.
            if token_norm.isdigit():
                continue

            if token_norm not in subjects:
                subjects.append(
                    token_norm
                )

        return subjects

    # ============================================================
    # SUBJECT EVIDENCE
    # ============================================================

    def _calculate_subject_gate(
        self,
        goal: str,
        derived_topics: List[Any],
        analysis: Dict[str, Any],
        topic_dict: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Determines whether a video actually matches the specific
        subject of the current goal.

        This prevents:

            "Kubernetes interview"

        from treating:

            "Java interview"

        as strongly relevant simply because both contain the word
        "interview".

        Evidence sources:

        1. Goal subject terms
        2. Derived goal topics
        3. Video topic scores
        4. Video title
        5. Video channel
        6. Video description
        """

        if not goal:

            return {
                "required": False,
                "passed": True,
                "score": 1.0,
                "matched_subjects": [],
                "goal_subject_terms": [],
                "reason": "No current goal subject gate required."
            }

        goal_subjects = self._extract_goal_subject_terms(
            goal
        )

        # Generic goal:
        #
        # "Prepare for interview"
        #
        # has no specific subject.
        if not goal_subjects:

            return {
                "required": False,
                "passed": True,
                "score": 1.0,
                "matched_subjects": [],
                "goal_subject_terms": [],
                "reason": "Current goal does not contain a specific subject."
            }

        title = self._normalise(
            analysis.get(
                "title",
                ""
            )
        )

        channel = self._normalise(
            analysis.get(
                "channel",
                ""
            )
        )

        description = self._normalise(
            analysis.get(
                "description",
                ""
            )
        )

        full_text = (
            f"{title} "
            f"{channel} "
            f"{description}"
        ).strip()

        matched_subjects = []
        evidence_scores = []

        # --------------------------------------------------------
        # Direct goal-subject evidence
        # --------------------------------------------------------

        for subject in goal_subjects:

            canonical_subject = self._canonical_topic(
                subject
            )

            topic_score = self._topic_score(
                topic_dict,
                canonical_subject
            )

            direct_title = self._contains(
                title,
                canonical_subject
            )

            direct_channel = self._contains(
                channel,
                canonical_subject
            )

            direct_description = self._contains(
                description,
                canonical_subject
            )

            if topic_score >= 0.45:

                matched_subjects.append(
                    canonical_subject
                )

                evidence_scores.append(
                    min(
                        1.0,
                        topic_score
                    )
                )

                continue

            if direct_title:

                matched_subjects.append(
                    canonical_subject
                )

                evidence_scores.append(
                    1.0
                )

                continue

            if direct_channel:

                matched_subjects.append(
                    canonical_subject
                )

                evidence_scores.append(
                    0.85
                )

                continue

            if direct_description:

                matched_subjects.append(
                    canonical_subject
                )

                evidence_scores.append(
                    0.70
                )

                continue

        # --------------------------------------------------------
        # Derived goal topics
        # --------------------------------------------------------

        derived_subject_matches = []

        for item in derived_topics:

            if isinstance(
                item,
                dict
            ):

                derived_topic = str(
                    item.get(
                        "topic",
                        ""
                    )
                ).strip()

            else:

                derived_topic = str(
                    item or ""
                ).strip()

            if not derived_topic:
                continue

            derived_norm = self._canonical_topic(
                derived_topic
            )

            # Do not treat generic intent words as subjects.
            if (
                derived_norm in self.GENERIC_GOAL_TERMS
                or derived_norm in self.GENERIC_TECHNICAL_TERMS
            ):
                continue

            topic_score = self._topic_score(
                topic_dict,
                derived_norm
            )

            title_match = self._contains(
                title,
                derived_norm
            )

            description_match = self._contains(
                description,
                derived_norm
            )

            channel_match = self._contains(
                channel,
                derived_norm
            )

            if topic_score >= 0.55:

                derived_subject_matches.append(
                    derived_norm
                )

                evidence_scores.append(
                    min(
                        1.0,
                        topic_score
                    )
                )

            elif title_match:

                derived_subject_matches.append(
                    derived_norm
                )

                evidence_scores.append(
                    0.90
                )

            elif channel_match:

                derived_subject_matches.append(
                    derived_norm
                )

                evidence_scores.append(
                    0.75
                )

            elif description_match:

                derived_subject_matches.append(
                    derived_norm
                )

                evidence_scores.append(
                    0.65
                )

        for item in derived_subject_matches:

            if item not in matched_subjects:

                matched_subjects.append(
                    item
                )

        # --------------------------------------------------------
        # Calculate final subject score
        # --------------------------------------------------------

        if evidence_scores:

            subject_score = max(
                evidence_scores
            )

        else:

            subject_score = 0.0

        subject_score = max(
            0.0,
            min(
                1.0,
                subject_score
            )
        )

        passed = subject_score >= 0.45

        if passed:

            reason = (
                "Video contains meaningful evidence for the "
                "current goal subject."
            )

        else:

            reason = (
                "Video does not contain sufficient evidence for "
                "the specific subject of the current goal."
            )

        return {
            "required": True,
            "passed": passed,
            "score": round(
                subject_score,
                4
            ),
            "matched_subjects": list(
                dict.fromkeys(
                    matched_subjects
                )
            ),
            "goal_subject_terms": goal_subjects,
            "reason": reason
        }

    # ============================================================
    # CONTENT TYPE COMPATIBILITY
    # ============================================================

    def check_content_type_compatibility(
        self,
        goal: str,
        content_type: str,
        goal_relevance: float = 0.0
    ) -> float:
        """
        Determines whether the detected content type is compatible
        with the user's CURRENT GOAL.

        Important:
        Content type alone does NOT establish subject relevance.

        Example:

            Goal:
                Prepare for Kubernetes interview

            Kubernetes tutorial:
                high compatibility

            Java interview:
                interview-compatible but subject mismatch

            Gaming:
                low compatibility
        """

        goal = self._normalise(
            goal
        )

        content_type = str(
            content_type or "Other"
        ).strip()

        if not goal:
            return 1.0

        # --------------------------------------------------------
        # CARTOON / ENTERTAINMENT GOAL
        # --------------------------------------------------------

        cartoon_goal_words = [
            "cartoon",
            "animation",
            "anime",
            "oggy",
            "tom and jerry",
            "kids entertainment",
        ]

        if any(
            word in goal
            for word in cartoon_goal_words
        ):

            if content_type in [
                "Cartoon",
                "Kids",
                "Entertainment",
            ]:
                return 1.0

            if content_type in [
                "Music",
                "Comedy",
            ]:
                return 0.70

            return 0.15

        # --------------------------------------------------------
        # SPORTS GOAL
        # --------------------------------------------------------

        sports_words = [
            "cricket",
            "football",
            "soccer",
            "basketball",
            "nba",
            "fifa",
            "tennis",
            "sports",
            "match",
            "ipl",
            "olympics",
            "athletics",
        ]

        if any(
            word in goal
            for word in sports_words
        ):

            if content_type == "Sports":
                return 1.0

            if content_type == "Fitness":
                return 0.70

            if content_type in [
                "News",
                "Education",
                "Tutorial",
            ]:
                return 0.70

            return 0.15

        # --------------------------------------------------------
        # COOKING GOAL
        # --------------------------------------------------------

        cooking_words = [
            "cook",
            "cooking",
            "recipe",
            "bake",
            "baking",
            "chef",
            "food",
            "kitchen",
            "culinary",
        ]

        if any(
            word in goal
            for word in cooking_words
        ):

            if content_type == "Cooking":
                return 1.0

            if content_type == "Tutorial":
                return 0.85

            if content_type == "Education":
                return 0.65

            return 0.15

        # --------------------------------------------------------
        # TRAVEL GOAL
        # --------------------------------------------------------

        travel_words = [
            "travel",
            "tourism",
            "trip",
            "vacation",
            "destination",
            "tour",
            "transport",
            "transit",
            "flight",
        ]

        if any(
            word in goal
            for word in travel_words
        ):

            if content_type == "Travel":
                return 1.0

            if content_type in [
                "Education",
                "Tutorial",
            ]:
                return 0.75

            return 0.15

        # --------------------------------------------------------
        # TECHNICAL / CAREER GOAL
        # --------------------------------------------------------

        technical_words = [
            "programming",
            "coding",
            "developer",
            "software",
            "python",
            "java",
            "javascript",
            "react",
            "node",
            "flutter",
            "android",
            "web development",
            "backend",
            "frontend",
            "devops",
            "docker",
            "kubernetes",
            "cloud",
            "aws",
            "azure",
            "gcp",
            "cybersecurity",
            "network security",
            "machine learning",
            "artificial intelligence",
            "data science",
            "sql",
            "database",
            "algorithm",
            "data structures",
            "system design",
            "software engineering",
            "interview",
            "placement",
            "engineering",
        ]

        if any(
            word in goal
            for word in technical_words
        ):

            if content_type in [
                "Programming",
                "Interview Preparation",
                "Tutorial",
                "Education",
            ]:
                return 1.0

            if content_type == "Technology":
                return 0.90

            if content_type == "News":

                if goal_relevance >= 0.65:
                    return 0.80

                return 0.55

            if content_type == "Science":

                if goal_relevance >= 0.70:
                    return 0.75

                return 0.45

            return 0.15

        # --------------------------------------------------------
        # DEFAULT
        # --------------------------------------------------------

        return 1.0

    # ============================================================
    # DERIVED TOPIC SCORE
    # ============================================================

    def _calculate_derived_score(
        self,
        derived_topics: List[Any],
        topic_dict: Dict[str, float]
    ) -> float:

        scores = []

        for item in derived_topics:

            if isinstance(
                item,
                dict
            ):

                topic = str(
                    item.get(
                        "topic",
                        ""
                    )
                ).strip()

                weight = self._safe_float(
                    item.get(
                        "weight",
                        0.5
                    ),
                    0.5
                )

            else:

                topic = str(
                    item or ""
                ).strip()

                weight = 0.5

            if not topic:
                continue

            score = self._topic_score(
                topic_dict,
                topic
            )

            weight = max(
                0.0,
                min(
                    1.0,
                    weight
                )
            )

            scores.append(
                score * weight
            )

        return min(
            1.0,
            max(
                scores,
                default=0.0
            )
        )

    # ============================================================
    # PREFERRED TOPIC SCORE
    # ============================================================

    def _calculate_preferred_score(
        self,
        preferred_topics: List[str],
        preferred_weights: Dict[str, float],
        topic_dict: Dict[str, float]
    ) -> float:

        scores = []

        for topic in preferred_topics:

            topic = str(
                topic or ""
            ).strip()

            if not topic:
                continue

            score = self._topic_score(
                topic_dict,
                topic
            )

            weight = self._safe_float(
                preferred_weights.get(
                    topic,
                    1.0
                ),
                1.0
            )

            weight = max(
                0.0,
                min(
                    2.0,
                    weight
                )
            )

            scores.append(
                score * weight
            )

        return min(
            1.0,
            max(
                scores,
                default=0.0
            )
        )

    # ============================================================
    # STABLE PROFILE SCORE
    # ============================================================

    def _calculate_interest_score(
        self,
        interests: List[str],
        profession: str,
        topic_dict: Dict[str, float]
    ) -> float:

        scores = []

        for interest in interests:

            score = self._topic_score(
                topic_dict,
                str(interest)
            )

            scores.append(
                score
            )

        if profession:

            scores.append(
                self._topic_score(
                    topic_dict,
                    profession
                )
            )

        return min(
            1.0,
            max(
                scores,
                default=0.0
            )
        )

    # ============================================================
    # HOBBY SCORE
    # ============================================================

    def _calculate_hobby_score(
        self,
        hobbies: List[str],
        topic_dict: Dict[str, float]
    ) -> float:

        scores = []

        for hobby in hobbies:

            scores.append(
                self._topic_score(
                    topic_dict,
                    str(hobby)
                )
            )

        return min(
            1.0,
            max(
                scores,
                default=0.0
            )
        )

    # ============================================================
    # NEGATIVE TOPIC ANALYSIS
    # ============================================================

    def _calculate_negative_score(
        self,
        unwanted_topics: List[str],
        topic_dict: Dict[str, float],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:

        negative_scores = []
        matched = []

        title = self._normalise(
            analysis.get(
                "title",
                ""
            )
        )

        description = self._normalise(
            analysis.get(
                "description",
                ""
            )
        )

        channel = self._normalise(
            analysis.get(
                "channel",
                ""
            )
        )

        content_type = self._normalise(
            analysis.get(
                "content_type",
                "Other"
            )
        )

        for topic in unwanted_topics:

            topic = str(
                topic or ""
            ).strip()

            if not topic:
                continue

            score = self._topic_score(
                topic_dict,
                topic
            )

            topic_norm = self._normalise(
                topic
            )

            # Explicit unwanted phrase in title/description/channel.
            if (
                self._contains(
                    title,
                    topic_norm
                )
                or self._contains(
                    description,
                    topic_norm
                )
                or self._contains(
                    channel,
                    topic_norm
                )
            ):

                score = max(
                    score,
                    0.90
                )

            # Content type match.
            if (
                self._normalise(
                    topic
                ) == content_type
            ):

                score = max(
                    score,
                    0.85
                )

            negative_scores.append(
                score
            )

            if score >= 0.35:

                matched.append(
                    topic
                )

        # Also retain matches supplied by ContentService.
        supplied_matches = (
            analysis.get(
                "matched_negative"
            )
            or []
        )

        for topic in supplied_matches:

            if topic not in matched:

                matched.append(
                    topic
                )

        negative = max(
            negative_scores,
            default=0.0
        )

        # Preserve stronger ContentService result.
        analysis_negative = self._safe_float(
            analysis.get(
                "negative_relevance",
                0.0
            )
        )

        negative = max(
            negative,
            analysis_negative
        )

        return {
            "score": round(
                min(
                    1.0,
                    negative
                ),
                4
            ),
            "matched": matched
        }

    # ============================================================
    # HARD / SOFT NEGATIVES
    # ============================================================

    def _classify_negative(
        self,
        matched_negatives: List[str],
        hard_negatives: Set[str],
        soft_negatives: Set[str]
    ) -> Dict[str, Any]:

        hard_topic = None
        soft_topics = []

        for negative in matched_negatives:

            negative_l = self._normalise(
                negative
            )

            for hard in hard_negatives:

                hard_l = self._normalise(
                    hard
                )

                if (
                    negative_l == hard_l
                    or hard_l in negative_l
                    or negative_l in hard_l
                ):

                    hard_topic = negative
                    break

            if hard_topic:
                break

            for soft in soft_negatives:

                soft_l = self._normalise(
                    soft
                )

                if (
                    negative_l == soft_l
                    or soft_l in negative_l
                    or negative_l in soft_l
                ):

                    if negative not in soft_topics:

                        soft_topics.append(
                            negative
                        )

        return {
            "hard": hard_topic is not None,
            "hard_topic": hard_topic,
            "soft": len(
                soft_topics
            ) > 0,
            "soft_topics": soft_topics
        }

    # ============================================================
    # MAIN ALIGNMENT
    # ============================================================

    def compute_alignment(
        self,
        analysis: Dict[str, Any],
        intent: Dict[str, Any],
        strictness: float = 0.65
    ) -> Dict[str, Any]:
        """
        Calculate final alignment.

        Priority:

        CURRENT GOAL
            >
        DERIVED GOAL TOPICS
            >
        PREFERRED TOPICS
            >
        STABLE PROFILE
            >
        HOBBIES

        V1.4 introduces the subject gate after the positive
        composite calculation.

        Alignment answers:

            "How relevant is this video?"

        Intervention answers:

            "What should the extension do with it?"
        """

        if not isinstance(
            analysis,
            dict
        ):
            analysis = {}

        if not isinstance(
            intent,
            dict
        ):
            intent = {}

        current_intent = (
            intent.get(
                "current_intent"
            )
            or {}
        )

        user_profile = (
            intent.get(
                "user_profile"
            )
            or intent.get(
                "profile"
            )
            or {}
        )

        settings = (
            intent.get(
                "settings"
            )
            or {}
        )

        # ========================================================
        # USER INPUT
        # ========================================================

        goal = str(
            current_intent.get(
                "current_goal"
            )
            or intent.get(
                "goal"
            )
            or ""
        ).strip()

        derived_topics = (
            current_intent.get(
                "derived_goal_topics"
            )
            or intent.get(
                "derivedTopics"
            )
            or []
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

        preferred_weights = (
            current_intent.get(
                "preferred_weights"
            )
            or {}
        )

        interests = (
            user_profile.get(
                "interests"
            )
            or []
        )

        profession = str(
            user_profile.get(
                "profession"
            )
            or ""
        ).strip()

        hobbies = (
            user_profile.get(
                "hobbies"
            )
            or []
        )

        # ========================================================
        # VIDEO ANALYSIS
        # ========================================================

        goal_rel = self._safe_float(
            analysis.get(
                "goal_relevance",
                0.0
            )
        )

        positive_from_analysis = (
            self._safe_float(
                analysis.get(
                    "positive_relevance",
                    0.0
                )
            )
        )

        negative_from_analysis = (
            self._safe_float(
                analysis.get(
                    "negative_relevance",
                    0.0
                )
            )
        )

        content_type = str(
            analysis.get(
                "content_type",
                "Other"
            )
        )

        is_incidental = bool(
            analysis.get(
                "is_incidental",
                False
            )
        )

        is_uncertain = bool(
            analysis.get(
                "is_uncertain",
                False
            )
        )

        # ========================================================
        # TOPIC DICTIONARY
        # ========================================================

        raw_topics = (
            analysis.get(
                "topic_dict"
            )
            or analysis.get(
                "topics"
            )
            or {}
        )

        if isinstance(
            raw_topics,
            list
        ):

            topic_dict = {}

            for item in raw_topics:

                if not isinstance(
                    item,
                    dict
                ):
                    continue

                topic = str(
                    item.get(
                        "topic",
                        ""
                    )
                ).strip()

                score = self._safe_float(
                    item.get(
                        "score",
                        0.0
                    )
                )

                if topic:

                    topic_dict[
                        topic
                    ] = score

        elif isinstance(
            raw_topics,
            dict
        ):

            topic_dict = {
                str(key): self._safe_float(
                    value
                )
                for key, value
                in raw_topics.items()
            }

        else:

            topic_dict = {}

        # ========================================================
        # TIER 1 - CURRENT GOAL
        # ========================================================

        tier1_goal = max(
            0.0,
            min(
                1.0,
                goal_rel
            )
        )

        # ========================================================
        # TIER 2 - DERIVED GOAL TOPICS
        # ========================================================

        tier2_derived = (
            self._calculate_derived_score(
                derived_topics,
                topic_dict
            )
        )

        # ========================================================
        # TIER 3 - PREFERRED TOPICS
        # ========================================================

        tier3_preferred = (
            self._calculate_preferred_score(
                preferred_topics,
                preferred_weights,
                topic_dict
            )
        )

        # ========================================================
        # TIER 4 - INTERESTS / PROFESSION
        # ========================================================

        tier4_interests = (
            self._calculate_interest_score(
                interests,
                profession,
                topic_dict
            )
        )

        # ========================================================
        # TIER 5 - HOBBIES
        # ========================================================

        tier5_hobbies = (
            self._calculate_hobby_score(
                hobbies,
                topic_dict
            )
        )

        # ========================================================
        # ACTIVE GOAL PROTECTION
        # ========================================================

        if goal:

            if (
                tier1_goal < 0.30
                and tier2_derived < 0.30
            ):

                tier4_interests = min(
                    tier4_interests,
                    0.15
                )

                tier5_hobbies = min(
                    tier5_hobbies,
                    0.10
                )

        # ========================================================
        # POSITIVE COMPOSITE
        # ========================================================

        weighted_sum = 0.0
        active_weight = 0.0

        if goal:

            weighted_sum += (
                self.w_goal
                * tier1_goal
            )

            weighted_sum += (
                self.w_derived
                * tier2_derived
            )

            active_weight += (
                self.w_goal
                + self.w_derived
            )

        if preferred_topics:

            weighted_sum += (
                self.w_preferred
                * tier3_preferred
            )

            active_weight += (
                self.w_preferred
            )

        if interests or profession:

            weighted_sum += (
                self.w_interests
                * tier4_interests
            )

            active_weight += (
                self.w_interests
            )

        if hobbies:

            weighted_sum += (
                self.w_hobbies
                * tier5_hobbies
            )

            active_weight += (
                self.w_hobbies
            )

        if active_weight > 0:

            composite_positive = (
                weighted_sum
                / active_weight
            )

        else:

            composite_positive = (
                positive_from_analysis
            )

        composite_positive = max(
            0.0,
            min(
                1.0,
                composite_positive
            )
        )

        # ========================================================
        # CONTENT TYPE COMPATIBILITY
        # ========================================================

        type_compatibility = (
            self.check_content_type_compatibility(
                goal,
                content_type,
                tier1_goal
            )
        )

        adjusted_positive = (
            composite_positive
            * type_compatibility
        )

        # ========================================================
        # CURRENT-GOAL SUBJECT GATE
        # ========================================================

        subject_gate = (
            self._calculate_subject_gate(
                goal,
                derived_topics,
                analysis,
                topic_dict
            )
        )

        subject_gate_applied = bool(
            subject_gate.get(
                "required",
                False
            )
        )

        subject_gate_passed = bool(
            subject_gate.get(
                "passed",
                True
            )
        )

        subject_gate_score = self._safe_float(
            subject_gate.get(
                "score",
                1.0
            ),
            1.0
        )

        subject_gate_strength = 1.0

        if subject_gate_applied:

            # Strong subject evidence.
            if subject_gate_score >= 0.75:

                subject_gate_strength = 1.0

            # Moderate subject evidence.
            elif subject_gate_score >= 0.45:

                subject_gate_strength = 0.85

            # Weak/no subject evidence.
            else:

                subject_gate_strength = 0.35

        adjusted_positive_before_subject_gate = (
            adjusted_positive
        )

        adjusted_positive *= (
            subject_gate_strength
        )

        # ========================================================
        # NEGATIVE ANALYSIS
        # ========================================================

        negative_info = (
            self._calculate_negative_score(
                unwanted_topics,
                topic_dict,
                analysis
            )
        )

        negative_score = max(
            negative_info["score"],
            negative_from_analysis
        )

        matched_negatives = list(
            negative_info["matched"]
        )

        # ========================================================
        # HARD / SOFT NEGATIVES
        # ========================================================

        hard_negatives = set(
            intent.get(
                "hard_negatives"
            )
            or settings.get(
                "hard_negatives"
            )
            or self.DEFAULT_HARD_NEGATIVES
        )

        soft_negatives = set(
            intent.get(
                "soft_negatives"
            )
            or settings.get(
                "soft_negatives"
            )
            or self.DEFAULT_SOFT_NEGATIVES
        )

        negative_classification = (
            self._classify_negative(
                matched_negatives,
                hard_negatives,
                soft_negatives
            )
        )

        is_hard_negative = (
            negative_classification[
                "hard"
            ]
        )

        hard_negative_topic = (
            negative_classification[
                "hard_topic"
            ]
        )

        has_soft_negative = (
            negative_classification[
                "soft"
            ]
        )

        # ========================================================
        # NEGATIVE PENALTY
        # ========================================================

        negative_penalty = (
            self.w_neg
            * negative_score
        )

        # ========================================================
        # HARD NEGATIVE RULE
        # ========================================================

        hard_negative_overridden = False

        if is_hard_negative:

            # Hard negative can only be overridden by an extremely
            # strong current-goal relationship.
            if (
                tier1_goal >= 0.85
                and subject_gate_passed
                and not is_incidental
            ):

                hard_negative_overridden = True

            else:

                adjusted_positive = min(
                    adjusted_positive,
                    0.15
                )

                negative_penalty = max(
                    negative_penalty,
                    0.70
                )

        # ========================================================
        # SOFT NEGATIVE RULE
        # ========================================================

        soft_negative_mitigated = False

        if (
            has_soft_negative
            and not is_hard_negative
            and not is_incidental
        ):

            goal_strength = max(
                tier1_goal,
                tier2_derived
            )

            # Soft negative is useful if it directly supports
            # the current subject.
            if (
                goal_strength >= 0.55
                and (
                    not subject_gate_applied
                    or subject_gate_passed
                )
            ):

                soft_negative_mitigated = True

                mitigation = min(
                    0.70,
                    goal_strength
                )

                negative_penalty *= (
                    1.0
                    - (
                        0.65
                        * mitigation
                    )
                )

        # ========================================================
        # FINAL ALIGNMENT
        # ========================================================

        alignment = (
            adjusted_positive
            - negative_penalty
        )

        # Analysis positive relevance can help when preferred
        # topics exist but exact topic keys were not available.
        #
        # IMPORTANT:
        # Do NOT allow this fallback to bypass the subject gate.
        if (
            preferred_topics
            and positive_from_analysis > alignment
        ):

            candidate_alignment = max(
                alignment,
                (
                    0.70 * alignment
                    + 0.30 * positive_from_analysis
                )
            )

            if (
                not subject_gate_applied
                or subject_gate_passed
            ):

                alignment = candidate_alignment

        # ========================================================
        # RE-APPLY SUBJECT CAP
        # ========================================================

        if (
            subject_gate_applied
            and not subject_gate_passed
        ):

            # A video without the actual current-goal subject
            # must not remain a high-alignment video.
            alignment = min(
                alignment,
                self.subject_gate_cap
            )

        elif (
            subject_gate_applied
            and subject_gate_score < 0.75
        ):

            # Weak subject evidence receives a softer cap.
            alignment = min(
                alignment,
                self.subject_gate_soft_cap
            )

        alignment = max(
            0.0,
            min(
                1.0,
                alignment
            )
        )

        # ========================================================
        # INCIDENTAL CONTENT
        # ========================================================

        if is_incidental:

            alignment = min(
                alignment,
                0.28
            )

        # ========================================================
        # UNCERTAIN CONTENT
        # ========================================================

        if is_uncertain:

            alignment = min(
                alignment,
                0.45
            )

        # ========================================================
        # STRICTNESS
        # ========================================================

        strictness = self._safe_float(
            strictness,
            0.65
        )

        strictness = max(
            0.0,
            min(
                1.0,
                strictness
            )
        )

        # Strictness does NOT arbitrarily reduce alignment.
        #
        # Alignment:
        #     How relevant?
        #
        # Intervention:
        #     What should happen?
        #
        # They remain separate.

        # ========================================================
        # CONFIDENCE SUPPORT
        # ========================================================

        confidence = self._safe_float(
            analysis.get(
                "confidence",
                0.0
            )
        )

        confidence = max(
            0.0,
            min(
                1.0,
                confidence
            )
        )

        # If a specific subject was directly detected, confidence
        # can be slightly improved because there is stronger
        # evidence for the active goal.
        if (
            subject_gate_applied
            and subject_gate_score >= 0.75
        ):

            confidence = min(
                1.0,
                max(
                    confidence,
                    0.60
                )
            )

        # ========================================================
        # COMPONENTS
        # ========================================================

        components = {

            "goal_tier_score":
                round(
                    tier1_goal,
                    4
                ),

            "derived_tier_score":
                round(
                    tier2_derived,
                    4
                ),

            "preferred_tier_score":
                round(
                    tier3_preferred,
                    4
                ),

            "interests_tier_score":
                round(
                    tier4_interests,
                    4
                ),

            "hobbies_tier_score":
                round(
                    tier5_hobbies,
                    4
                ),

            "composite_positive":
                round(
                    composite_positive,
                    4
                ),

            "analysis_positive_relevance":
                round(
                    positive_from_analysis,
                    4
                ),

            "negative_relevance":
                round(
                    negative_score,
                    4
                ),

            "negative_penalty":
                round(
                    negative_penalty,
                    4
                ),

            "content_type_compatibility":
                round(
                    type_compatibility,
                    4
                ),

            "subject_gate_score":
                round(
                    subject_gate_score,
                    4
                ),

            "subject_gate_strength":
                round(
                    subject_gate_strength,
                    4
                ),

            "adjusted_positive_before_subject_gate":
                round(
                    adjusted_positive_before_subject_gate,
                    4
                ),

            "adjusted_positive_after_subject_gate":
                round(
                    adjusted_positive,
                    4
                ),

            "strictness":
                round(
                    strictness,
                    4
                ),

            "confidence":
                round(
                    confidence,
                    4
                )
        }

        # ========================================================
        # RETURN
        # ========================================================

        return {

            "alignment":
                round(
                    float(alignment),
                    4
                ),

            "positive_alignment":
                round(
                    float(adjusted_positive),
                    4
                ),

            "negative_relevance":
                round(
                    float(negative_score),
                    4
                ),

            "matched_negative":
                matched_negatives,

            "is_hard_negative":
                is_hard_negative,

            "hard_negative_topic":
                hard_negative_topic,

            "hard_negative_overridden":
                hard_negative_overridden,

            "soft_negative_mitigated":
                soft_negative_mitigated,

            "content_type_compatibility":
                round(
                    float(type_compatibility),
                    4
                ),

            "is_incidental":
                is_incidental,

            "is_uncertain":
                is_uncertain,

            "confidence":
                round(
                    float(confidence),
                    4
                ),

            # ----------------------------------------------------
            # NEW V1.4 SUBJECT-GATE OUTPUT
            # ----------------------------------------------------

            "subject_gate_required":
                subject_gate_applied,

            "subject_gate_passed":
                subject_gate_passed,

            "subject_gate_score":
                round(
                    float(subject_gate_score),
                    4
                ),

            "subject_gate_strength":
                round(
                    float(subject_gate_strength),
                    4
                ),

            "goal_subject_terms":
                subject_gate.get(
                    "goal_subject_terms",
                    []
                ),

            "matched_goal_subjects":
                subject_gate.get(
                    "matched_subjects",
                    []
                ),

            "subject_gate_reason":
                subject_gate.get(
                    "reason",
                    ""
                ),

            "components":
                components
        }


# ================================================================
# OPTIONAL DIRECT TEST
# ================================================================

if __name__ == "__main__":

    service = AlignmentService()

    # ------------------------------------------------------------
    # Example intent
    # ------------------------------------------------------------

    example_intent = {

        "current_intent": {

            "current_goal":
                "Prepare for Kubernetes interview",

            "derived_goal_topics": [
                {
                    "topic": "Kubernetes",
                    "weight": 1.0
                },
                {
                    "topic": "DevOps",
                    "weight": 0.80
                },
                {
                    "topic": "Docker",
                    "weight": 0.70
                },
                {
                    "topic": "Linux",
                    "weight": 0.60
                }
            ],

            "preferred_topics": [
                "Cloud",
                "DevOps",
                "Kubernetes"
            ],

            "unwanted_topics": [
                "Gaming",
                "Entertainment"
            ],

            "preferred_weights": {
                "Cloud": 1.0,
                "DevOps": 1.0,
                "Kubernetes": 1.0
            }
        },

        "user_profile": {

            "profession":
                "BE IT Student",

            "interests": [
                "Web Development",
                "Machine Learning",
                "DevOps"
            ],

            "hobbies": [
                "Cricket"
            ]
        }
    }

    # ------------------------------------------------------------
    # Kubernetes video
    # ------------------------------------------------------------

    kubernetes_video = {

        "title":
            "Kubernetes Interview Questions and Architecture",

        "channel":
            "DevOps Learning",

        "description":
            "Complete Kubernetes interview preparation",

        "content_type":
            "Interview Preparation",

        "goal_relevance":
            0.90,

        "positive_relevance":
            0.90,

        "negative_relevance":
            0.0,

        "topic_dict": {

            "Kubernetes": 0.95,
            "DevOps": 0.80,
            "Docker": 0.60,
            "Linux": 0.40,
            "Interview Preparation": 0.90
        },

        "confidence":
            0.90
    }

    kubernetes_result = (
        service.compute_alignment(
            kubernetes_video,
            example_intent
        )
    )

    print("\nKUBERNETES VIDEO")
    print(
        kubernetes_result
    )

    # ------------------------------------------------------------
    # Java interview video
    # ------------------------------------------------------------

    java_video = {

        "title":
            "Java Interview Questions and Answers",

        "channel":
            "Java Careers",

        "description":
            "Top Java interview questions for freshers",

        "content_type":
            "Interview Preparation",

        "goal_relevance":
            0.50,

        "positive_relevance":
            0.60,

        "negative_relevance":
            0.0,

        "topic_dict": {

            "Java": 0.95,
            "Interview Preparation": 0.90
        },

        "confidence":
            0.90
    }

    java_result = (
        service.compute_alignment(
            java_video,
            example_intent
        )
    )

    print("\nJAVA VIDEO")
    print(
        java_result
    )

    # ------------------------------------------------------------
    # Generic interview goal test
    # ------------------------------------------------------------

    generic_intent = {

        "current_intent": {

            "current_goal":
                "Prepare for interview",

            "derived_goal_topics": [],

            "preferred_topics": [
                "Interview Preparation"
            ],

            "unwanted_topics": [],

            "preferred_weights": {
                "Interview Preparation": 1.0
            }
        },

        "user_profile": {

            "profession":
                "BE IT Student",

            "interests": [
                "Technology"
            ],

            "hobbies": []
        }
    }

    generic_result = (
        service.compute_alignment(
            java_video,
            generic_intent
        )
    )

    print("\nGENERIC INTERVIEW GOAL")
    print(
        generic_result
    )