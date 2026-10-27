"""
Divergence and Profiling Service for IntentAlign
V1.3 — Raw vs Controlled Dual Feed Profiling

Responsibilities:

1. Build the dynamic semantic vocabulary.
2. Build the normalized Intent Distribution P.
3. Build the Raw Observed Feed Distribution Q_raw.
4. Build the Controlled Feed Distribution Q_ctrl.
5. Calculate Raw JSD and Controlled JSD.
6. Calculate Raw and Controlled Alignment.
7. Calculate Alignment Improvement.
8. Calculate Negative / Preferred Exposure.
9. Calculate Exposure Reduction / Improvement.
10. Calculate Topic Diversity using normalized Shannon entropy.

Core experimental comparison:

        YouTube Raw Feed
              ↓
          Q_raw
              ↓
        Raw JSD / Alignment
              ↓
        IntentAlign
              ↓
      Controlled Feed
              ↓
          Q_ctrl
              ↓
    Controlled JSD / Alignment

The purpose is to measure the effect of intervention rather than
claiming that the extension changes YouTube's internal ranking.
"""

import math
from typing import Dict, List, Any, Optional

import numpy as np

from backend.services.intent_service import TAXONOMY


class DivergenceService:

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(
        self,
        vocabulary: Optional[List[str]] = None
    ):
        self.vocabulary = vocabulary

    # ============================================================
    # Utility Methods
    # ============================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0
    ) -> float:

        try:
            return float(value)
        except (
            TypeError,
            ValueError
        ):
            return default

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0
    ) -> float:

        try:
            value = float(value)
        except (
            TypeError,
            ValueError
        ):
            value = minimum

        return max(
            minimum,
            min(maximum, value)
        )

    @staticmethod
    def _topic_key(
        topic: Any
    ) -> str:

        if topic is None:
            return ""

        return " ".join(
            str(topic)
            .strip()
            .lower()
            .split()
        )

    @staticmethod
    def _extract_topic_items(
        topics: Any
    ) -> Dict[str, float]:
        """
        Convert supported topic formats into:

            {
                "topic": score
            }

        Supported:
            dict
            list[dict]
            list[str]
        """

        result = {}

        if isinstance(
            topics,
            dict
        ):

            for topic, score in topics.items():

                if not topic:
                    continue

                try:
                    score = float(score)
                except (
                    TypeError,
                    ValueError
                ):
                    score = 0.0

                score = max(
                    0.0,
                    score
                )

                key = str(
                    topic
                ).strip()

                if not key:
                    continue

                normalized_key = (
                    DivergenceService
                    ._topic_key(key)
                )

                if (
                    normalized_key not in result
                    or score > result[normalized_key]
                ):
                    result[
                        normalized_key
                    ] = score

        elif isinstance(
            topics,
            list
        ):

            for item in topics:

                if isinstance(
                    item,
                    dict
                ):

                    topic = (
                        item.get(
                            "topic"
                        )
                        or item.get(
                            "name"
                        )
                    )

                    score = item.get(
                        "score",
                        0.0
                    )

                else:

                    topic = item
                    score = 1.0

                if not topic:
                    continue

                try:
                    score = float(score)
                except (
                    TypeError,
                    ValueError
                ):
                    score = 0.0

                score = max(
                    0.0,
                    score
                )

                normalized_key = (
                    DivergenceService
                    ._topic_key(topic)
                )

                if not normalized_key:
                    continue

                if (
                    normalized_key not in result
                    or score > result[normalized_key]
                ):
                    result[
                        normalized_key
                    ] = score

        return result

    # ============================================================
    # Dynamic Vocabulary
    # ============================================================

    def get_dynamic_vocabulary(
        self,
        intent: Dict[str, Any],
        results: Optional[
            List[Dict[str, Any]]
        ] = None
    ) -> List[str]:
        """
        Build the semantic vocabulary.

        Priority:

            1. Current goal keywords / concepts
            2. User preferred topics
            3. User unwanted topics
            4. Derived goal topics
            5. Topics observed in the feed
            6. Supporting canonical taxonomy

        User-defined intent therefore remains the primary semantic
        space while the taxonomy acts as supporting coverage.
        """

        vocabulary_map = {}

        def add_topic(
            topic: Any,
            priority: int = 1
        ):

            if topic is None:
                return

            topic = str(
                topic
            ).strip()

            if not topic:
                return

            key = self._topic_key(
                topic
            )

            if not key:
                return

            existing = vocabulary_map.get(
                key
            )

            if (
                existing is None
                or priority > existing["priority"]
            ):

                vocabulary_map[
                    key
                ] = {
                    "topic": topic,
                    "priority": priority
                }

        current_intent = (
            intent.get(
                "current_intent"
            )
            or {}
        )

        # --------------------------------------------------------
        # 1. Goal keywords
        # --------------------------------------------------------

        goal_keywords = (
            current_intent.get(
                "goal_keywords"
            )
            or intent.get(
                "goalKeywords"
            )
            or []
        )

        for keyword in goal_keywords:

            if isinstance(
                keyword,
                dict
            ):
                keyword = (
                    keyword.get(
                        "keyword"
                    )
                    or keyword.get(
                        "topic"
                    )
                    or keyword.get(
                        "text"
                    )
                )

            add_topic(
                keyword,
                priority=6
            )

        # --------------------------------------------------------
        # 2. Current goal itself
        # --------------------------------------------------------

        current_goal = (
            current_intent.get(
                "current_goal"
            )
            or intent.get(
                "goal"
            )
            or ""
        )

        # Don't add the complete sentence as a normal topic if it
        # is extremely long. Goal keywords/derived concepts cover
        # the semantic space more effectively.
        if current_goal:

            words = str(
                current_goal
            ).split()

            if len(words) <= 5:

                add_topic(
                    current_goal,
                    priority=6
                )

        # --------------------------------------------------------
        # 3. Preferred topics
        # --------------------------------------------------------

        preferred = (
            current_intent.get(
                "preferred_topics"
            )
            or intent.get(
                "preferredTopics"
            )
            or []
        )

        for topic in preferred:

            add_topic(
                topic,
                priority=6
            )

        # --------------------------------------------------------
        # 4. Unwanted topics
        # --------------------------------------------------------

        unwanted = (
            current_intent.get(
                "unwanted_topics"
            )
            or intent.get(
                "unwantedTopics"
            )
            or []
        )

        for topic in unwanted:

            add_topic(
                topic,
                priority=5
            )

        # --------------------------------------------------------
        # 5. Derived goal topics
        # --------------------------------------------------------

        derived = (
            current_intent.get(
                "derived_goal_topics"
            )
            or intent.get(
                "derivedTopics"
            )
            or []
        )

        for item in derived:

            topic = (
                item.get(
                    "topic"
                )
                if isinstance(
                    item,
                    dict
                )
                else str(item)
            )

            add_topic(
                topic,
                priority=5
            )

        # --------------------------------------------------------
        # 6. Learned-weight topics
        # --------------------------------------------------------

        preferred_weights = (
            current_intent.get(
                "preferred_weights"
            )
            or intent.get(
                "preferred_weights"
            )
            or {}
        )

        unwanted_weights = (
            current_intent.get(
                "unwanted_weights"
            )
            or intent.get(
                "unwanted_weights"
            )
            or {}
        )

        for topic in preferred_weights.keys():

            add_topic(
                topic,
                priority=5
            )

        for topic in unwanted_weights.keys():

            add_topic(
                topic,
                priority=5
            )

        # --------------------------------------------------------
        # 7. Topics actually observed in the feed
        # --------------------------------------------------------

        if results:

            for result in results:

                if not isinstance(
                    result,
                    dict
                ):
                    continue

                topics = (
                    result.get(
                        "topics"
                    )
                    or result.get(
                        "topic_dict"
                    )
                    or {}
                )

                topic_items = (
                    self._extract_topic_items(
                        topics
                    )
                )

                for topic_key in topic_items:

                    add_topic(
                        topic_key,
                        priority=4
                    )

        # --------------------------------------------------------
        # 8. Supporting taxonomy
        # --------------------------------------------------------

        for taxonomy_topic in TAXONOMY.keys():

            add_topic(
                taxonomy_topic,
                priority=1
            )

        # Return deterministic order.
        ordered = sorted(
            vocabulary_map.values(),
            key=lambda item: (
                -item["priority"],
                item["topic"].lower()
            )
        )

        return [
            item["topic"]
            for item in ordered
        ]

    # ============================================================
    # Topic Matching
    # ============================================================

    def _resolve_vocab_topic(
        self,
        topic: str,
        vocabulary: List[str]
    ) -> Optional[str]:
        """
        Resolve a topic to the corresponding vocabulary entry
        using case-insensitive matching and safe phrase matching.
        """

        topic_key = self._topic_key(
            topic
        )

        if not topic_key:
            return None

        # Exact normalized match.
        for vocabulary_topic in vocabulary:

            if (
                self._topic_key(
                    vocabulary_topic
                )
                == topic_key
            ):
                return vocabulary_topic

        # Controlled substring match.
        for vocabulary_topic in vocabulary:

            vocabulary_key = (
                self._topic_key(
                    vocabulary_topic
                )
            )

            if (
                topic_key in vocabulary_key
                or vocabulary_key in topic_key
            ):
                return vocabulary_topic

        return None

    # ============================================================
    # Intent Distribution
    # ============================================================

    def get_intent_distribution(
        self,
        intent: Dict[str, Any],
        vocabulary: List[str]
    ) -> Dict[str, float]:
        """
        Build normalized Intent Distribution P.

        Hierarchical pools:

            Current Goal / Derived Topics : 0.40
            Preferred Topics              : 0.40
            Profession                    : 0.10
            Interests                     : 0.07
            Hobbies                       : 0.03

        Learned preferred weights modify the preferred contribution.

        Learned unwanted weights do NOT become positive intent.
        They remain negative/intervention evidence.

        Sum(P) = 1.0 when vocabulary is non-empty.
        """

        if not vocabulary:

            return {}

        counts = {
            topic: 0.0
            for topic in vocabulary
        }

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

        # --------------------------------------------------------
        # 1. Current Goal + Derived Goal Topics
        # --------------------------------------------------------

        derived = (
            current_intent.get(
                "derived_goal_topics"
            )
            or intent.get(
                "derivedTopics"
            )
            or []
        )

        derived_pool = 0.40

        for item in derived:

            if isinstance(
                item,
                dict
            ):

                topic = item.get(
                    "topic"
                )

                weight = self._safe_float(
                    item.get(
                        "weight",
                        0.5
                    ),
                    0.5
                )

            else:

                topic = str(
                    item
                )

                weight = 0.5

            resolved = (
                self._resolve_vocab_topic(
                    topic,
                    vocabulary
                )
            )

            if resolved is None:
                continue

            counts[
                resolved
            ] += (
                self._clamp(
                    weight
                )
                * derived_pool
            )

        # --------------------------------------------------------
        # Current goal itself gets a small direct contribution.
        # This prevents a goal from disappearing when semantic
        # derivation produces weak results.
        # --------------------------------------------------------

        current_goal = (
            current_intent.get(
                "current_goal"
            )
            or intent.get(
                "goal"
            )
            or ""
        )

        goal_keywords = (
            current_intent.get(
                "goal_keywords"
            )
            or intent.get(
                "goalKeywords"
            )
            or []
        )

        # Goal keyword matching.
        if goal_keywords:

            goal_keyword_pool = 0.08

            valid_keywords = []

            for keyword in goal_keywords:

                if isinstance(
                    keyword,
                    dict
                ):

                    keyword = (
                        keyword.get(
                            "keyword"
                        )
                        or keyword.get(
                            "topic"
                        )
                        or keyword.get(
                            "text"
                        )
                    )

                if keyword:
                    valid_keywords.append(
                        str(keyword)
                    )

            if valid_keywords:

                contribution = (
                    goal_keyword_pool
                    / len(valid_keywords)
                )

                for keyword in valid_keywords:

                    resolved = (
                        self._resolve_vocab_topic(
                            keyword,
                            vocabulary
                        )
                    )

                    if resolved:

                        counts[
                            resolved
                        ] += contribution

        # If there are no derived topics, map a short current goal
        # directly to the vocabulary.
        if (
            current_goal
            and not derived
        ):

            resolved = (
                self._resolve_vocab_topic(
                    current_goal,
                    vocabulary
                )
            )

            if resolved:

                counts[
                    resolved
                ] += 0.20

        # --------------------------------------------------------
        # 2. Preferred Topics
        # --------------------------------------------------------

        preferred = (
            current_intent.get(
                "preferred_topics"
            )
            or intent.get(
                "preferredTopics"
            )
            or []
        )

        preferred_weights = (
            current_intent.get(
                "preferred_weights"
            )
            or intent.get(
                "preferred_weights"
            )
            or {}
        )

        preferred_pool = 0.40

        for preferred_topic in preferred:

            resolved = (
                self._resolve_vocab_topic(
                    preferred_topic,
                    vocabulary
                )
            )

            if resolved is None:
                continue

            learned_weight = 1.0

            # Case-insensitive learned-weight lookup.
            for key, value in (
                preferred_weights.items()
            ):

                if (
                    self._topic_key(
                        key
                    )
                    == self._topic_key(
                        preferred_topic
                    )
                ):

                    learned_weight = (
                        self._safe_float(
                            value,
                            1.0
                        )
                    )
                    break

            learned_weight = max(
                0.10,
                min(
                    2.0,
                    learned_weight
                )
            )

            counts[
                resolved
            ] += (
                learned_weight
                * preferred_pool
            )

        # --------------------------------------------------------
        # 3. Profession
        # --------------------------------------------------------

        profession = (
            user_profile.get(
                "profession"
            )
            or ""
        )

        if profession:

            profession_key = (
                self._topic_key(
                    profession
                )
            )

            for vocabulary_topic in vocabulary:

                vocab_key = (
                    self._topic_key(
                        vocabulary_topic
                    )
                )

                if (
                    profession_key in vocab_key
                    or vocab_key in profession_key
                ):

                    counts[
                        vocabulary_topic
                    ] += 0.10

        # --------------------------------------------------------
        # 4. Interests
        # --------------------------------------------------------

        interests = (
            user_profile.get(
                "interests"
            )
            or []
        )

        if interests:

            per_interest = (
                0.07
                / len(interests)
            )

            for interest in interests:

                resolved = (
                    self._resolve_vocab_topic(
                        interest,
                        vocabulary
                    )
                )

                if resolved:

                    counts[
                        resolved
                    ] += per_interest

        # --------------------------------------------------------
        # 5. Hobbies
        # --------------------------------------------------------

        hobbies = (
            user_profile.get(
                "hobbies"
            )
            or []
        )

        if hobbies:

            per_hobby = (
                0.03
                / len(hobbies)
            )

            for hobby in hobbies:

                resolved = (
                    self._resolve_vocab_topic(
                        hobby,
                        vocabulary
                    )
                )

                if resolved:

                    counts[
                        resolved
                    ] += per_hobby

        # --------------------------------------------------------
        # Normalize
        # --------------------------------------------------------

        total = sum(
            counts.values()
        )

        if total <= 0:

            uniform = (
                1.0
                / len(vocabulary)
            )

            return {
                topic: round(
                    uniform,
                    6
                )
                for topic in vocabulary
            }

        distribution = {
            topic: (
                counts[topic]
                / total
            )
            for topic in vocabulary
        }

        # Correct floating-point drift.
        distribution = self._normalize_distribution(
            distribution,
            vocabulary
        )

        return {
            topic: round(
                value,
                6
            )
            for topic, value
            in distribution.items()
        }

    # ============================================================
    # Distribution Normalization
    # ============================================================

    def _normalize_distribution(
        self,
        distribution: Dict[str, float],
        vocabulary: List[str]
    ) -> Dict[str, float]:

        values = np.array(
            [
                max(
                    0.0,
                    self._safe_float(
                        distribution.get(
                            topic,
                            0.0
                        )
                    )
                )
                for topic in vocabulary
            ],
            dtype=np.float64
        )

        total = float(
            np.sum(values)
        )

        if total <= 0:

            uniform = (
                1.0
                / max(
                    1,
                    len(vocabulary)
                )
            )

            return {
                topic: uniform
                for topic in vocabulary
            }

        values = values / total

        return {
            topic: float(
                values[index]
            )
            for index, topic
            in enumerate(vocabulary)
        }

    # ============================================================
    # Feed Distribution
    # ============================================================

    def build_feed_distribution(
        self,
        videos: List[Dict[str, Any]],
        vocabulary: List[str],
        window_size: int = 50,
        filter_mode: str = "all"
    ) -> Dict[str, float]:
        """
        Build feed topic distribution Q.

        filter_mode:

            "all"
                Raw observed feed.

            "controlled"
                Visible ALLOW + DEPRIORITIZE content only.

        Raw feed is the baseline exposure.

        Controlled feed represents what the extension allowed
        the user to continue seeing.
        """

        counts = {
            topic: 0.0
            for topic in vocabulary
        }

        if not videos or not vocabulary:

            return counts

        # --------------------------------------------------------
        # Select sample
        # --------------------------------------------------------

        if filter_mode == "controlled":

            eligible = [

                video
                for video in videos

                if (
                    video.get(
                        "controlled_visible"
                    ) is True

                    or video.get(
                        "decision"
                    ) in {
                        "ALLOW",
                        "DEPRIORITIZE"
                    }
                )
            ]

        else:

            eligible = list(
                videos
            )

        # Use the most recent window.
        recent_window = (
            eligible[-window_size:]
            if window_size > 0
            else eligible
        )

        if not recent_window:

            return counts

        # --------------------------------------------------------
        # Accumulate topic evidence
        # --------------------------------------------------------

        for video in recent_window:

            if not isinstance(
                video,
                dict
            ):
                continue

            decision = str(
                video.get(
                    "decision",
                    "ALLOW"
                )
            ).upper()

            # Suppressed content must never contribute to the
            # controlled distribution.
            if (
                filter_mode == "controlled"
                and decision == "SUPPRESS"
            ):
                continue

            # Deprioritized content remains visible, but has lower
            # prominence than fully allowed content.
            if decision == "DEPRIORITIZE":

                prominence = 0.35

            else:

                prominence = 1.0

            topics = (
                video.get(
                    "topics"
                )
                or video.get(
                    "topic_dict"
                )
                or {}
            )

            topic_items = (
                self._extract_topic_items(
                    topics
                )
            )

            for topic_key, score in (
                topic_items.items()
            ):

                resolved = (
                    self._resolve_vocab_topic(
                        topic_key,
                        vocabulary
                    )
                )

                if resolved is None:
                    continue

                counts[
                    resolved
                ] += (
                    max(
                        0.0,
                        score
                    )
                    * prominence
                )

        total = sum(
            counts.values()
        )

        if total <= 0:

            return {
                topic: 0.0
                for topic in vocabulary
            }

        distribution = {
            topic: (
                counts[topic]
                / total
            )
            for topic in vocabulary
        }

        return {
            topic: round(
                value,
                6
            )
            for topic, value
            in distribution.items()
        }

    # ============================================================
    # Jensen-Shannon Divergence
    # ============================================================

    def compute_jsd(
        self,
        p_dist: Dict[str, float],
        q_dist: Dict[str, float],
        vocabulary: Optional[List[str]] = None,
        epsilon: float = 1e-10
    ) -> float:
        """
        Jensen-Shannon Divergence.

        Base-2 logarithm gives:

            0 <= JSD <= 1

        Interpretation:

            0
                Identical distributions.

            Higher value
                Greater distributional divergence.
        """

        vocab = (
            vocabulary
            or self.vocabulary
            or sorted(
                set(
                    p_dist.keys()
                )
                | set(
                    q_dist.keys()
                )
            )
        )

        if not vocab:
            return 0.0

        p_values = np.array(
            [
                max(
                    0.0,
                    self._safe_float(
                        p_dist.get(
                            topic,
                            0.0
                        )
                    )
                )
                for topic in vocab
            ],
            dtype=np.float64
        )

        q_values = np.array(
            [
                max(
                    0.0,
                    self._safe_float(
                        q_dist.get(
                            topic,
                            0.0
                        )
                    )
                )
                for topic in vocab
            ],
            dtype=np.float64
        )

        p_sum = float(
            np.sum(
                p_values
            )
        )

        q_sum = float(
            np.sum(
                q_values
            )
        )

        # If no controlled content exists, divergence cannot
        # meaningfully be calculated from an empty distribution.
        if p_sum <= 0:

            return 0.0

        if q_sum <= 0:

            return 1.0

        p = (
            p_values
            / p_sum
        )

        q = (
            q_values
            / q_sum
        )

        midpoint = (
            0.5
            * (
                p
                + q
            )
        )

        def kl_divergence(
            a,
            b
        ):

            value = 0.0

            for ai, bi in zip(
                a,
                b
            ):

                if ai <= 0:
                    continue

                if bi <= 0:
                    bi = epsilon

                value += (
                    ai
                    * math.log2(
                        (
                            ai
                            + epsilon
                        )
                        /
                        (
                            bi
                            + epsilon
                        )
                    )
                )

            return max(
                0.0,
                value
            )

        jsd = (
            0.5
            * kl_divergence(
                p,
                midpoint
            )
            +
            0.5
            * kl_divergence(
                q,
                midpoint
            )
        )

        return self._clamp(
            float(jsd)
        )

    # ============================================================
    # Shannon Entropy Diversity
    # ============================================================

    def compute_entropy_diversity(
        self,
        dist: Dict[str, float]
    ) -> float:
        """
        Normalized Shannon entropy:

            0 = no topic diversity
            1 = maximum diversity
        """

        if not dist:
            return 0.0

        values = [
            self._safe_float(
                value
            )
            for value in dist.values()
            if self._safe_float(
                value
            ) > 0
        ]

        if not values:
            return 0.0

        if len(values) <= 1:
            return 0.0

        total = sum(
            values
        )

        if total <= 0:
            return 0.0

        values = [
            value / total
            for value in values
        ]

        entropy = -sum(
            probability
            * math.log2(
                probability
            )
            for probability in values
            if probability > 0
        )

        max_entropy = math.log2(
            len(dist)
        )

        if max_entropy <= 0:
            return 0.0

        return round(
            self._clamp(
                entropy
                / max_entropy
            ),
            4
        )

    def compute_topic_diversity(
        self,
        dist: Dict[str, float]
    ) -> float:
        """Backward-compatible alias."""

        return self.compute_entropy_diversity(
            dist
        )

    # ============================================================
    # Exposure Metrics
    # ============================================================

    def _calculate_exposure(
        self,
        videos: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate preferred and negative exposure.

        Denominator:
            Number of videos in the respective sample.

        Threshold:
            0.35 relevance.

        This makes raw and controlled exposure rates directly
        comparable.
        """

        count = len(
            videos
        )

        if count <= 0:

            return {
                "negative_exposure": 0.0,
                "preferred_exposure": 0.0
            }

        negative_hits = 0
        preferred_hits = 0

        for video in videos:

            negative = self._safe_float(
                video.get(
                    "negative_relevance",
                    0.0
                )
            )

            preferred = self._safe_float(
                video.get(
                    "positive_relevance",
                    0.0
                )
            )

            if negative >= 0.35:
                negative_hits += 1

            if preferred >= 0.35:
                preferred_hits += 1

        return {
            "negative_exposure": round(
                negative_hits / count,
                4
            ),

            "preferred_exposure": round(
                preferred_hits / count,
                4
            )
        }

    # ============================================================
    # Dual Feed Evaluation
    # ============================================================

    def evaluate_dual_feed(
        self,
        intent: Dict[str, Any],
        feed_history: List[Dict[str, Any]],
        window_size: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate the same observed sample before and after
        IntentAlign intervention.

        Raw sample:
            All exposed videos.

        Controlled sample:
            ALLOW + DEPRIORITIZE / controlled_visible videos.

        Returns:

            raw_jsd
            controlled_jsd
            jsd_improvement
            raw_alignment
            controlled_alignment
            alignment_improvement
            raw_negative_exposure
            controlled_negative_exposure
            negative_exposure_reduction
            raw_preferred_exposure
            controlled_preferred_exposure
            preferred_exposure_improvement
            topic_diversity
        """

        if not isinstance(
            feed_history,
            list
        ):
            feed_history = []

        # ========================================================
        # Dynamic Vocabulary
        # ========================================================

        vocabulary = (
            self.get_dynamic_vocabulary(
                intent,
                feed_history
            )
        )

        # ========================================================
        # Intent Distribution
        # ========================================================

        intent_distribution = (
            self.get_intent_distribution(
                intent,
                vocabulary
            )
        )

        # ========================================================
        # Raw Window
        # ========================================================

        raw_window = (
            feed_history[-window_size:]
            if feed_history
            and window_size > 0
            else list(feed_history)
        )

        # ========================================================
        # Controlled Window
        # ========================================================
        #
        # Important:
        # Controlled feed is derived from the SAME raw observation
        # window. We do not take a second independent time window.
        # ========================================================

        controlled_window = [

            video
            for video in raw_window

            if (
                video.get(
                    "controlled_visible"
                ) is True

                or str(
                    video.get(
                        "decision",
                        ""
                    )
                ).upper()
                in {
                    "ALLOW",
                    "DEPRIORITIZE"
                }
            )
        ]

        raw_count = len(
            raw_window
        )

        controlled_count = len(
            controlled_window
        )

        # ========================================================
        # Empty Data
        # ========================================================

        if raw_count == 0:

            return {
                "has_data": False,

                "raw_count": 0,

                "controlled_count": 0,

                "suppressed_count": 0,

                "intent_distribution": {},

                "raw_feed_distribution": {},

                "controlled_feed_distribution": {},

                "raw_jsd": 0.0,

                "controlled_jsd": 0.0,

                "jsd_improvement": 0.0,

                "raw_alignment": 0.0,

                "controlled_alignment": 0.0,

                "alignment_improvement": 0.0,

                "raw_negative_exposure": 0.0,

                "controlled_negative_exposure": 0.0,

                "negative_exposure_reduction": 0.0,

                "raw_preferred_exposure": 0.0,

                "controlled_preferred_exposure": 0.0,

                "preferred_exposure_improvement": 0.0,

                "topic_diversity": 0.0
            }

        # ========================================================
        # Raw Distribution
        # ========================================================

        raw_q = self.build_feed_distribution(
            raw_window,
            vocabulary,
            window_size=len(raw_window),
            filter_mode="all"
        )

        # ========================================================
        # Controlled Distribution
        # ========================================================

        ctrl_q = self.build_feed_distribution(
            controlled_window,
            vocabulary,
            window_size=len(controlled_window),
            filter_mode="controlled"
        )

        # ========================================================
        # JSD
        # ========================================================

        raw_jsd = self.compute_jsd(
            intent_distribution,
            raw_q,
            vocabulary
        )

        if controlled_count > 0:

            controlled_jsd = (
                self.compute_jsd(
                    intent_distribution,
                    ctrl_q,
                    vocabulary
                )
            )

        else:

            # Empty controlled feed means maximum divergence from
            # a non-empty intent distribution.
            controlled_jsd = 1.0

        # ========================================================
        # Alignment
        # ========================================================

        raw_alignment = self._clamp(
            1.0 - raw_jsd
        )

        controlled_alignment = self._clamp(
            1.0 - controlled_jsd
        )

        alignment_improvement = (
            controlled_alignment
            - raw_alignment
        )

        # Positive = improvement.
        # Negative = controlled feed moved farther away.
        alignment_improvement = round(
            alignment_improvement,
            4
        )

        # ========================================================
        # JSD Improvement
        # ========================================================

        jsd_improvement = (
            raw_jsd
            - controlled_jsd
        )

        # Positive = divergence reduced.
        jsd_improvement = round(
            jsd_improvement,
            4
        )

        # ========================================================
        # Exposure
        # ========================================================

        raw_exposure = (
            self._calculate_exposure(
                raw_window
            )
        )

        controlled_exposure = (
            self._calculate_exposure(
                controlled_window
            )
        )

        raw_negative_exposure = (
            raw_exposure[
                "negative_exposure"
            ]
        )

        controlled_negative_exposure = (
            controlled_exposure[
                "negative_exposure"
            ]
        )

        raw_preferred_exposure = (
            raw_exposure[
                "preferred_exposure"
            ]
        )

        controlled_preferred_exposure = (
            controlled_exposure[
                "preferred_exposure"
            ]
        )

        # Positive = unwanted exposure decreased.
        negative_exposure_reduction = (
            raw_negative_exposure
            - controlled_negative_exposure
        )

        # Positive = preferred exposure increased.
        preferred_exposure_improvement = (
            controlled_preferred_exposure
            - raw_preferred_exposure
        )

        # ========================================================
        # Topic Diversity
        # ========================================================

        if controlled_count > 0:

            diversity_distribution = ctrl_q

        else:

            diversity_distribution = raw_q

        topic_diversity = (
            self.compute_entropy_diversity(
                diversity_distribution
            )
        )

        # ========================================================
        # Suppression Statistics
        # ========================================================

        suppressed_count = sum(

            1

            for video in raw_window

            if str(
                video.get(
                    "decision",
                    ""
                )
            ).upper()
            == "SUPPRESS"

            or (
                video.get(
                    "controlled_visible"
                )
                is False
            )
        )

        # Never allow the count to become negative.
        suppressed_count = max(
            0,
            min(
                raw_count,
                suppressed_count
            )
        )

        # ========================================================
        # Final Result
        # ========================================================

        intent_output = {
            key: round(
                value,
                4
            )
            for key, value
            in intent_distribution.items()
            if value > 0
        }

        raw_output = {
            key: round(
                value,
                4
            )
            for key, value
            in raw_q.items()
            if value > 0
        }

        controlled_output = {
            key: round(
                value,
                4
            )
            for key, value
            in ctrl_q.items()
            if value > 0
        }

        return {

            "has_data": True,

            # ----------------------------------------------------
            # Counts
            # ----------------------------------------------------

            "raw_count": raw_count,

            "controlled_count": controlled_count,

            "suppressed_count": suppressed_count,

            # ----------------------------------------------------
            # Distributions
            # ----------------------------------------------------

            "intent_distribution": intent_output,

            "raw_feed_distribution": raw_output,

            "controlled_feed_distribution": (
                controlled_output
            ),

            # ----------------------------------------------------
            # Divergence
            # ----------------------------------------------------

            "raw_jsd": round(
                raw_jsd,
                4
            ),

            "controlled_jsd": round(
                controlled_jsd,
                4
            ),

            "jsd_improvement": jsd_improvement,

            # ----------------------------------------------------
            # Alignment
            # ----------------------------------------------------

            "raw_alignment": round(
                raw_alignment,
                4
            ),

            "controlled_alignment": round(
                controlled_alignment,
                4
            ),

            "alignment_improvement": (
                alignment_improvement
            ),

            # ----------------------------------------------------
            # Negative Exposure
            # ----------------------------------------------------

            "raw_negative_exposure": (
                raw_negative_exposure
            ),

            "controlled_negative_exposure": (
                controlled_negative_exposure
            ),

            "negative_exposure_reduction": round(
                negative_exposure_reduction,
                4
            ),

            # ----------------------------------------------------
            # Preferred Exposure
            # ----------------------------------------------------

            "raw_preferred_exposure": (
                raw_preferred_exposure
            ),

            "controlled_preferred_exposure": (
                controlled_preferred_exposure
            ),

            "preferred_exposure_improvement": round(
                preferred_exposure_improvement,
                4
            ),

            # ----------------------------------------------------
            # Diversity
            # ----------------------------------------------------

            "topic_diversity": topic_diversity,

            # ----------------------------------------------------
            # Metadata
            # ----------------------------------------------------

            "vocabulary_size": len(
                vocabulary
            ),

            # ----------------------------------------------------
            # Backward-compatible aliases
            # ----------------------------------------------------

            "jsDivergence": round(
                controlled_jsd,
                4
            ),

            "intentAlignment": round(
                controlled_alignment,
                4
            ),

            "intentDistribution": intent_output,

            "feedDistribution": controlled_output,

            "negativeExposure": (
                controlled_negative_exposure
            ),

            "preferredExposure": (
                controlled_preferred_exposure
            ),

            "topicDiversity": topic_diversity,

            "sampleSize": raw_count
        }

    # ============================================================
    # Backward-Compatible Feed Evaluation
    # ============================================================

    def evaluate_feed(
        self,
        intent: Dict[str, Any],
        feed_history: List[Dict[str, Any]],
        window_size: int = 50
    ) -> Dict[str, Any]:
        """
        Backward-compatible wrapper.

        Returns controlled-feed metrics as the standard feed
        evaluation values while retaining raw-vs-controlled metrics.
        """

        result = self.evaluate_dual_feed(
            intent,
            feed_history,
            window_size=window_size
        )

        result[
            "js_divergence"
        ] = result.get(
            "controlled_jsd",
            result.get(
                "raw_jsd",
                0.0
            )
        )

        result[
            "intent_alignment"
        ] = result.get(
            "controlled_alignment",
            result.get(
                "raw_alignment",
                0.0
            )
        )

        result[
            "preferred_exposure"
        ] = result.get(
            "controlled_preferred_exposure",
            0.0
        )

        result[
            "unwanted_exposure"
        ] = result.get(
            "controlled_negative_exposure",
            0.0
        )

        return result