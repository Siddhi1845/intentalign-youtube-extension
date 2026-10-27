"""
Feedback and Adaptive Learning Service for IntentAlign
V1.3 — Session-Aware Closed-Loop Feedback Learning

Handles:
    relevant
    not_relevant
    wrongly_hidden
    wrongfully_hidden
    show_anyway

Learning loop:

    User Feedback
          ↓
    Topic Weight Update
          ↓
    Updated Intent Profile
          ↓
    Rebuild / Recalculate Intent Profile
          ↓
    Recalculate Video Alignment
          ↓
    Recalculate Feed Distribution
          ↓
    Recalculate JSD / Intervention
          ↓
    Future Feed Decisions

Important:
- Explicit user preferences are preserved.
- Learned weights are kept separately from explicit topic lists.
- Weight adaptation is bounded.
- Feedback is isolated by intent session.
- This service does not itself perform semantic analysis.
  It returns recalculation metadata for the caller.
"""

from typing import Dict, List, Any, Optional
import time


class FeedbackService:

    VALID_ACTIONS = {
        "relevant",
        "not_relevant",
        "wrongly_hidden",
        "wrongfully_hidden",
        "show_anyway"
    }

    WRONGFULLY_HIDDEN_ACTIONS = {
        "wrongly_hidden",
        "wrongfully_hidden",
        "show_anyway"
    }

    def __init__(
        self,
        default_learning_rate: float = 0.05,
        min_weight: float = 0.10,
        max_weight: float = 2.00
    ):
        """
        Initialize bounded adaptive learning.

        Weight range:
            min_weight <= weight <= max_weight
        """

        self.learning_rate = self._clamp(
            default_learning_rate,
            0.001,
            0.50
        )

        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

        if self.min_weight < 0:
            self.min_weight = 0.0

        if self.max_weight <= self.min_weight:
            self.max_weight = max(
                1.0,
                self.min_weight + 0.10
            )

        # Global audit trail.
        #
        # Each entry contains session_id so history can be
        # filtered without mixing different current intents.
        self.feedback_history: List[
            Dict[str, Any]
        ] = []

    # ============================================================
    # Utility Functions
    # ============================================================

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float
    ) -> float:
        """Safely clamp a numeric value."""

        try:
            value = float(value)
        except (TypeError, ValueError):
            value = minimum

        return max(
            minimum,
            min(maximum, value)
        )

    @staticmethod
    def _normalize_topic(topic: Any) -> str:
        """
        Normalize topic names so:

            "Machine Learning"
            "machine learning"
            " Machine Learning "

        do not become three separate learned topics.
        """

        if topic is None:
            return ""

        return " ".join(
            str(topic).strip().split()
        )

    @classmethod
    def _topic_key(cls, topic: Any) -> str:
        """Create case-insensitive topic lookup key."""

        return cls._normalize_topic(topic).lower()

    def _bounded_weight(
        self,
        value: float
    ) -> float:
        """Apply configured weight bounds."""

        return round(
            self._clamp(
                value,
                self.min_weight,
                self.max_weight
            ),
            4
        )

    # ============================================================
    # Topic Normalization
    # ============================================================

    def _normalize_topic_scores(
        self,
        topics: Any
    ) -> Dict[str, float]:
        """
        Normalize supported topic formats.

        Supported:

            {
                "Python": 0.8,
                "Machine Learning": 0.6
            }

        or:

            [
                {"topic": "Python", "score": 0.8},
                {"topic": "Machine Learning", "score": 0.6}
            ]

        or:

            ["Python", "Machine Learning"]
        """

        normalized: Dict[str, float] = {}

        if isinstance(topics, list):

            for item in topics:

                if isinstance(item, dict):

                    topic = item.get(
                        "topic",
                        item.get("name", "")
                    )

                    score = item.get(
                        "score",
                        1.0
                    )

                else:

                    topic = item
                    score = 1.0

                topic = self._normalize_topic(topic)

                if not topic:
                    continue

                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = 1.0

                score = self._clamp(
                    score,
                    0.0,
                    1.0
                )

                key = self._topic_key(topic)

                # Preserve the strongest evidence if the same
                # topic occurs more than once.
                if (
                    key not in normalized
                    or score > normalized[key]
                ):
                    normalized[key] = score

        elif isinstance(topics, dict):

            for topic, score in topics.items():

                topic = self._normalize_topic(topic)

                if not topic:
                    continue

                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = 1.0

                score = self._clamp(
                    score,
                    0.0,
                    1.0
                )

                key = self._topic_key(topic)

                if (
                    key not in normalized
                    or score > normalized[key]
                ):
                    normalized[key] = score

        # Convert normalized keys back to readable topic names.
        readable = {}

        for topic_key, score in normalized.items():

            readable[topic_key] = round(
                score,
                4
            )

        return readable

    # ============================================================
    # Weight Dictionary Helpers
    # ============================================================

    def _normalize_weight_dictionary(
        self,
        weights: Any
    ) -> Dict[str, float]:
        """
        Normalize learned weight dictionaries.

        Topic names are stored in readable form while lookup is
        performed case-insensitively.
        """

        output: Dict[str, float] = {}

        if not isinstance(weights, dict):
            return output

        for topic, weight in weights.items():

            topic_name = self._normalize_topic(topic)

            if not topic_name:
                continue

            try:
                weight = float(weight)
            except (TypeError, ValueError):
                weight = 1.0

            output[topic_name] = self._bounded_weight(
                weight
            )

        return output

    def _find_existing_topic(
        self,
        weights: Dict[str, float],
        topic: str
    ) -> Optional[str]:
        """
        Find an existing topic ignoring capitalization.
        """

        target = self._topic_key(topic)

        for existing in weights.keys():

            if self._topic_key(existing) == target:
                return existing

        return None

    def _get_or_create_weight(
        self,
        weights: Dict[str, float],
        topic: str
    ) -> tuple:
        """
        Return:

            actual_topic_name,
            current_weight

        If topic does not exist, initialize it at 1.0.
        """

        existing = self._find_existing_topic(
            weights,
            topic
        )

        if existing is not None:

            return (
                existing,
                self._bounded_weight(
                    weights[existing]
                )
            )

        topic = self._normalize_topic(topic)

        weights[topic] = 1.0

        return (
            topic,
            1.0
        )

    # ============================================================
    # Feedback Recording
    # ============================================================

    def record_feedback(
        self,
        feedback_data: Dict[str, Any],
        intent_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Record feedback and adapt learned topic weights.

        Feedback actions:

            relevant
                Increase preferred-topic influence.

            not_relevant
                Increase unwanted-topic influence when appropriate,
                otherwise decrease learned preferred influence.

            wrongly_hidden / wrongfully_hidden / show_anyway
                Reduce unwanted influence and strengthen preferred
                influence because the user explicitly recovered a
                suppressed item.

        The returned object includes:
            - updated intent profile
            - weight changes
            - session information
            - recalculation_required
            - restored_card_id
        """

        if not isinstance(feedback_data, dict):
            return {
                "status": "invalid_feedback",
                "reason": "feedback_data must be a dictionary.",
                "intent_profile": intent_profile,
                "weight_changes": {},
                "recalculation_required": False
            }

        if not isinstance(intent_profile, dict):
            return {
                "status": "invalid_intent_profile",
                "reason": "intent_profile must be a dictionary.",
                "weight_changes": {},
                "recalculation_required": False
            }

        # ========================================================
        # Read feedback event
        # ========================================================

        video_id = str(
            feedback_data.get(
                "video_id",
                ""
            ) or ""
        )

        title = str(
            feedback_data.get(
                "title",
                ""
            ) or ""
        )

        action = str(
            feedback_data.get(
                "feedback",
                feedback_data.get(
                    "action",
                    ""
                )
            )
            or ""
        ).strip().lower()

        session_id = str(
            feedback_data.get(
                "session_id",
                feedback_data.get(
                    "intent_session_id",
                    "session_default"
                )
            )
            or "session_default"
        )

        timestamp = (
            feedback_data.get(
                "timestamp"
            )
            or time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime()
            )
        )

        topics = feedback_data.get(
            "topics",
            feedback_data.get(
                "topic_scores",
                {}
            )
        )

        topic_scores = self._normalize_topic_scores(
            topics
        )

        # ========================================================
        # Validate action
        # ========================================================

        if action not in self.VALID_ACTIONS:

            # Still keep invalid events out of the learning loop.
            invalid_entry = {
                "session_id": session_id,
                "video_id": video_id,
                "title": title,
                "feedback": action,
                "topics": topic_scores,
                "timestamp": timestamp,
                "status": "invalid_action"
            }

            self.feedback_history.append(
                invalid_entry
            )

            return {
                "status": "invalid_feedback",
                "reason": (
                    f"Unsupported feedback action: '{action}'."
                ),
                "allowed_actions": sorted(
                    self.VALID_ACTIONS
                ),
                "intent_profile": intent_profile,
                "weight_changes": {},
                "recalculation_required": False,
                "total_feedback": len(
                    self.feedback_history
                )
            }

        # ========================================================
        # Store audit event
        # ========================================================

        history_entry = {
            "session_id": session_id,
            "video_id": video_id,
            "title": title,
            "feedback": action,
            "topics": topic_scores,
            "timestamp": timestamp,
            "status": "recorded"
        }

        self.feedback_history.append(
            history_entry
        )

        # ========================================================
        # Read learning settings
        # ========================================================

        settings = (
            intent_profile.get(
                "settings"
            )
            or {}
        )

        learning_enabled = settings.get(
            "feedback_learning_enabled",
            True
        )

        try:
            learning_rate = float(
                settings.get(
                    "learning_rate",
                    self.learning_rate
                )
            )
        except (TypeError, ValueError):
            learning_rate = self.learning_rate

        learning_rate = self._clamp(
            learning_rate,
            0.001,
            0.50
        )

        # ========================================================
        # Learning disabled
        # ========================================================

        if not learning_enabled:

            return {
                "status": "recorded_without_weight_update",

                "intent_profile": intent_profile,

                "weight_changes": {},

                "recalculation_required": False,

                "learning_enabled": False,

                "session_id": session_id,

                "restored_card_id": (
                    video_id
                    if action in self.WRONGFULLY_HIDDEN_ACTIONS
                    else None
                ),

                "total_feedback": len(
                    self.feedback_history
                )
            }

        # ========================================================
        # Get current intent
        # ========================================================

        current_intent = (
            intent_profile.get(
                "current_intent"
            )
            or {}
        )

        if not isinstance(
            current_intent,
            dict
        ):
            current_intent = {}

        # ========================================================
        # IMPORTANT:
        # Keep explicit topics separate from learned weights.
        # ========================================================

        explicit_preferred = (
            intent_profile.get(
                "preferredTopics",
                []
            )
            or []
        )

        explicit_unwanted = (
            intent_profile.get(
                "unwantedTopics",
                []
            )
            or []
        )

        # Existing learned weights.
        preferred_weights = self._normalize_weight_dictionary(
            current_intent.get(
                "preferred_weights",
                intent_profile.get(
                    "preferred_weights",
                    {}
                )
            )
        )

        unwanted_weights = self._normalize_weight_dictionary(
            current_intent.get(
                "unwanted_weights",
                intent_profile.get(
                    "unwanted_weights",
                    {}
                )
            )
        )

        # Initialize explicit topics if a weight does not yet exist.
        for topic in explicit_preferred:

            topic = self._normalize_topic(topic)

            if not topic:
                continue

            if self._find_existing_topic(
                preferred_weights,
                topic
            ) is None:

                preferred_weights[topic] = 1.0

        for topic in explicit_unwanted:

            topic = self._normalize_topic(topic)

            if not topic:
                continue

            if self._find_existing_topic(
                unwanted_weights,
                topic
            ) is None:

                unwanted_weights[topic] = 1.0

        # ========================================================
        # Apply feedback
        # ========================================================

        weight_changes: Dict[
            str,
            Dict[str, Any]
        ] = {}

        # Only sufficiently strong topic evidence should update
        # the model.
        minimum_topic_evidence = 0.22

        for topic_key, score in topic_scores.items():

            if score < minimum_topic_evidence:
                continue

            topic = self._normalize_topic(
                topic_key
            )

            if not topic:
                continue

            # ----------------------------------------------------
            # RELEVANT
            # ----------------------------------------------------

            if action == "relevant":

                actual_topic, old_weight = (
                    self._get_or_create_weight(
                        preferred_weights,
                        topic
                    )
                )

                delta = (
                    learning_rate * score
                )

                new_weight = self._bounded_weight(
                    old_weight + delta
                )

                preferred_weights[
                    actual_topic
                ] = new_weight

                weight_changes[
                    actual_topic
                ] = {
                    "old": round(
                        old_weight,
                        4
                    ),
                    "new": new_weight,
                    "delta": round(
                        new_weight - old_weight,
                        4
                    ),
                    "type": "preferred",
                    "feedback": "relevant"
                }

            # ----------------------------------------------------
            # NOT RELEVANT
            # ----------------------------------------------------

            elif action == "not_relevant":

                # If this is already an unwanted topic, strengthen
                # its penalty.
                unwanted_topic = (
                    self._find_existing_topic(
                        unwanted_weights,
                        topic
                    )
                )

                if unwanted_topic is not None:

                    old_weight = self._bounded_weight(
                        unwanted_weights[
                            unwanted_topic
                        ]
                    )

                    delta = (
                        learning_rate * score
                    )

                    new_weight = self._bounded_weight(
                        old_weight + delta
                    )

                    unwanted_weights[
                        unwanted_topic
                    ] = new_weight

                    weight_changes[
                        unwanted_topic
                    ] = {
                        "old": round(
                            old_weight,
                            4
                        ),
                        "new": new_weight,
                        "delta": round(
                            new_weight - old_weight,
                            4
                        ),
                        "type": "unwanted",
                        "feedback": "not_relevant"
                    }

                else:

                    # If the topic was learned as preferred,
                    # reduce its learned preference rather than
                    # silently adding it to the user's explicit
                    # unwantedTopics list.
                    preferred_topic = (
                        self._find_existing_topic(
                            preferred_weights,
                            topic
                        )
                    )

                    if preferred_topic is not None:

                        old_weight = self._bounded_weight(
                            preferred_weights[
                                preferred_topic
                            ]
                        )

                        delta = (
                            learning_rate
                            * score
                            * 0.50
                        )

                        new_weight = self._bounded_weight(
                            old_weight - delta
                        )

                        preferred_weights[
                            preferred_topic
                        ] = new_weight

                        weight_changes[
                            preferred_topic
                        ] = {
                            "old": round(
                                old_weight,
                                4
                            ),
                            "new": new_weight,
                            "delta": round(
                                new_weight - old_weight,
                                4
                            ),
                            "type": "preferred_reduced",
                            "feedback": "not_relevant"
                        }

                    else:

                        # New negative evidence is learned in the
                        # adaptive unwanted-weight layer.
                        unwanted_weights[
                            topic
                        ] = self._bounded_weight(
                            1.0
                            + (
                                learning_rate
                                * score
                            )
                        )

                        weight_changes[
                            topic
                        ] = {
                            "old": 1.0,
                            "new": unwanted_weights[
                                topic
                            ],
                            "delta": round(
                                unwanted_weights[
                                    topic
                                ] - 1.0,
                                4
                            ),
                            "type": "learned_unwanted",
                            "feedback": "not_relevant"
                        }

            # ----------------------------------------------------
            # WRONGLY HIDDEN / SHOW ANYWAY
            # ----------------------------------------------------

            elif action in self.WRONGFULLY_HIDDEN_ACTIONS:

                # First reduce unwanted influence.
                unwanted_topic = (
                    self._find_existing_topic(
                        unwanted_weights,
                        topic
                    )
                )

                unwanted_change = None

                if unwanted_topic is not None:

                    old_unwanted = self._bounded_weight(
                        unwanted_weights[
                            unwanted_topic
                        ]
                    )

                    delta = (
                        learning_rate
                        * score
                        * 1.50
                    )

                    new_unwanted = self._bounded_weight(
                        old_unwanted - delta
                    )

                    unwanted_weights[
                        unwanted_topic
                    ] = new_unwanted

                    unwanted_change = {
                        "old": round(
                            old_unwanted,
                            4
                        ),
                        "new": new_unwanted,
                        "delta": round(
                            new_unwanted
                            - old_unwanted,
                            4
                        ),
                        "type": "unwanted_reduced"
                    }

                # Then strengthen preferred influence.
                preferred_topic = (
                    self._find_existing_topic(
                        preferred_weights,
                        topic
                    )
                )

                if preferred_topic is not None:

                    old_preferred = self._bounded_weight(
                        preferred_weights[
                            preferred_topic
                        ]
                    )

                    delta = (
                        learning_rate
                        * score
                        * 1.20
                    )

                    new_preferred = self._bounded_weight(
                        old_preferred + delta
                    )

                    preferred_weights[
                        preferred_topic
                    ] = new_preferred

                    preferred_change = {
                        "old": round(
                            old_preferred,
                            4
                        ),
                        "new": new_preferred,
                        "delta": round(
                            new_preferred
                            - old_preferred,
                            4
                        ),
                        "type": "preferred_boosted"
                    }

                else:

                    old_preferred = 1.0

                    delta = (
                        learning_rate
                        * score
                        * 1.20
                    )

                    new_preferred = self._bounded_weight(
                        old_preferred + delta
                    )

                    preferred_weights[
                        topic
                    ] = new_preferred

                    preferred_change = {
                        "old": old_preferred,
                        "new": new_preferred,
                        "delta": round(
                            new_preferred
                            - old_preferred,
                            4
                        ),
                        "type": "preferred_created"
                    }

                weight_changes[
                    topic
                ] = {
                    "feedback": action,
                    "type": "wrongly_hidden_recovery",
                    "unwanted": unwanted_change,
                    "preferred": preferred_change
                }

        # ========================================================
        # Save learned weights
        # ========================================================

        current_intent[
            "preferred_weights"
        ] = preferred_weights

        current_intent[
            "unwanted_weights"
        ] = unwanted_weights

        intent_profile[
            "current_intent"
        ] = current_intent

        # Keep backward-compatible top-level fields.
        intent_profile[
            "preferred_weights"
        ] = preferred_weights

        intent_profile[
            "unwanted_weights"
        ] = unwanted_weights

        # ========================================================
        # Store learning metadata
        # ========================================================

        learning_metadata = (
            intent_profile.get(
                "learning_metadata"
            )
            or {}
        )

        if not isinstance(
            learning_metadata,
            dict
        ):
            learning_metadata = {}

        learning_metadata[
            "last_feedback_action"
        ] = action

        learning_metadata[
            "last_feedback_session_id"
        ] = session_id

        learning_metadata[
            "last_feedback_timestamp"
        ] = timestamp

        learning_metadata[
            "last_feedback_video_id"
        ] = video_id

        learning_metadata[
            "feedback_learning_enabled"
        ] = True

        learning_metadata[
            "recalculation_required"
        ] = bool(weight_changes)

        intent_profile[
            "learning_metadata"
        ] = learning_metadata

        # ========================================================
        # Closed-loop recalculation signal
        # ========================================================
        #
        # This service does not directly call AlignmentService or
        # DivergenceService. The application layer should use this
        # flag to trigger:
        #
        #   rebuild intent
        #       ↓
        #   recalculate video alignment
        #       ↓
        #   recalculate feed profile
        #       ↓
        #   recalculate JSD
        #       ↓
        #   recalculate intervention
        #
        # ========================================================

        recalculation_required = bool(
            weight_changes
        )

        restored_card_id = (
            video_id
            if action in self.WRONGFULLY_HIDDEN_ACTIONS
            else None
        )

        # ========================================================
        # Return
        # ========================================================

        return {
            "status": (
                "updated"
                if weight_changes
                else "recorded_no_material_change"
            ),

            "intent_profile": intent_profile,

            "weight_changes": weight_changes,

            "recalculation_required": (
                recalculation_required
            ),

            "recalculation": {
                "required": recalculation_required,
                "reason": (
                    "Feedback changed learned topic weights."
                    if recalculation_required
                    else "No material weight change occurred."
                ),
                "session_id": session_id
            },

            "learning_enabled": True,

            "learning_rate": round(
                learning_rate,
                4
            ),

            "session_id": session_id,

            "feedback_action": action,

            "restored_card_id": restored_card_id,

            "total_feedback": len(
                self.feedback_history
            )
        }

    # ============================================================
    # Feedback History
    # ============================================================

    def get_history(
        self,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Return feedback history.

        If session_id is supplied, only feedback belonging to that
        intent session is returned.

        This prevents feedback from an old goal being mixed with
        the current goal.
        """

        if session_id is None:
            return list(
                self.feedback_history
            )

        return [
            feedback
            for feedback in self.feedback_history
            if feedback.get(
                "session_id"
            ) == session_id
        ]

    # ============================================================
    # Session Statistics
    # ============================================================

    def get_session_statistics(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Return basic feedback statistics for one intent session.
        """

        history = self.get_history(
            session_id
        )

        counts = {
            "relevant": 0,
            "not_relevant": 0,
            "wrongly_hidden": 0,
            "wrongfully_hidden": 0,
            "show_anyway": 0
        }

        for item in history:

            action = item.get(
                "feedback"
            )

            if action in counts:
                counts[action] += 1

        return {
            "session_id": session_id,
            "total_feedback": len(history),
            "counts": counts
        }

    # ============================================================
    # Clear Session History
    # ============================================================

    def clear_session_history(
        self,
        session_id: str
    ) -> int:
        """
        Remove feedback audit entries for a specific session.

        Returns the number of removed entries.

        This does NOT modify the intent profile itself.
        """

        before = len(
            self.feedback_history
        )

        self.feedback_history = [
            feedback
            for feedback in self.feedback_history
            if feedback.get(
                "session_id"
            ) != session_id
        ]

        return (
            before
            - len(self.feedback_history)
        )