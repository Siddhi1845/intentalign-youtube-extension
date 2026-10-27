"""
Adaptive Intervention Engine for IntentAlign
V1.3 — Goal-Aware Adaptive Intervention & Confidence

Responsibilities:
1. Calculate feed-level intervention pressure.
2. Decide per-video:
      ALLOW
      DEPRIORITIZE
      SUPPRESS
3. Respect the user's current goal above stable preferences.
4. Handle hard and soft negative topics contextually.
5. Handle uncertain / ambiguous content conservatively.
6. Keep alignment separate from intervention pressure.
7. Return clear explanations for UI and analytics.

Important:
- This service does NOT modify YouTube's internal recommendation ranking.
- It only determines how the browser extension should handle an observed video.
"""

from typing import Dict, Any, List


class InterventionService:

    def __init__(
        self,
        alpha: float = 0.60,
        beta: float = 0.40
    ):
        """
        Feed-level intervention score:

            InterventionScore =
                alpha * FeedDivergence
                + beta * NegativeExposure

        Default:
            alpha = 0.60
            beta  = 0.40
        """

        self.alpha = float(alpha)
        self.beta = float(beta)

        # Normalize weights if custom values are supplied.
        total = self.alpha + self.beta

        if total <= 0:
            self.alpha = 0.60
            self.beta = 0.40
        else:
            self.alpha /= total
            self.beta /= total

    # ============================================================
    # Utility Methods
    # ============================================================

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0
    ) -> float:
        """Clamp a numeric value to a safe range."""

        try:
            value = float(value)
        except (TypeError, ValueError):
            value = minimum

        return max(minimum, min(maximum, value))

    @staticmethod
    def _safe_bool(value: Any) -> bool:
        """Safely convert a value to boolean."""

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            return value.strip().lower() in {
                "true",
                "1",
                "yes",
                "y"
            }

        return bool(value)

    @staticmethod
    def _format_percent(value: float) -> int:
        """Convert 0-1 value into an integer percentage."""

        value = max(0.0, min(1.0, float(value)))
        return int(round(value * 100))

    @staticmethod
    def _get_matched_negative_topics(
        content_analysis: Dict[str, Any]
    ) -> List[str]:
        """Safely extract matched negative topics."""

        matched = content_analysis.get("matched_negative")

        if matched is None:
            matched = content_analysis.get("matched_negative_topics")

        if matched is None:
            return []

        if isinstance(matched, str):
            return [matched]

        if isinstance(matched, (list, tuple, set)):
            return [
                str(item)
                for item in matched
                if item is not None and str(item).strip()
            ]

        return []

    # ============================================================
    # Feed-Level Intervention
    # ============================================================

    def compute_feed_intervention(
        self,
        feed_metrics: Dict[str, Any],
        strictness: float = 0.65
    ) -> Dict[str, Any]:
        """
        Calculate feed-level intervention pressure.

        Formula:

            InterventionScore =
                alpha * RawFeedJSD
                + beta * RawNegativeExposure

        Higher score means the observed YouTube feed is further
        from the user's current intent.

        strictness controls when LOW / MEDIUM / HIGH intervention
        levels are triggered.

        Important:
        Strictness changes intervention sensitivity.
        It does NOT change the semantic alignment score itself.
        """

        strictness = self._clamp(strictness)

        # Prefer raw feed metrics because raw exposure represents
        # what YouTube actually exposed before intervention.
        divergence = feed_metrics.get(
            "raw_jsd",
            feed_metrics.get(
                "js_divergence",
                feed_metrics.get("divergence", 0.0)
            )
        )

        negative_exposure = feed_metrics.get(
            "raw_negative_exposure",
            feed_metrics.get(
                "unwanted_exposure",
                feed_metrics.get("negative_exposure", 0.0)
            )
        )

        divergence = self._clamp(divergence)
        negative_exposure = self._clamp(negative_exposure)

        # --------------------------------------------------------
        # Base intervention score
        # --------------------------------------------------------

        raw_score = (
            self.alpha * divergence
            + self.beta * negative_exposure
        )

        intervention_score = self._clamp(raw_score)

        # --------------------------------------------------------
        # Adaptive thresholds
        #
        # Higher strictness:
        #   LOW range becomes smaller
        #   MEDIUM begins earlier
        #   HIGH begins earlier
        #
        # These are experimental thresholds and should be tuned
        # using evaluation data.
        # --------------------------------------------------------

        medium_threshold = 0.30 - (
            0.10 * (strictness - 0.50)
        )

        high_threshold = 0.60 - (
            0.15 * (strictness - 0.50)
        )

        medium_threshold = max(
            0.20,
            min(0.35, medium_threshold)
        )

        high_threshold = max(
            0.45,
            min(0.65, high_threshold)
        )

        # Make sure HIGH cannot start before MEDIUM.
        if high_threshold <= medium_threshold:
            high_threshold = min(
                0.70,
                medium_threshold + 0.20
            )

        # --------------------------------------------------------
        # Intervention level
        # --------------------------------------------------------

        if intervention_score >= high_threshold:
            level = "HIGH"

        elif intervention_score >= medium_threshold:
            level = "MEDIUM"

        else:
            level = "LOW"

        return {
            "intervention_score": round(
                intervention_score,
                4
            ),

            "intervention_level": level,

            "thresholds": {
                "medium": round(
                    medium_threshold,
                    4
                ),
                "high": round(
                    high_threshold,
                    4
                )
            },

            "alpha": round(
                self.alpha,
                4
            ),

            "beta": round(
                self.beta,
                4
            ),

            "strictness": round(
                strictness,
                4
            ),

            "raw_jsd": round(
                divergence,
                4
            ),

            "raw_negative_exposure": round(
                negative_exposure,
                4
            )
        }

    # ============================================================
    # Per-Video Decision
    # ============================================================

    def decide_video(
        self,
        content_analysis: Dict[str, Any],
        alignment_result: Dict[str, Any],
        intervention_profile: Dict[str, Any],
        strictness: float = 0.65
    ) -> Dict[str, Any]:
        """
        Determine how a video should be handled.

        Possible decisions:

            ALLOW
            DEPRIORITIZE
            SUPPRESS

        Decision logic considers:

        - semantic alignment
        - current goal relevance
        - preferred-topic relevance
        - negative relevance
        - confidence
        - content-type compatibility
        - hard negative topics
        - soft negative topics
        - incidental topic matches
        - uncertain state
        - feed-level intervention pressure
        - strictness
        """

        # ========================================================
        # Read and sanitize inputs
        # ========================================================

        strictness = self._clamp(strictness)

        alignment = self._clamp(
            alignment_result.get(
                "alignment",
                0.0
            )
        )

        confidence = self._clamp(
            content_analysis.get(
                "confidence",
                0.50
            )
        )

        goal_relevance = self._clamp(
            content_analysis.get(
                "goal_relevance",
                alignment_result.get(
                    "goal_relevance",
                    0.0
                )
            )
        )

        positive_relevance = self._clamp(
            content_analysis.get(
                "positive_relevance",
                alignment_result.get(
                    "positive_relevance",
                    0.0
                )
            )
        )

        negative_relevance = self._clamp(
            content_analysis.get(
                "negative_relevance",
                alignment_result.get(
                    "negative_relevance",
                    0.0
                )
            )
        )

        type_compatibility = self._clamp(
            alignment_result.get(
                "content_type_compatibility",
                1.0
            )
        )

        is_uncertain = self._safe_bool(
            content_analysis.get(
                "is_uncertain",
                False
            )
        )

        is_incidental = self._safe_bool(
            content_analysis.get(
                "is_incidental",
                False
            )
        )

        is_hard_negative = self._safe_bool(
            alignment_result.get(
                "is_hard_negative",
                False
            )
        )

        hard_negative_topic = alignment_result.get(
            "hard_negative_topic"
        )

        soft_negative_mitigated = self._safe_bool(
            alignment_result.get(
                "soft_negative_mitigated",
                False
            )
        )

        content_type = content_analysis.get(
            "content_type",
            "Other"
        )

        matched_negative = self._get_matched_negative_topics(
            content_analysis
        )

        intervention_level = intervention_profile.get(
            "intervention_level",
            "LOW"
        )

        intervention_score = self._clamp(
            intervention_profile.get(
                "intervention_score",
                0.0
            )
        )

        # ========================================================
        # Adaptive thresholds
        # ========================================================

        # Higher strictness means stronger filtering.

        # Base ALLOW threshold.
        allow_threshold = (
            0.40
            + (0.12 * strictness)
        )

        # Feed-level pressure slightly increases filtering.
        if intervention_level == "MEDIUM":
            allow_threshold += 0.03

        elif intervention_level == "HIGH":
            allow_threshold += 0.07

        # Keep threshold bounded.
        allow_threshold = self._clamp(
            allow_threshold,
            0.35,
            0.65
        )

        # SUPPRESS threshold.
        suppress_threshold = (
            0.18
            + (0.12 * strictness)
        )

        if intervention_level == "HIGH":
            suppress_threshold += 0.04

        suppress_threshold = self._clamp(
            suppress_threshold,
            0.18,
            0.40
        )

        # ========================================================
        # Helper for consistent return format
        # ========================================================

        def result(
            decision: str,
            label: str,
            reason: str,
            uncertain: bool = False,
            extra: Dict[str, Any] = None
        ) -> Dict[str, Any]:

            output = {
                "decision": decision,
                "label": label,
                "reason": reason,

                "alignment": round(
                    alignment,
                    4
                ),

                "confidence": round(
                    confidence,
                    4
                ),

                "goal_relevance": round(
                    goal_relevance,
                    4
                ),

                "positive_relevance": round(
                    positive_relevance,
                    4
                ),

                "negative_relevance": round(
                    negative_relevance,
                    4
                ),

                "content_type_compatibility": round(
                    type_compatibility,
                    4
                ),

                "intervention_level": intervention_level,

                "intervention_score": round(
                    intervention_score,
                    4
                ),

                "is_uncertain": uncertain,

                "thresholds": {
                    "allow": round(
                        allow_threshold,
                        4
                    ),
                    "suppress": round(
                        suppress_threshold,
                        4
                    )
                }
            }

            if extra:
                output.update(extra)

            return output

        # ========================================================
        # 1. HARD NEGATIVE CONTENT
        # ========================================================
        #
        # Hard negative should normally be suppressed.
        #
        # However, current goal relevance can override a hard
        # negative category when the content is demonstrably
        # relevant to the user's explicit goal.
        #
        # Example:
        # User blocks "News"
        # Current goal = "Prepare for Kubernetes security interview"
        # Video = "Latest Kubernetes Security Vulnerabilities"
        #
        # This should NOT automatically disappear.
        # ========================================================

        if is_hard_negative:

            strong_goal_context = (
                goal_relevance >= 0.70
                and alignment >= 0.60
                and confidence >= 0.55
                and not is_incidental
            )

            if strong_goal_context:

                return result(
                    "ALLOW",
                    "Goal Override",
                    (
                        f"Allowed because the video matches the "
                        f"current goal strongly ({self._format_percent(goal_relevance)}% "
                        f"goal relevance), despite matching the blocked "
                        f"category '{hard_negative_topic or 'Hard Negative'}'."
                    ),
                    False,
                    {
                        "hard_negative_override": True,
                        "hard_negative_topic": hard_negative_topic
                    }
                )

            # Otherwise suppress hard negative content.
            return result(
                "SUPPRESS",
                "Blocked Topic",
                (
                    "Suppressed because it strongly matches a "
                    f"blocked topic "
                    f"({hard_negative_topic or 'Hard Negative'}) "
                    "and does not have sufficient current-goal "
                    "evidence to override the block."
                ),
                False,
                {
                    "hard_negative_override": False,
                    "hard_negative_topic": hard_negative_topic
                }
            )

        # ========================================================
        # 2. INCIDENTAL / HOMONYM / CONTRADICTION
        # ========================================================
        #
        # Example:
        # "Oggy cartoon" should not be classified as a video about
        # the programming concept "Oggy" merely because of a word
        # overlap.
        # ========================================================

        if is_incidental:

            debug_analysis = (
                content_analysis.get(
                    "debug_analysis",
                    {}
                )
            )

            incidental_details = (
                debug_analysis.get(
                    "incidental_details",
                    {}
                )
                if isinstance(debug_analysis, dict)
                else {}
            )

            flag_reasons = []

            if isinstance(
                incidental_details,
                dict
            ):
                for details in incidental_details.values():

                    if (
                        isinstance(details, dict)
                        and details.get("reason")
                    ):
                        flag_reasons.append(
                            str(details["reason"])
                        )

            reason = (
                flag_reasons[0]
                if flag_reasons
                else (
                    f"Primary topic appears to be "
                    f"{content_type}; apparent intent-topic "
                    "matches are incidental."
                )
            )

            # Do not suppress incidental content blindly.
            # At lower/medium strictness, de-emphasize it.
            # At very high strictness, suppress it.
            if strictness >= 0.82:

                return result(
                    "SUPPRESS",
                    "Incidental / Contradictory",
                    (
                        "Suppressed because the apparent topic "
                        "match is incidental and strict filtering "
                        "is enabled. "
                        + reason
                    ),
                    False,
                    {
                        "is_incidental": True
                    }
                )

            return result(
                "DEPRIORITIZE",
                "Incidental / Deprioritized",
                (
                    "Deprioritized because the apparent intent "
                    "match appears incidental. "
                    + reason
                ),
                False,
                {
                    "is_incidental": True
                }
            )

        # ========================================================
        # 3. SOFT NEGATIVE WITH STRONG GOAL CONTEXT
        # ========================================================
        #
        # Soft negatives are not absolute blocks.
        #
        # Example:
        # unwanted = News
        # goal = Kubernetes Security
        # video = Kubernetes Security News
        #
        # If semantic alignment is high, ALLOW.
        # ========================================================

        if soft_negative_mitigated:

            if (
                goal_relevance >= 0.65
                and alignment >= 0.60
                and confidence >= 0.50
            ):

                return result(
                    "ALLOW",
                    "Goal-Relevant Soft Negative",
                    (
                        "Allowed because the video contains a "
                        "soft-unwanted category but is strongly "
                        "relevant to the current goal."
                    ),
                    False,
                    {
                        "soft_negative_mitigated": True
                    }
                )

            if (
                goal_relevance >= 0.45
                and alignment >= 0.45
                and confidence >= 0.45
            ):

                return result(
                    "DEPRIORITIZE",
                    "Contextual Goal Match",
                    (
                        "Deprioritized because the video has "
                        "meaningful goal relevance but also "
                        "contains a soft-unwanted topic."
                    ),
                    False,
                    {
                        "soft_negative_mitigated": True
                    }
                )

            # Weak soft-negative content should continue through
            # the normal suppression logic below rather than being
            # automatically allowed.

        # ========================================================
        # 4. UNCERTAIN / AMBIGUOUS CONTENT
        # ========================================================
        #
        # Important principle:
        # Low confidence alone should not cause aggressive
        # suppression.
        #
        # Strong negative evidence + low alignment can suppress.
        # Otherwise prefer DEPRIORITIZE.
        # ========================================================

        if is_uncertain:

            # Strong goal match should protect uncertain content.
            if (
                goal_relevance >= 0.70
                and alignment >= 0.60
            ):

                return result(
                    "ALLOW",
                    "Uncertain but Goal-Relevant",
                    (
                        "Allowed because semantic evidence is "
                        "uncertain, but the available evidence "
                        "still shows strong relevance to the "
                        "current goal."
                    ),
                    True
                )

            # Very relaxed strictness -> allow.
            if strictness <= 0.35:

                return result(
                    "ALLOW",
                    "Uncertain / Permitted",
                    (
                        "Allowed because the content is ambiguous "
                        "and the filtering strictness is relaxed."
                    ),
                    True
                )

            # Strong unwanted evidence + poor alignment.
            if (
                negative_relevance >= 0.65
                and alignment < 0.35
            ):

                return result(
                    "SUPPRESS",
                    "Uncertain / Strong Negative Evidence",
                    (
                        "Suppressed because the content is "
                        "uncertain, has low intent alignment, and "
                        "has strong evidence for unwanted topics."
                    ),
                    True
                )

            # Default conservative action.
            return result(
                "DEPRIORITIZE",
                "Low Confidence",
                (
                    "Deprioritized because the semantic evidence "
                    "is ambiguous or incomplete. The system avoids "
                    "aggressive suppression when confidence is low."
                ),
                True
            )

        # ========================================================
        # 5. STRONG CURRENT-GOAL MATCH
        # ========================================================

        strong_goal_match = (
            goal_relevance >= 0.70
            and alignment >= 0.60
            and confidence >= 0.55
            and type_compatibility >= 0.60
        )

        if strong_goal_match:

            # Negative relevance should not be ignored completely.
            if negative_relevance < 0.45:

                return result(
                    "ALLOW",
                    "High Intent Match",
                    (
                        "Allowed because the video has strong "
                        "semantic alignment with your current goal."
                    ),
                    False
                )

        # ========================================================
        # 6. HIGH OVERALL ALIGNMENT
        # ========================================================

        high_alignment = (
            alignment >= 0.70
            and confidence >= 0.60
            and type_compatibility >= 0.70
            and negative_relevance < 0.35
        )

        if high_alignment:

            if goal_relevance >= 0.45:

                reason = (
                    "Strong semantic match with your current goal."
                )

            elif positive_relevance >= 0.50:

                reason = (
                    "Strong match with your preferred topics "
                    "and intent profile."
                )

            else:

                reason = (
                    "Strong semantic alignment with the current "
                    "intent profile."
                )

            return result(
                "ALLOW",
                "High Intent Match",
                reason,
                False
            )

        # ========================================================
        # 7. STRONG NEGATIVE CONTENT WITH LOW GOAL RELEVANCE
        # ========================================================

        if (
            negative_relevance >= 0.65
            and goal_relevance < 0.35
            and alignment < 0.45
        ):

            matched_text = (
                ", ".join(matched_negative)
                if matched_negative
                else "unwanted topics"
            )

            return result(
                "SUPPRESS",
                "Unwanted Content",
                (
                    f"Suppressed because negative relevance is "
                    f"high ({self._format_percent(negative_relevance)}%) "
                    f"while current-goal relevance is low. "
                    f"Matched: {matched_text}."
                ),
                False
            )

        # ========================================================
        # 8. LOW ALIGNMENT
        # ========================================================

        if alignment < suppress_threshold:

            matched_text = (
                ", ".join(matched_negative)
                if matched_negative
                else "unwanted topics"
            )

            if matched_negative:

                reason = (
                    f"Low intent alignment "
                    f"({self._format_percent(alignment)}%) "
                    f"and meaningful relevance to "
                    f"{matched_text}."
                )

            else:

                reason = (
                    f"Low intent alignment "
                    f"({self._format_percent(alignment)}%) "
                    "with the current goal and preferences."
                )

            return result(
                "SUPPRESS",
                "Low Intent Alignment",
                reason,
                False
            )

        # ========================================================
        # 9. MODERATE ALIGNMENT
        # ========================================================

        if (
            alignment >= allow_threshold
            and type_compatibility >= 0.60
            and negative_relevance < 0.40
        ):

            if confidence >= 0.45:

                if goal_relevance >= 0.30:

                    reason = (
                        "Moderate semantic match with your "
                        "current goal."
                    )

                elif positive_relevance >= 0.35:

                    reason = (
                        "Relevant to your preferred topics "
                        "and intent profile."
                    )

                else:

                    reason = (
                        "Moderately aligned with your current "
                        "intent profile."
                    )

                return result(
                    "ALLOW",
                    "Moderate Intent Match",
                    reason,
                    False
                )

            return result(
                "DEPRIORITIZE",
                "Moderate Intent / Low Confidence",
                (
                    f"Moderate alignment "
                    f"({self._format_percent(alignment)}%) "
                    f"but classification confidence is only "
                    f"{self._format_percent(confidence)}%."
                ),
                False
            )

        # ========================================================
        # 10. BORDERLINE CONTENT
        # ========================================================

        if (
            alignment >= 0.30
            and negative_relevance < 0.60
        ):

            return result(
                "DEPRIORITIZE",
                "Borderline Intent Match",
                (
                    f"Deprioritized because alignment is "
                    f"borderline "
                    f"({self._format_percent(alignment)}%) "
                    "with the current intent."
                ),
                False
            )

        # ========================================================
        # 11. FINAL FALLBACK
        # ========================================================

        return result(
            "SUPPRESS",
            "Low Intent Relevance",
            (
                f"Suppressed because the video has insufficient "
                f"relevance to the current intent "
                f"({self._format_percent(alignment)}% alignment)."
            ),
            False
        )