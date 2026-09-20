"""Deterministic follow-up policy shared by graph routing and engine callers.

These are the existing engine thresholds, not calibrated model-quality claims.
"""

from collections.abc import Mapping
from typing import Any

POLICY_VERSION = "follow-up-v1"
SUFFICIENT_SCORE_THRESHOLD = 0.58
HIGH_SCORE_THRESHOLD = 0.72
SIGNIFICANT_GAPS = frozenset(
    {
        "no_specific_example",
        "no_metrics",
        "unclear_role",
        "no_outcome",
        "resume_inconsistent",
    }
)


def should_follow_up(
    evaluation: Mapping[str, Any], follow_up_count: int, max_follow_ups: int
) -> bool:
    """Apply the cap first, then sufficiency, score and gap rules without side effects."""
    if follow_up_count >= max_follow_ups:
        return False
    score = evaluation["score"]
    if follow_up_count == 0 and score < HIGH_SCORE_THRESHOLD:
        return True
    if not evaluation["is_sufficient"] or score < SUFFICIENT_SCORE_THRESHOLD:
        return True
    gaps = evaluation.get("gap_analysis", {}).get("detected_gaps", [])
    return follow_up_count < 2 and bool(SIGNIFICANT_GAPS.intersection(gaps))
