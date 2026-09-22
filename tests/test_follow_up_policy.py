"""Deterministic policy cases, not a model-quality benchmark."""

from unittest.mock import Mock

import pytest

from backend.app.core.grilling_engine import AnswerEvaluation, GrillingEngine
from backend.app.graph.edges import route_after_evaluate


@pytest.mark.parametrize(
    "score,sufficient,gaps,count,cap,expected",
    [
        (0.9, True, [], 0, 3, False),
        (0.71, True, [], 0, 3, True),
        (0.72, True, [], 0, 3, False),
        (0.9, False, [], 1, 3, True),
        (0.5, True, [], 1, 3, True),
        (0.9, True, ["no_metrics"], 1, 3, True),
        (0.9, True, ["no_metrics"], 2, 3, False),
        (0.9, True, ["too_generic"], 1, 3, False),
        (0.1, False, ["too_generic"], 0, 0, False),
        (0.1, False, ["too_generic"], 3, 3, False),
    ],
)
def test_engine_and_graph_use_one_policy(score, sufficient, gaps, count, cap, expected):
    evaluation = {
        "outcome": "valid",
        "score": score,
        "is_sufficient": sufficient,
        "gap_analysis": {"detected_gaps": gaps},
    }
    engine = GrillingEngine(Mock())
    assert (
        engine.should_grill(AnswerEvaluation.from_dict(evaluation), count, cap)
        is expected
    )
    result = route_after_evaluate(
        {
            "current_evaluation": evaluation,
            "current_follow_up_count": count,
            "max_follow_ups": cap,
            "questions": ["Synthetic question"],
            "current_question_index": 0,
        }
    )
    assert result == ("grill" if expected else "done")
