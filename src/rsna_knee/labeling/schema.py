"""The label set and the value objects the labeler passes around.

Split from the matching logic so that a change to the scale -- adding a
certainty level, retuning what "probable" is worth -- touches one small file
with no regexes in it.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LABELS", "Certainty", "Mention", "LabelResult"]


# The 12 findings. Order is the submission column order; nothing else in this
# module hardcodes a label list.
LABELS: list[str] = [
    "ACL", "MCL", "medial_meniscus", "lateral_meniscus",
    "medial_OA", "lateral_OA", "patellofemoral_OA",
    "effusion", "synovitis", "bakers_cyst", "bone_contusion", "fracture",
]


class Certainty:
    """The ordinal scale a report can express about a finding.

    Six levels rather than a boolean, because reports hedge constantly and
    collapsing "definite tear" and "tear cannot be excluded" to the same 1
    throws away information the model can use.

    Categorical rather than a raw float because the mapping to numbers is a
    tunable you want in one place, calibrated against a gold set, not scattered
    through the matching code.
    """

    DEFINITE = "definite"
    PROBABLE = "probable"
    POSSIBLE = "possible"
    UNLIKELY = "unlikely"
    NEGATED = "negated"
    NOT_MENTIONED = "not_mentioned"
    UNSUPPORTED = "script_unsupported"     # see ClinicalNoteLabeler.extract

    # Strength ordering. Used to pick a winner when one report mentions the
    # same finding more than once.
    RANK = {
        DEFINITE: 5, PROBABLE: 4, POSSIBLE: 3,
        UNLIKELY: 2, NEGATED: 1, NOT_MENTIONED: 0, UNSUPPORTED: 0,
    }

    # Default categorical -> probability map. Override per project.
    DEFAULT_VALUES = {
        DEFINITE: 0.95, PROBABLE: 0.80, POSSIBLE: 0.50,
        UNLIKELY: 0.20, NEGATED: 0.02, NOT_MENTIONED: 0.02, UNSUPPORTED: 0.02,
    }

    @classmethod
    def stronger(cls, a: str, b: str) -> str:
        return a if cls.RANK[a] >= cls.RANK[b] else b


@dataclass
class Mention:
    """One occurrence of one finding inside one text unit.

    Carries the unit it was found in so downstream code can show evidence
    without re-parsing, and so negation can be scoped without passing the whole
    document around.
    """

    label: str
    matched_text: str
    span: tuple[int, int]
    unit: str
    certainty: str = Certainty.DEFINITE
    negation_source: str = ""       # "", "pre", "post"

    def evidence(self, max_chars: int = 200) -> str:
        return self.unit.strip()[:max_chars]


@dataclass
class LabelResult:
    """Final verdict for one label on one report."""

    label: str
    certainty: str
    value: float
    evidence: str = ""
    needs_review: bool = False

    @property
    def is_positive(self) -> bool:
        return self.value > 0.5
