"""Asserted, denied, or hedged -- the decision the whole labeler turns on."""
from __future__ import annotations

import re

from .schema import Certainty, Mention
from .vocabulary import Vocabulary

__all__ = ["NegationDetector"]


class NegationDetector:
    """Decides whether a mention is asserted, denied, or hedged.

    THE CENTRAL PROBLEM. "No evidence of ACL tear" contains "ACL tear". Since
    most mentions of most findings in radiology reports are negative, a matcher
    that cannot tell assertion from denial is worse than predicting the base
    rate.

    Two directions, because languages put the cue on different sides:
        pre-posed   "no fracture"            (English, German, French, Spanish)
        post-posed  "the ACL is intact"      (English)
                    "перелом не выявлен"     (Russian -- fracture not detected)

    A single backwards search gets post-posed negation exactly backwards,
    labelling intact structures as torn.
    """

    # Clause boundaries within a unit. "No fracture, but ACL torn" must not
    # negate the ACL.
    CLAUSE_BREAK = re.compile(
        r"[,;:]|\bbut\b|\bhowever\b|\baber\b|\bjedoch\b|\bmais\b|\bpero\b|\bmaar\b|\bно\b|\bαλλα\b",
        re.I | re.U,
    )

    def __init__(self, vocab: Vocabulary, window: int = 80):
        self.vocab = vocab
        self.window = window

    def _scopes(self, unit: str, span: tuple[int, int]) -> tuple[str, str]:
        """Left and right context, truncated at the nearest clause break."""
        left = unit[max(0, span[0] - self.window): span[0]].lower()
        breaks = list(self.CLAUSE_BREAK.finditer(left))
        if breaks:
            left = left[breaks[-1].end():]

        right = unit[span[1]: span[1] + self.window].lower()
        brk = self.CLAUSE_BREAK.search(right)
        if brk:
            right = right[: brk.start()]
        return left, right

    def detect(self, mention: Mention) -> str:
        """Returns '' (not negated), 'pre', or 'post'."""
        left, right = self._scopes(mention.unit, mention.span)
        if any(re.search(c, left, re.U) for c in self.vocab.pre_negation):
            return "pre"
        if any(re.search(c, right, re.U) for c in self.vocab.post_negation):
            return "post"
        return ""

    def certainty(self, mention: Mention) -> str:
        """Hedge level for a non-negated mention."""
        left, right = self._scopes(mention.unit, mention.span)
        ctx = left + " " + right
        for bucket, cues in self.vocab.hedges.items():
            if any(re.search(c, ctx, re.U) for c in cues):
                return bucket
        return Certainty.DEFINITE
