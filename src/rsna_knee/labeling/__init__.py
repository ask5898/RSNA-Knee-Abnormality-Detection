# -*- coding: utf-8 -*-
"""Rule-based label extraction from multilingual radiology reports.

Turns a free-text knee MRI report, in any of ~12 languages, into 12 soft
labels in [0, 1] suitable for training a vision model.

    labeler = ClinicalNoteLabeler()
    labeler.to_soft_labels("No ACL tear. Medial meniscus posterior horn tear.")
    # {'ACL': 0.02, 'medial_meniscus': 0.95, ...}

Stdlib only. No model, no network, no GPU.

DESIGN
------
Six small pieces rather than one big class, so each can be tested and swapped
independently:

    Certainty          the six-level ordinal scale and its mapping to numbers
    Mention            one occurrence of one finding in one text unit
    LabelResult        the final per-label verdict, with provenance
    Vocabulary         all language-specific data, isolated from all logic
    TextNormalizer     script detection, accent folding, abbreviation expansion
    NegationDetector   scope windows and polarity
    ClinicalNoteLabeler  orchestrates the above

The split that matters most is Vocabulary vs the detectors. Adding a language
should mean adding data, never touching logic. If you find yourself editing
NegationDetector to support Polish, something is wrong with the boundary.
"""
from __future__ import annotations

from .labeler import ClinicalNoteLabeler
from .negation import NegationDetector
from .schema import LABELS, Certainty, LabelResult, Mention
from .text import TextNormalizer
from .vocabulary import Vocabulary, build_default_vocabulary

__all__ = [
    "LABELS", "Certainty", "Mention", "LabelResult",
    "Vocabulary", "build_default_vocabulary",
    "TextNormalizer", "NegationDetector", "ClinicalNoteLabeler",
]
