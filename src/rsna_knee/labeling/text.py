"""Script detection, accent folding, abbreviation expansion, unit splitting."""
from __future__ import annotations

import re
import unicodedata

__all__ = ["TextNormalizer"]


class TextNormalizer:
    """Script detection, accent folding, abbreviation expansion, unit splitting."""

    SCRIPT_RANGES = {
        "greek": (0x0370, 0x03FF), "cyrillic": (0x0400, 0x04FF),
        "arabic": (0x0600, 0x06FF), "hebrew": (0x0590, 0x05FF),
        "devanagari": (0x0900, 0x097F), "han": (0x4E00, 0x9FFF),
        "kana": (0x3040, 0x30FF), "hangul": (0xAC00, 0xD7AF),
    }

    # A unit boundary. Negation must not cross one.
    UNIT_SPLIT = re.compile(r"[\n\r]+|(?<=[.;:])\s+")

    def __init__(self, abbreviations: dict[str, str] | None = None):
        self.abbreviations = abbreviations or {}

    def detect_script(self, text: str) -> str:
        """Dominant script by letter census.

        Dominance, not presence: reports routinely mix Cyrillic prose with
        Latin units and drug names, so an any() test would misclassify them.
        """
        counts = {k: 0 for k in self.SCRIPT_RANGES}
        counts["latin"] = 0
        for ch in text or "":
            if not ch.isalpha():
                continue
            o = ord(ch)
            if o < 0x0250:
                counts["latin"] += 1
                continue
            for name, (lo, hi) in self.SCRIPT_RANGES.items():
                if lo <= o <= hi:
                    counts[name] += 1
                    break
        return max(counts, key=counts.get) if any(counts.values()) else "unknown"

    def normalize(self, text: str) -> str:
        """Fold accents, normalise Greek sigma, expand abbreviations, tidy space.

        NFKD decomposition plus combining-mark stripping folds Latin diacritics
        (é -> e) AND Greek tonos (ά -> α), which is exactly what we want since
        the Greek vocabulary is written unaccented. For Cyrillic it folds
        ё -> е and й -> и, harmless because the vocabulary goes through the
        same function.
        """
        if not text:
            return ""
        t = unicodedata.normalize("NFKD", text)
        t = "".join(c for c in t if not unicodedata.combining(c))
        t = t.replace("\u03c2", "\u03c3")        # Greek final sigma -> sigma
        for pat, rep in self.abbreviations.items():
            t = re.sub(pat, rep, t, flags=re.I)
        return re.sub(r"[ \t]+", " ", t)

    def split_units(self, text: str) -> list[str]:
        """Sentences or lines. The unit is the negation scope."""
        return [u.strip() for u in self.UNIT_SPLIT.split(text) if u.strip()]
