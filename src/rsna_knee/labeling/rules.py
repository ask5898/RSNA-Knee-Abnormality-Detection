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

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

__all__ = [
    "LABELS", "Certainty", "Mention", "LabelResult",
    "Vocabulary", "TextNormalizer", "NegationDetector", "ClinicalNoteLabeler",
]

# The 12 findings. Order is the submission column order; nothing else in this
# module hardcodes a label list.
LABELS: list[str] = [
    "ACL", "MCL", "medial_meniscus", "lateral_meniscus",
    "medial_OA", "lateral_OA", "patellofemoral_OA",
    "effusion", "synovitis", "bakers_cyst", "bone_contusion", "fracture",
]


# =============================================================================
# 1. Certainty
# =============================================================================
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


# =============================================================================
# 2/3. Value objects
# =============================================================================
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


# =============================================================================
# 4. Vocabulary — all language data, no logic
# =============================================================================
@dataclass
class Vocabulary:
    """Language-specific data for matching.

    TERM FORMAT: space-separated STEMS, not dictionary forms.
    "медиальн мениск" not "медиальный мениск". The compiler below turns each
    stem into `stem\\w*`, so every inflected form matches. This is essential for
    Russian and Greek, where an adjective-noun pair inflects on BOTH words and
    a literal substring search finds nothing.

    ACCURACY WARNING: the non-English entries below are a starting point, not
    validated terminology. Check them against your actual corpus before
    trusting them. A wrong stem produces no match, and no match looks exactly
    like a negative finding.
    """

    findings: dict[str, list[str]] = field(default_factory=dict)
    pre_negation: list[str] = field(default_factory=list)
    post_negation: list[str] = field(default_factory=list)
    hedges: dict[str, list[str]] = field(default_factory=dict)
    abbreviations: dict[str, str] = field(default_factory=dict)
    supported_scripts: set[str] = field(default_factory=lambda: {"latin", "greek", "cyrillic"})

    _compiled: dict[str, list[re.Pattern]] = field(default_factory=dict, repr=False)

    # -- compilation ------------------------------------------------------
    @staticmethod
    def compile_term(term: str) -> re.Pattern:
        r"""Turn "медиальн мениск" into `медиальн\w*[\s\-]*мениск\w*`.

        Each stem gets a trailing \w* so any case/number ending matches, and
        the separator tolerates a space or hyphen. Latin terms are unaffected:
        "medial meniscus" still matches itself, and now also "mediale
        meniscus" for free.
        """
        stems = [re.escape(s) for s in term.split()]
        return re.compile(r"\w*[\s\-]*".join(stems) + r"\w*", re.I | re.U)

    def compiled(self, label: str) -> list[re.Pattern]:
        """Lazily compile and cache patterns for one label."""
        if label not in self._compiled:
            self._compiled[label] = [self.compile_term(t) for t in self.findings.get(label, [])]
        return self._compiled[label]

    def add_language(self, findings: dict[str, list[str]],
                     pre: Sequence[str] = (), post: Sequence[str] = (),
                     hedges: dict[str, list[str]] | None = None) -> "Vocabulary":
        """Merge another language in. Returns self so calls chain.

        Scripts cannot collide -- a Cyrillic stem will never match Latin text --
        so everything lives in one merged pool and there is no routing by
        language at match time. That also means a mixed-language report works
        without any special handling.
        """
        for lab, terms in findings.items():
            self.findings.setdefault(lab, []).extend(terms)
        self.pre_negation.extend(pre)
        self.post_negation.extend(post)
        for bucket, cues in (hedges or {}).items():
            self.hedges.setdefault(bucket, []).extend(cues)
        self._compiled.clear()
        return self


def build_default_vocabulary() -> Vocabulary:
    """English/German/French/Spanish/Dutch + Greek + Russian."""
    v = Vocabulary(
        findings={
            "ACL": ["anterior cruciate"],
            "MCL": ["medial collateral", "tibial collateral"],
            "medial_meniscus": ["medial meniscus", "medial meniscal"],
            "lateral_meniscus": ["lateral meniscus", "lateral meniscal"],
            "medial_OA": ["medial osteoarthritis", "medial compartment osteoarthritis",
                          "medial chondral loss"],
            "lateral_OA": ["lateral osteoarthritis", "lateral compartment osteoarthritis",
                           "lateral chondral loss"],
            "patellofemoral_OA": ["patellofemoral", "retropatellar", "chondromalacia patell"],
            "effusion": ["effusion", "joint fluid"],
            "synovitis": ["synovitis", "synovial thickening", "synovial proliferation"],
            "bakers_cyst": ["baker cyst", "bakers cyst", "baker's cyst", "baker\u2019s cyst", "popliteal cyst"],
            "bone_contusion": ["bone marrow edema", "bone marrow oedema", "marrow edema", "marrow oedema", "bone contusion",
                               "bone bruise"],
            "fracture": ["fracture", "avulsion"],
        },
        pre_negation=[r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative for\b",
                      r"\babsence of\b", r"\bfree of\b", r"\bruled out\b", r"\bexcluded\b"],
        post_negation=[r"\bintact\b", r"\bnone\b", r"\babsent\b", r"\bnegative\b", r"\bnormal\b", r"\bunremarkable\b", r"\bpreserved\b",
                       r"\bwithin normal limits\b", r"\bwnl\b"],
        hedges={
            "probable": [r"likely", r"probable", r"consistent with", r"suggestive of"],
            "possible": [r"possible", r"cannot be (excluded|ruled out)", r"suspicion",
                         r"query", r"may represent", r"\bversus\b", r"\bvs\b"],
            "unlikely": [r"unlikely", r"doubtful"],
        },
        abbreviations={
            r"\bACL\b": "anterior cruciate ligament", r"\bVKB\b": "anterior cruciate ligament",
            r"\bLCA\b": "anterior cruciate ligament", r"\bMCL\b": "medial collateral ligament",
            r"\bMM\b": "medial meniscus", r"\bLM\b": "lateral meniscus",
            r"\bPF\b": "patellofemoral", r"\bOA\b": "osteoarthritis",
            r"\beff\b": "effusion", r"\bBME\b": "bone marrow edema",
            r"\bfx\b": "fracture", r"\bsyn\b": "synovitis",
        },
    )

    v.add_language(
        {   # German
            "ACL": ["vorder kreuzband", "kreuzband vorder"],
            "MCL": ["innenband", "mediale kollateralband"],
            "medial_meniscus": ["innenmeniskus"],
            "lateral_meniscus": ["aussenmeniskus", "außenmeniskus"],
            "medial_OA": ["mediale gonarthrose", "mediale arthrose"],
            "lateral_OA": ["laterale gonarthrose", "laterale arthrose"],
            "effusion": ["erguss", "gelenkerguss"],
            "bakers_cyst": ["bakerzyste"],
            "bone_contusion": ["knochenmarkodem", "knochenmarködem"],
            "fracture": ["fraktur"],
        },
        pre=[r"\bkein\b", r"\bkeine\b", r"\bohne\b"],
        post=[r"\bunauffallig\b", r"\bintakt\b", r"\bregelrecht\b"],
        hedges={"possible": [r"verdacht", r"moglich"], "probable": [r"wahrscheinlich"]},
    ).add_language(
        {   # French
            "ACL": ["ligament croise anterieur"],
            "medial_meniscus": ["menisque medial", "menisque interne"],
            "lateral_meniscus": ["menisque lateral", "menisque externe"],
            "effusion": ["epanchement"],
            "bakers_cyst": ["kyste de baker"],
            "bone_contusion": ["oedeme osseux"],
        },
        pre=[r"\bpas de\b", r"\bsans\b", r"\baucun\b"],
        post=[r"\bsans particularite\b", r"\bnormale?\b"],
        hedges={"possible": [r"possible", r"ne peut etre exclu", r"suspicion"],
                "probable": [r"compatible avec", r"en faveur de"]},
    ).add_language(
        {   # Spanish
            "ACL": ["ligamento cruzado anterior"],
            "medial_meniscus": ["menisco medial", "menisco interno"],
            "lateral_meniscus": ["menisco lateral", "menisco externo"],
            "effusion": ["derrame"],
            "bakers_cyst": ["quiste de baker"],
            "bone_contusion": ["edema oseo"],
            "fracture": ["fisura"],
        },
        pre=[r"\bno hay\b", r"\bsin\b", r"\bausencia\b"],
        post=[r"\bintacto\b", r"\bnormales?\b"],
        hedges={"possible": [r"posible", r"no se puede excluir", r"sospecha"],
                "probable": [r"compatible con", r"sugestivo de"]},
    ).add_language(
        {   # Dutch
            "ACL": ["voorste kruisband"],
            "medial_meniscus": ["mediale meniscus", "binnenmeniscus"],
            "lateral_meniscus": ["laterale meniscus", "buitenmeniscus"],
        },
        pre=[r"\bgeen\b", r"\bzonder\b"],
        post=[r"\bintact\b", r"\bnormaal\b"],
    ).add_language(
        {   # Russian (Cyrillic) -- stems
            "ACL": ["передн крестообразн", "пкс"],
            "MCL": ["внутренн боков", "медиальн боков"],
            "medial_meniscus": ["медиальн мениск", "внутренн мениск"],
            "lateral_meniscus": ["латеральн мениск", "наружн мениск"],
            "medial_OA": ["медиальн артроз", "внутренн артроз"],
            "lateral_OA": ["латеральн артроз", "наружн артроз"],
            "patellofemoral_OA": ["пателлофеморальн", "ретропателляр"],
            "effusion": ["выпот", "жидкост в полост"],
            "synovitis": ["синовит"],
            "bakers_cyst": ["киста бейкер", "подколенн киста"],
            "bone_contusion": ["отек костн мозг", "трабекулярн отек"],
            "fracture": ["перелом", "трещин"],
        },
        pre=[r"\bне\b", r"\bнет\b", r"\bбез\b"],
        # These are POST-posed in Russian: "перелом не выявлен" puts the
        # negation after the finding, unlike English "no fracture".
        post=[r"интактн", r"сохранн", r"не изменен", r"в норме", r"нормальн",
              r"не выявлен", r"не определя", r"не отмеча", r"не обнаружен", r"отсутств"],
        hedges={"possible": [r"возможн", r"подозрени", r"не исключ"],
                "probable": [r"вероятн", r"соответству"]},
    ).add_language(
        {   # Greek -- stems, written unaccented (normaliser strips tonos)
            "ACL": ["προσθι χιαστ", "χιαστου συνδεσμ"],
            "MCL": ["εσω πλαγι συνδεσμ"],
            "medial_meniscus": ["εσω μηνισκ", "εσωτερικ μηνισκ"],
            "lateral_meniscus": ["εξω μηνισκ", "εξωτερικ μηνισκ"],
            "medial_OA": ["εσω οστεοαρθρι"],
            "lateral_OA": ["εξω οστεοαρθρι"],
            "patellofemoral_OA": ["επιγονατιδομηριαι", "οπισθοεπιγονατιδ"],
            "effusion": ["αρθρικ συλλογ", "ενδαρθρικ υγρ"],
            "synovitis": ["υμενιτιδ"],
            "bakers_cyst": ["κυστ baker", "ιγνυακ κυστ"],
            "bone_contusion": ["οιδημα μυελ", "οστικ οιδημα"],
            "fracture": ["καταγμα", "ρωγμ"],
        },
        pre=[r"\bδεν\b", r"\bχωρις\b", r"απουσι", r"ουδεμι"],
        post=[r"ακεραι", r"φυσιολογικ", r"ανευ ευρηματ"],
        hedges={"possible": [r"πιθανον", r"δεν αποκλειετ"], "probable": [r"συμβατ"]},
    )
    return v


# =============================================================================
# 5. TextNormalizer
# =============================================================================
class TextNormalizer:
    """Script detection, accent folding, abbreviation expansion, unit splitting."""

    SCRIPT_RANGES = {
        "greek": (0x0370, 0x03FF), "cyrillic": (0x0400, 0x04FF),
        "arabic": (0x0600, 0x06FF), "hebrew": (0x0590, 0x05FF),
        "devanagari": (0x0900, 0x097F), "han": (0x4E00, 0x9FFF),
        "kana": (0x3040, 0x30FF), "hangul": (0xAC00, 0xD7AF),
    }

    # A unit boundary. Negation must not cross one.
    UNIT_SPLIT = re.compile(r"[\n\r]+|(?<=[.;])\s+")

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


# =============================================================================
# 6. NegationDetector
# =============================================================================
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
        right = re.sub(r"^(\s+(?:ligament|ligaments|compartment))?\s*:", r"\1", right)   # "ACL: intact": the cue after the colon belongs to the ACL
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


# =============================================================================
# 7. ClinicalNoteLabeler
# =============================================================================
class ClinicalNoteLabeler:
    """Orchestrates normalisation, matching, negation and scoring.

        labeler = ClinicalNoteLabeler()
        results = labeler.extract(report)          # dict[label] -> LabelResult
        soft    = labeler.to_soft_labels(report)   # dict[label] -> float
        df_rows = list(labeler.batch(pairs))       # for a training CSV
    """

    def __init__(self, vocabulary: Vocabulary | None = None,
                 certainty_values: dict[str, float] | None = None,
                 omission_priors: dict[str, float] | None = None,
                 labels: Sequence[str] = LABELS):
        """
        certainty_values  categorical -> probability. Tune on a gold set.
        omission_priors   per-label p(present | never mentioned). See below.
        """
        self.vocab = vocabulary or build_default_vocabulary()
        self.values = dict(Certainty.DEFAULT_VALUES)
        if certainty_values:
            self.values.update(certainty_values)
        self.omission_priors = omission_priors or {}
        self.labels = list(labels)
        self.normalizer = TextNormalizer(self.vocab.abbreviations)
        self.negation = NegationDetector(self.vocab)

    # -- matching ---------------------------------------------------------
    def find_mentions(self, unit: str) -> list[Mention]:
        """All findings mentioned in one text unit.

        First matching term per label wins and we move on -- the terms within a
        label are synonyms, so a second hit adds nothing.
        """
        out = []
        for label in self.labels:
            for pat in self.vocab.compiled(label):
                m = pat.search(unit)
                if m:
                    out.append(Mention(label, m.group(0), m.span(), unit))
                    break
        return out

    # -- main entry point -------------------------------------------------
    def extract(self, text: str) -> dict[str, LabelResult]:
        script = self.normalizer.detect_script(text)
        normalized = self.normalizer.normalize(text)

        mentions: list[Mention] = []
        for unit in self.normalizer.split_units(normalized):
            for m in self.find_mentions(unit):
                src = self.negation.detect(m)
                m.negation_source = src
                m.certainty = Certainty.NEGATED if src else self.negation.certainty(m)
                mentions.append(m)

        # One report can mention a finding several times ("possible medial
        # meniscus tear" in findings, "medial meniscus tear" in impression).
        # The strongest assertion wins: a report that both hedges and asserts
        # has resolved its own uncertainty by the time it asserts.
        best: dict[str, Mention] = {}
        for m in mentions:
            cur = best.get(m.label)
            if cur is None or Certainty.RANK[m.certainty] > Certainty.RANK[cur.certainty]:
                best[m.label] = m

        # COVERAGE GUARD.
        # Zero mentions across a whole report means one of two things: a
        # genuinely unremarkable study, or a language this vocabulary does not
        # cover. Those produce identical output -- all negative -- and must not
        # be conflated, because the second silently poisons the training set
        # with an entire language's worth of false negatives. Flagging it turns
        # a silent failure into a visible one.
        unsupported = not mentions and script not in self.vocab.supported_scripts

        results = {}
        for label in self.labels:
            m = best.get(label)
            if m is None:
                cert = Certainty.UNSUPPORTED if unsupported else Certainty.NOT_MENTIONED
                val = self.omission_priors.get(label, self.values[Certainty.NOT_MENTIONED])
                results[label] = LabelResult(label, cert, val, "", unsupported)
            else:
                results[label] = LabelResult(
                    label, m.certainty, self.values[m.certainty], m.evidence(), False
                )
        return results

    def to_soft_labels(self, text: str) -> dict[str, float]:
        return {k: v.value for k, v in self.extract(text).items()}

    # -- batch helpers ----------------------------------------------------
    def batch(self, reports: Iterable[tuple[str, str]]) -> Iterator[dict]:
        """Yield one flat row per report. Feed straight to pd.DataFrame.

        A generator so a corpus of any size streams rather than materialising.
        """
        for study_id, text in reports:
            res = self.extract(text)
            yield {"study_id": study_id,
                   **{lab: r.value for lab, r in res.items()}}

    def review_queue(self, reports: Iterable[tuple[str, str]]) -> list[dict]:
        """Reports the extractor could not handle, for a human to look at.

        With a hand-built multilingual vocabulary this is the most useful
        diagnostic in the module: it tells you which languages or templates you
        are silently failing on, ranked so the worst come first.
        """
        rows = []
        for study_id, text in reports:
            res = self.extract(text)
            flagged = [r.label for r in res.values() if r.needs_review]
            if flagged:
                rows.append({
                    "study_id": study_id,
                    "script": self.normalizer.detect_script(text),
                    "n_flagged": len(flagged),
                    "preview": (text or "")[:80],
                })
        return sorted(rows, key=lambda r: -r["n_flagged"])

    def coverage_report(self, reports: Iterable[tuple[str, str]]) -> dict:
        """Per-script hit rate. Run this before trusting any output.

        A script with a near-zero mention rate is a vocabulary gap, not a
        population of healthy knees.
        """
        stats: dict[str, dict] = {}
        for _, text in reports:
            script = self.normalizer.detect_script(text)
            s = stats.setdefault(script, {"n": 0, "with_mentions": 0})
            s["n"] += 1
            res = self.extract(text)
            if any(r.certainty not in (Certainty.NOT_MENTIONED, Certainty.UNSUPPORTED)
                   for r in res.values()):
                s["with_mentions"] += 1
        for s in stats.values():
            s["hit_rate"] = round(s["with_mentions"] / max(s["n"], 1), 3)
        return stats
