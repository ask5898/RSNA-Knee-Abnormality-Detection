"""All language-specific data, and no logic.

The boundary this file defends: adding a language means adding data here,
never editing a detector. If you find yourself opening negation.py to support
Polish, something has gone wrong with the split.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

__all__ = ["Vocabulary", "build_default_vocabulary"]


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
            "MCL": ["medial collateral"],
            "medial_meniscus": ["medial meniscus"],
            "lateral_meniscus": ["lateral meniscus"],
            "medial_OA": ["medial osteoarthritis", "medial compartment osteoarthritis",
                          "medial chondral loss"],
            "lateral_OA": ["lateral osteoarthritis", "lateral compartment osteoarthritis",
                           "lateral chondral loss"],
            "patellofemoral_OA": ["patellofemoral", "retropatellar", "chondromalacia patell"],
            "effusion": ["effusion", "joint fluid"],
            "synovitis": ["synovitis", "synovial thickening"],
            "bakers_cyst": ["baker cyst", "bakers cyst", "popliteal cyst"],
            "bone_contusion": ["bone marrow edema", "bone marrow oedema", "bone contusion",
                               "bone bruise"],
            "fracture": ["fracture", "avulsion"],
        },
        pre_negation=[r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative for\b",
                      r"\babsence of\b", r"\bfree of\b", r"\bruled out\b", r"\bexcluded\b"],
        post_negation=[r"\bintact\b", r"\bnormal\b", r"\bunremarkable\b", r"\bpreserved\b",
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
