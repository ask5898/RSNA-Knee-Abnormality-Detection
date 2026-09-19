"""Orchestration: normalise, match, negate, score."""
from __future__ import annotations

from typing import Iterable, Iterator, Sequence

from .negation import NegationDetector
from .schema import LABELS, Certainty, LabelResult, Mention
from .text import TextNormalizer
from .vocabulary import Vocabulary, build_default_vocabulary

__all__ = ["ClinicalNoteLabeler"]


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
