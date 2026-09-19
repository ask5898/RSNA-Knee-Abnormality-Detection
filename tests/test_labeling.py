# -*- coding: utf-8 -*-
"""The labeler: matching, negation, hedging, and the coverage guard."""
from __future__ import annotations

import pytest

from rsna_knee import LABELS, Certainty, ClinicalNoteLabeler


@pytest.fixture(scope="module")
def labeler():
    return ClinicalNoteLabeler()


def test_all_labels_present(labeler):
    assert set(labeler.to_soft_labels("anything")) == set(LABELS)


@pytest.mark.parametrize("text,label,certainty", [
    # The point of the whole module: the finding is named in both cases, and
    # only one of them means the patient has it.
    ("Medial meniscus posterior horn tear.", "medial_meniscus", Certainty.DEFINITE),
    ("No ACL tear.", "ACL", Certainty.NEGATED),
    ("The ACL is intact.", "ACL", Certainty.NEGATED),          # post-posed, English
    ("Перелом не выявлен.", "fracture", Certainty.NEGATED),    # post-posed, Russian
    ("Possible lateral meniscus tear.", "lateral_meniscus", Certainty.POSSIBLE),
    ("Findings likely represent a fracture.", "fracture", Certainty.PROBABLE),
    ("Verdacht auf Bakerzyste.", "bakers_cyst", Certainty.POSSIBLE),
    ("Kein Erguss.", "effusion", Certainty.NEGATED),
    ("Moderate joint effusion.", "effusion", Certainty.DEFINITE),
])
def test_certainty(labeler, text, label, certainty):
    assert labeler.extract(text)[label].certainty == certainty


def test_clause_break_stops_negation(labeler):
    """"No fracture, but ACL torn" must not negate the ACL."""
    res = labeler.extract("No fracture, but ACL is torn.")
    assert res["fracture"].certainty == Certainty.NEGATED
    assert res["ACL"].certainty == Certainty.DEFINITE


def test_strongest_mention_wins(labeler):
    """A report that hedges and then asserts has resolved its own uncertainty."""
    res = labeler.extract("Possible medial meniscus tear.\n"
                          "Impression: medial meniscus tear.")
    assert res["medial_meniscus"].certainty == Certainty.DEFINITE


def test_russian_stems_inflect(labeler):
    """Stem matching, not substring: both words inflect in Russian."""
    res = labeler.extract("Медиальный мениск: разрыв заднего рога.")
    assert res["medial_meniscus"].certainty == Certainty.DEFINITE


def test_greek_accents_are_folded(labeler):
    """The vocabulary is written unaccented; the normaliser strips tonos."""
    assert labeler.extract("Ρήξη έσω μηνίσκου.")["medial_meniscus"].is_positive


def test_abbreviations_expand(labeler):
    assert labeler.extract("ACL tear noted.")["ACL"].certainty == Certainty.DEFINITE


def test_unsupported_script_is_flagged_not_silently_negative(labeler):
    """The failure mode that would poison a training set with false negatives."""
    res = labeler.extract("前十字靭帯断裂を認める。")
    assert all(r.certainty == Certainty.UNSUPPORTED for r in res.values())
    assert all(r.needs_review for r in res.values())


def test_quiet_supported_report_is_not_flagged(labeler):
    """An unremarkable English study must not look like a vocabulary gap."""
    res = labeler.extract("Normal study. No abnormality.")
    assert not any(r.needs_review for r in res.values())


def test_empty_text(labeler):
    res = labeler.extract("")
    assert set(res) == set(LABELS)
    assert not any(r.is_positive for r in res.values())


def test_soft_labels_are_probabilities(labeler):
    for value in labeler.to_soft_labels("Medial meniscus tear. No ACL tear.").values():
        assert 0.0 <= value <= 1.0


def test_omission_prior_is_respected():
    """Per-label p(present | never mentioned) must override the default."""
    lab = ClinicalNoteLabeler(omission_priors={"effusion": 0.25})
    assert lab.to_soft_labels("Normal study.")["effusion"] == 0.25


def test_certainty_values_are_tunable():
    lab = ClinicalNoteLabeler(certainty_values={Certainty.DEFINITE: 0.99})
    assert lab.to_soft_labels("Medial meniscus tear.")["medial_meniscus"] == 0.99


def test_batch_rows_are_flat(labeler):
    rows = list(labeler.batch([("s1", "ACL tear."), ("s2", "Normal.")]))
    assert [r["study_id"] for r in rows] == ["s1", "s2"]
    assert rows[0]["ACL"] > rows[1]["ACL"]


def test_coverage_report_separates_scripts(labeler):
    stats = labeler.coverage_report([
        ("a", "ACL tear."),
        ("b", "前十字靭帯断裂を認める。"),
    ])
    assert stats["latin"]["hit_rate"] == 1.0
    assert stats["han"]["hit_rate"] == 0.0


def test_review_queue_ranks_worst_first(labeler):
    queue = labeler.review_queue([("a", "ACL tear."), ("b", "前十字靭帯断裂。")])
    assert [r["study_id"] for r in queue] == ["b"]
