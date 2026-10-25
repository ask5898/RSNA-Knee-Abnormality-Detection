"""The rule-based labeler: the sentence checks used while fixing it, and the
worked example traced in the README."""
import pytest

from rsna_knee.labeling.rules import LABELS, ClinicalNoteLabeler

CASES = [
    ("Baker's cyst.", ["bakers_cyst"]), ("Baker’s cyst is present", ["bakers_cyst"]),
    ("Small Baker's cyst in the popliteal fossa.", ["bakers_cyst"]),
    ("No Baker's cyst.", []), ("There is no evidence of a Baker’s cyst.", []),
    ("Medial meniscal tear.", ["medial_meniscus"]), ("Lateral meniscal tear.", ["lateral_meniscus"]),
    ("The medial meniscal body is intact.", []), ("No medial meniscal tear.", []),
    ("Sprain of the tibial collateral ligament", ["MCL"]), ("The tibial collateral ligament is intact.", []),
    ("Mild synovial proliferation.", ["synovitis"]), ("Subchondral marrow oedema.", ["bone_contusion"]),
    ("No marrow edema.", []),
    ("ACL: intact", []), ("ACL: torn", ["ACL"]), ("ACL: complete tear, MCL: intact", ["ACL"]),
    ("MCL: sprain, ACL: intact.", ["MCL"]),
    ("Effusion: none", []), ("Effusion: moderate", ["effusion"]), ("Fracture: absent", []),
    ("Fracture: negative", []), ("Joint fluid: normal", []),
    ("Medial meniscus: posterior horn tear", ["medial_meniscus"]),
    ("Impression: complete ACL tear. Moderate effusion.", ["ACL", "effusion"]),
    ("Findings: no fracture, but ACL torn", ["ACL"]),
    ("Complete ACL tear. MCL sprain. Tear of the medial meniscus.", ["ACL", "MCL", "medial_meniscus"]),
    ("Normal knee MRI. No effusion.", []),
    ("FINDINGS:\nACL: intact.\nMCL: grade 2 sprain.\nEffusion: small.\nIMPRESSION:\nMCL sprain with small effusion.",
     ["MCL", "effusion"]),
    ("Impression: no fracture or effusion.", []), ("Impression: tibial plateau fracture, no effusion.", ["fracture"]),
    ("Ligaments: ACL intact, MCL intact. Menisci: medial meniscal tear.", ["medial_meniscus"]),
    ("Kein Erguss. Innenband intakt.", []), ("Gelenkerguss: mäßig", ["effusion"]),
]

REPORT = ("FINDINGS: Complete tear of the ACL. MCL: intact. Possible tear of the posterior horn of the medial "
          "meniscus. Small joint effusion. No fracture, but bone bruise of the lateral femoral condyle. "
          "Patellofemoral cartilage is preserved. Small Baker's cyst. Possible synovitis.\n"
          "IMPRESSION: ACL rupture with bone contusion. Medial meniscus tear.")


@pytest.fixture(scope="module")
def labeler():
    return ClinicalNoteLabeler()


@pytest.mark.parametrize("text,positives", CASES)
def test_sentence(labeler, text, positives):
    found = {lab for lab, v in labeler.to_soft_labels(text).items() if v >= 0.5}
    assert found == set(positives)


def test_worked_example(labeler):
    soft = labeler.to_soft_labels(REPORT)
    assert soft == {"ACL": 0.95, "MCL": 0.02, "medial_meniscus": 0.95, "lateral_meniscus": 0.02,
                    "medial_OA": 0.02, "lateral_OA": 0.02, "patellofemoral_OA": 0.02, "effusion": 0.95,
                    "synovitis": 0.5, "bakers_cyst": 0.95, "bone_contusion": 0.95, "fracture": 0.02}
    assert list(soft) == LABELS


def test_strongest_mention_wins(labeler):
    res = labeler.extract("Possible tear of the medial meniscus. Medial meniscus tear.")
    assert res["medial_meniscus"].certainty == "definite"


def test_negation_beats_hedge(labeler):
    assert labeler.extract("No possible medial meniscus tear.")["medial_meniscus"].certainty == "negated"


def test_unsupported_script_is_flagged(labeler):
    res = labeler.extract("前十字韧带撕裂。")
    assert all(r.needs_review for r in res.values())


def test_known_gap_unlikely_reads_as_probable(labeler):
    # Documented in the README: the probable cue "likely" also matches inside "unlikely".
    assert labeler.extract("ACL tear is unlikely.")["ACL"].certainty == "probable"
