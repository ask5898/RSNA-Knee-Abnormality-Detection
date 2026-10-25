"""A tiny fake competition folder, shaped like the real one, for tests.

    python tests/make_synthetic_data.py /tmp/comp

70 training studies (3 series each: sagittal PD fat-sat, coronal T1, a localizer)
with English reports and 12 label columns, of which only the first --gold
studies keep their values (like the 58 official ones); 4 test studies, the last
one with unreadable files. Pixels are noise brightened by the number of
findings, so a model can learn a little. Deterministic: same seed, same files.
"""
import argparse
import copy
import os
import random

import numpy as np
import pandas as pd
import pydicom
from pydicom.data import get_testdata_file
from pydicom.uid import generate_uid

COMP = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
        "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PHRASES = ["Complete ACL tear.", "MCL sprain.", "Tear of the medial meniscus.", "Lateral meniscus tear.",
           "Medial compartment osteoarthritis.", "Lateral compartment osteoarthritis.",
           "Patellofemoral osteoarthritis.", "Moderate joint effusion.", "Synovitis.", "Baker's cyst.",
           "Bone contusion of the lateral femoral condyle.", "Tibial plateau fracture."]
SERIES = [("SAG PD FS", [0, 1, 0, 0, 0, -1], 10), ("COR T1", [1, 0, 0, 0, 0, -1], 10),
          ("LOC 3-PLANE", [1, 0, 0, 0, 1, 0], 3)]


def make(root, n_train=70, n_test=4, n_gold=15, seed=3):
    random.seed(seed)
    rng = np.random.default_rng(seed)
    base = pydicom.dcmread(get_testdata_file("MR_small.dcm"))
    # UIDs from the seeded generator, so two runs build identical folders
    uid = lambda: generate_uid(entropy_srcs=[str(rng.random())])

    def write_study(split, study, positives, broken=False):
        out = []
        for desc, iop, n in SERIES:
            series = uid()
            d = f"{root}/{split}_series/{study}/{series}"
            os.makedirs(d)
            out.append({"StudyInstanceUID": study, "SeriesInstanceUID": series})
            if broken:
                open(f"{d}/0.dcm", "wb").write(b"junk")
                continue
            for k in range(n):
                ds = copy.deepcopy(base)
                ds.Rows = ds.Columns = 64
                img = rng.random((64, 64)) * 400 + 300 * len(positives)
                ds.PixelData = img.astype(np.int16).tobytes()
                ds.SeriesDescription = desc
                ds.ImageOrientationPatient = iop
                nrm = np.cross(iop[:3], iop[3:])
                ds.ImagePositionPatient = list(k * 3.0 * nrm)
                ds.InstanceNumber = k + 1
                ds.RepetitionTime = 3000 if "PD" in desc else 500
                ds.EchoTime = 30 if "PD" in desc else 12
                ds.save_as(f"{d}/{k}.dcm")
        return out

    train_rows, train_series = [], []
    for s in range(n_train):
        study = uid()
        pos = [i for i in range(12) if rng.random() < 0.3]
        report = " ".join(PHRASES[i] for i in pos) or "Normal knee MRI. No effusion."
        labels = {c: (float(i in pos) if s < n_gold else np.nan) for i, c in enumerate(COMP)}
        train_rows.append({"StudyInstanceUID": study, "Report": report, **labels})
        train_series += write_study("train", study, pos)
    test_rows, test_series = [], []
    for s in range(n_test):
        study = uid()
        test_rows.append({"StudyInstanceUID": study})
        test_series += write_study("test", study, [0, 7] if s == 0 else [], broken=(s == n_test - 1))
    pd.DataFrame(train_rows).to_csv(f"{root}/train.csv", index=False)
    pd.DataFrame(train_series).to_csv(f"{root}/train_series.csv", index=False)
    pd.DataFrame(test_rows).to_csv(f"{root}/test.csv", index=False)
    pd.DataFrame(test_series).to_csv(f"{root}/test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": [r["StudyInstanceUID"] for r in test_rows], **{c: 0.5 for c in COMP}}
                 ).to_csv(f"{root}/sample_submission.csv", index=False)
    return root


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--gold", type=int, default=15)
    a = p.parse_args()
    os.makedirs(a.root, exist_ok=True)
    make(a.root, n_gold=a.gold)
    print("synthetic competition data in", a.root)
