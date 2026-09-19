# RSNA Knee Abnormality Detection

Knee MRI studies come with pixels in one place and a radiologist's report in
another, and nothing in the competition data joins them. This repo is that
join: DICOM series on one side, 12 soft labels parsed out of multilingual
free-text reports on the other, one table with both.

```python
from rsna_knee import KneeDatasetBuilder

builder = KneeDatasetBuilder("/kaggle/input/competitions/rsna-knee-abnormality-detection")
df = builder.build()        # one row per series: metadata + labels
ds = builder.dataset()      # indexable; volumes read on demand
```

## Layout

```
src/rsna_knee/
    dicom.py            where the pixels are, and every header except the pixels
    labeling/           free-text reports -> 12 soft labels, in ~12 languages
        schema.py         the label set, the certainty scale, the value objects
        vocabulary.py     all language data, no logic
        text.py           script detection, accent folding, abbreviations
        negation.py       asserted / denied / hedged
        labeler.py        orchestration
    records.py          one series, pixels on demand; an indexable view
    builder.py          the join, and the cached table it produces
    config.py           where the data is
    cli.py              python -m rsna_knee ...
notebooks/              exploration; imports the package, defines nothing
tests/                  pytest, against synthetic DICOM files
```

## Install

```bash
pip install -e ".[dev]"     # numpy, pandas, pydicom, pytest, pyarrow
pytest
```

On Kaggle there is nothing to install — `sys.path.insert(0, "src")` is enough.

## The three ideas

**The row is a series, the label is a study.** Reports are written per study,
pixels live per series, and a knee study is typically 4–8 sequences. Labels
broadcast down to every series, and `df["group"]` carries the study id. Split
on that: a row-level split puts one report on both sides and flatters your
validation score.

**Metadata is eager, pixels are lazy.** The scan reads headers only
(`stop_before_pixels=True`) and keeps the sorted file paths;
`SeriesRecord.volume()` reads pixels when you ask. So the index caches to
parquet and reloads in seconds instead of re-scanning.

**A failed series is data, not an exception.** Missing folders, unreadable
files, absent geometry tags, ragged slice shapes — all of these happen on a
real corpus. Each is recorded on the row and surfaced in `builder.errors`,
so a bad series is something you count, not something that ends the run.

## Check the labels before you train on them

The labeler is rules and vocabulary, not a model, and its failure mode is
quiet: a report in a script it does not cover produces exactly the same
all-negative output as a genuinely normal knee. That is flagged rather than
hidden.

```python
builder.review_queue()          # reports the labeler could not read
builder.labeler.coverage_report(zip(df["StudyInstanceUID"], df["report"]))
```

A script with a near-zero hit rate is a vocabulary gap, not a population of
healthy knees. Exclude those studies; do not train on them as negatives.

The non-English vocabulary is a starting point, not validated terminology —
check it against your corpus. A wrong stem produces no match, and no match
looks exactly like a negative finding.

## Command line

```bash
python -m rsna_knee check                      # dry run on two studies
python -m rsna_knee labels --out labels.csv    # reports only, no DICOM scan
python -m rsna_knee build  --out index.parquet # the full index
```

`--data-path` overrides the location; otherwise `$RSNA_DATA_PATH`, then the
Kaggle mounts, then `./data`.

## Labels

`ACL`, `MCL`, `medial_meniscus`, `lateral_meniscus`, `medial_OA`,
`lateral_OA`, `patellofemoral_OA`, `effusion`, `synovitis`, `bakers_cyst`,
`bone_contusion`, `fracture`.

Values are probabilities, not booleans: reports hedge constantly, and
collapsing "definite tear" and "tear cannot be excluded" to the same 1 throws
away information the model can use. Each label also carries a
`<label>_certainty` column, so you can weight a definite finding above a
hedged one — or drop hedged rows — without re-running the labeler.
