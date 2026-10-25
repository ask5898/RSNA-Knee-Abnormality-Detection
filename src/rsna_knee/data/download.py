"""Download the competition data, and your own Kaggle datasets or models, with
the Kaggle API.

    python scripts/download_data.py                      # the whole competition (large: all DICOMs)
    python scripts/download_data.py --csv-only           # only the CSVs: reports, labels, series lists
    python scripts/download_data.py --dataset ask5898/train-set --dest data/
    python scripts/download_data.py --model ask5898/knee-model-convnext/pytorch/default/1 --dest outputs/knee_model

Sign-in (Kaggle client 2.x tries these in order):
  1. an API token: the KAGGLE_API_TOKEN environment variable, or the token saved
     in ~/.kaggle/access_token (generate one at https://www.kaggle.com/settings/api)
  2. a legacy key: ~/.kaggle/kaggle.json, or KAGGLE_USERNAME and KAGGLE_KEY
  3. `kaggle auth login`, which signs in through the browser and caches the result

Accept the competition's rules on its Kaggle page once before the first
download; until then the API refuses with 403 Forbidden.
"""
import os
import shutil

from .. import config as C

COMPETITION = "rsna-knee-abnormality-detection"
CSV_FILES = ("train.csv", "test.csv", "train_series.csv", "test_series.csv", "sample_submission.csv")
# What a complete download looks like (dicom.py and predict.py read these)
EXPECTED = ("train.csv", "test.csv", "train_series.csv", "test_series.csv", "train_series", "test_series")


def get_api():
    """An authenticated KaggleApi.

    The client calls exit() when no credentials work; that is turned into an
    error that says what to do, so a calling script is not silently killed."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise ImportError("The Kaggle client is not installed: pip install kaggle") from exc
    api = KaggleApi()
    try:
        api.authenticate()
    except SystemExit as exc:
        raise RuntimeError(
            "Kaggle sign-in failed. Set KAGGLE_API_TOKEN (from https://www.kaggle.com/settings/api), "
            "save the token to ~/.kaggle/access_token, put kaggle.json in ~/.kaggle/, "
            "or run `kaggle auth login`.") from exc
    return api


def _explain(exc, competition):
    """Re-raise an API error with the usual cause spelled out."""
    text = str(exc)
    if "403" in text or "Forbidden" in text:
        raise RuntimeError(
            f"Kaggle refused the download (403). Accept the rules of the competition first: "
            f"https://www.kaggle.com/competitions/{competition}/rules") from exc
    if "404" in text or "Not Found" in text:
        raise RuntimeError(f"Kaggle has no competition, dataset or model called '{competition}' "
                           "that this account can see (404).") from exc
    raise exc


def list_competition_files(competition=COMPETITION, api=None, max_files=None):
    """[(name, size in bytes), ...] for the competition's files, page by page.

    The competition holds one file per DICOM slice, so a full listing is long;
    max_files stops early."""
    api = api or get_api()
    out, token = [], None
    while True:
        try:
            response = api.competition_list_files(competition, page_token=token, page_size=200)
        except Exception as exc:                       # noqa: BLE001 - re-raised with a reason
            _explain(exc, competition)
        out += [(f.name, int(f.total_bytes or 0)) for f in response.files]
        token = response.next_page_token
        if not token or (max_files and len(out) >= max_files):
            return out[:max_files] if max_files else out


def missing_parts(dest):
    """The expected files and folders that are not (yet) in dest."""
    return [name for name in EXPECTED if not os.path.exists(os.path.join(dest, name))]


def download_competition(dest=None, competition=COMPETITION, files=None, force=False, api=None, quiet=False):
    """Download the competition data into dest (default: config.DATA_PATH).

    files=None   everything: one archive, extracted into dest and then deleted.
                 Extraction needs room for the archive and its contents at once.
    files=[...]  only these files (e.g. CSV_FILES), each extracted if Kaggle zipped it.

    Files already in dest are kept unless force=True. Returns dest."""
    dest = C.DATA_PATH if dest is None else dest
    os.makedirs(dest, exist_ok=True)
    if files is None:
        if not force and not missing_parts(dest):
            print(f"{dest} already holds the competition data; pass force=True to download again")
            return dest
        free = shutil.disk_usage(dest).free
        print(f"downloading all of {competition} into {dest} ({free / 1e9:.0f} GB free) ...")
        api = api or get_api()
        try:
            api.competition_download_files(competition, path=dest, force=force, quiet=quiet, unzip=True)
        except Exception as exc:                       # noqa: BLE001
            _explain(exc, competition)
    else:
        api = api or get_api()
        for name in files:
            if not force and os.path.exists(os.path.join(dest, name)):
                print(f"  {name}: already there")
                continue
            print(f"  {name}: downloading")
            try:
                api.competition_download_file(competition, name, path=dest, force=force, quiet=quiet, unzip=True)
            except Exception as exc:                   # noqa: BLE001
                _explain(exc, competition)
    missing = missing_parts(dest)
    if missing and files is None:
        print(f"warning: {dest} is missing {missing} after the download")
    return dest


def download_dataset(slug, dest, force=False, api=None, quiet=False):
    """A Kaggle dataset ('owner/name', e.g. your train_set.csv dataset) into dest."""
    os.makedirs(dest, exist_ok=True)
    api = api or get_api()
    try:
        api.dataset_download_files(slug, path=dest, force=force, quiet=quiet, unzip=True)
    except Exception as exc:                           # noqa: BLE001
        _explain(exc, slug)
    return dest


def download_model(handle, dest, force=False, api=None, quiet=False):
    """A Kaggle model version ('owner/model/framework/variation/version', e.g. a
    knee_model uploaded as a Model) into dest, unpacked."""
    os.makedirs(dest, exist_ok=True)
    api = api or get_api()
    try:
        api.model_instance_version_download(handle, path=dest, force=force, quiet=quiet, untar=True)
    except Exception as exc:                           # noqa: BLE001
        _explain(exc, handle)
    return dest


def summarize(dest=None):
    """A line about what dest holds: studies, series, how many have reports."""
    import pandas as pd

    dest = C.DATA_PATH if dest is None else dest
    parts = []
    for split in ("train", "test"):
        csv = os.path.join(dest, f"{split}.csv")
        series = os.path.join(dest, f"{split}_series.csv")
        if os.path.exists(csv):
            parts.append(f"{split}: {len(pd.read_csv(csv))} studies")
        if os.path.exists(series):
            parts.append(f"{len(pd.read_csv(series))} {split} series")
    missing = missing_parts(dest)
    return f"{dest}: " + (", ".join(parts) or "no CSVs") + (f" | missing {missing}" if missing else "")
