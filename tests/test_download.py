"""The Kaggle downloader, against a stand-in for the Kaggle API (no network)."""
import os
from types import SimpleNamespace

import pytest

from rsna_knee.data import download as D


class FakeApi:
    def __init__(self, error=None, pages=None):
        self.calls, self.error, self.pages = [], error, pages or []

    def competition_download_files(self, competition, path, force, quiet, unzip):
        self.calls.append(("all", competition, unzip))
        if self.error:
            raise Exception(self.error)
        for name in D.EXPECTED:
            target = os.path.join(path, name)
            if "." in name:
                open(target, "w").write("StudyInstanceUID\n1\n")
            else:
                os.makedirs(target, exist_ok=True)

    def competition_download_file(self, competition, name, path, force, quiet, unzip):
        self.calls.append(("file", name, unzip))
        if self.error:
            raise Exception(self.error)
        open(os.path.join(path, name), "w").write("StudyInstanceUID\n1\n")

    def competition_list_files(self, competition, page_token=None, page_size=200):
        i = int(page_token or 0)
        files, nxt = self.pages[i]
        return SimpleNamespace(files=[SimpleNamespace(name=n, total_bytes=b) for n, b in files], next_page_token=nxt)

    def dataset_download_files(self, slug, path, force, quiet, unzip):
        self.calls.append(("dataset", slug, unzip))
        open(os.path.join(path, "train_set.csv"), "w").write("study\n")


def test_csv_only_downloads_each_file_once(tmp_path):
    api = FakeApi()
    D.download_competition(str(tmp_path), files=D.CSV_FILES, api=api)
    assert [c[1] for c in api.calls] == list(D.CSV_FILES) and all(c[2] for c in api.calls)
    D.download_competition(str(tmp_path), files=D.CSV_FILES, api=api)          # second time: all there
    assert len(api.calls) == len(D.CSV_FILES)


def test_full_download_and_skip_when_complete(tmp_path):
    api = FakeApi()
    D.download_competition(str(tmp_path), api=api)
    assert api.calls == [("all", D.COMPETITION, True)]
    assert D.missing_parts(str(tmp_path)) == []
    D.download_competition(str(tmp_path), api=api)                             # complete: no API call
    assert len(api.calls) == 1
    D.download_competition(str(tmp_path), api=api, force=True)                 # unless forced
    assert len(api.calls) == 2


def test_rules_not_accepted_gives_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="Accept the rules"):
        D.download_competition(str(tmp_path), api=FakeApi(error="403 Client Error: Forbidden"))


def test_unknown_name_gives_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="404"):
        D.download_competition(str(tmp_path), files=["train.csv"], api=FakeApi(error="404 Not Found"))


def test_failed_sign_in_raises_instead_of_exiting(monkeypatch):
    from kaggle.api.kaggle_api_extended import KaggleApi

    def fail(self):
        raise SystemExit(1)

    monkeypatch.setattr(KaggleApi, "authenticate", fail)
    with pytest.raises(RuntimeError, match="sign-in failed"):
        D.get_api()


def test_listing_follows_pages():
    api = FakeApi(pages=[([("train.csv", 10), ("test.csv", 5)], "1"), ([("train_series.csv", 7)], None)])
    assert D.list_competition_files(api=api) == [("train.csv", 10), ("test.csv", 5), ("train_series.csv", 7)]
    assert D.list_competition_files(api=api, max_files=2) == [("train.csv", 10), ("test.csv", 5)]


def test_dataset_download(tmp_path):
    api = FakeApi()
    D.download_dataset("ask5898/train-set", str(tmp_path), api=api)
    assert api.calls == [("dataset", "ask5898/train-set", True)]
    assert (tmp_path / "train_set.csv").exists()
