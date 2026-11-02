"""HEEDB access point -> private bucket copy (dataset_maker/download_heedb.py),
against an in-memory fake S3 client."""
import pandas as pd
import pytest

from dataset_maker import download_heedb as dh
from dataset_maker import heedb_index as hx

KEYS = ["EEG/bids/S0001/sub-S0001101/ses-1/eeg/sub-S0001101_ses-1_task-EEG_eeg.edf",
        "EEG/bids/S0001/sub-S0001101/ses-1/eeg/sub-S0001101_ses-1_task-EEG_channels.tsv",
        "EEG/bids/S0002/sub-S0002201/ses-3/eeg/sub-S0002201_ses-3_task-EEG_eeg.edf"]


class FakeS3:
    def __init__(self, source, fail_once=()):
        self.source, self.dest, self.fail_once = dict(source), {}, set(fail_once)

    def copy_object(self, Bucket, Key, CopySource):
        assert CopySource["Bucket"] == hx.ACCESS_POINT
        if CopySource["Key"] in self.fail_once:
            self.fail_once.discard(CopySource["Key"])
            raise RuntimeError("SlowDown")
        self.dest[(Bucket, Key)] = self.source[CopySource["Key"]]

    def get_paginator(self, name):
        fake = self

        class Pager:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k, "Size": s} for (b, k), s in fake.dest.items()
                                    if b == Bucket and k.startswith(Prefix)]}
        return Pager()


@pytest.fixture
def objects():
    return pd.DataFrame({"key": KEYS, "bytes": [100, 2, 50]})


def test_dest_key_keeps_bids_path():
    assert dh.dest_key(KEYS[0]) == "HEEDB/v4.1/raw/bids/S0001/sub-S0001101/ses-1/eeg/" \
                                   "sub-S0001101_ses-1_task-EEG_eeg.edf"
    with pytest.raises(ValueError):
        dh.dest_key("other/key")


class _Exceptions:
    class NoSuchBucketPolicy(Exception):
        pass

    def from_code(self, code):
        return getattr(self, code)


class FakeAccess:
    exceptions = _Exceptions()

    def __init__(self, blocks, policy_public=None):
        self.blocks, self.policy_public = blocks, policy_public

    def get_public_access_block(self, Bucket):
        if self.blocks is None:
            raise RuntimeError("NoSuchPublicAccessBlockConfiguration")
        return {"PublicAccessBlockConfiguration": self.blocks}

    def get_bucket_policy_status(self, Bucket):
        if self.policy_public is None:
            raise self.exceptions.NoSuchBucketPolicy()
        return {"PolicyStatus": {"IsPublic": self.policy_public}}


ALL_ON = dict.fromkeys(["BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets"], True)


def test_private_bucket_passes_and_public_ones_are_refused():
    dh.check_private(FakeAccess(ALL_ON), "b")                        # no policy at all
    dh.check_private(FakeAccess(ALL_ON, policy_public=False), "b")
    for client in (FakeAccess(None), FakeAccess({**ALL_ON, "RestrictPublicBuckets": False}),
                   FakeAccess(ALL_ON, policy_public=True)):
        with pytest.raises(SystemExit):
            dh.check_private(client, "b")


def test_copy_retries_failures_and_resumes(objects):
    s3 = FakeS3(dict(zip(objects["key"], objects["bytes"])), fail_once=[KEYS[2]])
    summary = dh.run_copy(s3, objects, "bkt", workers=2)
    assert summary["failed"] == 0 and summary["copied_bytes"] == 152
    existing = dh.list_dest(s3, "bkt", "HEEDB/v4.1/raw/")
    assert dh.todo(objects, existing).empty                     # rerun copies nothing
    s3.dest[("bkt", dh.dest_key(KEYS[1]))] = 1                  # truncated copy
    assert list(dh.todo(objects, dh.list_dest(s3, "bkt", "HEEDB/v4.1/raw/"))["key"]) == [KEYS[1]]


def test_verify_counts_per_site(objects):
    s3 = FakeS3(dict(zip(objects["key"], objects["bytes"])))
    dh.run_copy(s3, objects.iloc[:2], "bkt", workers=2)
    t = dh.verify(objects, dh.list_dest(s3, "bkt", "HEEDB/v4.1/raw/"))
    assert t.loc["S0001", "missing"] == 0 and t.loc["S0002", "missing"] == 1
    assert t.loc["total", "expected"] == 3 and t.loc["total", "present"] == 2
