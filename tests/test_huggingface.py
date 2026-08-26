# The mirror side: shards, the configs block, the card, the assets.

import re

import pytest
from conftest import CONSTITUENT_NAMES, JET_NAMES, LABEL_NAMES

from jet_tagging_datamaker.publish import card, export, huggingface

JET_COLUMNS = list(JET_NAMES[: -len(LABEL_NAMES)])


@pytest.fixture
def mirror(raw_dir, tmp_path):
    """A whole mirror of the synthetic archives, card and assets included."""
    out = tmp_path / "hf"
    written = huggingface.build(raw_dir, out)
    huggingface.write_card(out, written, huggingface.shard_names(out))

    return out, written


def test_build_writes_one_shard_per_split_and_returns_its_pattern(raw_dir, tmp_path):
    out = tmp_path / "hf"

    written = huggingface.build(raw_dir, out)

    assert written == {
        "default": {
            "train": "data/train-*.parquet",
            "validation": "data/validation-*.parquet",
            "test": "data/test-*.parquet",
        }
    }
    for split in huggingface.SPLITS:
        assert (out / "data" / f"{split}-00000-of-00001.parquet").exists()


def test_split_rows_add_up_to_the_archives(raw_dir, tmp_path):
    out = tmp_path / "hf"
    huggingface.build(raw_dir, out)

    counts = huggingface.split_counts(out)

    train_rows = export.archive_table(raw_dir, "train").num_rows
    assert counts["train"]["rows"] + counts["validation"]["rows"] == train_rows
    assert counts["test"]["rows"] == export.archive_table(raw_dir, "val").num_rows


def test_shard_names_recover_the_column_names(mirror):
    out, _ = mirror

    particle, jet = huggingface.shard_names(out)

    assert particle == list(CONSTITUENT_NAMES)
    assert jet == JET_COLUMNS


def test_configs_block_names_every_split_under_one_config():
    written = {
        "default": {"train": "data/train-*.parquet", "test": "data/test-*.parquet"}
    }

    assert huggingface.configs_block(written) == (
        "configs:\n"
        "- config_name: default\n"
        "  data_files:\n"
        "  - split: train\n"
        "    path: data/train-*.parquet\n"
        "  - split: test\n"
        "    path: data/test-*.parquet\n"
    )


def test_card_opens_with_the_front_matter_the_hub_reads(mirror):
    out, written = mirror

    text = (out / "README.md").read_text()

    assert text.startswith("---\nlicense: cc-by-4.0")
    assert huggingface.configs_block(written) in text
    assert text.split("---\n\n", 1)[1].startswith("# hls4ml LHC jet dataset")


def test_card_quotes_the_split_table_and_the_licence(mirror):
    out, _ = mirror
    counts = huggingface.split_counts(out)

    text = (out / "README.md").read_text()

    for split in huggingface.SPLITS:
        assert f"| `{split}` | {counts[split]['rows']:,} |" in text
    assert (out / "LICENSE").read_text().startswith("Creative Commons Attribution 4.0")


def test_card_leaves_no_placeholder_unfilled(mirror):
    out, _ = mirror

    body = card.render(huggingface.split_counts(out), *huggingface.shard_names(out))

    assert re.search(r"\{[a-z_]+\}", body) is None
    assert card.REPO_ID in body and card.ZENODO_DOI in body


def test_copy_assets_replaces_what_an_earlier_run_left(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    (assets / "loader").mkdir(parents=True)
    (assets / "loader" / "__init__.py").write_text("VERSION = 1\n")
    (assets / "configs" / "paths").mkdir(parents=True)
    (assets / "configs" / "paths" / "default.yaml").write_text("root_dir: .\n")
    (assets / "requirements.txt").write_text("numpy\n")
    monkeypatch.setattr(huggingface, "ASSETS", assets)
    out = tmp_path / "hf"
    (out / "loader").mkdir(parents=True)
    (out / "loader" / "gone.py").write_text("")

    copied = huggingface.copy_assets(out)

    assert copied == ["loader", "configs", "requirements.txt"]
    assert (out / "loader" / "__init__.py").read_text() == "VERSION = 1\n"
    assert not (out / "loader" / "gone.py").exists()
    assert (out / "configs" / "paths" / "default.yaml").exists()
    assert (out / "requirements.txt").read_text() == "numpy\n"


def test_a_rerun_leaves_no_stale_shards(raw_dir, tmp_path):
    out = tmp_path / "hf"
    huggingface.build(raw_dir, out)
    stale = out / "data" / "train-00001-of-00002.parquet"
    stale.write_bytes(b"")

    huggingface.build(raw_dir, out)

    assert not stale.exists()
    assert sorted(p.name for p in (out / "data").glob("train-*.parquet")) == [
        "train-00000-of-00001.parquet"
    ]


def test_prune_stray_removes_caches_and_os_files(tmp_path):
    (tmp_path / "loader" / "__pycache__").mkdir(parents=True)
    (tmp_path / "loader" / "__pycache__" / "x.pyc").write_bytes(b"")
    (tmp_path / ".DS_Store").write_bytes(b"")

    huggingface.prune_stray(tmp_path)

    assert not (tmp_path / "loader" / "__pycache__").exists()
    assert not (tmp_path / ".DS_Store").exists()
