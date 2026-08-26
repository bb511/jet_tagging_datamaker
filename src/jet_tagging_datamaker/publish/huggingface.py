# Building the HuggingFace mirror from the Zenodo archives.
#
# Same jets, same order, same values: only the shape differs, and the split the legacy
# pipeline drew at every run is drawn once here and frozen into the file names.

import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq

from jet_tagging_datamaker.publish import card, export

SHARD_BYTES = 500 * 1024**2

SPLITS = ("train", "validation", "test")

CONFIG = "default"

# HuggingFace reads this block and nothing else to file the page. It takes the licence
# from the identifier here and never opens the LICENSE file, so a record without one
# renders as unlicensed.
#
# Everything the Hub filters its dataset listing by lives here rather than only on the
# Hub itself, because write_card replaces the whole README: a field set through the web
# interface and not repeated here is dropped by the next export.
FRONT_MATTER = """license: cc-by-4.0
language:
- en
task_categories:
- tabular-classification
pretty_name: hls4ml LHC jet dataset (150 particles)
size_categories:
- 100K<n<1M
tags:
- physics
- particle-physics
- jet-tagging
- hls4ml
- lhc
"""

# Directories of publish/assets that ship with the mirror: the loading pipeline and the
# configuration tree that drives it.
ASSET_DIRS = ("loader", "configs")

# Files of publish/assets that ship beside those directories, at the mirror's root.
ASSET_FILES = ("requirements.txt",)

# Module level so a test can point it somewhere else.
ASSETS = Path(__file__).resolve().parent / "assets"

STRAY_GLOBS = (".DS_Store", "._*", ".AppleDouble", "Thumbs.db")


def build(raw_dir: Path, hf_root: Path) -> dict:
    """Write the mirror's shards and report the files each split resolves to."""
    hf_root = Path(hf_root)
    table = export.archive_table(raw_dir, "train")
    train_idx, val_idx = export.split_indices(table.num_rows)
    _write(table.take(train_idx), hf_root, "train")
    _write(table.take(val_idx), hf_root, "validation")
    del table
    _write(export.archive_table(raw_dir, "val"), hf_root, "test")

    return {CONFIG: {split: f"data/{split}-*.parquet" for split in SPLITS}}


def write_shards(table: pa.Table, out_dir: Path, split: str) -> int:
    """Write a table as HuggingFace-style ``<split>-NNNNN-of-NNNNN.parquet`` shards.

    The row count per shard is estimated from the table in memory, so the files on disk
    land near SHARD_BYTES rather than on it: parquet compresses and the ratio is not
    known until the bytes are written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_shards(out_dir, split)
    per_shard = max(1, int(table.num_rows * SHARD_BYTES / max(table.nbytes, 1)))
    chunks = [table.slice(s, per_shard) for s in range(0, table.num_rows, per_shard)]
    if not chunks:
        raise ValueError(f"{split} has no rows; the card would point at no files")
    for index, chunk in enumerate(chunks):
        pq.write_table(
            chunk,
            out_dir / f"{split}-{index:05d}-of-{len(chunks):05d}.parquet",
            compression="snappy",
        )

    return len(chunks)


def split_counts(hf_root: Path) -> dict[str, dict]:
    """Rows and per-class counts per split, read back from the shards on disk."""
    return {split: _counts(Path(hf_root) / "data", split) for split in SPLITS}


def shard_names(hf_root: Path) -> tuple[list[str], list[str]]:
    """The constituent and jet-level column names of a mirror already written.

    Taken from the parquet schema rather than from the h5 files, so the card can be
    rewritten for a record whose sources are no longer on disk.
    """
    shard = sorted((Path(hf_root) / "data").glob("*.parquet"))[0]
    fields = [f for f in pq.read_schema(shard) if f.name not in card.EXTRA_COLUMNS]

    return (
        [f.name for f in fields if pa.types.is_list(f.type)],
        [f.name for f in fields if not pa.types.is_list(f.type)],
    )


def configs_block(written: dict) -> str:
    """The ``configs:`` YAML that lets ``load_dataset`` find the files without a script."""
    lines = ["configs:"]
    for name in sorted(written):
        lines.append(f"- config_name: {name}")
        lines.append("  data_files:")
        for split, pattern in written[name].items():
            lines.append(f"  - split: {split}")
            lines.append(f"    path: {pattern}")

    return "\n".join(lines) + "\n"


def write_card(
    hf_root: Path, written: dict, names: tuple[list[str], list[str]]
) -> None:
    """Write the card, whose front matter is what HuggingFace reads to find the data.

    A card copied without that block renders but does not load.
    """
    front = f"{FRONT_MATTER}{configs_block(written)}"
    body = card.render(split_counts(hf_root), *names)
    (hf_root / "README.md").write_text(f"---\n{front}---\n\n{body}\n")
    (hf_root / "LICENSE").write_text(card.LICENCE)


def copy_assets(hf_root: Path) -> list[str]:
    """Copy the loading pipeline, its configuration tree and its requirements over.

    Each directory is replaced rather than written over, so a file renamed or dropped
    here does not survive in a mirror built on top of an earlier one.
    """
    for name in ASSET_DIRS:
        shutil.rmtree(hf_root / name, ignore_errors=True)
        shutil.copytree(
            ASSETS / name, hf_root / name, ignore=shutil.ignore_patterns("__pycache__")
        )
    for name in ASSET_FILES:
        shutil.copy(ASSETS / name, hf_root / name)

    return [*ASSET_DIRS, *ASSET_FILES]


def prune_stray(root: Path) -> list[Path]:
    """Delete the byte caches and the OS listing files, which would otherwise upload."""
    root = Path(root)
    removed = []
    for path in sorted(root.rglob("__pycache__")):
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)
    for pattern in STRAY_GLOBS:
        for path in root.rglob(pattern):
            path.unlink()
            removed.append(path)

    return removed


def finish(hf_root: Path, written: dict, names: tuple[list[str], list[str]]) -> None:
    """Write the card, copy the loader, and clear the stray files before upload."""
    write_card(hf_root, written, names)
    copy_assets(hf_root)
    prune_stray(hf_root)


def _write(table: pa.Table, hf_root: Path, split: str) -> None:
    """One split, sharded, with a line about it."""
    shards = write_shards(table, hf_root / "data", split)
    print(f"  {split:11s} {table.num_rows:>9,} rows  {shards:>3d} shards")


def _counts(data_dir: Path, split: str) -> dict:
    """Rows and class counts of one split, reading the label column alone."""
    shards = sorted(data_dir.glob(f"{split}-*.parquet"))
    table = pds.dataset(shards, format="parquet").to_table(columns=["label"])
    values, counts = np.unique(table["label"].to_numpy(), return_counts=True)

    labels = dict(zip(values.tolist(), counts.tolist()))

    return {"rows": table.num_rows, "labels": labels}


def _clear_shards(out_dir: Path, split: str) -> None:
    """Drop shards an earlier run left, which a rerun writing fewer would not overwrite.

    They carry the old shard count in their names, so nothing overwrites them and the
    config's glob would read them as extra rows.
    """
    for shard in out_dir.glob(f"{split}-*.parquet"):
        shard.unlink()
