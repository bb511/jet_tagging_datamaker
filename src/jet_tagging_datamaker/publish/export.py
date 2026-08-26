# Zenodo archives to arrow tables, one row per jet.
#
# The two tarballs of record 3602260 hold HDF5 files of padded (jets, 150, 16) blocks.
# This module fetches them, drops the padding, and hands the result to huggingface.py as
# tables that can be sharded straight into parquet. The jet images the same files carry
# are never read: they are two orders of magnitude larger than everything else here.

import hashlib
import tarfile
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
from sklearn.model_selection import train_test_split

ZENODO = "https://zenodo.org/api/records/3602260/files/{name}/content"

# Zenodo's own checksums, so a truncated download fails here rather than hours later.
ARCHIVES = {
    "train": ("hls4ml_LHCjet_150p_train.tar.gz", "af12d3a500414924e70a73b56fc49aec"),
    "val": ("hls4ml_LHCjet_150p_val.tar.gz", "ff902e6a02bfb4247a9068bf7ff7f6db"),
}

H5_GLOB = "jetImage_*_150p_*.h5"

# particleFeatureNames has 17 entries for 16 columns: j1_pdgid is a name without data.
DROPPED_PARTICLE_NAMES = ("j1_pdgid",)

# The five one-hot class columns and j_undef, at the end of `jets`.
N_LABEL_COLUMNS = 6

# The split the legacy pipeline draws, reproduced here so the record carries it.
SPLIT_SEED = 42
VAL_FRACTION = 0.2

# The h5 arrays are float64, but every value in them is float32-representable, so the
# mirror halves in size without losing a bit. See the verified facts in the README.
JET_DTYPE = np.float32

BLOCK = 1024**2


def download(name: str, md5: str, raw_dir: Path) -> Path:
    """Stream one Zenodo archive into raw_dir, hashing the bytes as they land."""
    path = Path(raw_dir) / name
    if path.exists() and _md5(path) == md5:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5()
    with urllib.request.urlopen(ZENODO.format(name=name)) as source:
        with open(path, "wb") as out:
            for block in iter(lambda: source.read(BLOCK), b""):
                digest.update(block)
                out.write(block)
    if digest.hexdigest() != md5:
        raise ValueError(f"{name}: md5 {digest.hexdigest()} does not match {md5}")

    return path


def extract(archive: Path, raw_dir: Path) -> None:
    """Unpack one archive. Both hold a single top-level train/ or val/ directory."""
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(raw_dir, filter="data")


def fetch(raw_dir: Path) -> None:
    """Put both archives on disk, skipping whichever is already unpacked."""
    raw_dir = Path(raw_dir)
    for split, (name, md5) in ARCHIVES.items():
        if any((raw_dir / split).glob(H5_GLOB)):
            print(f"  {split:5s} already unpacked, skipping")
            continue
        extract(download(name, md5, raw_dir), raw_dir)
        print(f"  {split:5s} {len(h5_files(raw_dir, split))} files")


def h5_files(raw_dir: Path, split: str) -> list[Path]:
    """One archive's h5 files, in the order everything downstream concatenates them."""
    files = sorted((Path(raw_dir) / split).glob(H5_GLOB))
    if not files:
        raise FileNotFoundError(f"No {H5_GLOB} under {raw_dir}/{split}")

    return files


def feature_names(h5) -> tuple[list[str], list[str]]:
    """The constituent and jet-level column names the record keeps, in h5 order."""
    particle = _decoded(h5["particleFeatureNames"][:])
    jet = _decoded(h5["jetFeatureNames"][:])

    return (
        [name for name in particle if name not in DROPPED_PARTICLE_NAMES],
        jet[:-N_LABEL_COLUMNS],
    )


def jagged(x: np.ndarray, names: list[str]) -> dict[str, pa.ListArray]:
    """One variable-length column per constituent feature, with the padding removed.

    A padded slot is a row of zeros in every feature at once, so one mask serves all
    sixteen columns and the lists of one jet stay the same length.
    """
    keep = np.any(x != 0, axis=2)
    # A kept slot with pT == 0 would tie with the padding under the loader's stable
    # -pT sort, so its position would no longer match the h5 block.
    if not np.array_equal(keep, x[:, :, names.index("j1_pt")] > 0):
        raise ValueError("a non-padding constituent has j1_pt == 0")
    offsets = pa.array(np.concatenate(([0], keep.sum(1).cumsum())).astype(np.int32))

    # x[:, :, f][keep] flattens in (jet, slot) order, which is the order the lists want.
    return {
        name: pa.ListArray.from_arrays(
            offsets, pa.array(x[:, :, f][keep].astype(np.float32))
        )
        for f, name in enumerate(names)
    }


def labels(jets: np.ndarray, path: Path) -> np.ndarray:
    """The class index of each jet, from the one-hot block of the ``jets`` table."""
    one_hot = jets[:, -N_LABEL_COLUMNS:-1]
    # argmax would silently label a flagless jet 0, where the legacy one-hot is all zeros.
    if not np.array_equal(one_hot.sum(1), np.ones(len(jets))):
        raise ValueError(f"{path.name}: a jet does not carry exactly one class flag")

    return one_hot.argmax(1).astype(np.int8)


def jets_table(path: Path) -> pa.Table:
    """One h5 file as one row per jet, provenance included."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        particle, jet = feature_names(handle)
        x = np.asarray(handle["jetConstituentList"])
        jets = np.asarray(handle["jets"])

    columns = jagged(x, particle)
    columns |= {n: pa.array(jets[:, i].astype(JET_DTYPE)) for i, n in enumerate(jet)}
    columns["label"] = pa.array(labels(jets, path))
    columns["source_file"] = pa.array([path.stem] * len(jets), type=pa.string())
    columns["source_row"] = pa.array(np.arange(len(jets), dtype=np.int32))

    return pa.table(columns)


def archive_table(raw_dir: Path, split: str) -> pa.Table:
    """One whole archive, its files concatenated in sorted-filename order.

    That order is what the split indices are drawn against, so it is part of what the
    record is rather than an implementation detail.
    """
    tables = [jets_table(path) for path in h5_files(raw_dir, split)]

    return pa.concat_tables(tables).combine_chunks()


def split_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The legacy train/validation split, drawn on row numbers rather than on the data.

    train_test_split permutes indices and slices them, so splitting arange(n) and taking
    those rows reproduces splitting the arrays themselves exactly.
    """
    train, val = train_test_split(
        np.arange(n), test_size=VAL_FRACTION, random_state=SPLIT_SEED
    )

    return train, val


def _decoded(values) -> list[str]:
    """h5py hands back bytes for these name datasets; older files hand back str."""
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def _md5(path: Path) -> str:
    """md5 of a file already on disk, read a megabyte at a time."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(BLOCK), b""):
            digest.update(block)

    return digest.hexdigest()
