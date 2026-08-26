# Reading the published shards back into the padded arrays the study's pipeline works on.
#
# One row of the record is one jet and its constituent columns are jagged: only the real
# constituents are stored, in the order they had on disk. Padding them back out to a
# fixed width is what the study's normalisation and its models expect.

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def shards(data_dir, split: str) -> list[Path]:
    """The shards of one split, in the order the record numbers them."""
    found = sorted(Path(data_dir).glob(f"{split}-*.parquet"))
    if not found:
        raise FileNotFoundError(f"No {split} shards under {data_dir}")

    return found


def dense(table, names, width: int) -> np.ndarray:
    """Rebuild the (jets, width, features) array, zero-padding the empty slots.

    float64 throughout: the study fits its normalisation on the h5 arrays, which numpy
    reads as float64, so anything narrower here would shift the scales.
    """
    # The 16 columns of a row are one jet's constituents, so they share their offsets and
    # the first column places the values of all of them.
    row, slot, n = _positions(_list_array(table[names[0]]), width)
    out = np.zeros((n, width, len(names)), np.float64)
    for f, name in enumerate(names):
        out[row, slot, f] = np.asarray(_list_array(table[name]).flatten())

    return out


def labels(table, n_classes: int) -> np.ndarray:
    """The label column as one-hot rows, the shape the models are trained against."""
    return np.eye(n_classes, dtype=np.float32)[np.asarray(table["label"]).astype(np.intp)]


def read_split(
    data_dir, split: str, names, width: int, n_classes: int, transform
) -> tuple[np.ndarray, np.ndarray]:
    """Every shard of one split, transformed shard by shard and concatenated.

    *transform* is applied before the concatenation, so peak memory holds one dense shard
    rather than the whole split at full width.
    """
    x, y = [], []
    for shard in shards(data_dir, split):
        table = pq.read_table(shard, columns=[*names, "label"])
        x.append(transform(dense(table, names, width)))
        y.append(labels(table, n_classes))

    return np.concatenate(x), np.concatenate(y)


def _positions(column, width: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Row and slot of every constituent within the flattened value array.

    The offsets of a sliced or combined list array need not start at zero and its
    ``.values`` may hold entries the array itself does not own, so the slots are counted
    from ``offsets[0]`` and the values come from ``.flatten()``, which respects the slice.
    """
    offsets = np.asarray(column.offsets).astype(np.int64)
    counts = np.diff(offsets)
    if counts.size and counts.max() > width:
        raise ValueError(
            f"A jet carries {counts.max()} constituents, more than the width {width}"
        )

    starts = offsets[:-1] - offsets[0]
    row = np.repeat(np.arange(len(counts)), counts)
    slot = np.arange(counts.sum()) - np.repeat(starts, counts)

    return row, slot, len(counts)


def _list_array(column) -> pa.ListArray:
    """One contiguous list array, whether the table handed over chunks or an array."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()

    return column.chunk(0) if isinstance(column, pa.ChunkedArray) else column
