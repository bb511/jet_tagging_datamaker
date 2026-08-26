# The Zenodo side: names, schema, padding, split, order.

import h5py
import numpy as np
import pyarrow as pa
import pytest
from conftest import CONSTITUENT_NAMES, FILES, JET_NAMES, LABEL_NAMES, N_JETS
from sklearn.model_selection import train_test_split

from jet_tagging_datamaker.publish import export

JET_COLUMNS = list(JET_NAMES[: -len(LABEL_NAMES)])


def test_feature_names_drop_the_names_without_columns(raw_dir):
    with h5py.File(export.h5_files(raw_dir, "train")[0], "r") as handle:
        particle, jet = export.feature_names(handle)

    assert particle == list(CONSTITUENT_NAMES)
    assert jet == JET_COLUMNS
    assert not set(particle) & set(export.DROPPED_PARTICLE_NAMES)
    assert not set(jet) & set(LABEL_NAMES)


def test_jets_table_schema(raw_dir):
    schema = export.jets_table(export.h5_files(raw_dir, "train")[0]).schema

    assert schema.names == [
        *CONSTITUENT_NAMES, *JET_COLUMNS, "label", "source_file", "source_row"
    ]
    assert {schema.field(n).type for n in CONSTITUENT_NAMES} == {pa.list_(pa.float32())}
    assert {schema.field(n).type for n in JET_COLUMNS} == {pa.float32()}
    assert schema.field("label").type == pa.int8()
    assert schema.field("source_file").type == pa.string()
    assert schema.field("source_row").type == pa.int32()


def test_label_is_the_argmax_of_the_one_hot_block(raw_dir):
    path = export.h5_files(raw_dir, "train")[0]
    with h5py.File(path, "r") as handle:
        jets = np.asarray(handle["jets"])

    labels = export.jets_table(path)["label"].to_numpy()

    assert labels.tolist() == jets[:, -6:-1].argmax(1).tolist()


def test_provenance_columns_name_the_row_they_came_from(raw_dir):
    path = export.h5_files(raw_dir, "train")[0]
    table = export.jets_table(path)

    assert table["source_file"].to_pylist() == [path.stem] * table.num_rows
    assert table["source_row"].to_numpy().tolist() == list(range(table.num_rows))


def test_jagged_strips_exactly_the_padded_slots(raw_dir):
    with h5py.File(export.h5_files(raw_dir, "train")[0], "r") as handle:
        x = np.asarray(handle["jetConstituentList"])
    keep = np.any(x != 0, axis=2)

    columns = export.jagged(x, list(CONSTITUENT_NAMES))

    for feature, name in enumerate(CONSTITUENT_NAMES):
        lengths = [len(entry) for entry in columns[name].to_pylist()]
        assert lengths == keep.sum(1).tolist()
        np.testing.assert_array_equal(
            columns[name].flatten().to_numpy(),
            x[:, :, feature][keep].astype(np.float32),
        )


def test_split_indices_reproduce_the_array_split():
    n = 137
    x = np.arange(3 * n).reshape(n, 3)
    y = np.arange(n)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=export.VAL_FRACTION, random_state=export.SPLIT_SEED
    )

    train_idx, val_idx = export.split_indices(n)

    np.testing.assert_array_equal(x[train_idx], x_train)
    np.testing.assert_array_equal(x[val_idx], x_val)
    np.testing.assert_array_equal(y[train_idx], y_train)
    np.testing.assert_array_equal(y[val_idx], y_val)


def test_archive_table_concatenates_in_sorted_file_order(raw_dir):
    files = export.h5_files(raw_dir, "train")
    table = export.archive_table(raw_dir, "train")

    assert [path.name for path in files] == sorted(path.name for path in files)
    assert table.num_rows == N_JETS * len(files)
    assert table["source_file"].to_pylist() == [
        path.stem for path in files for _ in range(N_JETS)
    ]
    assert table["source_row"].to_numpy().tolist() == list(range(N_JETS)) * len(files)


def test_h5_files_raises_on_an_empty_split(tmp_path):
    with pytest.raises(FileNotFoundError):
        export.h5_files(tmp_path, "train")


def test_fetch_skips_the_archives_already_unpacked(raw_dir, monkeypatch):
    def unreachable(*args, **kwargs):
        raise AssertionError("fetch went to the network for files already on disk")

    monkeypatch.setattr(export, "download", unreachable)

    export.fetch(raw_dir)

    assert {split for split, _, _ in FILES} == set(export.ARCHIVES)
