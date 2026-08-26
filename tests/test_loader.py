"""The loader that ships inside the record, against a mini mirror written here.

The mirror follows the published schema (16 jagged constituent columns, a couple of jet
columns, label, provenance) but holds a handful of jets, so the whole record pipeline
runs offline in a fraction of a second.
"""

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

ASSETS = Path(__file__).resolve().parents[1] / "src/jet_tagging_datamaker/publish/assets"

# The consumer recipe: the record's root on sys.path, then loader.* imports.
sys.path.insert(0, str(ASSETS))

from loader import physics, reading  # noqa: E402

FEATURES = [
    "j1_px", "j1_py", "j1_pz", "j1_e", "j1_erel", "j1_pt", "j1_ptrel", "j1_eta",
    "j1_etarel", "j1_etarot", "j1_phi", "j1_phirel", "j1_phirot", "j1_deltaR",
    "j1_costheta", "j1_costhetarel",
]
JET_COLUMNS = ["j_pt", "j_mass"]
WIDTH = 150
N_CLASSES = 5
NCONST = 4

# Multiplicities per shard: empty jets, jets below NCONST and jets above it.
MULTIPLICITIES = {
    "train": [[0, 3, 7, 1], [5, 2, 0, 9, 4]],
    "validation": [[6, 1, 3]],
    "test": [[2, 8, 0, 5]],
}

TIE_PT = np.float32(5.0)  # larger than any generated pT, so the ties lead the jet


class Mirror:
    """The written record plus the dense arrays it was written from."""

    def __init__(self, root: Path, x: dict, y: dict):
        self.root = root
        self.data_dir = root / "data"
        self.x = x
        self.y = y


@pytest.fixture
def mirror(tmp_path) -> Mirror:
    rng = np.random.default_rng(0)
    root = tmp_path / "record"
    (root / "data").mkdir(parents=True)
    x, y = {}, {}
    for split, shards in MULTIPLICITIES.items():
        blocks = [_shard(rng, counts) for counts in shards]
        for i, (counts, (xi, yi)) in enumerate(zip(shards, blocks)):
            path = root / "data" / f"{split}-{i:05d}-of-{len(shards):05d}.parquet"
            pq.write_table(_table(xi, np.array(counts), yi, path.stem), path, compression="snappy")
        x[split] = np.concatenate([xi for xi, _ in blocks])
        y[split] = np.concatenate([yi for _, yi in blocks])

    return Mirror(root, x, y)


def _shard(rng, counts: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Dense (jets, 150, 16) float64 holding float32 values, and the labels."""
    x = np.zeros((len(counts), WIDTH, len(FEATURES)), np.float64)
    positive = [FEATURES.index(name) for name in sorted(physics.POSITIVE_FEATURES)]
    for i, m in enumerate(counts):
        block = rng.normal(size=(m, len(FEATURES)))
        block[:, positive] = np.abs(block[:, positive]) % 3.0 + 0.5
        x[i, :m] = block.astype(np.float32)
        if m >= 3:  # equal pT within a jet, so the stable sort has ties to keep in order
            x[i, [0, 2, m - 1], FEATURES.index("j1_pt")] = TIE_PT

    return x, rng.integers(0, N_CLASSES, len(counts)).astype(np.int8)


def _table(x: np.ndarray, counts: np.ndarray, labels: np.ndarray, stem: str) -> pa.Table:
    """The published row layout: one list entry per real constituent, in slot order."""
    keep = np.arange(WIDTH)[None, :] < counts[:, None]
    offsets = pa.array(np.concatenate([[0], np.cumsum(counts)]).astype(np.int32))
    columns = {
        name: pa.ListArray.from_arrays(offsets, pa.array(x[:, :, f][keep].astype(np.float32)))
        for f, name in enumerate(FEATURES)
    }
    for j, name in enumerate(JET_COLUMNS):
        columns[name] = pa.array((counts + j).astype(np.float32))
    columns["label"] = pa.array(labels, pa.int8())
    columns["source_file"] = pa.array([stem] * len(counts), pa.string())
    columns["source_row"] = pa.array(np.arange(len(counts), dtype=np.int32))

    return pa.table(columns)


def _shift_offsets(table: pa.Table, pad: int = 3) -> pa.Table:
    """Rewrite the list columns so their offsets start past a run of junk values.

    Slicing and combining chunks both produce arrays like this, and reading ``.values``
    or trusting ``offsets[0] == 0`` would then pick the junk up.
    """
    columns = {}
    for name in table.column_names:
        column = table[name].combine_chunks()
        if name not in FEATURES:
            columns[name] = column
            continue
        array = column.chunk(0) if isinstance(column, pa.ChunkedArray) else column
        offsets = pa.array((np.asarray(array.offsets) + pad).astype(np.int32))
        junk = pa.array(np.full(pad, -99.0, np.float32))
        values = pa.concat_arrays([junk, array.flatten()])
        columns[name] = pa.ListArray.from_arrays(offsets, values)

    return pa.table(columns)


def _read(mirror: Mirror, split: str) -> list[pa.Table]:
    return [pq.read_table(path) for path in reading.shards(mirror.data_dir, split)]


# ----------------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------------
def test_shards_are_found_in_name_order(mirror):
    found = reading.shards(mirror.data_dir, "train")

    assert [path.name for path in found] == [
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
    ]


def test_shards_of_a_missing_split_raise(mirror):
    with pytest.raises(FileNotFoundError):
        reading.shards(mirror.data_dir, "nosuchsplit")


@pytest.mark.parametrize("split", list(MULTIPLICITIES))
def test_dense_rebuilds_the_source_arrays(mirror, split):
    rebuilt = np.concatenate([reading.dense(t, FEATURES, WIDTH) for t in _read(mirror, split)])

    assert rebuilt.dtype == np.float64
    assert np.array_equal(rebuilt, mirror.x[split])


def test_dense_on_a_table_with_several_chunks(mirror):
    table = pa.concat_tables(_read(mirror, "train"))

    assert table[FEATURES[0]].num_chunks == 2
    assert np.array_equal(reading.dense(table, FEATURES, WIDTH), mirror.x["train"])


def test_dense_on_a_sliced_table(mirror):
    table = pa.concat_tables(_read(mirror, "train")).slice(2, 4)

    assert np.array_equal(reading.dense(table, FEATURES, WIDTH), mirror.x["train"][2:6])


def test_dense_ignores_values_the_offsets_do_not_reach(mirror):
    table = _shift_offsets(pa.concat_tables(_read(mirror, "train")))

    assert np.array_equal(reading.dense(table, FEATURES, WIDTH), mirror.x["train"])


def test_dense_rejects_jets_wider_than_the_target(mirror):
    table = _read(mirror, "train")[1]  # holds a jet of 9 constituents

    with pytest.raises(ValueError, match="9 constituents"):
        reading.dense(table, FEATURES, 8)


def test_labels_are_one_hot(mirror):
    one_hot = np.concatenate([reading.labels(t, N_CLASSES) for t in _read(mirror, "test")])

    assert one_hot.dtype == np.float32
    assert np.array_equal(one_hot.argmax(axis=1), mirror.y["test"])
    assert np.array_equal(one_hot.sum(axis=1), np.ones(len(one_hot), np.float32))


def test_read_split_applies_the_transform_shard_by_shard(mirror):
    seen = []

    def transform(x):
        seen.append(x.shape)
        return x[:, :2, :]

    x, y = reading.read_split(mirror.data_dir, "train", FEATURES, WIDTH, N_CLASSES, transform)

    assert seen == [(4, WIDTH, 16), (5, WIDTH, 16)]
    assert x.shape == (9, 2, 16) and y.shape == (9, N_CLASSES)


# ----------------------------------------------------------------------------------
# physics
# ----------------------------------------------------------------------------------
def _tiny() -> np.ndarray:
    """One jet, pT in column 0 and a tag in column 1 that identifies each slot."""
    return np.array([[[3.0, 0.0], [5.0, 1.0], [5.0, 2.0], [1.0, 3.0], [0.0, 4.0]]])


def test_sort_is_descending_and_keeps_ties_in_on_disk_order():
    sorted_x = physics.restrict_and_sort_by_pt(_tiny(), 0, 5)

    assert sorted_x[0, :, 1].tolist() == [1.0, 2.0, 0.0, 3.0, 4.0]


def test_sort_truncates_to_the_leading_constituents():
    assert physics.restrict_and_sort_by_pt(_tiny(), 0, 3)[0, :, 1].tolist() == [1.0, 2.0, 0.0]


def test_sort_pads_with_zeros_when_asked_for_more_than_there_are():
    padded = physics.restrict_and_sort_by_pt(_tiny(), 0, 7)

    assert padded.shape == (1, 7, 2)
    assert np.array_equal(padded[0, 5:], np.zeros((2, 2)))


@pytest.mark.parametrize("nconstituents", [0, -1, None])
def test_sort_keeps_every_slot_when_no_count_is_asked_for(nconstituents):
    assert physics.restrict_and_sort_by_pt(_tiny(), 0, nconstituents).shape == (1, 5, 2)


def test_fit_takes_half_the_range_for_centered_and_the_maximum_for_positive():
    x = np.array([[[-1.0, 2.0], [3.0, 4.0]]])  # j1_eta in [-1, 3], j1_pt in [2, 4]

    params = physics.fit_physics_norm(x, ["j1_eta", "j1_pt"])

    assert params == {"j1_eta": ("centered", 2.0), "j1_pt": ("positive", 4.0)}


def test_fit_rejects_a_feature_it_cannot_categorise():
    with pytest.raises(ValueError, match="not categorised"):
        physics.fit_physics_norm(np.zeros((1, 1, 1)), ["j1_nonsense"])


def test_apply_divides_by_the_scale_and_leaves_padding_at_zero():
    x = np.array([[[4.0, 8.0], [0.0, 0.0]]])
    params = {"j1_eta": ("centered", 2.0), "j1_pt": ("positive", 4.0)}

    normalised = physics.apply_physics_norm(x, ["j1_eta", "j1_pt"], params, eps=0.0)

    assert normalised.tolist() == [[[2.0, 2.0], [0.0, 0.0]]]
    assert x.tolist() == [[[4.0, 8.0], [0.0, 0.0]]]  # the input is left alone


def test_apply_uses_eps_to_survive_a_zero_scale():
    normalised = physics.apply_physics_norm(
        np.array([[[1.0]]]), ["j1_pt"], {"j1_pt": ("positive", 0.0)}
    )

    assert normalised[0, 0, 0] == pytest.approx(1e8)


def test_norm_params_round_trip_in_the_legacy_npz_layout(tmp_path):
    params = {"j1_eta": ("centered", 0.5), "j1_pt": ("positive", 2.0)}
    path = tmp_path / "norm_params.npz"

    physics.save_norm_params(path, params)

    npz = np.load(path, allow_pickle=False)
    assert npz["feature_names"].dtype == np.dtype("U64")
    assert npz["kinds"].dtype == np.dtype("U16")
    assert npz["scales"].dtype == np.float32
    assert physics.load_norm_params(path) == params


# ----------------------------------------------------------------------------------
# datamodule, through the recipe the dataset card gives
# ----------------------------------------------------------------------------------
def build(root: Path, cache: Path, nconstituents: int = NCONST):
    """Compose the record's own config and instantiate what it names."""
    with initialize_config_dir(config_dir=str(ASSETS / "configs"), version_base=None):
        cfg = compose(
            "config",
            overrides=[
                f"paths.root_dir={root}",
                f"paths.base_data_dir={cache}",
                f"data.nconstituents={nconstituents}",
            ],
        )

    return instantiate(cfg.data)


def reference(mirror: Mirror, nconstituents: int = NCONST) -> tuple[dict, dict]:
    """What the study's own functions make of the fixture's dense arrays."""
    pt_idx = FEATURES.index("j1_pt")
    x = {s: physics.restrict_and_sort_by_pt(v, pt_idx, nconstituents) for s, v in mirror.x.items()}
    params = physics.fit_physics_norm(x["train"], FEATURES)

    return {
        split: physics.apply_physics_norm(v, FEATURES, params).astype(np.float32)
        for split, v in x.items()
    }, params


def _assert_params_match(actual: dict, expected: dict) -> None:
    """The npz keeps the scales as float32, so the fitted float64 ones only round trip."""
    assert list(actual) == list(expected)
    for name, (kind, scale) in expected.items():
        assert actual[name] == (kind, pytest.approx(scale, rel=1e-6))


def test_config_names_the_record_directories_and_the_16_features(mirror, tmp_path):
    data = build(mirror.root, tmp_path / "cache")

    assert Path(data.data_dir) == mirror.data_dir
    assert data.feature_names == FEATURES
    assert data.pt_idx == FEATURES.index("j1_pt")
    assert data.cache_folder == tmp_path / "cache" / f"nconst_{NCONST}"
    assert data.max_constituents == WIDTH and data.n_classes == N_CLASSES


def test_prepare_writes_the_seven_cache_files(mirror, tmp_path):
    data = build(mirror.root, tmp_path / "cache")
    data.prepare()

    assert sorted(p.name for p in data.cache_folder.iterdir()) == [
        "norm_params.npz", "x_test.npy", "x_train.npy", "x_validation.npy",
        "y_test.npy", "y_train.npy", "y_validation.npy",
    ]


@pytest.mark.parametrize("split", list(MULTIPLICITIES))
def test_loaded_tensors_match_the_study_functions(mirror, tmp_path, split):
    data = build(mirror.root, tmp_path / "cache")
    data.prepare()
    expected_x, expected_params = reference(mirror)

    tensors = data.load(split)

    assert tensors.x.dtype == tensors.y.dtype == torch.float32
    assert np.array_equal(tensors.x.numpy(), expected_x[split])
    assert np.array_equal(tensors.y.numpy().argmax(axis=1), mirror.y[split])
    _assert_params_match(data.norm_params, expected_params)


def test_saved_names_are_the_16_features_in_config_order(mirror, tmp_path):
    data = build(mirror.root, tmp_path / "cache")
    data.prepare()

    npz = np.load(data.cache_folder / "norm_params.npz", allow_pickle=False)

    assert list(npz["feature_names"]) == FEATURES


def test_prepare_is_a_no_op_once_the_cache_is_there(mirror, tmp_path):
    data = build(mirror.root, tmp_path / "cache")
    data.prepare()
    before = {p.name: p.stat().st_mtime_ns for p in data.cache_folder.iterdir()}

    data.prepare()

    assert {p.name: p.stat().st_mtime_ns for p in data.cache_folder.iterdir()} == before


def test_no_constituent_count_keeps_the_full_width(mirror, tmp_path):
    data = build(mirror.root, tmp_path / "cache", nconstituents=0)
    data.prepare()
    expected_x, _ = reference(mirror, nconstituents=0)

    assert data.cache_folder == tmp_path / "cache" / "full"
    assert data.load("train").x.shape == (9, WIDTH, 16)
    assert np.array_equal(data.load("train").x.numpy(), expected_x["train"])


def test_norm_params_are_absent_before_prepare(mirror, tmp_path):
    assert build(mirror.root, tmp_path / "cache").norm_params is None


def test_load_rejects_an_unknown_split(mirror, tmp_path):
    data = build(mirror.root, tmp_path / "cache")

    with pytest.raises(ValueError, match="Unknown split"):
        data.load("val")
