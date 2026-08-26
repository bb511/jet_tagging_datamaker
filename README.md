[![Python: version](https://img.shields.io/badge/python-3.11-blue?style=flat-square&logo=python)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.3602260-blue?style=flat-square&logo=doi)](https://doi.org/10.5281/zenodo.3602260)
[![Dataset on HF](https://img.shields.io/badge/Dataset-hls4ml__lhc__jets__150p-blue?style=flat-square&logo=huggingface)](https://huggingface.co/datasets/podagiu/hls4ml_lhc_jets_150p)

# hls4ml LHC jets datamaker

This repository turns the Zenodo release of the hls4ml LHC jet dataset (150 particles),
[10.5281/zenodo.3602260](https://doi.org/10.5281/zenodo.3602260), into a HuggingFace
record: `podagiu/hls4ml_lhc_jets_150p`.

The source is two tarballs of HDF5 files. Each file holds a padded
`(jets, 150, 16)` block of constituents, a `(jets, 59)` block of jet-level observables
whose last six columns are the one-hot class labels, and three 100x100 images per jet.
This code drops the padding, drops the images, drops the label columns in favour of one
`int8`, and writes the result as parquet shards with a dataset card, a loading pipeline
and the Hydra configs that drive it.

Nothing is recomputed: the values in the mirror are the values in the HDF5 files.

## The record

```
data/{train,validation,test}-NNNNN-of-NNNNN.parquet   one row per jet, snappy, ~500 MB each
loader/            the pipeline the jet-tagging studies run on this data
configs/           the Hydra tree that drives it
requirements.txt   what that pipeline needs
README.md          the dataset card, whose front matter makes the record load without a script
LICENSE            CC BY 4.0, the licence of the original record
```

A row holds the 16 `j1_*` constituent features as `list<float32>`, the 53 `j_*` jet-level
features as `float32`, `label` (0 gluon, 1 light quark, 2 W, 3 Z, 4 top), and the pair
`source_file` / `source_row` saying where the jet came from.

`train` and `validation` are the Zenodo *train* archive, its files concatenated in
sorted-filename order and cut with
`train_test_split(np.arange(n), test_size=0.2, random_state=42)`, stored in the order
those indices come back in. `test` is the Zenodo *val* archive in file order. That is the
split `jet_tagging_gdl` draws at every run, drawn once here instead.

## Install

The export needs `numpy`, `h5py`, `pyarrow` and `scikit-learn`. In an environment that
has them:

```
pip install -e . --no-deps
```

Otherwise `poetry install` builds the environment, and `poetry install --with dev` adds
what the tests need (`pytest`, and `torch` / `hydra-core` / `omegaconf` for the pipeline
the record ships).

## Usage

```
./scripts/publish/export_hf --raw /data/deodagiu/hls4ml_jets_data/raw \
                            --out /data/deodagiu/hls4ml_jets_data/hf
```

The script downloads both archives from Zenodo (checking their md5), unpacks them into
`--raw/{train,val}`, reads every h5 file, writes the shards into `--out/data`, then
renders the card and copies the loader and the configs in. `--skip-download` takes the h5
files under `--raw` as they are; `--card-only` rewrites the card and the assets over
shards that are already there, without reading a single h5 file.

Expect about 3.9 GB of download and about 3 GB of parquet out; snappy barely compresses floats.

### Tests

```
pytest
```

They build a handful of synthetic h5 files and run the whole export over them, so they
need neither the network nor the real data.

## Upload

The mirror is uploaded by hand, once it has been checked:

```
hf auth whoami
hf repos create podagiu/hls4ml_lhc_jets_150p --repo-type dataset
hf upload podagiu/hls4ml_lhc_jets_150p /data/deodagiu/hls4ml_jets_data/hf . \
    --repo-type dataset --exclude "**/__pycache__/*" \
    --commit-message "hls4ml LHC jets (150p): one row per jet, train/validation/test, shipped loader"
```

## Verified facts

Checked on every h5 file of both archives (62 train, 26 val) before the export was run,
with a read-only pass over `jetConstituentList`, `jets` and the two name datasets.

| fact | value |
|---|---|
| jets per archive | 620,000 train, 260,000 val (10,000 per file) |
| `jetConstituentList` | `(jets, 150, 16)` float64 |
| `jets` | `(jets, 59)` float64 |
| float32-representable | yes, both arrays, bit for bit |
| `particleFeatureNames` | 17 names, `j1_pdgid` last, for 16 columns; identical in every file |
| `jetFeatureNames` | 59 names, ending `j_g j_q j_w j_z j_t j_undef`; identical in every file |
| one class flag per jet | yes, and `j_undef` is always zero |
| padding | an all-zero suffix in every jet, equal to `j1_pt == 0`; no real constituent has `j1_pt == 0` |
| constituents per jet | 5 to 150, mean 49.4 in both archives |
| class counts, train archive | g 124,848, q 120,211, W 124,937, Z 124,654, t 125,350 |
| class counts, val archive | g 52,404, q 50,468, W 52,235, Z 52,298, t 52,595 |

The dtype finding is what sets `export.JET_DTYPE`: the h5 arrays are float64, every value
in them is float32-representable, so the mirror stores float32 and a pipeline reading it
back in float64 sees the original values bit for bit.
