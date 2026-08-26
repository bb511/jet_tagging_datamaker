[![Python: version](https://img.shields.io/badge/python-3.11-blue?style=flat-square&logo=python)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.3602260-blue?style=flat-square&logo=doi)](https://doi.org/10.5281/zenodo.3602260)
[![Dataset on HF](https://img.shields.io/badge/Dataset-hls4ml__lhc__jets__150p-blue?style=flat-square&logo=huggingface)](https://huggingface.co/datasets/fastmachinelearning/hls4ml_lhc_jets_150p)

# hls4ml LHC jets datamaker

This repository converts the Zenodo release of the hls4ml LHC jet dataset (150 particles),
[10.5281/zenodo.3602260](https://doi.org/10.5281/zenodo.3602260), into a HuggingFace
record: `fastmachinelearning/hls4ml_lhc_jets_150p`.

The Zenodo source serves two tarballs of HDF5 files.
Each file contains a padded `(jets, 150, 16)` block of constituents, a `(jets, 59)`
block of jet-level observables whose last six columns are the one-hot class labels,
and three 100x100 images per jet.
In This code, we apply the following operations to the `(jets, 150, 16)` subset only:
drop the padding, drop the label columns in favour of one `int8`,
and writes the result as parquet shards with a dataset card.
We also add a loading pipeline with differnt Hydra configs that prescribe different kinds of preprocessing.

## Data Record

```
data/{train,validation,test}-NNNNN-of-NNNNN.parquet   one row per jet, snappy, ~500 MB each
loader/            the pipeline the jet-tagging studies run on this data
configs/           the Hydra tree that drives it
requirements.txt   what that pipeline needs
README.md          the dataset card, whose front matter makes the record load without a script
LICENSE            CC BY 4.0, the licence of the original record
```

One row in the parquest files is comprised of the 16 `j1_*` constituent features as `list<float32>`,
the 53 `j_*` jet-level features as `float32`, `label` (0 gluon, 1 light quark, 2 W, 3 Z, 4 top),
and the pair `source_file` / `source_row` documenting where the jet came from.
`train` and `validation` in the HF record belong to the Zenodo *train* archive.
The files in this archive are concatenated in sorted-filename order and cut with `train_test_split(np.arange(n), test_size=0.2, random_state=42)`,
then stored in the order those indices come back in.
`test` is the Zenodo *val* archive in file order.

## Install

The project uses `poetry`.
Run `poetry install` to build the env, and `poetry install --with dev` to add additional packages
that are used in testing.
Otherwise, `pyproject.toml` contains

## Usage

```
./scripts/publish/export_hf --raw [raw_folder_path] \
                            --out [hf_folder_path]
```

The script downloads both archives from Zenodo (checking their md5), unpacks them into
`--raw/{train,val}`, reads every h5 file, writes the shards into `--out/data`, then
renders the card and copies the loader and the configs in.
`--skip-download` takes the h5
files under `--raw` as they are; `--card-only` rewrites the card and the assets over
shards that are already there, without reading a single h5 file.

Expect about 3.9 GB of download and about 3 GB of parquet out.

### Tests

```
pytest
```

They build a handful of synthetic h5 files and run the whole export over them.
The tests do not need neither the real data.
