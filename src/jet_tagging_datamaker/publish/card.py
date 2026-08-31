# The dataset card and the licence that ship inside the published mirror.
#
# Every number the card quotes is read back from the shards that were just written, and
# every claim about the data itself is one the Zenodo record already makes. Nothing here
# describes physics the mirror did not receive.

import textwrap

import numpy as np

from jet_tagging_datamaker.publish import export

# The mirror the card's examples load from.
REPO_ID = "fastmachinelearning/hls4ml_lhc_jets_150p"

# The datamaker commit that produced the shards. Stamp it before every export.
BUILT_AT_COMMIT = "6ab2f90"

ZENODO_DOI = "10.5281/zenodo.3602260"

# Width the card's prose is reflowed to, so a reader opening the raw markdown sees lines
# of an even length whatever the substituted counts came out to.
WRAP = 88

# The five one-hot columns of `jets`, in the order argmax turns them into the label.
CLASSES = ("gluon", "light quark", "W boson", "Z boson", "top quark")

# Columns of the mirror that are neither constituent nor jet-level features.
EXTRA_COLUMNS = ("label", "source_file", "source_row")

LICENCE = """Creative Commons Attribution 4.0 International (CC BY 4.0)

This dataset is a mirror of

    Pierini, Maurizio; Duarte, Javier; Tran, Nhan; Freytsis, Marat
    HLS4ML LHC Jet dataset (150 particles)
    Zenodo, 2020
    https://doi.org/10.5281/zenodo.3602260

which is published under the Creative Commons Attribution 4.0 International
licence. The mirror is published under the same licence.

You are free to share and adapt the material for any purpose, including
commercially, provided you give appropriate credit to the original authors,
link to the licence, and indicate whether changes were made. The rows here are
the rows of the original HDF5 files, reshaped; no values were recomputed.

Licence: https://creativecommons.org/licenses/by/4.0/
"""

CARD_HF = """# hls4ml LHC jet dataset (150 particles)

*A mirror of the Zenodo record [{doi}](https://doi.org/{doi}), reshaped to one row per jet.*

The data are simulated high transverse-momentum (~1 TeV) jets from proton-proton collisions at the Large Hadron Collider, labelled by what produced them: a light quark, a gluon, a W boson, a Z boson or a top quark.
Each jet is constituted of up to 150 constituents together with a set of jet-level observables, and the set was prepared for the hls4ml jet-tagging studies.
The mirror holds {total:,} jets.

Every value is the value of the original HDF5 files.
The jet images from the original data files are dropped.

## One row is one jet

| column | type | holds |
|---|---|---|
| the {n_constituent} `j1_*` columns | `list<{constituent_type}>` | one entry per constituent of that jet, so the lists of one row all have the same length |
| the {n_jet} `j_*` columns | `{jet_type}` | jet-level observables, one value per jet |
| `label` | `int8` | {label_map} |
| `source_file` | `string` | the HDF5 file the jet was read from, without its extension |
| `source_row` | `int32` | the jet's row within that file |

The constituent features, in the order the original files list them, are {constituent_columns}.
The jet-level features are {jet_columns}.
The label is the argmax over the five one-hot columns `j_g`, `j_q`, `j_w`, `j_z`, `j_t` that the original `jets` array ends with.
The one-hot columns and the always-zero `j_undef` beside them are not mirrored, but are replaced by `label`.

## Splits

`train` and `validation` come from the Zenodo *train* archive.
Its files are concatenated in sorted-filename order and cut with

```python
train_idx, validation_idx = train_test_split(np.arange(n), test_size=0.2, random_state=42)
```

`test` is the Zenodo *val* archive, in sorted-filename order.

| split | jets | {class_header} |
|---|---|{class_rule}
{split_table}

## Layout

```
data/train-NNNNN-of-NNNNN.parquet
data/validation-NNNNN-of-NNNNN.parquet
data/test-NNNNN-of-NNNNN.parquet
loader/            the pipeline described below
configs/           the Hydra tree that drives it
requirements.txt   what that pipeline needs
```

## Loading

The tables need only the `datasets` package:

```python
from datasets import load_dataset
jets = load_dataset("{repo}", split="train")
jets[0]["j1_pt"]    # the transverse momenta of that jet's constituents
```

This data record also contains a data processing pipeline that implements a few basic recommended processing steps.
It reads the parquet files, keeps the leading constituents by transverse momentum, normalises each feature by a scale fitted on the training split alone, caches the result as `.npy`, and hands back torch tensors:

```python
import sys
from huggingface_hub import snapshot_download
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
record = snapshot_download("{repo}", repo_type="dataset")
sys.path.insert(0, record)  # the configs name loader.*, so the record has to be importable
with initialize_config_dir(config_dir=record + "/configs", version_base=None):
    cfg = compose("config", overrides=["paths.root_dir=" + record, "data.nconstituents=32"])
data = instantiate(cfg.data)
data.prepare()
train = data.load("train")
```

The `train.x` object is a `(jets, 32, {n_constituent})` float32 tensor of normalised constituents and `train.y` a `(jets, {n_classes})` float32 one-hot tensor of the labels.
The `data.load("validation")` and the `data.load("test")` objects return the other two splits, normalised with the scales fitted on `train`.
The `data.nconstituents` is used to set the maximum number of constituents per jet; `0` or less keeps all 150.
Again, these are ordered by descending transverse momentum by the data loader.
Data that is processed in different ways using the record's dataloader are cached on disk in different ways; hence, there's no duplicate preprocessing.

The `prepare` method writes the caches under the `./cache` folder.
Please set the `HLS4ML_JETS_CACHE` environment to move this elsewhere.
Additionally, `prepare` reads the shards from `HLS4ML_JETS_ROOT` when the overrides above are left out.
Allow the cache a few times the record's size on disk.
The pipeline needs python 3.11 or newer with `numpy`, `pyarrow`, `torch`, `omegaconf` and `hydra-core`, which `pip install -r requirements.txt` at the record's root installs.
**If you only want the raw tables, you need none of the dataloader functionality. Just call `load_dataset`.**

## Caveats

**Constituents are stored in descending transverse momentum, the order they sit in the original files.**
Every jet of all three splits was checked: no list holds a constituent with a higher `j1_pt` than the one before it.
The loader in this record still stable-sorts each jet by descending `j1_pt` before it truncates to `nconstituents` or pads up to it, because the reference pipeline does.
The sort of the dataloader acts as a double-check that constiutents are stored in descending order.
Truncating the lists as they come therefore keeps the same leading constituents.

**The lists have no padding or fixed length.**
A jet holds as many entries as it has constituents, up to 150.
The original files pad every jet to 150 slots with rows of zeros; those slots are dropped here and rebuilt by the loader.

**The jet images are not mirrored.**
The original files carry `jetImage`, `jetImageECAL` and `jetImageHCAL`, three 100x100 arrays per jet, which dwarf everything else.
Take them from Zenodo if you need them.

## Provenance

Simulated proton-proton collisions at the LHC, produced for the hls4ml jet-tagging studies and published on Zenodo in 2020 by Maurizio Pierini, Javier Duarte, Nhan Tran and Marat Freytsis.
This mirror was built from the two archives of that record, `hls4ml_LHCjet_150p_train.tar.gz` and `hls4ml_LHCjet_150p_val.tar.gz`, by the code at https://github.com/bb511/jet_tagging_datamaker.
The data were produced at commit [`{built_at_commit}`](https://github.com/bb511/jet_tagging_datamaker/tree/{built_at_commit}) of that repository.

## Citation

Cite the Zenodo record this dataset mirrors.

```bibtex
@dataset{{pierini_hls4ml_lhc_jets_150p_2020,
  author    = {{Pierini, Maurizio and Duarte, Javier and Tran, Nhan and Freytsis, Marat}},
  title     = {{HLS4ML LHC Jet dataset (150 particles)}},
  year      = {{2020}},
  publisher = {{Zenodo}},
  doi       = {{{doi}}},
  url       = {{https://doi.org/{doi}}}
}}
```

## Licence

CC BY 4.0, the licence of the original record.
See `LICENSE`.
Use it for anything, including commercially, as long as you credit Pierini, Duarte, Tran and Freytsis, link the licence at https://creativecommons.org/licenses/by/4.0/, and say what you changed.

## Contact

Questions and problems are welcome as a discussion on this dataset's page, or as an issue on the repository that produced it: https://github.com/bb511/jet_tagging_datamaker.
"""


def render(counts: dict, particle_names: list[str], jet_names: list[str]) -> str:
    """Fill the card from the shards just written and the column names they carry."""
    return _rewrap(
        CARD_HF.format(
            repo=REPO_ID,
            built_at_commit=BUILT_AT_COMMIT,
            doi=ZENODO_DOI,
            total=sum(split["rows"] for split in counts.values()),
            n_constituent=len(particle_names),
            n_jet=len(jet_names),
            n_classes=len(CLASSES),
            constituent_type="float32",
            jet_type=np.dtype(export.JET_DTYPE).name,
            constituent_columns=_listed(particle_names),
            jet_columns=_listed(jet_names),
            label_map=", ".join(f"{i} {name}" for i, name in enumerate(CLASSES)),
            class_header=" | ".join(CLASSES),
            class_rule="---|" * len(CLASSES),
            split_table=split_table(counts),
        )
    )


def split_table(counts: dict) -> str:
    """One row per split: its jets and how they fall over the five classes."""
    return "\n".join(
        f"| `{split}` | {counts[split]['rows']:,} | "
        + " | ".join(f"{counts[split]['labels'].get(i, 0):,}" for i in range(len(CLASSES)))
        + " |"
        for split in counts
    )


def _listed(names: list[str]) -> str:
    """Join column names as prose, with a final 'and' rather than a bullet list."""
    quoted = [f"`{name}`" for name in names]

    return ", ".join(quoted[:-1]) + f" and {quoted[-1]}" if len(quoted) > 1 else quoted[0]


def _rewrap(text: str) -> str:
    """Reflow the prose, leaving headings, tables and fenced blocks as they are."""
    return "\n\n".join(
        block if block.startswith(("#", "|", "```")) else _fill(block)
        for block in text.split("\n\n")
    )


def _fill(block: str) -> str:
    """Wrap one paragraph without splitting identifiers such as ``load_dataset``."""
    return textwrap.fill(
        " ".join(block.split()),
        width=WRAP,
        break_long_words=False,
        break_on_hyphens=False,
    )
