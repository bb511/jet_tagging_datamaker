# A miniature stand-in for the unpacked Zenodo archives.
#
# Three h5 files with the same datasets and the same name conventions as the real ones,
# small enough that the whole export runs over them in a second and offline.

import h5py
import numpy as np
import pytest

# The 16 columns of jetConstituentList, in the order the real files name them.
CONSTITUENT_NAMES = (
    "j1_px", "j1_py", "j1_pz", "j1_e", "j1_erel", "j1_pt", "j1_ptrel", "j1_eta",
    "j1_etarel", "j1_etarot", "j1_phi", "j1_phirel", "j1_phirot", "j1_deltaR",
    "j1_costheta", "j1_costhetarel",
)

# particleFeatureNames carries one more name than there are columns.
PARTICLE_NAMES = CONSTITUENT_NAMES + ("j1_pdgid",)

LABEL_NAMES = ("j_g", "j_q", "j_w", "j_z", "j_t", "j_undef")

JET_NAMES = tuple(f"j_feat{i:02d}" for i in range(53)) + LABEL_NAMES

MAX_CONSTITUENTS = 150

PT_INDEX = CONSTITUENT_NAMES.index("j1_pt")

# One entry per file: where it goes, what it is called, and how many real constituents
# each of its six jets has. Every file holds an empty jet and a full one, the two edges
# the padding mask has to survive.
FILES = (
    ("train", "jetImage_0_150p_0_6", (0, 1, 3, 7, 150, 12)),
    ("train", "jetImage_0_150p_6_12", (5, 2, 150, 0, 9, 4)),
    ("val", "jetImage_1_150p_0_6", (11, 150, 0, 1, 6, 3)),
)

N_JETS = 6


@pytest.fixture
def raw_dir(tmp_path):
    """The three files, laid out as the archives unpack: raw/train and raw/val."""
    raw = tmp_path / "raw"
    rng = np.random.default_rng(0)
    for split, name, multiplicities in FILES:
        (raw / split).mkdir(parents=True, exist_ok=True)
        _write(raw / split / f"{name}.h5", rng, multiplicities)

    return raw


def _write(path, rng, multiplicities) -> None:
    """One h5 file, images included so the export can be seen ignoring them."""
    with h5py.File(path, "w") as handle:
        handle["jetConstituentList"] = _constituents(rng, multiplicities)
        handle["jets"] = _jets(rng, len(multiplicities))
        handle["particleFeatureNames"] = np.array([n.encode() for n in PARTICLE_NAMES])
        handle["jetFeatureNames"] = np.array([n.encode() for n in JET_NAMES])
        handle["jetImage"] = rng.uniform(0, 1, (len(multiplicities), 2, 2))


def _constituents(rng, multiplicities) -> np.ndarray:
    """Real constituents first, an all-zero suffix after, as the real files pad."""
    x = np.zeros((len(multiplicities), MAX_CONSTITUENTS, len(CONSTITUENT_NAMES)))
    for jet, count in enumerate(multiplicities):
        shape = (count, len(CONSTITUENT_NAMES))
        # Bounded away from zero, so that a real slot is never mistaken for padding.
        block = rng.uniform(0.5, 1.5, shape) * rng.choice([-1.0, 1.0], shape)
        block[:, PT_INDEX] = np.abs(block[:, PT_INDEX])
        x[jet, :count] = block

    return x


def _jets(rng, n_jets) -> np.ndarray:
    """53 observables, then one class flag set per jet, then an always-zero j_undef."""
    jets = np.zeros((n_jets, len(JET_NAMES)))
    jets[:, : len(JET_NAMES) - len(LABEL_NAMES)] = rng.uniform(-5, 5, (n_jets, 53))
    jets[np.arange(n_jets), 53 + rng.integers(0, 5, n_jets)] = 1.0

    return jets
