# Restricting to the leading constituents and normalising them.
#
# Copied from the study's own data module (jet_tagging_gdl,
# src/data/hls4ml_datamodule.py) rather than rewritten, so that arrays built from the
# record and arrays built from the original h5 files agree bit for bit: the same stable
# argsort over all 150 slots, the same truncate-or-pad, the same python-float scales and
# the same eps.

from pathlib import Path

import numpy as np

# Features that live on both sides of zero, so the scale is half the observed range.
CENTERED_FEATURES = {
    "j1_px", "j1_py", "j1_pz",
    "j1_eta", "j1_etarel", "j1_etarot",
    "j1_phi", "j1_phirel", "j1_phirot",
    "j1_costheta", "j1_costhetarel",
}

# Features that are non-negative by construction, so the scale is the observed maximum.
POSITIVE_FEATURES = {
    "j1_pt", "j1_ptrel",
    "j1_e",  "j1_erel",
    "j1_deltaR",
}


def restrict_and_sort_by_pt(x, pt_idx: int, nconstituents: int | None) -> np.ndarray:
    """Keep the leading *nconstituents* by pT, in descending pT order.

    The full constituent list is sorted before truncation, so the constituents kept are
    genuinely the highest-pT ones whatever order they appear in on disk. Negating and
    using a stable sort gives a deterministic order for equal-pT constituents (ties keep
    their on-disk order); ``argsort(...)[::-1]`` would instead reverse an unstable
    ascending sort, making tie order arbitrary run to run. Padded slots carry pT = 0 and
    therefore sort to the end.
    """
    sort_idx = np.argsort(-x[:, :, pt_idx], axis=1, kind="stable")
    x = np.take_along_axis(x, sort_idx[:, :, np.newaxis], axis=1)

    target = x.shape[1] if (nconstituents is None or nconstituents <= 0) else nconstituents
    if x.shape[1] < target:
        padding = np.zeros((x.shape[0], target - x.shape[1], x.shape[2]), dtype=x.dtype)
        return np.concatenate((x, padding), axis=1)

    return x[:, :target, :]


def fit_physics_norm(x_train, feature_names) -> dict[str, tuple[str, float]]:
    """One scale per feature, fitted on the restricted training array alone."""
    params: dict[str, tuple[str, float]] = {}
    for i, name in enumerate(feature_names):
        vals = x_train[:, :, i]
        vmin, vmax = float(vals.min()), float(vals.max())
        if name in CENTERED_FEATURES:
            params[name] = ("centered", 0.5 * (vmax - vmin))
        elif name in POSITIVE_FEATURES:
            params[name] = ("positive", vmax)
        else:
            raise ValueError(f"Feature '{name}' not categorised in CENTERED or POSITIVE sets.")

    return params


def apply_physics_norm(x, feature_names, params, eps: float = 1e-8) -> np.ndarray:
    """Divide each feature by its scale. Zero padding stays zero, so it survives."""
    x_norm = x.copy()
    for i, name in enumerate(feature_names):
        kind, scale = params[name]
        if kind not in {"centered", "positive"}:
            raise ValueError(f"Unknown normalisation kind '{kind}' for feature '{name}'.")
        x_norm[:, :, i] /= (scale + eps)

    return x_norm


def save_norm_params(path, params: dict[str, tuple[str, float]]) -> None:
    """The study's own npz layout, which its diagnostics and feature lookup read back."""
    names = np.array(list(params.keys()), dtype="U64")
    kinds = np.array([params[n][0] for n in names], dtype="U16")
    scales = np.array([params[n][1] for n in names], dtype=np.float32)
    np.savez(Path(path), feature_names=names, kinds=kinds, scales=scales)


def load_norm_params(path) -> dict[str, tuple[str, float]]:
    npz = np.load(Path(path), allow_pickle=False)

    return {
        str(name): (str(kind), float(scale))
        for name, kind, scale in zip(npz["feature_names"], npz["kinds"], npz["scales"])
    }
