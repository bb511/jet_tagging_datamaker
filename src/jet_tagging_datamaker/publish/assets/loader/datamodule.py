# Driving the two stages, for a caller who wants tensors and nothing else.

import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch

from . import physics, reading

log = logging.getLogger(__name__)

# The record is published pre-split: train and validation are the Zenodo train archive
# split 80/20, test is the Zenodo validation archive.
SPLITS = ("train", "validation", "test")

CACHE_FILES = (
    *(f"x_{split}.npy" for split in SPLITS),
    *(f"y_{split}.npy" for split in SPLITS),
    "norm_params.npz",
)


@dataclass(frozen=True)
class SplitTensors:
    """One split as a model sees it: the constituents and the one-hot flavour label."""

    x: torch.Tensor
    y: torch.Tensor


@dataclass
class JetData:
    """Restrict, normalise and cache the published jets, then hand back their tensors.

    The normalisation is fitted on the training split alone and applied unchanged to
    validation and to test, which is what the study does.

    :param data_dir: The record's ``data/``, holding the parquet shards.
    :param cache_root_dir: Where the caches are built. They are written to and roughly
        the size of the record, so they stay outside the downloaded copy.
    :param constituent_features: The 16 columns, in the order the h5 files carry them.
    :param nconstituents: How many leading constituents by pT to keep; 0 or less keeps
        all ``max_constituents``.
    """

    data_dir: str
    cache_root_dir: str
    constituent_features: list[str]
    nconstituents: int = 32
    max_constituents: int = 150
    n_classes: int = 5

    @property
    def cache_folder(self) -> Path:
        """One directory per constituent count, so two of them never share a cache."""
        if self.nconstituents <= 0:
            return Path(self.cache_root_dir) / "full"

        return Path(self.cache_root_dir) / f"nconst_{self.nconstituents}"

    @property
    def feature_names(self) -> list[str]:
        return list(self.constituent_features)

    @property
    def pt_idx(self) -> int:
        return self.feature_names.index("j1_pt")

    @property
    def norm_params(self) -> dict[str, tuple[str, float]] | None:
        """The fitted scales, once prepare() has written them."""
        path = self.cache_folder / "norm_params.npz"

        return physics.load_norm_params(path) if path.is_file() else None

    def prepare(self) -> None:
        """Build the cache. Cached, so reruns are cheap."""
        if all((self.cache_folder / name).is_file() for name in CACHE_FILES):
            log.info(f"Cache already built in {self.cache_folder}")
            return

        self.cache_folder.mkdir(parents=True, exist_ok=True)
        params = self._prepare_train()
        for split in SPLITS[1:]:
            self._prepare_split(split, params)

    def load(self, split: str) -> SplitTensors:
        """One cached split, x as (jets, constituents, features) and y one-hot."""
        if split not in SPLITS:
            raise ValueError(f"Unknown split '{split}', expected one of {SPLITS}")

        return SplitTensors(x=self._tensor(f"x_{split}"), y=self._tensor(f"y_{split}"))

    def _prepare_train(self) -> dict[str, tuple[str, float]]:
        """The training split, which is also where the normalisation is fitted."""
        x, y = self._read("train")
        params = physics.fit_physics_norm(x, self.feature_names)
        physics.save_norm_params(self.cache_folder / "norm_params.npz", params)
        self._save("train", physics.apply_physics_norm(x, self.feature_names, params), y)

        return params

    def _prepare_split(self, split: str, params: dict[str, tuple[str, float]]) -> None:
        x, y = self._read(split)
        self._save(split, physics.apply_physics_norm(x, self.feature_names, params), y)

    def _read(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """The shards of one split, each restricted to the leading constituents."""
        transform = partial(
            physics.restrict_and_sort_by_pt,
            pt_idx=self.pt_idx,
            nconstituents=self.nconstituents,
        )

        return reading.read_split(
            self.data_dir,
            split,
            self.feature_names,
            self.max_constituents,
            self.n_classes,
            transform,
        )

    def _save(self, split: str, x: np.ndarray, y: np.ndarray) -> None:
        """float32 on disk: the pipeline runs in float64, the models train in float32."""
        np.save(self.cache_folder / f"x_{split}.npy", x.astype(np.float32))
        np.save(self.cache_folder / f"y_{split}.npy", y.astype(np.float32))

    def _tensor(self, name: str) -> torch.Tensor:
        return torch.from_numpy(np.load(self.cache_folder / f"{name}.npy")).float()
