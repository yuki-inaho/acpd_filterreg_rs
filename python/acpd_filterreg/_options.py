"""Stage-specific immutable options; variance units are documented in API.md."""
from __future__ import annotations
from dataclasses import dataclass
from . import _validation as v


@dataclass(frozen=True)
class FilterRegOptions:
    max_iterations: int = 60
    tolerance: float = 1e-7
    w: float = 0.1
    sigma2: float | None = None
    min_sigma2: float = 1e-8
    update_sigma2: bool = True
    solver: str = "twist"
    objective: str = "point_to_point"
    inner_iterations: int = 1

    def __post_init__(self) -> None:
        _common(self)
        v.boolean("update_sigma2", self.update_sigma2)
        if self.solver not in ("twist", "kabsch"):
            raise ValueError("solver must be 'twist' or 'kabsch'")
        if self.objective not in ("point_to_point", "point_to_plane"):
            raise ValueError("objective must be 'point_to_point' or 'point_to_plane'")
        if self.objective == "point_to_plane" and self.solver != "twist":
            raise ValueError("point-to-plane requires the twist solver")
        v.integer("inner_iterations", self.inner_iterations, 1, 100)


@dataclass(frozen=True)
class FgtOptions:
    """Improved Fast Gauss Transform controls.

    Only read when a stage selects the ``fgt`` backend. The transform is an
    approximation with an explicit error/cost trade-off; it is never selected
    implicitly and never substituted for a failed exact computation.
    """
    order: int = 5
    max_clusters: int = 4096
    cluster_radius: float = 0.25
    cutoff_radius: float = 4.0

    def __post_init__(self) -> None:
        v.integer("order", self.order, 0, 12)
        v.integer("max_clusters", self.max_clusters, 1, 1000000)
        v.real("cluster_radius", self.cluster_radius)
        v.real("cutoff_radius", self.cutoff_radius)


@dataclass(frozen=True)
class CudaOptions:
    """GPU Gaussian-sum controls, read only when a stage selects the ``cuda`` backend.

    The device operator is exact pair evaluation, not an approximation. It is
    never selected implicitly, and a build or machine without CUDA raises rather
    than computing on the CPU.
    """
    single_precision: bool = False

    def __post_init__(self) -> None:
        v.boolean("single_precision", self.single_precision)


@dataclass(frozen=True)
class AnalyticOptions:
    max_iterations: int = 220
    min_degree: int = 1
    max_degree: int = 10
    tolerance: float = 1e-7
    w: float = 0.1
    sigma2: float | None = None
    min_sigma2: float = 1e-12
    rank_tolerance: float = 1e-12
    min_mass: float = 1e-12
    backend: str = "direct"
    initialization: str = "auto"
    stable_patience: int = 5
    no_improve_patience: int = 8
    min_iterations: int = 6
    improvement_relative: float = 1e-6
    rebound_relative: float = 1e-3
    divergence_radius: float = 100.0

    def __post_init__(self) -> None:
        _common(self)
        low = v.integer("min_degree", self.min_degree, 1, 10)
        v.integer("max_degree", self.max_degree, low, 10)
        if v.real("rank_tolerance", self.rank_tolerance) >= 1:
            raise ValueError("rank_tolerance must be < 1")
        v.real("min_mass", self.min_mass)
        if self.backend not in ("direct", "fgt", "cuda"):
            raise ValueError("analytic backend must be 'direct', 'fgt' or 'cuda'; the lattice "
                             "backends normalize the posterior in the other direction")
        if self.initialization not in ("auto", "cpd", "filterreg"):
            raise ValueError("initialization must be 'auto', 'cpd' or 'filterreg'")
        for name in ("stable_patience", "no_improve_patience", "min_iterations"):
            v.integer(name, getattr(self, name), 1, 100000)
        v.real("improvement_relative", self.improvement_relative, positive=False)
        v.real("rebound_relative", self.rebound_relative, positive=False)
        if v.real("divergence_radius", self.divergence_radius) <= 1:
            raise ValueError("divergence_radius must be greater than one")


def _common(option: FilterRegOptions | AnalyticOptions) -> None:
    v.integer("max_iterations", option.max_iterations, 1, 100000)
    v.real("tolerance", option.tolerance)
    v.probability(option.w)
    v.real("min_sigma2", option.min_sigma2)
    if option.sigma2 is not None:
        v.real("sigma2", option.sigma2)
