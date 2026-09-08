"""Thin, shared public interface; every registration call uses native numerics."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any, Literal, Protocol
import numpy as np
from . import _validation as v
from ._options import AnalyticOptions, CudaOptions, FgtOptions, FilterRegOptions
from ._result import RegistrationResult

Engine = Literal["cpp", "rust"]
Backend = Literal["direct", "permutohedral", "permutohedral_noblur", "probreg", "fgt", "cuda"]
Method = Literal["rigid", "analytic", "nonrigid"]


class NativeEngine(Protocol):
    """Small structural contract shared by nanobind and PyO3 modules."""
    def registration(self, fixed: np.ndarray, moving: np.ndarray, options: dict[str, Any],
                     rotation: np.ndarray, translation: np.ndarray, target_normals: np.ndarray) -> dict[str, Any]: ...
    def gaussian_sum(self, sources: np.ndarray, queries: np.ndarray, values: np.ndarray,
                     sigma2: float, backend: str, fgt: dict[str, Any]) -> np.ndarray: ...
    def posterior_stats(self, fixed: np.ndarray, moving: np.ndarray, sigma2: float,
                        w: float, filterreg: bool, backend: str) -> dict[str, Any]: ...
    def permutohedral_filter(self, features: np.ndarray, values: np.ndarray, with_blur: bool,
                             start: int, reverse: bool) -> dict[str, Any]: ...
    def basis(self, points: np.ndarray, degree: int) -> np.ndarray: ...


class NativeExtensionUnavailableError(ImportError):
    """The selected native engine is absent; no alternate engine was selected."""


def method_names() -> tuple[str, ...]:
    return ("rigid", "analytic", "nonrigid")


def backend_names() -> tuple[str, ...]:
    return ("direct", "permutohedral", "permutohedral_noblur", "probreg", "fgt", "cuda")


def engine_names() -> tuple[str, ...]:
    return ("cpp", "rust")


def cuda_available(engine: Engine = "cpp") -> bool:
    """Whether the selected engine can run the GPU Gaussian-sum operator.

    False means the backend raises rather than computing on the CPU. Only the
    C++ engine has a device path; the Rust engine always reports False.
    """
    if engine not in engine_names():
        raise ValueError(f"engine must be one of {engine_names()}")
    if engine != "cpp":
        return False
    module = _get_engine(engine)
    query = getattr(module, "cuda_available", None)
    return bool(query()) if query is not None else False


def cuda_device_name(engine: Engine = "cpp") -> str:
    if not cuda_available(engine):
        return ""
    return str(_get_engine(engine).cuda_device_name())


def _get_engine(engine: str) -> NativeEngine:
    if engine not in engine_names():
        raise ValueError("engine must be 'cpp' or 'rust'")
    module = "acpd_filterreg_cpp" if engine == "cpp" else "acpd_filterreg_rs"
    try:
        return import_module(module)
    except ImportError as error:
        task = "build-cpp" if engine == "cpp" else "build-rust"
        raise NativeExtensionUnavailableError(
            f"{module} could not be imported. Run `pixi run {task}`. No fallback was used. Original error: {error}"
        ) from error


def _fgt(fgt: FgtOptions | None, cuda: CudaOptions | None = None) -> dict[str, Any]:
    fgt = FgtOptions() if fgt is None else fgt
    cuda = CudaOptions() if cuda is None else cuda
    if not isinstance(fgt, FgtOptions):
        raise TypeError("fgt must be FgtOptions")
    if not isinstance(cuda, CudaOptions):
        raise TypeError("cuda must be CudaOptions")
    out = {f"fgt_{key}": value for key, value in asdict(fgt).items()}
    out.update({f"cuda_{key}": value for key, value in asdict(cuda).items()})
    return out


def _options(method: str, backend: str, rigid: FilterRegOptions,
             analytic: AnalyticOptions, fgt: FgtOptions | None = None,
             cuda: CudaOptions | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"method": method, "backend": backend}
    for prefix, stage in (("rigid", rigid), ("analytic", analytic)):
        for key, value in asdict(stage).items():
            if value is None:
                value = -1.0
            elif isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.floating):
                value = float(value)
            out[f"{prefix}_{key}"] = value
    out.update(_fgt(fgt, cuda))
    return out


def registration(fixed: Any, moving: Any, *, method: Method = "nonrigid", engine: Engine = "cpp",
                 backend: Backend = "permutohedral",
                 rigid: FilterRegOptions | None = None, analytic: AnalyticOptions | None = None,
                 fgt: FgtOptions | None = None, cuda: CudaOptions | None = None,
                 initial_rotation: Any = None, initial_translation: Any = None, target_normals: Any = None,
                 copy: bool = False) -> RegistrationResult:
    """Register moving -> fixed in 2D or 3D. The first argument is ALWAYS fixed.

    rigid: FilterReg point-to-point rigid EM, no scale.
    analytic: standalone compositional Analytic-CPD from the supplied/identity pose.
    nonrigid: FilterReg first, then freeze that pose and fit analytic residual maps.
    sigma2 options use input squared units; min_sigma2 and tolerance are normalized.
    backend selects the FilterReg E-step; analytic.backend selects the ACPD E-step
    ("direct" exact by default, "fgt" for the explicitly chosen IFGT approximation).
    target_normals is required for point-to-plane (point-to-line in 2D).
    """
    if method not in method_names():
        raise ValueError(f"method must be one of {method_names()}")
    if engine not in engine_names():
        raise ValueError(f"engine must be one of {engine_names()}")
    backend = v.backend(backend)
    x, y = v.points("fixed", fixed, copy=copy), v.points("moving", moving, copy=copy)
    d = x.shape[1]
    if y.shape[1] != d:
        raise ValueError("fixed and moving dimensions must match")
    if min(len(x), len(y)) < d + 1:
        raise ValueError("registration requires at least d+1 points in each cloud")
    rigid = FilterRegOptions() if rigid is None else rigid
    analytic = AnalyticOptions() if analytic is None else analytic
    fgt = FgtOptions() if fgt is None else fgt
    cuda = CudaOptions() if cuda is None else cuda
    if not isinstance(rigid, FilterRegOptions) or not isinstance(analytic, AnalyticOptions):
        raise TypeError("rigid/analytic must be FilterRegOptions/AnalyticOptions")
    if not isinstance(fgt, FgtOptions):
        raise TypeError("fgt must be FgtOptions")
    rotation = np.eye(d) if initial_rotation is None else v.array("initial_rotation", initial_rotation, 2, copy=copy)
    translation = np.zeros(d) if initial_translation is None else v.array("initial_translation", initial_translation, 1, copy=copy)
    if rotation.shape != (d, d) or translation.shape != (d,):
        raise ValueError("invalid initial pose dimensions")
    if not np.allclose(rotation.T @ rotation, np.eye(d), atol=1e-8, rtol=0) or abs(np.linalg.det(rotation) - 1) > 1e-8:
        raise ValueError("initial_rotation must belong to SO(d); reflections and scaling are rejected")
    normals = np.empty((0, d)) if target_normals is None else v.points("target_normals", target_normals, copy=copy)
    if target_normals is not None and (normals.shape != x.shape or not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError("target_normals must match fixed points and contain unit normals")
    if method != "analytic" and rigid.objective == "point_to_plane" and target_normals is None:
        raise ValueError("point-to-plane requires target_normals")
    if method == "analytic" and analytic.initialization == "filterreg" and analytic.sigma2 is None:
        raise ValueError("standalone analytic mode has no FilterReg variance to inherit")
    raw = _get_engine(engine).registration(x, y, _options(method, backend, rigid, analytic, fgt, cuda), rotation, translation, normals)
    return RegistrationResult.from_native(raw, engine)


def registration_rigid(fixed: Any, moving: Any, **kwargs: Any) -> RegistrationResult:
    """FilterReg rigid registration (not rigid CPD)."""
    return registration(fixed, moving, method="rigid", **kwargs)


def registration_analytic(fixed: Any, moving: Any, **kwargs: Any) -> RegistrationResult:
    """Standalone compositional Analytic-CPD, without a FilterReg stage."""
    return registration(fixed, moving, method="analytic", **kwargs)


def registration_nonrigid(fixed: Any, moving: Any, **kwargs: Any) -> RegistrationResult:
    """FilterReg -> frozen rigid pose -> residual Analytic-CPD."""
    return registration(fixed, moving, method="nonrigid", **kwargs)


def gaussian_sum(sources: Any, queries: Any, values: Any, *, sigma2: float,
                 engine: Engine = "cpp", backend: Backend = "direct",
                 fgt: FgtOptions | None = None, cuda: CudaOptions | None = None,
                 copy: bool = False) -> np.ndarray:
    """Direct Gaussian sum, or its explicitly selected lattice/IFGT approximation.

    No row normalization: backend-specific lattice scaling/gain is preserved.
    fgt is read only when backend == "fgt".
    """
    b = v.backend(backend)
    variance = v.real("sigma2", sigma2)
    s, q = v.points("sources", sources, copy=copy), v.points("queries", queries, copy=copy)
    value = v.array("values", values, 2, copy=copy)
    if s.shape[1] != q.shape[1] or len(value) != len(s):
        raise ValueError("inconsistent Gaussian transform dimensions")
    return np.asarray(_get_engine(engine).gaussian_sum(s, q, value, variance, b, _fgt(fgt, cuda)))


@dataclass(frozen=True)
class PosteriorStatistics:
    rho: np.ndarray
    px: np.ndarray
    x2: np.ndarray
    mass: float
    nll: float | None
    vertices: int = 0
    unsupported: int = 0
    lattice_mode: str = "direct"


def posterior_stats(fixed: Any, moving: Any, *, sigma2: float, w: float = 0.1,
                    kind: Literal["cpd", "filterreg"] = "cpd", engine: Engine = "cpp",
                    backend: Backend = "direct", copy: bool = False) -> PosteriorStatistics:
    """Inspect raw-coordinate posterior moments. Registration normalizes separately."""
    if kind not in ("cpd", "filterreg"):
        raise ValueError("kind must be 'cpd' or 'filterreg'")
    b = v.backend(backend)
    if kind == "cpd" and b != "direct":
        raise ValueError("paper-faithful Analytic-CPD uses direct posterior computation")
    variance, outlier = v.real("sigma2", sigma2), v.probability(w)
    x, y = v.points("fixed", fixed, copy=copy), v.points("moving", moving, copy=copy)
    if x.shape[1] != y.shape[1]:
        raise ValueError("point dimensions must match")
    raw = _get_engine(engine).posterior_stats(x, y, variance, outlier, kind == "filterreg", b)
    return PosteriorStatistics(np.asarray(raw["rho"]), np.asarray(raw["px"]), np.asarray(raw["x2"]),
                               float(raw["mass"]), None if raw["nll"] is None else float(raw["nll"]),
                               int(raw["vertices"]), int(raw["unsupported"]), str(raw["lattice_mode"]))


@dataclass(frozen=True)
class LatticeResult:
    values: np.ndarray
    vertices: int


def permutohedral_filter(features: Any, values: Any, *, with_blur: bool = True,
                         start: int = 0, reverse: bool = False, engine: Engine = "cpp",
                         copy: bool = False) -> LatticeResult:
    """Reusable native Splat/Blur/Slice on 1..16 dimensional whitened features.

    Only value rows start..n are splatted. All feature rows are sliced.
    No row normalization is applied; the probreg gain is retained in both modes.
    """
    f = v.array("features", features, 2, copy=copy)
    value = v.array("values", values, 2, copy=copy)
    if f.shape[1] > 16 or len(f) != len(value):
        raise ValueError("features require 1..16 columns and values require the same row count")
    v.boolean("with_blur", with_blur)
    v.boolean("reverse", reverse)
    start = v.integer("start", start, 0, len(f))
    raw = _get_engine(engine).permutohedral_filter(f, value, with_blur, start, reverse)
    return LatticeResult(np.asarray(raw["values"]), int(raw["vertices"]))
