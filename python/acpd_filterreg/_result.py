"""Owned result snapshots, reproducible map evaluation, and pickle-free storage."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any
import numpy as np
from . import _validation as v
from ._mapping import basis, basis_derivative, exponents


def _owned(value: Any, ndim: int) -> np.ndarray:
    output = v.array("result array", value, ndim, copy=True)
    output.flags.writeable = False
    return output


@dataclass(frozen=True)
class Iteration:
    iteration: int
    raw_degree: int
    degree: int
    active: int
    rank: int
    sigma2: float
    nll_before: float | None
    step_rms: float
    fit_rms: float
    lattice_vertices: int
    lattice_mode: str


@dataclass(frozen=True)
class StageResult:
    history: tuple[Iteration, ...]
    initial_sigma2: float
    final_sigma2: float
    converged: bool
    stop_reason: str
    best_iteration: int
    index_builds: int

    @property
    def iterations(self) -> int:
        return len(self.history)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StageResult:
        history = value["history"]
        if not isinstance(history, (list, tuple)) or len(history) > 100000:
            raise ValueError("invalid saved iteration history")
        rows = []
        for index, item in enumerate(history, start=1):
            row = Iteration(**item)
            if v.integer("saved iteration", row.iteration, 1, 100000) != index:
                raise ValueError("iteration history must be consecutive")
            for name in ("raw_degree", "degree"):
                v.integer(name, getattr(row, name), 0, 10)
            for name in ("active", "rank", "lattice_vertices"):
                v.integer(name, getattr(row, name), 0, 2**63-1)
            v.real("saved iteration sigma2", row.sigma2)
            v.real("saved step RMS", row.step_rms, positive=False)
            v.real("saved fit RMS", row.fit_rms, positive=False)
            if row.nll_before is not None and (isinstance(row.nll_before, bool)
                    or not isinstance(row.nll_before, (int, float)) or not np.isfinite(row.nll_before)):
                raise ValueError("saved nll must be finite or None")
            if not isinstance(row.lattice_mode, str):
                raise ValueError("saved lattice mode must be a string")
            rows.append(row)
        initial = v.real("stage initial sigma2", value["initial_sigma2"], positive=False)
        final = v.real("stage final sigma2", value["final_sigma2"], positive=False)
        converged = v.boolean("stage converged", value["converged"])
        stop = value["stop_reason"]
        if stop not in ("not_run", "iteration_limit", "tolerance", "residual_tolerance",
                        "stable_tolerance", "internal_rebound", "no_improvement",
                        "insufficient_posterior_mass", "numerical_divergence"):
            raise ValueError("unknown saved stopping reason")
        best = v.integer("stage best iteration", value["best_iteration"], 0, len(rows))
        builds = v.integer("stage index builds", value["index_builds"], 0, 200000)
        if rows and (initial <= 0 or final <= 0):
            raise ValueError("executed stages need positive variance")
        return cls(tuple(rows), initial, final, converged, stop, best, builds)


@dataclass(frozen=True)
class AnalyticStep:
    degree: int
    coefficients: np.ndarray


@dataclass(frozen=True)
class RegistrationResult:
    rotation: np.ndarray
    translation: np.ndarray
    center: np.ndarray
    normalization_scale: float
    transformed: np.ndarray
    rigid_transformed: np.ndarray
    steps: tuple[AnalyticStep, ...]
    rigid_stage: StageResult
    analytic_stage: StageResult
    sigma2: float
    method: str
    backend: str
    engine: str

    @classmethod
    def from_native(cls, raw: dict[str, Any], engine: str) -> RegistrationResult:
        rotation = _owned(raw["rotation"], 2)
        translation, center = _owned(raw["translation"], 1), _owned(raw["center"], 1)
        d = len(translation)
        if d not in (2, 3) or rotation.shape != (d, d) or center.shape != (d,):
            raise ValueError("inconsistent result coordinate dimensions")
        if not np.allclose(rotation.T @ rotation, np.eye(d), atol=1e-8, rtol=0) or abs(np.linalg.det(rotation) - 1) > 1e-8:
            raise ValueError("result rotation is not in SO(d)")
        transformed, rigid = _owned(raw["transformed"], 2), _owned(raw["rigid_transformed"], 2)
        if transformed.shape[1] != d or rigid.shape != transformed.shape:
            raise ValueError("inconsistent result cloud dimensions")
        steps = []
        for item in raw["steps"]:
            degree = v.integer("saved degree", item["degree"], 1, 10)
            coefficient = _owned(item["coefficients"], 2)
            if coefficient.shape != (len(exponents(d, degree)), d):
                raise ValueError("inconsistent saved coefficient dimensions")
            steps.append(AnalyticStep(degree, coefficient))
        scale = v.real("normalization_scale", raw["normalization_scale"])
        sigma2 = v.real("result sigma2", raw["sigma2"])
        if raw["method"] not in ("rigid", "analytic", "nonrigid") or raw["backend"] not in ("direct", "permutohedral", "permutohedral_noblur", "probreg"):
            raise ValueError("unknown result method/backend")
        if raw["method"] == "rigid" and steps:
            raise ValueError("a rigid result cannot contain analytic steps")
        rigid_stage = StageResult.from_dict(raw["rigid_stage"])
        analytic_stage = StageResult.from_dict(raw["analytic_stage"])
        if analytic_stage.best_iteration != len(steps):
            raise ValueError("saved map length disagrees with the best iteration")
        if engine not in ("cpp", "rust"):
            raise ValueError("unknown saved engine")
        final_stage = rigid_stage if raw["method"] == "rigid" else analytic_stage
        # Native implementations may associate sigma2 * scale * scale differently.
        if not np.isclose(final_stage.final_sigma2, sigma2,
                          rtol=8 * np.finfo(np.float64).eps, atol=0.0):
            raise ValueError("saved result variance disagrees with the returned stage")
        return cls(rotation, translation, center, scale, transformed, rigid, tuple(steps),
                   rigid_stage, analytic_stage, sigma2, raw["method"], raw["backend"], engine)

    @property
    def iterations(self) -> int:
        return self.rigid_stage.iterations + self.analytic_stage.iterations

    @property
    def converged(self) -> bool:
        stages = [stage for stage in (self.rigid_stage, self.analytic_stage) if stage.iterations]
        return bool(stages) and all(stage.converged for stage in stages)

    @property
    def degree_history(self) -> tuple[int, ...]:
        return tuple(row.degree for row in self.analytic_stage.history)

    @property
    def scale(self) -> float:
        """Rigid scale is always one; similarity registration is not implemented."""
        return 1.0

    def _points(self, points: Any, copy: bool) -> np.ndarray:
        result = v.points("points", points, copy=copy)
        if result.shape[1] != len(self.translation):
            raise ValueError("point dimension differs from the saved map")
        return result

    def transform(self, points: Any, *, copy: bool = False) -> np.ndarray:
        """Apply the frozen pose and all saved residual maps to arbitrary points."""
        p = self._points(points, copy)
        with np.errstate(over="raise", invalid="raise"):
            value = (p @ self.rotation.T + self.translation - self.center) / self.normalization_scale
            for step in self.steps:
                value = value + basis(value, step.degree) @ step.coefficients
            result = value * self.normalization_scale + self.center
        if not np.isfinite(result).all():
            raise FloatingPointError("map evaluation overflow; extrapolation is not bounded")
        return result

    def residual_displacement(self, points: Any, *, copy: bool = False) -> np.ndarray:
        """Displacement relative to the frozen FilterReg/user-supplied rigid pose."""
        p = self._points(points, copy)
        return self.transform(p) - (p @ self.rotation.T + self.translation)

    def displacement(self, points: Any, *, copy: bool = False) -> np.ndarray:
        p = self._points(points, copy)
        return self.transform(p) - p

    def jacobian(self, points: Any, *, copy: bool = False) -> np.ndarray:
        """Return (n,d,d) Jacobians, rows=output coordinates, columns=input."""
        p = self._points(points, copy)
        d = p.shape[1]
        value = (p @ self.rotation.T + self.translation - self.center) / self.normalization_scale
        jac = np.broadcast_to(self.rotation, (len(p), d, d)).copy()
        with np.errstate(over="raise", invalid="raise"):
            for step in self.steps:
                derivative = np.einsum("nkj,kb->nbj", basis_derivative(value, step.degree), step.coefficients)
                derivative += np.eye(d)
                jac = derivative @ jac
                value = value + basis(value, step.degree) @ step.coefficients
        if not np.isfinite(jac).all():
            raise FloatingPointError("Jacobian evaluation overflow")
        return jac

    def save(self, path: str | Path) -> None:
        """Write an NPZ archive without pickle or executable object serialization."""
        metadata = {"format_version": 2, "engine": self.engine, "method": self.method,
                    "backend": self.backend, "sigma2": self.sigma2,
                    "normalization_scale": self.normalization_scale,
                    "degrees": [step.degree for step in self.steps],
                    "rigid_stage": asdict(self.rigid_stage), "analytic_stage": asdict(self.analytic_stage)}
        arrays = {key: getattr(self, key) for key in
                  ("rotation", "translation", "center", "transformed", "rigid_transformed")}
        arrays.update({f"coeff_{i}": step.coefficients for i, step in enumerate(self.steps)})
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, metadata=np.array(json.dumps(metadata, allow_nan=False)), **arrays)


def load_result(path: str | Path) -> RegistrationResult:
    """Load a versioned saved map. Native modules are not required for evaluation."""
    with np.load(path, allow_pickle=False) as archive:
        raw = json.loads(str(archive["metadata"].item()))
        if raw.pop("format_version", None) != 2:
            raise ValueError("unsupported saved-map format version")
        engine = raw.pop("engine")
        degrees = raw.pop("degrees")
        if len(degrees) > 100000:
            raise ValueError("too many saved analytic steps")
        for key in ("rotation", "translation", "center", "transformed", "rigid_transformed"):
            raw[key] = archive[key]
        raw["steps"] = [{"degree": degree, "coefficients": archive[f"coeff_{i}"]}
                        for i, degree in enumerate(degrees)]
        return RegistrationResult.from_native(raw, engine)
