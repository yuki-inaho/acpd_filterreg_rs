"""Test-only adapter to the C++ executable; never imported by the public package."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import numpy as np


class Driver:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def _run(self, header: list, *arrays: np.ndarray):
        def scalar(value):
            return str(int(value)) if isinstance(value, (bool, np.bool_)) else str(value)
        fields = [scalar(value) for value in header]
        fields.extend(repr(float(value)) for array in arrays for value in np.asarray(array).ravel())
        done = subprocess.run([str(self.path)], input=" ".join(fields), text=True,
                              capture_output=True, check=False, timeout=45)
        if done.returncode:
            exception = ValueError if done.returncode == 2 else RuntimeError
            raise exception(done.stderr.strip())
        return json.loads(done.stdout)

    def registration(self, x, y, o, r, t, normals):
        keys = ("rigid_max_iterations", "rigid_tolerance", "rigid_w", "rigid_sigma2", "rigid_min_sigma2",
                "rigid_update_sigma2", "rigid_solver", "rigid_objective", "rigid_inner_iterations",
                "analytic_max_iterations", "analytic_tolerance", "analytic_w", "analytic_sigma2",
                "analytic_min_sigma2", "analytic_min_degree", "analytic_max_degree",
                "analytic_rank_tolerance", "analytic_min_mass", "analytic_initialization",
                "analytic_stable_patience", "analytic_no_improve_patience", "analytic_min_iterations",
                "analytic_improvement_relative", "analytic_rebound_relative", "analytic_divergence_radius",
                "analytic_backend", "fgt_order", "fgt_max_clusters", "fgt_cluster_radius", "fgt_cutoff_radius", "cuda_single_precision")
        return self._run(["registration", x.shape[1], len(x), len(y), len(normals), o["method"], o["backend"],
                          *(o[key] for key in keys)], x, y, r, t, normals)

    def gaussian_sum(self, sources, queries, values, sigma2, backend, fgt):
        return np.asarray(self._run(["gaussian", sources.shape[1], len(sources), len(queries), values.shape[1],
                                     sigma2, backend, fgt["fgt_order"], fgt["fgt_max_clusters"],
                                     fgt["fgt_cluster_radius"], fgt["fgt_cutoff_radius"],
                                     int(fgt["cuda_single_precision"])],
                                    sources, queries, values))

    def posterior_stats(self, x, y, sigma2, w, filterreg, backend):
        return self._run(["stats", x.shape[1], len(x), len(y), sigma2, w, filterreg, backend], x, y)

    def basis(self, points, degree):
        return np.asarray(self._run(["basis", points.shape[1], len(points), degree], points))

    def permutohedral_filter(self, features, values, with_blur, start, reverse):
        return self._run(["lattice", features.shape[1], len(features), values.shape[1], with_blur, start, reverse], features, values)

    def schedule(self, iterations, low, high):
        return self._run(["schedule", iterations, low, high])

    def fit(self, moving, targets, weights, degree):
        return self._run(["fit", moving.shape[1], len(moving), degree], moving, targets, weights)

    def rigid_fit(self, moving, targets, weights, inner=1, solver="twist", normals=None):
        arrays = [moving, targets, weights]
        if normals is not None:
            arrays.append(normals)
        return self._run(["rigid_fit", moving.shape[1], len(moving), inner, normals is not None, solver], *arrays)

    def twist(self, point, delta):
        return self._run(["twist", len(point)], point, delta)
