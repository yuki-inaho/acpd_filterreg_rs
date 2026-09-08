"""2D/3D FilterReg + Analytic-CPD with interchangeable C++ and Rust engines."""
from ._api import (NativeExtensionUnavailableError, PosteriorStatistics, backend_names,
                   engine_names, gaussian_sum, method_names, posterior_stats, registration,
                   registration_analytic, registration_nonrigid, registration_rigid, permutohedral_filter, LatticeResult)
from ._options import AnalyticOptions, FilterRegOptions
from ._result import AnalyticStep, Iteration, RegistrationResult, StageResult, load_result
__version__ = "0.2.1"
__all__ = ["AnalyticOptions", "FilterRegOptions", "RegistrationResult", "StageResult", "Iteration",
           "AnalyticStep", "NativeExtensionUnavailableError", "PosteriorStatistics", "registration",
           "registration_rigid", "registration_analytic", "registration_nonrigid", "gaussian_sum",
           "posterior_stats", "method_names", "backend_names", "engine_names", "load_result", "permutohedral_filter", "LatticeResult"]
