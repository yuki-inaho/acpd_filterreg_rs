//! Reusable, Python-independent 2D/3D FilterReg and Analytic-CPD.
//! All Gaussian filtering, linear algebra and registration are implemented in
//! Rust; neither C++ calls nor a Python numerical fallback are used.
#![forbid(unsafe_code)]
pub mod analytic;
pub mod gaussian;
pub mod lattice;
pub mod registration;
pub mod rigid;
pub mod types;
pub use analytic::{
    basis, degree_schedule, exponents
};
pub use gaussian::{
    gaussian_sum, posterior_statistics, Statistics
};
pub use lattice::{
    Permutohedral, FixedNoBlurLattice
};
pub use registration::registration;
pub use types::*;
