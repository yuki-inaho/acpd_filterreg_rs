use nalgebra::{
    DMatrix, DVector
};
use std::{
    error, fmt
};
pub type Matrix = DMatrix<f64>;
pub type Vector = DVector<f64>;
pub type RegResult<T> = Result<T, Error>;
#[derive(Debug, Clone)]
pub enum Error {
    Invalid(String), Numerical(String)
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) | Self::Numerical(s) => write!(f, "{s}")
        }
    }
}
impl error::Error for Error {
}
pub(crate) fn invalid(s: &str) -> Error {
    Error::Invalid(s.to_owned())
}
pub(crate) fn numerical(s: &str) -> Error {
    Error::Numerical(s.to_owned())
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Direct, Permutohedral, PermutohedralNoBlur, Probreg
}
impl Backend {
    pub fn parse(s: &str) -> RegResult<Self> {
        match s {
            "direct" => Ok(Self::Direct), "permutohedral" => Ok(Self::Permutohedral),
            "permutohedral_noblur" => Ok(Self::PermutohedralNoBlur), "probreg" => Ok(Self::Probreg),
            _ => Err(invalid("unknown Gaussian backend; no automatic fallback")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Direct => "direct", Self::Permutohedral => "permutohedral",
            Self::PermutohedralNoBlur => "permutohedral_noblur", Self::Probreg => "probreg",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Rigid, Analytic, Nonrigid
}
impl Method {
    pub fn parse(s: &str) -> RegResult<Self> {
        match s {
            "rigid" => Ok(Self::Rigid), "analytic" => Ok(Self::Analytic), "nonrigid" => Ok(Self::Nonrigid),
            _ => Err(invalid("method must be rigid, analytic, or nonrigid"))
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Rigid => "rigid", Self::Analytic => "analytic", Self::Nonrigid => "nonrigid"
        }
    }
}
#[derive(Debug, Clone)]
pub struct FilterOptions {
    pub max_iterations: usize, pub tolerance: f64, pub w: f64,
    /// -1 means automatic; otherwise world squared units.
    pub sigma2: f64,
    /// Normalized squared units, not world units.
    pub min_sigma2: f64, pub update_sigma2: bool,
    pub solver: String, pub objective: String, pub inner_iterations: usize,
}
impl Default for FilterOptions {
    fn default() -> Self {
        Self {
            max_iterations: 60, tolerance: 1e-7, w: 0.1,
            sigma2: -1.0, min_sigma2: 1e-8, update_sigma2: true,
            solver: "twist".into(), objective: "point_to_point".into(), inner_iterations: 1
        }
    }
}
#[derive(Debug, Clone)]
pub struct AnalyticOptions {
    pub max_iterations: usize, pub min_degree: usize, pub max_degree: usize,
    pub tolerance: f64, pub w: f64, pub sigma2: f64, pub min_sigma2: f64,
    pub rank_tolerance: f64, pub min_mass: f64,
    pub initialization: String, pub stable_patience: usize,
    pub no_improve_patience: usize, pub min_iterations: usize,
    pub improvement_relative: f64, pub rebound_relative: f64,
}
impl Default for AnalyticOptions {
    fn default() -> Self {
        Self {
            max_iterations: 220, min_degree: 1, max_degree: 10,
            tolerance: 1e-7, w: 0.1, sigma2: -1.0, min_sigma2: 1e-12,
            rank_tolerance: 1e-12, min_mass: 1e-12, initialization: "auto".into(),
            stable_patience: 5, no_improve_patience: 8, min_iterations: 6,
            improvement_relative: 1e-6, rebound_relative: 1e-3
        }
    }
}
#[derive(Debug, Clone)]
pub struct Options {
    pub method: Method, pub backend: Backend,
    pub rigid: FilterOptions, pub analytic: AnalyticOptions
}
impl Default for Options {
    fn default() -> Self {
        Self {
            method: Method::Nonrigid, backend: Backend::Permutohedral,
            rigid: FilterOptions::default(), analytic: AnalyticOptions::default()
        }
    }
}
pub(crate) fn positive(x: f64, label: &str) -> RegResult<()> {
    if !x.is_finite() || x <= 0.0 {
        return Err(invalid(&format!("{label} must be finite and positive")));
    }
    Ok(())
}
fn common(n: usize, tol: f64, w: f64, sigma2: f64, floor: f64) -> RegResult<()> {
    if !(1..=100000).contains(&n) {
        return Err(invalid("max_iterations must be in [1,100000]"));
    }
    positive(tol, "tolerance")?;
    positive(floor, "min_sigma2")?;
    if !w.is_finite() || !(0.0..1.0).contains(&w) {
        return Err(invalid("w must be in [0,1)"));
    }
    if sigma2 != -1.0 {
        positive(sigma2, "sigma2")?;
    }
    Ok(())
}
impl FilterOptions {
    pub fn validate(&self) -> RegResult<()> {
        common(self.max_iterations, self.tolerance, self.w, self.sigma2, self.min_sigma2)?;
        if !["twist", "kabsch"].contains(&self.solver.as_str()) {
            return Err(invalid("invalid rigid solver"));
        }
        if !["point_to_point", "point_to_plane"].contains(&self.objective.as_str()) {
            return Err(invalid("invalid rigid objective"));
        }
        if self.objective == "point_to_plane" && self.solver != "twist" {
            return Err(invalid("point-to-plane requires twist"));
        }
        if !(1..=100).contains(&self.inner_iterations) {
            return Err(invalid("inner_iterations must be in [1,100]"));
        }
        Ok(())
    }
}
impl AnalyticOptions {
    pub fn validate(&self) -> RegResult<()> {
        common(self.max_iterations, self.tolerance, self.w, self.sigma2, self.min_sigma2)?;
        if self.min_degree < 1 || self.max_degree > 10 || self.min_degree > self.max_degree {
            return Err(invalid("analytic degrees must satisfy 1 <= min <= max <= 10"));
        }
        positive(self.rank_tolerance, "rank_tolerance")?;
        if self.rank_tolerance >= 1.0 {
            return Err(invalid("rank_tolerance must be less than one"));
        }
        positive(self.min_mass, "min_mass")?;
        if !["auto", "cpd", "filterreg"].contains(&self.initialization.as_str()) {
            return Err(invalid("invalid analytic initialization"));
        }
        if self.stable_patience == 0 || self.no_improve_patience == 0 || self.min_iterations == 0 {
            return Err(invalid("stopping counts must be positive"));
        }
        if !self.improvement_relative.is_finite() || self.improvement_relative < 0.0
        || !self.rebound_relative.is_finite() || self.rebound_relative < 0.0 {
            return Err(invalid("relative stopping thresholds must be finite and nonnegative"));
        }
        Ok(())
    }
}
impl Options {
    pub fn validate(&self) -> RegResult<()> {
        self.rigid.validate()?;
        self.analytic.validate()?;
        if self.method == Method::Analytic && self.analytic.initialization == "filterreg" && self.analytic.sigma2 < 0.0 {
            return Err(invalid("standalone analytic mode has no FilterReg variance to inherit"));
        }
        Ok(())
    }
}
#[derive(Debug, Clone)]
pub struct Iteration {
    pub iteration: usize, pub raw_degree: usize, pub degree: usize, pub active: usize, pub rank: usize,
    pub sigma2: f64, pub nll_before: f64, pub step_rms: f64, pub fit_rms: f64,
    pub lattice_vertices: usize, pub lattice_mode: String,
}
#[derive(Debug, Clone)]
pub struct Stage {
    pub history: Vec<Iteration>, pub initial_sigma2: f64, pub final_sigma2: f64,
    pub converged: bool, pub stop_reason: String, pub best_iteration: usize, pub index_builds: usize
}
impl Default for Stage {
    fn default() -> Self {
        Self {
            history: Vec::new(), initial_sigma2: 0.0, final_sigma2: 0.0,
            converged: false, stop_reason: "not_run".to_owned(), best_iteration: 0, index_builds: 0
        }
    }
}
#[derive(Debug, Clone)]
pub struct AnalyticStep {
    pub degree: usize, pub coefficients: Matrix
}
#[derive(Debug, Clone)]
pub struct RegistrationResult {
    pub rotation: Matrix, pub translation: Vector, pub center: Vector, pub normalization_scale: f64,
    pub transformed: Matrix, pub rigid_transformed: Matrix, pub steps: Vec<AnalyticStep>,
    pub rigid_stage: Stage, pub analytic_stage: Stage, pub sigma2: f64,
    pub method: Method, pub backend: Backend,
}
pub fn validate_cloud(p: &Matrix, label: &str) -> RegResult<()> {
    if p.nrows() == 0 || ![2, 3].contains(&p.ncols()) {
        return Err(invalid(&format!("{label} needs nonempty shape (n,2) or (n,3)")));
    }
    if !p.iter().all(|x| x.is_finite()) {
        return Err(invalid(&format!("{label} must be finite")));
    }
    Ok(())
}
pub fn validate_pair(x: &Matrix, y: &Matrix, registration: bool) -> RegResult<()> {
    validate_cloud(x, "fixed")?;
    validate_cloud(y, "moving")?;
    if x.ncols() != y.ncols() {
        return Err(invalid("point dimensions must match"));
    }
    if registration && (x.nrows() < x.ncols()+1 || y.nrows() < y.ncols()+1) {
        return Err(invalid("registration requires at least d+1 points per cloud"));
    }
    Ok(())
}
pub fn validate_pose(r: &Matrix, t: &Vector, d: usize) -> RegResult<()> {
    if r.shape() != (d,d) || t.len() != d || !r.iter().chain(t.iter()).all(|x| x.is_finite()) {
        return Err(invalid("invalid initial pose dimensions or values"));
    }
    if (r.transpose()*r - Matrix::identity(d,d)).norm() > 1e-8 || (r.determinant()-1.0).abs() > 1e-8 {
        return Err(invalid("initial_rotation must belong to SO(d)"));
    }
    Ok(())
}
pub fn validate_normals(normals: &Matrix, fixed: &Matrix) -> RegResult<()> {
    if normals.shape() != fixed.shape() || !normals.iter().all(|x| x.is_finite()) {
        return Err(invalid("target normals must match fixed points"));
    }
    for i in 0..normals.nrows() {
        if (normals.row(i).norm() - 1.0).abs() > 1e-6 {
            return Err(invalid("each target normal must have unit norm"));
        }
    }
    Ok(())
}
pub(crate) fn finite(p: &Matrix, context: &str) -> RegResult<()> {
    if !p.iter().all(|v| v.is_finite()) {
        return Err(numerical(&format!("{context}: non-finite numerical result")));
    }
    Ok(())
}
pub(crate) fn mean(p: &Matrix) -> Vector {
    Vector::from_fn(p.ncols(), |a,_| (0..p.nrows()).map(|i| p[(i,a)]).sum::<f64>() / p.nrows() as f64)
}
pub fn apply_pose(p: &Matrix, r: &Matrix, t: &Vector) -> RegResult<Matrix> {
    validate_cloud(p,"pose points")?;
    validate_pose(r,t,p.ncols())?;
    let out = Matrix::from_fn(p.nrows(), p.ncols(), |i,a| t[a] + (0..p.ncols()).map(|b| r[(a,b)]*p[(i,b)]).sum::<f64>());
    finite(&out, "pose")?;
    Ok(out)
}
impl RegistrationResult {
    pub fn apply(&self, p: &Matrix) -> RegResult<Matrix> {
        validate_cloud(p, "points")?;
        if p.ncols() != self.rotation.ncols() {
            return Err(invalid("transform dimension mismatch"));
        }
        let mut out = apply_pose(p, &self.rotation, &self.translation)?;
        for i in 0..out.nrows() {
            for a in 0..out.ncols() {
                out[(i,a)] = (out[(i,a)]-self.center[a])/self.normalization_scale;
            }
        }
        for step in &self.steps {
            let delta = crate::analytic::basis(&out, step.degree)? * &step.coefficients;
            out += delta;
            finite(&out, "analytic composition")?;
        }
        for i in 0..out.nrows() {
            for a in 0..out.ncols() {
                out[(i,a)] = out[(i,a)]*self.normalization_scale+self.center[a];
            }
        }
        finite(&out, "denormalization")?;
        Ok(out)
    }
}
