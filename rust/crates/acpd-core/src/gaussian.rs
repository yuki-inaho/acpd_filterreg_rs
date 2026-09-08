//! Direct CPD posterior statistics and FilterReg's inverse posterior moments.
use crate::lattice::{
    lattice_transform, FilteredValues, FixedNoBlurLattice
};
use crate::types::*;
use std::f64::consts::PI;
#[derive(Debug, Clone)]
pub struct Statistics {
    pub rho: Vector, pub px: Matrix, pub x2: Vector, pub normals: Matrix,
    pub mass: f64, pub nll: f64, pub vertices: usize, pub unsupported: usize,
    pub lattice_mode: String,
}
pub fn gaussian_sum(s: &Matrix, q: &Matrix, v: &Matrix, sigma2: f64, backend: Backend,
fgt: &FgtOptions) -> RegResult<Matrix> {
    Ok(lattice_transform(s,q,v,sigma2,backend,fgt)?.values)
}
pub fn moment_values(x: &Matrix, normals: Option<&Matrix>) -> RegResult<Matrix> {
    validate_cloud(x,"fixed")?;
    if let Some(n) = normals {
        validate_normals(n,x)?;
    }
    let d = x.ncols();
    let values = Matrix::from_fn(x.nrows(),d+2+normals.map_or(0,|_|d),|i,a| {
        if a == 0 {
            1.0
        }
        else if a <= d {
            x[(i,a-1)]
        }
        else if a == d+1 {
            x.row(i).norm_squared()
        }
        else {
            normals.expect("normal channel only allocated when provided")[(i,a-d-2)]
        }
    });
    finite(&values,"Gaussian moment values")?;
    Ok(values)
}
pub fn initial_variance(x: &Matrix, y: &Matrix) -> RegResult<f64> {
    validate_pair(x,y,false)?;
    let mx = mean(x);
    let my = mean(y);
    let sx = (0..x.nrows()).map(|i| (0..x.ncols()).map(|a| (x[(i,a)]-mx[a]).powi(2)).sum::<f64>()).sum::<f64>()/x.nrows() as f64;
    let sy = (0..y.nrows()).map(|i| (0..y.ncols()).map(|a| (y[(i,a)]-my[a]).powi(2)).sum::<f64>()).sum::<f64>()/y.nrows() as f64;
    let result = (sx+sy+(&mx-&my).norm_squared())/x.ncols() as f64;
    if !result.is_finite() {
        return Err(numerical("initial variance overflow"));
    }
    Ok(result)
}
pub fn posterior_statistics(
x: &Matrix, y: &Matrix, sigma2: f64, w: f64, inverse: bool, backend: Backend,
normals: Option<&Matrix>, cache: Option<&FixedNoBlurLattice>, fgt: &FgtOptions,
) -> RegResult<Statistics> {
    validate_pair(x,y,false)?;
    positive(sigma2,"sigma2")?;
    if !w.is_finite() || !(0.0..1.0).contains(&w) {
        return Err(invalid("w must be in [0,1)"));
    }
    if let Some(n) = normals {
        validate_normals(n,x)?;
    }
    if !inverse && backend != Backend::Direct && backend != Backend::Fgt {
        return Err(invalid("Analytic-CPD accepts the exact direct posterior or the explicitly selected fgt approximation; lattice backends normalize in the other direction"));
    }
    let (m,n,d) = (y.nrows(),x.nrows(),x.ncols());
    let (centers,queries) = if inverse {
        (n,m)
    } else {
        (m,n)
    };
    let normalizer = 0.5*d as f64*((2.0*PI).ln()+sigma2.ln());
    let logc = if w == 0.0 {
        f64::NEG_INFINITY
    } else {
        normalizer+w.ln()-(-w).ln_1p()+(centers as f64/queries as f64).ln()
    };
    let logfactor = (-w).ln_1p()-(centers as f64).ln()-normalizer;
    let mut out = Statistics {
        rho:Vector::zeros(m),px:Matrix::zeros(m,d),x2:Vector::zeros(m),
        normals:Matrix::zeros(if normals.is_some() {
            m
        }else {
            0
        },d),mass:0.0,nll:0.0,vertices:0,unsupported:0,lattice_mode:"direct".into()
    };
    if !inverse && backend == Backend::Fgt {
        // Two O(N+M) transforms replace the O(NM) streaming loop: first the per-fixed-point
        // support G_j = sum_i K_ij, then the moving-point moments weighted by 1/(G_j + C).
        // Same posterior as the direct branch, evaluated approximately. This branch works in
        // linear space, so it needs a representable outlier constant; direct stays the
        // log-sum-exp reference.
        let outlier = if w == 0.0 {
            0.0
        } else {
            logc.exp()
        };
        if !outlier.is_finite() {
            return Err(numerical("fgt posterior needs a representable outlier constant; use the direct backend"));
        }
        let (support,_) = crate::fgt::fgt_transform(y,x,&Matrix::from_element(m,1,1.0),sigma2,fgt)?;
        let mut weighted = moment_values(x,None)?;
        for j in 0..n {
            let denominator = support[(j,0)]+outlier;
            if !(denominator > 0.0) {
                return Err(numerical("fgt posterior has no representable support"));
            }
            out.nll -= denominator.ln()+logfactor;
            for c in 0..weighted.ncols() {
                weighted[(j,c)] /= denominator;
            }
        }
        let (moments,cost) = crate::fgt::fgt_transform(x,y,&weighted,sigma2,fgt)?;
        out.vertices = cost.clusters;
        out.lattice_mode = "fgt".into();
        for i in 0..m {
            out.rho[i] = moments[(i,0)].max(0.0);
            for a in 0..d {
                out.px[(i,a)] = moments[(i,a+1)];
            }
            out.x2[i] = moments[(i,d+1)].max(0.0);
        }
    } else if inverse && backend != Backend::Direct {
        let filtered = if let (Backend::PermutohedralNoBlur,Some(cache)) = (backend,cache) {
            FilteredValues {
                values:cache.slice(&(y/sigma2.sqrt()))?,vertices:cache.lattice_size(),mode:"original_noblur".into()
            }
        } else {
            lattice_transform(x,y,&moment_values(x,normals)?,sigma2,backend,fgt)?
        };
        out.vertices = filtered.vertices;
        out.lattice_mode = filtered.mode;
        let moments = filtered.values;
        for i in 0..m {
            let m0 = moments[(i,0)];
            if m0 < 0.0 {
                return Err(numerical("negative permutohedral zeroth moment"));
            }
            if m0 == 0.0 {
                out.unsupported += 1;
                out.nll -= if w > 0.0 {
                    w.ln()-(m as f64).ln()
                } else {
                    f64::NEG_INFINITY
                };
                continue;
            }
            let logm = m0.ln();
            let maxlog = logm.max(logc);
            let logden = maxlog+((logm-maxlog).exp()+(logc-maxlog).exp()).ln();
            out.rho[i] = (logm-logden).exp();
            for a in 0..d {
                out.px[(i,a)] = out.rho[i]*moments[(i,1+a)]/m0;
            }
            out.x2[i] = out.rho[i]*moments[(i,d+1)]/m0;
            if normals.is_some() {
                for a in 0..d {
                    out.normals[(i,a)] = moments[(i,d+2+a)]/m0;
                }
            }
            out.nll -= logden+logfactor;
        }
    } else {
        // Stream one posterior column (CPD) or row (FilterReg) at a time.
        // Log-sum-exp changes no finite non-underflow probability.
        let mut weights = vec![0.0;
        centers];
        for q in 0..queries {
            let mut maxlog = logc;
            for c in 0..centers {
                let (i,j) = if inverse {
                    (q,c)
                } else {
                    (c,q)
                };
                let distance = (0..d).map(|a| (y[(i,a)]-x[(j,a)]).powi(2)).sum::<f64>();
                if !distance.is_finite() {
                    return Err(numerical("squared distance overflow"));
                }
                weights[c] = -0.5*(distance/sigma2);
                maxlog = maxlog.max(weights[c]);
            }
            if !maxlog.is_finite() {
                return Err(numerical("posterior has no representable support"));
            }
            let mut denominator = (logc-maxlog).exp();
            for v in &mut weights {
                *v = (*v-maxlog).exp();
                denominator += *v;
            }
            out.nll -= maxlog+denominator.ln()+logfactor;
            for c in 0..centers {
                let p = weights[c]/denominator;
                let (i,j) = if inverse {
                    (q,c)
                } else {
                    (c,q)
                };
                out.rho[i] += p;
                for a in 0..d {
                    out.px[(i,a)] += p*x[(j,a)];
                    if let Some(normals) = normals {
                        out.normals[(i,a)] += p*normals[(j,a)];
                    }
                }
                out.x2[i] += p*x.row(j).norm_squared();
            }
        }
        if normals.is_some() {
            for i in 0..m {
                if out.rho[i] > 0.0 {
                    for a in 0..d {
                        out.normals[(i,a)] /= out.rho[i];
                    }
                }
            }
        }
    }
    out.mass = out.rho.sum();
    if !out.mass.is_finite() || !out.px.iter().chain(out.x2.iter()).all(|v| v.is_finite()) {
        return Err(numerical("non-finite posterior moments"));
    }
    Ok(out)
}
pub fn variance_from_statistics(y: &Matrix, stats: &Statistics, floor: f64) -> RegResult<f64> {
    positive(floor,"variance floor")?;
    if y.nrows() != stats.rho.len() || stats.px.shape() != y.shape() || stats.x2.len() != y.nrows() {
        return Err(invalid("variance statistic dimensions mismatch"));
    }
    validate_cloud(y,"variance points")?;
    if !stats.mass.is_finite() || stats.mass <= 1e-14 || stats.rho.iter().any(|&v| !v.is_finite() || v<0.0)
    || !stats.px.iter().chain(stats.x2.iter()).all(|v| v.is_finite()) {
        return Err(numerical("invalid or insufficient posterior mass"));
    }
    let (mut residual,mut magnitude) = (0.0,0.0);
    for i in 0..y.nrows() {
        let a = stats.x2[i];
        let b = 2.0*(0..y.ncols()).map(|j| y[(i,j)]*stats.px[(i,j)]).sum::<f64>();
        let c = stats.rho[i]*y.row(i).norm_squared();
        residual += a-b+c;
        magnitude += a.abs()+b.abs()+c.abs();
    }
    if !residual.is_finite() || residual < -1e-10*magnitude.max(1.0) {
        return Err(numerical("invalid second-moment residual"));
    }
    Ok(floor.max(residual.max(0.0)/(y.ncols() as f64*stats.mass)))
}
