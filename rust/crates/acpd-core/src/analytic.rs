//! Factorial-scaled Taylor mappings and exact, unregularized weighted fitting.
use crate::gaussian::Statistics;
use crate::types::*;
pub fn exponents(d: usize, degree: usize) -> RegResult<Vec<[usize;
3]>> {
    if ![2,3].contains(&d) || degree > 10 {
        return Err(invalid("basis requires dimension 2/3 and degree 0..10"));
    }
    let mut out = Vec::new();
    for r in 0..=degree {
        for a in (0..=r).rev() {
            if d == 2 {
                out.push([a,r-a,0]);
            }
            else {
                for b in (0..=r-a).rev() {
                    out.push([a,b,r-a-b]);
                }
            }
        }
    }
    Ok(out)
}
pub fn basis(points: &Matrix, degree: usize) -> RegResult<Matrix> {
    validate_cloud(points,"basis points")?;
    let powers = exponents(points.ncols(),degree)?;
    let phi = Matrix::from_fn(points.nrows(),powers.len(),|i,k| {
        let mut term = 1.0;
        for a in 0..points.ncols() {
            for j in 1..=powers[k][a] {
                term *= points[(i,a)]/j as f64;
            }
        }
        term
    });
    finite(&phi,"Taylor basis")?;
    Ok(phi)
}
/// Exact triangular-prefix remainder allocation from supplied Algo.h.
pub fn degree_schedule(iterations: usize, low: usize, high: usize) -> RegResult<Vec<usize>> {
    if iterations == 0 || low < 1 || high < low || high > 10 {
        return Err(invalid("invalid degree schedule"));
    }
    let count = high-low+1;
    let unit = count*(count+1)/2;
    let mut lengths: Vec<usize> = (0..count).map(|k| (count-k)*(iterations/unit)).collect();
    let mut remaining = iterations%unit;
    for prefix in (1..=count).rev() {
        let take = prefix.min(remaining);
        for length in lengths.iter_mut().take(take) {
            *length += 1;
        }
        remaining -= take;
        if remaining == 0 {
            break;
        }
    }
    let mut out = Vec::with_capacity(iterations);
    for (k,&length) in lengths.iter().enumerate() {
        out.extend(std::iter::repeat_n(low+k,length));
    }
    Ok(out)
}
#[derive(Debug, Clone)]
pub struct Fit {
    pub step: AnalyticStep, pub next: Matrix, pub active: usize, pub rank: usize, pub fit_rms: f64
}
pub fn fit_analytic(y: &Matrix, stats: &Statistics, mut degree: usize, o: &AnalyticOptions) -> RegResult<Fit> {
    validate_cloud(y,"analytic points")?;
    o.validate()?;
    if stats.rho.len() != y.nrows() || stats.px.shape() != y.shape()
    || !stats.rho.iter().chain(stats.px.iter()).all(|v| v.is_finite()) {
        return Err(invalid("invalid analytic statistics"));
    }
    if degree < o.min_degree || degree > o.max_degree {
        return Err(invalid("requested degree outside options"));
    }
    if stats.rho.iter().any(|v| *v < 0.0) {
        return Err(invalid("negative analytic fitting weight"));
    }
    let active: Vec<usize> = (0..y.nrows()).filter(|&i| stats.rho[i] > o.min_mass).collect();
    let n = active.len();
    let d = y.ncols();
    while degree > o.min_degree && exponents(d,degree)?.len() > n {
        degree -= 1;
    }
    let phi = basis(y,degree)?;
    let k = phi.ncols();
    if n < k {
        return Err(numerical("insufficient active rows for the minimum analytic degree"));
    }
    let design = Matrix::from_fn(n,k,|r,c| stats.rho[active[r]].sqrt()*phi[(active[r],c)]);
    let targets = Matrix::from_fn(n,d,|r,a| stats.px[(active[r],a)]/stats.rho[active[r]].sqrt());
    finite(&design,"weighted Taylor design")?;
    finite(&targets,"weighted targets")?;
    let svd = design.svd(true,true);
    let threshold = o.rank_tolerance*svd.singular_values.iter().copied().fold(0.0,f64::max);
    let rank = svd.singular_values.iter().filter(|&&s| s > threshold).count();
    // Minimum-norm ABSOLUTE mapping for a rank-deficient design, not an
    // identity-biased solution. No ridge, projection, cap, or damping.
    let absolute = svd.solve(&targets,threshold).map_err(numerical)?;
    let next = &phi*&absolute;
    finite(&next,"analytic M-step")?;
    let mut coefficients = absolute;
    for a in 0..d {
        coefficients[(1+a,a)] -= 1.0;
    }
    let mass: f64 = active.iter().map(|&i| stats.rho[i]).sum();
    let sse: f64 = active.iter().map(|&i| stats.rho[i]*(0..d).map(|a| (stats.px[(i,a)]/stats.rho[i]-next[(i,a)]).powi(2)).sum::<f64>()).sum();
    Ok(Fit {
        step:AnalyticStep {
            degree,coefficients
        },next,active:n,rank,fit_rms:(sse/mass).sqrt()
    })
}
