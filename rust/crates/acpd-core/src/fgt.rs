//! Improved Fast Gauss Transform: cluster the sources, expand the Gaussian in a
//! truncated Taylor series about each cluster centre, evaluate per query.
//!
//! ```text
//! exp(-|q-s|^2/h^2) = exp(-|q-c|^2/h^2) exp(-|s-c|^2/h^2)
//!                     sum_alpha (2^|alpha|/alpha!) ((q-c)/h)^alpha ((s-c)/h)^alpha
//! ```
//!
//! with `h^2 = 2 sigma^2`, so the kernel matches the exact `exp(-|q-s|^2/(2 sigma^2))`
//! used everywhere else. This is an approximation with an explicit, reported cost and a
//! controllable truncation error; it is never selected implicitly and never substituted
//! for a failed exact computation. It is not part of either source paper's own
//! implementation. Enumeration and accumulation order mirror `cpp/src/fgt.cpp` so the two
//! engines agree to rounding.
use crate::types::*;
#[derive(Debug, Clone, Default)]
pub struct FgtCost {
    pub clusters: usize, pub terms: usize, pub cluster_query_pairs: u64,
    pub covering_radius_over_h: f64, pub radius_target_met: bool,
}
/// Multi-indices alpha with total degree <= order, and their 2^|alpha|/alpha! factor.
/// The odometer advances the last axis fastest, matching the C++ enumeration.
fn build_terms(d: usize, order: usize) -> (Vec<Vec<usize>>, Vec<f64>) {
    let mut exponents = Vec::new();
    let mut factors = Vec::new();
    let mut alpha = vec![0usize; d];
    loop {
        let total: usize = alpha.iter().sum();
        if total <= order {
            let mut factorial = 1.0;
            for a in 0..d {
                for j in 2..=alpha[a] {
                    factorial *= j as f64;
                }
            }
            exponents.push(alpha.clone());
            factors.push((2.0_f64).powi(total as i32)/factorial);
        }
        let mut axis = d;
        while axis > 0 && alpha[axis-1] == order {
            alpha[axis-1] = 0;
            axis -= 1;
        }
        if axis == 0 {
            break;
        }
        alpha[axis-1] += 1;
    }
    (exponents, factors)
}
struct Clustering {
    centre: Matrix, owner: Vec<usize>, radius: f64, radius_met: bool,
}
/// Uniform-grid clustering in O(n): sources are bucketed into axis-aligned cells whose
/// half-diagonal is the requested covering radius, so every source is within `target` of
/// its cell centre by construction. Farthest-point clustering would give tighter centres
/// but costs O(k n), which degenerates to O(n^2) once sigma is small enough that nearly
/// every source needs its own centre.
fn grid_clusters(points: &Matrix, cap: usize, target: f64) -> RegResult<Clustering> {
    use std::collections::HashMap;
    let (n, d) = (points.nrows(), points.ncols());
    let cell = 2.0*target/(d as f64).sqrt();
    let mut owner = vec![0usize; n];
    let mut index: HashMap<Vec<i64>, usize> = HashMap::with_capacity(n);
    let mut cells: Vec<Vec<i64>> = Vec::new();
    let mut radius_met = true;
    for i in 0..n {
        let mut key = Vec::with_capacity(d);
        for a in 0..d {
            let scaled = (points[(i,a)]/cell).floor();
            if !scaled.is_finite() || scaled.abs() > 70368744177664.0 {
                return Err(numerical("fgt grid coordinate overflow: rescale points or raise cluster_radius"));
            }
            key.push(scaled as i64);
        }
        match index.get(&key) {
            Some(&existing) => owner[i] = existing,
            None => {
                if cells.len() >= cap {
                    // Cap reached: keep the coarse cluster set rather than silently mixing
                    // resolutions. Reported through radius_target_met.
                    radius_met = false;
                    owner[i] = 0;
                    continue;
                }
                index.insert(key.clone(), cells.len());
                owner[i] = cells.len();
                cells.push(key);
            }
        }
    }
    let clusters = cells.len();
    let centre = Matrix::from_fn(clusters,d,|c,a| (cells[c][a] as f64+0.5)*cell);
    let mut worst: f64 = 0.0;
    for i in 0..n {
        let c = owner[i];
        let distance: f64 = (0..d).map(|a| (points[(i,a)]-centre[(c,a)]).powi(2)).sum();
        worst = worst.max(distance);
    }
    Ok(Clustering {
        centre, owner, radius: worst.sqrt(), radius_met
    })
}
pub fn fgt_transform(s: &Matrix, q: &Matrix, v: &Matrix, sigma2: f64, options: &FgtOptions)
-> RegResult<(Matrix, FgtCost)> {
    options.validate()?;
    if s.nrows() == 0 || q.nrows() == 0 || s.ncols() != q.ncols() || s.nrows() != v.nrows()
    || v.ncols() == 0 || !s.iter().all(|x| x.is_finite()) || !q.iter().all(|x| x.is_finite())
    || !v.iter().all(|x| x.is_finite()) || !sigma2.is_finite() || sigma2 <= 0.0 {
        return Err(invalid("invalid fast Gauss transform arguments"));
    }
    let (d, n, m, columns) = (s.ncols(), s.nrows(), q.nrows(), v.ncols());
    let h = (2.0*sigma2).sqrt();
    let (exponents, factors) = build_terms(d,options.order);
    let width = exponents.len();
    let clustering = grid_clusters(s,options.max_clusters.min(n),options.cluster_radius*h)?;
    let centre = &clustering.centre;
    let clusters = centre.nrows();
    let mut coefficients = Matrix::zeros(clusters*width,columns);
    let mut power = vec![0.0f64; d*(options.order+1)];
    let mut basis = vec![0.0f64; width];
    for i in 0..n {
        let c = clustering.owner[i];
        let mut squared = 0.0;
        for a in 0..d {
            let u = (s[(i,a)]-centre[(c,a)])/h;
            squared += u*u;
            power[a*(options.order+1)] = 1.0;
            for k in 1..=options.order {
                power[a*(options.order+1)+k] = power[a*(options.order+1)+k-1]*u;
            }
        }
        let weight = (-squared).exp();
        for t in 0..width {
            let mut value = factors[t];
            for a in 0..d {
                value *= power[a*(options.order+1)+exponents[t][a]];
            }
            basis[t] = value;
        }
        for t in 0..width {
            let scale = weight*basis[t];
            for col in 0..columns {
                coefficients[(c*width+t,col)] += scale*v[(i,col)];
            }
        }
    }
    // The Taylor factor is symmetric in the source and query monomials, so it is folded
    // into the source coefficients only and the query side uses plain powers.
    let mut out = Matrix::zeros(m,columns);
    let cutoff = options.cutoff_radius*options.cutoff_radius;
    let mut pairs = 0u64;
    let mut query_basis = vec![0.0f64; width];
    for i in 0..m {
        for c in 0..clusters {
            let mut squared = 0.0;
            for a in 0..d {
                let z = (q[(i,a)]-centre[(c,a)])/h;
                squared += z*z;
            }
            // Prune before building the power table: a pruned cluster costs only d flops.
            if squared > cutoff {
                continue;
            }
            pairs += 1;
            for a in 0..d {
                let z = (q[(i,a)]-centre[(c,a)])/h;
                power[a*(options.order+1)] = 1.0;
                for k in 1..=options.order {
                    power[a*(options.order+1)+k] = power[a*(options.order+1)+k-1]*z;
                }
            }
            let gaussian = (-squared).exp();
            for t in 0..width {
                let mut value = 1.0;
                for a in 0..d {
                    value *= power[a*(options.order+1)+exponents[t][a]];
                }
                query_basis[t] = gaussian*value;
            }
            for t in 0..width {
                let scale = query_basis[t];
                for col in 0..columns {
                    out[(i,col)] += scale*coefficients[(c*width+t,col)];
                }
            }
        }
    }
    finite(&out,"fast Gauss transform")?;
    Ok((out, FgtCost {
        clusters, terms: width, cluster_query_pairs: pairs,
        covering_radius_over_h: clustering.radius/h, radius_target_met: clustering.radius_met,
    }))
}
