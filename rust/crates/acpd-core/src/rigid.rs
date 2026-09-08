//! Left-composed twist increments and the small normal systems of FilterReg.
use crate::gaussian::Statistics;
use crate::types::*;
#[derive(Debug, Clone)]
pub struct Pose {
    pub rotation: Matrix, pub translation: Vector
}
#[derive(Debug, Clone)]
pub struct RigidFit {
    pub rotation: Matrix, pub translation: Vector, pub next: Matrix,
    pub rank: usize, pub active: usize, pub fit_rms: f64,
}
pub fn twist_jacobian(p: &Vector) -> RegResult<Matrix> {
    if ![2,3].contains(&p.len()) || !p.iter().all(|x| x.is_finite()) {
        return Err(invalid("twist point needs 2/3 finite coordinates"));
    }
    if p.len() == 2 {
        return Ok(Matrix::from_row_slice(2,3,&[-p[1],1.0,0.0,p[0],0.0,1.0]));
    }
    Ok(Matrix::from_row_slice(3,6,&[
    0.0,p[2],-p[1],1.0,0.0,0.0,
    -p[2],0.0,p[0],0.0,1.0,0.0,
    p[1],-p[0],0.0,0.0,0.0,1.0,
    ]))
}
pub fn twist_pose(delta: &Vector, d: usize) -> RegResult<Pose> {
    let expected = if d==2 {
        3
    }else {
        6
    };
    if ![2,3].contains(&d) || delta.len() != expected || !delta.iter().all(|x| x.is_finite()) {
        return Err(invalid("invalid twist size/values"));
    }
    let rotation = if d == 2 {
        let (s,c) = delta[0].sin_cos();
        Matrix::from_row_slice(2,2,&[c,-s,s,c])
    } else {
        let theta2 = delta.rows(0,3).norm_squared();
        let theta = theta2.sqrt();
        let a = Matrix::from_row_slice(3,3,&[
        0.0,-delta[2],delta[1], delta[2],0.0,-delta[0], -delta[1],delta[0],0.0,
        ]);
        let sinc = if theta < 1e-4 {
            1.0-theta2/6.0+theta2*theta2/120.0
        }else {
            theta.sin()/theta
        };
        let cosc = if theta < 1e-4 {
            0.5-theta2/24.0+theta2*theta2/720.0
        }else {
            (1.0-theta.cos())/theta2
        };
        Matrix::identity(3,3)+sinc*&a+cosc*(&a*&a)
    };
    // The original code uses exp(rotation-vector) and an independent translation
    // increment. It is a retraction, not the full SE(d) exponential's V*v.
    let translation = delta.rows(delta.len()-d,d).into_owned();
    Ok(Pose {
        rotation,translation
    })
}
pub fn fit_rigid(y: &Matrix, stats: &Statistics, options: &FilterOptions) -> RegResult<RigidFit> {
    validate_cloud(y,"rigid points")?;
    options.validate()?;
    let d = y.ncols();
    if stats.rho.len() != y.nrows() || stats.px.shape() != y.shape() || !stats.mass.is_finite() || stats.mass <= 1e-14 || stats.rho.iter().any(|&v| v<0.0)
    || !stats.rho.iter().chain(stats.px.iter()).all(|x| x.is_finite()) {
        return Err(numerical("invalid or insufficient FilterReg posterior mass"));
    }
    let plane = options.objective == "point_to_plane";
    if plane && (stats.normals.shape() != y.shape() || !stats.normals.iter().all(|x| x.is_finite())) {
        return Err(invalid("point-to-plane requires filtered target normals"));
    }
    let active: Vec<usize> = (0..y.nrows()).filter(|&i| stats.rho[i] > 1e-12
    && (!plane || stats.normals.row(i).norm() > 1e-12)).collect();
    if active.len() < d {
        return Err(numerical("insufficient effective FilterReg correspondences"));
    }
    let mut rotation = Matrix::identity(d,d);
    let mut translation = Vector::zeros(d);
    let mut next = y.clone();
    let rank;
    if options.solver == "kabsch" {
        let mass: f64 = active.iter().map(|&i| stats.rho[i]).sum();
        let cy = Vector::from_fn(d,|a,_| active.iter().map(|&i| stats.rho[i]*y[(i,a)]).sum::<f64>()/mass);
        let cz = Vector::from_fn(d,|a,_| active.iter().map(|&i| stats.px[(i,a)]).sum::<f64>()/mass);
        let h = Matrix::from_fn(d,d,|a,b| active.iter().map(|&i| (y[(i,a)]-cy[a])*(stats.px[(i,b)]-stats.rho[i]*cz[b])).sum());
        let svd = h.svd(true,true);
        let threshold = 1e-12*svd.singular_values.iter().copied().fold(0.0,f64::max);
        rank = svd.singular_values.iter().filter(|&&s| s>threshold).count();
        if rank < d-1 {
            return Err(numerical("rigid pose is not identifiable"));
        }
        let u = svd.u.ok_or_else(|| numerical("missing SVD U"))?;
        let vt = svd.v_t.ok_or_else(|| numerical("missing SVD V"))?;
        let mut sign = Matrix::identity(d,d);
        if (vt.transpose()*u.transpose()).determinant() < 0.0 {
            sign[(d-1,d-1)] = -1.0;
        }
        rotation = vt.transpose()*sign*u.transpose();
        translation = cz-&rotation*cy;
        next = apply_pose(y,&rotation,&translation)?;
    } else {
        let parameters = if d==2 {
            3
        }else {
            6
        };
        // Eq.(18): accumulate only a 3x3 or 6x6 system, independent of point count.
        for _ in 0..options.inner_iterations {
            let mut normal = Matrix::zeros(parameters,parameters);
            let mut gradient = Vector::zeros(parameters);
            for &i in &active {
                let point = next.row(i).transpose().into_owned();
                let jp = twist_jacobian(&point)?;
                let residual = Vector::from_fn(d,|a,_| next[(i,a)]-stats.px[(i,a)]/stats.rho[i]);
                if plane {
                    let projected = stats.normals.row(i)*&jp;
                    let value: f64 = (0..d).map(|a| stats.normals[(i,a)]*residual[a]).sum();
                    normal += stats.rho[i]*(projected.transpose()*&projected);
                    gradient += stats.rho[i]*value*projected.transpose();
                } else {
                    normal += stats.rho[i]*(jp.transpose()*&jp);
                    gradient += stats.rho[i]*(jp.transpose()*residual);
                }
            }
            finite(&normal,"twist normal matrix")?;
            if !gradient.iter().all(|x| x.is_finite()) {
                return Err(numerical("non-finite twist gradient"));
            }
            let svd = normal.svd(true,true);
            let threshold = 1e-12*svd.singular_values.iter().copied().fold(0.0,f64::max);
            let r = svd.singular_values.iter().filter(|&&s| s>threshold).count();
            if r < parameters {
                return Err(numerical("twist normal system is rank deficient"));
            }
            let delta = svd.solve(&(-gradient),threshold).map_err(numerical)?;
            let step = twist_pose(&delta,d)?;
            next = apply_pose(&next,&step.rotation,&step.translation)?;
            rotation = &step.rotation*rotation;
            translation = &step.rotation*translation+step.translation;
            if delta.norm() < options.tolerance {
                break;
            }
        }
        rank = parameters;
    }
    let mut error = 0.0;
    let mut mass = 0.0;
    for &i in &active {
        let residual = Vector::from_fn(d,|a,_| next[(i,a)]-stats.px[(i,a)]/stats.rho[i]);
        let value = if plane {
            (0..d).map(|a| stats.normals[(i,a)]*residual[a]).sum::<f64>().powi(2)
        }
        else {
            residual.norm_squared()
        };
        error += stats.rho[i]*value;
        mass += stats.rho[i];
    }
    Ok(RigidFit {
        rotation,translation,next,rank,active:active.len(),fit_rms:(error/mass).sqrt()
    })
}
