//! Two-stage registration. The rigid pose is frozen before the analytic stage.
use crate::analytic::{
    degree_schedule, exponents, fit_analytic, Fit
};
use crate::gaussian::{
    initial_variance, moment_values, posterior_statistics, variance_from_statistics, Statistics
};
use crate::lattice::FixedNoBlurLattice;
use crate::rigid::fit_rigid;
use crate::types::*;
fn world_units(stage: &mut Stage, scale: f64) {
    stage.initial_sigma2 *= scale*scale;
    stage.final_sigma2 *= scale*scale;
    for h in &mut stage.history {
        h.sigma2 *= scale*scale;
        h.step_rms *= scale;
        h.fit_rms *= scale;
    }
}
fn motion_rms(a: &Matrix,b: &Matrix) -> f64 {
    (a-b).norm()/(a.nrows() as f64).sqrt()
}
pub fn registration(
fixed: &Matrix, moving: &Matrix, options: &Options,
initial_rotation: &Matrix, initial_translation: &Vector,
target_normals: Option<&Matrix>,
) -> RegResult<RegistrationResult> {
    validate_pair(fixed,moving,true)?;
    options.validate()?;
    let d = fixed.ncols();
    validate_pose(initial_rotation,initial_translation,d)?;
    if let Some(normals) = target_normals {
        validate_normals(normals,fixed)?;
    }
    if options.method != Method::Analytic && options.rigid.objective == "point_to_plane" && target_normals.is_none() {
        return Err(invalid("point-to-plane requires target_normals"));
    }
    let center = mean(fixed);
    let mut x = Matrix::from_fn(fixed.nrows(),d,|i,a| fixed[(i,a)]-center[a]);
    let mut y0 = Matrix::from_fn(moving.nrows(),d,|i,a| moving[(i,a)]-center[a]);
    let scale = x.norm()/(x.nrows() as f64).sqrt();
    if !scale.is_finite() || scale <= 1e-150 {
        return Err(numerical("fixed cloud has zero or unrepresentable extent"));
    }
    x /= scale;
    y0 /= scale;
    finite(&x,"normalization")?;
    finite(&y0,"normalization")?;
    let mut rotation = initial_rotation.clone();
    let mut translation = (initial_rotation*&center+initial_translation-&center)/scale;
    let mut y = apply_pose(&y0,&rotation,&translation)?;
    let mut sigma2 = 0.0;
    let mut rigid_stage = Stage::default();
    let mut analytic_stage = Stage::default();
    let mut steps = Vec::new();
    if options.method != Method::Analytic {
        let opt = &options.rigid;
        sigma2 = if opt.sigma2 > 0.0 {
            opt.sigma2/scale/scale
        }else {
            initial_variance(&x,&y)?
        };
        sigma2 = sigma2.max(opt.min_sigma2);
        rigid_stage.initial_sigma2 = sigma2;
        rigid_stage.stop_reason = "iteration_limit".into();
        let mut cache: Option<FixedNoBlurLattice> = None;
        let mut indexed_sigma = -1.0;
        let normals = if opt.objective == "point_to_plane" {
            target_normals
        }else {
            None
        };
        for it in 0..opt.max_iterations {
            if options.backend == Backend::PermutohedralNoBlur && (cache.is_none() || sigma2 != indexed_sigma) {
                cache = Some(FixedNoBlurLattice::new(&(&x/sigma2.sqrt()),&moment_values(&x,normals)?)?);
                indexed_sigma = sigma2;
                rigid_stage.index_builds += 1;
            }
            let stats = posterior_statistics(&x,&y,sigma2,opt.w,true,options.backend,normals,cache.as_ref(),&options.fgt)?;
            if options.backend == Backend::Permutohedral || options.backend == Backend::Fgt {
                rigid_stage.index_builds += 1;
            }
            if options.backend == Backend::Probreg {
                rigid_stage.index_builds += if stats.lattice_mode == "probreg_noblur" {
                    2
                }else {
                    1
                };
            }
            let fit = fit_rigid(&y,&stats,opt)?;
            // Updated coordinates and ambient dimension, not the old-coordinate
            // / literal-three update found in the attached probreg Python code.
            let next_sigma = if opt.update_sigma2 {
                variance_from_statistics(&fit.next,&stats,opt.min_sigma2)?
            }else {
                sigma2
            };
            let motion = motion_rms(&y,&fit.next);
            let change = (next_sigma-sigma2).abs()/(sigma2+1e-12);
            rotation = &fit.rotation*rotation;
            translation = &fit.rotation*translation+&fit.translation;
            y = fit.next;
            rigid_stage.history.push(Iteration {
                iteration:it+1,raw_degree:0,degree:0,active:fit.active,rank:fit.rank,
                sigma2:next_sigma,nll_before:stats.nll,step_rms:motion,fit_rms:fit.fit_rms,
                lattice_vertices:stats.vertices,lattice_mode:stats.lattice_mode,
            });
            sigma2 = next_sigma;
            rigid_stage.best_iteration = it+1;
            if motion <= opt.tolerance && change <= opt.tolerance {
                rigid_stage.converged = true;
                rigid_stage.stop_reason = "tolerance".into();
                break;
            }
        }
        rigid_stage.final_sigma2 = sigma2;
    }
    // No assignment to rotation or world_translation after this point.
    let world_translation = scale*&translation+&center-&rotation*&center;
    let rigid_transformed = apply_pose(moving,&rotation,&world_translation)?;
    if options.method != Method::Rigid {
        let opt = &options.analytic;
        // "auto" is resolved from the declared method, never from a runtime quality test.
        // Nonrigid inherits the variance the rigid stage annealed to; standalone analytic
        // uses the ACPD/CPD all-pair initialization of the paper.
        let resolved_initialization = if opt.initialization == "auto" {
            if options.method == Method::Nonrigid {
                "filterreg"
            }else {
                "cpd"
            }
        }else {
            opt.initialization.as_str()
        };
        if opt.sigma2 > 0.0 {
            sigma2 = opt.sigma2/scale/scale;
        }
        else if resolved_initialization == "cpd" {
            sigma2 = initial_variance(&x,&y)?;
        }
        sigma2 = sigma2.max(opt.min_sigma2);
        analytic_stage.initial_sigma2 = sigma2;
        analytic_stage.stop_reason = "iteration_limit".into();
        let mut best_y = y.clone();
        let mut best_sigma = sigma2;
        let mut best_score = (d as f64*sigma2).sqrt();
        let mut previous_score = best_score;
        let mut best_steps = 0;
        let mut stable = 0;
        let mut no_improve = 0;
        let mut previous_degree: Option<usize> = None;
        let schedule = degree_schedule(opt.max_iterations,opt.min_degree,opt.max_degree)?;
        // Degree continuation: the schedule bounds how long each degree MAY run, it does
        // not require running that long. Convergence at a degree below max_degree advances
        // the continuation instead of ending the whole stage; only the highest scheduled
        // degree may terminate it. See docs/SOURCE_DIFFERENCES.md, Analytic-CPD 10.
        fn advance_degree(schedule: &[usize], at: usize) -> usize {
            let degree = schedule[at];
            let mut next = at;
            while next < schedule.len() && schedule[next] <= degree {
                next += 1;
            }
            next
        }
        // Dimensionless: the fixed cloud is normalized to unit RMS radius, and the limit
        // tracks its own largest radius so an outlier-heavy cloud is not penalised.
        let divergence_limit = opt.divergence_radius
        *x.row_iter().map(|r| r.norm()).fold(1.0_f64,f64::max);
        let mut cursor = 0usize;
        let mut it = 0usize;
        while it < opt.max_iterations && cursor < schedule.len() {
            let raw_degree = schedule[cursor];
            let attempt = (|| -> RegResult<Option<(Statistics,Fit,f64)>> {
                let stats = posterior_statistics(&x,&y,sigma2,opt.w,false,opt.backend,None,None,&options.fgt)?;
                let active = stats.rho.iter().filter(|&&v| v>opt.min_mass).count();
                if stats.mass <= opt.min_mass || active < exponents(d,opt.min_degree)?.len() {
                    return Ok(None);
                }
                let fit = fit_analytic(&y,&stats,raw_degree,opt)?;
                let next_sigma = variance_from_statistics(&fit.next,&stats,opt.min_sigma2)?;
                Ok(Some((stats,fit,next_sigma)))
            })();
            let (stats,fit,next_sigma) = match attempt {
                Ok(Some(value)) => value,
                Ok(None) => {
                    analytic_stage.stop_reason = "insufficient_posterior_mass".into();
                    break;
                }
                // The analytic map is unregularized by design, so an ill-posed pair can drive
                // the Taylor basis or the second-moment residual out of range. Report it as a
                // stop reason and return the best state that was actually evaluated; never a
                // silently substituted or re-tuned computation. A failure before any iteration
                // succeeded is a genuine input error and still propagates.
                Err(Error::Numerical(_)) if !analytic_stage.history.is_empty() => {
                    analytic_stage.stop_reason = "numerical_divergence".into();
                    break;
                }
                Err(error) => return Err(error),
            };
            if fit.next.row_iter().map(|r| r.norm()).fold(0.0_f64,f64::max) > divergence_limit {
                // The analytic map is a global polynomial constrained only where the posterior
                // supports it. Once a point loses support it can be extrapolated arbitrarily
                // far, and the rho-weighted variance cannot see it, so a diverging state could
                // otherwise be recorded as best. Refuse the iterate; the fit is left as computed.
                analytic_stage.stop_reason = "numerical_divergence".into();
                break;
            }
            if previous_degree.is_some_and(|degree| degree != fit.step.degree) {
                stable = 0;
                no_improve = 0;
            }
            previous_degree = Some(fit.step.degree);
            let score = (d as f64*next_sigma).sqrt();
            let delta_y = (&fit.next-&y).norm()/(y.norm()+1e-12);
            let delta_sigma = (next_sigma-sigma2).abs()/(sigma2.abs()+1e-12);
            let delta_score = (score-previous_score).abs()/(previous_score.abs()+1e-12);
            let motion = motion_rms(&y,&fit.next);
            let significant = best_score-score > 1e-12_f64.max(opt.improvement_relative*best_score.abs());
            no_improve = if significant {
                0
            }else {
                no_improve+1
            };
            let degree = fit.step.degree;
            steps.push(fit.step);
            y = fit.next;
            sigma2 = next_sigma;
            analytic_stage.history.push(Iteration {
                iteration:it+1,raw_degree,degree,active:fit.active,rank:fit.rank,
                sigma2,nll_before:stats.nll,step_rms:motion,fit_rms:fit.fit_rms,
                lattice_vertices:stats.vertices,lattice_mode:stats.lattice_mode.clone(),
            });
            // Actual minimum is retained; significant improvement controls only patience.
            if score < best_score {
                best_score = score;
                best_y = y.clone();
                best_sigma = sigma2;
                best_steps = steps.len();
                analytic_stage.best_iteration = it+1;
            }
            stable = if delta_y<opt.tolerance && delta_sigma<opt.tolerance && delta_score<opt.tolerance {
                stable+1
            }else {
                0
            };
            previous_score = score;
            if score < opt.tolerance {
                analytic_stage.converged = true;
                analytic_stage.stop_reason = "residual_tolerance".into();
                break;
            }
            if it+1 >= opt.min_iterations {
                let is_stable = stable >= opt.stable_patience;
                let is_rebound = no_improve >= opt.stable_patience
                && score > best_score*(1.0+opt.rebound_relative)+1e-12;
                let is_stalled = no_improve >= opt.no_improve_patience;
                if is_stable || is_rebound || is_stalled {
                    let next = advance_degree(&schedule,cursor);
                    if next < schedule.len() {
                        // Converged at this degree with higher degrees still scheduled:
                        // continue the sequence rather than reporting the whole stage done.
                        cursor = next;
                        stable = 0;
                        no_improve = 0;
                        previous_degree = None;
                        it += 1;
                        continue;
                    }
                    if is_stable {
                        analytic_stage.converged = true;
                        analytic_stage.stop_reason = "stable_tolerance".into();
                    }
                    else {
                        analytic_stage.stop_reason = if is_rebound {
                            "internal_rebound".into()
                        }else {
                            "no_improvement".into()
                        };
                    }
                    break;
                }
            }
            it += 1;
            cursor += 1;
        }
        y = best_y;
        sigma2 = best_sigma;
        steps.truncate(best_steps);
        analytic_stage.final_sigma2 = sigma2;
    }
    let transformed = Matrix::from_fn(y.nrows(),d,|i,a| y[(i,a)]*scale+center[a]);
    finite(&transformed,"result")?;
    if !(sigma2*scale*scale).is_finite() {
        return Err(numerical("world-coordinate variance overflow"));
    }
    world_units(&mut rigid_stage,scale);
    world_units(&mut analytic_stage,scale);
    Ok(RegistrationResult {
        rotation,translation:world_translation,center,normalization_scale:scale,
        transformed,rigid_transformed,steps,rigid_stage,analytic_stage,
        sigma2:sigma2*scale*scale,method:options.method,backend:options.backend,
    })
}
