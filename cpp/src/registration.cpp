#include "acpd/analytic.hpp"
#include "acpd/rigid.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
namespace acpd {
    namespace {
        void world_units(Stage& s,double scale) {
            s.initial_sigma2*=scale*scale;
            s.final_sigma2*=scale*scale;
            for(auto& h:s.history) {
                h.sigma2*=scale*scale;
                h.step_rms*=scale;
                h.fit_rms*=scale;
            }
        }
        double motion_rms(const Matrix& a,const Matrix& b) {
            return (a-b).norm()/std::sqrt(double(a.rows()));
        }
    }
    Result registration(const Matrix& fixed,const Matrix& moving,const Options& o,const Matrix& r0,const Vector& t0,const Matrix& normals) {
        validate_pair(fixed,moving,true);
        o.validate();
        const int d=static_cast<int>(fixed.cols());
        validate_pose(r0,t0,d);
        if(normals.size()) validate_normals(normals,fixed);
        if(o.method!=Method::Analytic && o.rigid.objective=="point_to_plane" && !normals.size())
        throw std::invalid_argument("point-to-plane requires target_normals");
        const Vector center=fixed.colwise().mean();
        Matrix x=fixed.rowwise()-center.transpose(),y0=moving.rowwise()-center.transpose();
        const double scale=x.norm()/std::sqrt(double(x.rows()));
        if(!std::isfinite(scale)||scale<=1e-150) throw NumericalError("fixed cloud has zero or unrepresentable extent");
        x/=scale;
        y0/=scale;
        require_finite(x,"normalization");
        require_finite(y0,"normalization");
        Matrix rotation=r0;
        Vector translation=(r0*center+t0-center)/scale;
        Matrix y=apply_pose(y0,rotation,translation);
        Result out;
        out.center=center;
        out.normalization_scale=scale;
        out.method=o.method;
        out.backend=o.backend;
        double sigma2=0;
        if(o.method!=Method::Analytic) {
            Stage& stage=out.rigid_stage;
            sigma2=o.rigid.sigma2>0?o.rigid.sigma2/scale/scale:initial_variance(x,y);
            sigma2=std::max(sigma2,o.rigid.min_sigma2);
            stage.initial_sigma2=sigma2;
            stage.stop_reason="iteration_limit";
            std::unique_ptr<FixedNoBlurLattice> cache;
            double indexed_sigma=-1;
            const Matrix nv=o.rigid.objective=="point_to_plane"?normals:Matrix();
            for(int it=0;it<o.rigid.max_iterations;++it) {
                if(o.backend==Backend::PermutohedralNoBlur && (!cache||sigma2!=indexed_sigma)) {
                    cache=std::make_unique<FixedNoBlurLattice>(x/std::sqrt(sigma2),moment_values(x,nv));
                    indexed_sigma=sigma2;
                    ++stage.index_builds;
                }
                const Statistics stats=posterior_statistics(x,y,sigma2,o.rigid.w,true,o.backend,nv,cache.get(),o.fgt);
                if(o.backend==Backend::Permutohedral||o.backend==Backend::Fgt) ++stage.index_builds;
                if(o.backend==Backend::Probreg) stage.index_builds+=(stats.lattice_mode=="probreg_noblur"?2:1);
                const RigidFit fit=fit_rigid(y,stats,o.rigid);
                // Use UPDATED coordinates and ambient dimension d. The attached
                // probreg.py uses old coordinates and literal 3; see SOURCE_DIFFERENCES.md.
                const double next_sigma=o.rigid.update_sigma2?variance_from_statistics(fit.next,stats,o.rigid.min_sigma2):sigma2;
                const double motion=motion_rms(y,fit.next),change=std::abs(next_sigma-sigma2)/(sigma2+1e-12);
                rotation=fit.rotation*rotation;
                translation=fit.rotation*translation+fit.translation;
                y=fit.next;
                stage.history.push_back({
                    it+1,0,0,fit.active,fit.rank,next_sigma,stats.nll,motion,fit.fit_rms,stats.vertices,stats.lattice_mode
                });
                sigma2=next_sigma;
                stage.best_iteration=it+1;
                if(motion<=o.rigid.tolerance&&change<=o.rigid.tolerance) {
                    stage.converged=true;
                    stage.stop_reason="tolerance";
                    break;
                }
            }
            stage.final_sigma2=sigma2;
        }
        // The rigid pose is immutable below this line; residual maps are composed
        // AFTER it, not absorbed back into rotation/translation.
        out.rotation=rotation;
        out.translation=scale*translation+center-rotation*center;
        out.rigid_transformed=apply_pose(moving,out.rotation,out.translation);
        if(o.method!=Method::Rigid) {
            const auto& opt=o.analytic;
            Stage& stage=out.analytic_stage;
            // "auto" is resolved from the declared method, never from a runtime quality
            // test. Nonrigid inherits the variance the rigid stage annealed to; standalone
            // analytic uses the ACPD/CPD all-pair initialization of the paper.
            const std::string initialization=opt.initialization=="auto"
            ?(o.method==Method::Nonrigid?std::string("filterreg"):std::string("cpd")):opt.initialization;
            if(opt.sigma2>0) sigma2=opt.sigma2/scale/scale;
            else if(initialization=="cpd") sigma2=initial_variance(x,y);
            sigma2=std::max(sigma2,opt.min_sigma2);
            stage.initial_sigma2=sigma2;
            stage.stop_reason="iteration_limit";
            Matrix best_y=y;
            double best_sigma=sigma2,best_score=std::sqrt(d*sigma2),previous_score=best_score;
            std::size_t best_steps=0;
            int stable=0,no_improve=0,previous_degree=-1;
            const auto schedule=degree_schedule(opt.max_iterations,opt.min_degree,opt.max_degree);
            // Degree continuation: the schedule bounds how long each degree MAY run,
            // it does not require running that long. Convergence at a degree below
            // max_degree advances the continuation instead of ending the whole stage;
            // only the highest scheduled degree may terminate it. See
            // docs/SOURCE_DIFFERENCES.md, Analytic-CPD 10.
            // Dimensionless: the fixed cloud is normalized to unit RMS radius, and the
            // limit tracks its own largest radius so an outlier-heavy cloud is not
            // penalised. See the divergence_radius note in types.hpp.
            const double divergence_limit=opt.divergence_radius*std::max(1.0,x.rowwise().norm().maxCoeff());
            std::size_t cursor=0;
            const auto advance_degree=[&schedule](std::size_t at)->std::size_t {
                const int degree=schedule[at];
                while(at<schedule.size()&&schedule[at]<=degree) ++at;
                return at;
            };
            for(int it=0;it<opt.max_iterations&&cursor<schedule.size();++it,++cursor) {
                try {
                    const Statistics stats=posterior_statistics(x,y,sigma2,opt.w,false,opt.backend,Matrix(),nullptr,o.fgt);
                    int active=0;
                    for(int i=0;i<stats.rho.size();++i) if(stats.rho[i]>opt.min_mass) ++active;
                    if(stats.mass<=opt.min_mass||active<static_cast<int>(exponents(d,opt.min_degree).size())) {
                        stage.stop_reason="insufficient_posterior_mass";
                        break;
                    }
                    const Fit fit=fit_analytic(y,stats,schedule[cursor],opt);
                    if(fit.next.rowwise().norm().maxCoeff()>divergence_limit) {
                        // The analytic map is a global polynomial constrained only where the
                        // posterior supports it. Once a point loses support it can be
                        // extrapolated arbitrarily far, and the rho-weighted variance cannot
                        // see it, so a diverging state could otherwise be recorded as best.
                        // Refuse the iterate; the fit itself is left exactly as computed.
                        stage.stop_reason="numerical_divergence";
                        break;
                    }
                    if(previous_degree>=0&&previous_degree!=fit.step.degree) {
                        stable=0;
                        no_improve=0;
                    }
                    previous_degree=fit.step.degree;
                    const double next_sigma=variance_from_statistics(fit.next,stats,opt.min_sigma2),score=std::sqrt(d*next_sigma);
                    const double delta_y=(fit.next-y).norm()/(y.norm()+1e-12);
                    const double delta_sigma=std::abs(next_sigma-sigma2)/(std::abs(sigma2)+1e-12);
                    const double delta_score=std::abs(score-previous_score)/(std::abs(previous_score)+1e-12);
                    const double motion=motion_rms(y,fit.next);
                    const bool significant=best_score-score>std::max(1e-12,opt.improvement_relative*std::abs(best_score));
                    no_improve=significant?0:no_improve+1;
                    out.steps.push_back(fit.step);
                    y=fit.next;
                    sigma2=next_sigma;
                    stage.history.push_back({
                        it+1,schedule[cursor],fit.step.degree,fit.active,fit.rank,sigma2,stats.nll,motion,fit.fit_rms,
                            stats.vertices,stats.lattice_mode
                    });
                    // Fig.1: save the actual best state. Patience uses the source's
                    // significant-improvement threshold, but never loses a lower score.
                    if(score<best_score) {
                        best_score=score;
                        best_y=y;
                        best_sigma=sigma2;
                        best_steps=out.steps.size();
                        stage.best_iteration=it+1;
                    }
                    stable=(delta_y<opt.tolerance&&delta_sigma<opt.tolerance&&delta_score<opt.tolerance)?stable+1:0;
                    previous_score=score;
                    if(score<opt.tolerance) {
                        stage.converged=true;
                        stage.stop_reason="residual_tolerance";
                        break;
                    }
                    if(it+1>=opt.min_iterations) {
                        const bool is_stable=stable>=opt.stable_patience;
                        const bool is_rebound=no_improve>=opt.stable_patience&&score>best_score*(1+opt.rebound_relative)+1e-12;
                        const bool is_stalled=no_improve>=opt.no_improve_patience;
                        if(is_stable||is_rebound||is_stalled) {
                            const std::size_t next=advance_degree(cursor);
                            if(next<schedule.size()) {
                                // Converged at this degree with higher degrees still scheduled:
                                // continue the sequence rather than reporting the whole stage done.
                                cursor=next-1;
                                stable=0;
                                no_improve=0;
                                previous_degree=-1;
                                continue;
                            }
                            if(is_stable) {
                                stage.converged=true;
                                stage.stop_reason="stable_tolerance";
                            }
                            else stage.stop_reason=is_rebound?"internal_rebound":"no_improvement";
                            break;
                        }
                    }
                }
                catch(const NumericalError&) {
                    // The analytic map is unregularized by design, so an ill-posed pair can
                    // drive the Taylor basis or the second-moment residual out of range. Report
                    // it as a stop reason and return the best state that was actually
                    // evaluated; never a silently substituted or re-tuned computation. A
                    // failure before any iteration succeeded is a genuine input error and
                    // still propagates.
                    if(stage.history.empty()) throw;
                    stage.stop_reason="numerical_divergence";
                    break;
                }
            }
            y=best_y;
            sigma2=best_sigma;
            out.steps.resize(best_steps);
            stage.final_sigma2=sigma2;
        }
        out.sigma2=sigma2*scale*scale;
        out.transformed=y*scale;
        out.transformed.rowwise()+=center.transpose();
        require_finite(out.transformed,"result");
        if(!std::isfinite(out.sigma2)) throw NumericalError("world-coordinate variance overflow");
        world_units(out.rigid_stage,scale);
        world_units(out.analytic_stage,scale);
        return out;
    }
}
// namespace acpd
