//! Compiled and executed since 0.2.1 (see validation/rerun_2026-09-08/cargo_test.log).
//! Frozen upstream expectations are shared with the executed C++ comparison.
use acpd_core::*;
use acpd_core::analytic::fit_analytic;
use acpd_core::gaussian::{initial_variance, variance_from_statistics};
use acpd_core::lattice::enclosing_simplex;
use acpd_core::rigid::{fit_rigid, twist_jacobian, twist_pose};

fn close(a: &Matrix, b: &Matrix, atol: f64, rtol: f64) {
    assert_eq!(a.shape(),b.shape());
    for (x,y) in a.iter().zip(b.iter()) {
        assert!((x-y).abs()<=atol+rtol*y.abs(),"{x} != {y}");
    }
}
fn cloud(n: usize,d: usize) -> Matrix {
    Matrix::from_fn(n,d,|i,a| (0.7*i as f64+0.9*a as f64).sin()
        +0.3*(i as f64*(a+1) as f64*0.17).cos())
}
fn targets(z: &Matrix,w: &Vector,normals: Option<&Matrix>) -> Statistics {
    Statistics {rho:w.clone(),px:Matrix::from_fn(z.nrows(),z.ncols(),|i,a|w[i]*z[(i,a)]),
        x2:Vector::from_fn(z.nrows(),|i,_|w[i]*z.row(i).norm_squared()),
        normals:normals.cloned().unwrap_or_else(||Matrix::zeros(0,z.ncols())),
        mass:w.sum(),nll:0.0,vertices:0,unsupported:0,lattice_mode:"test_targets".into()}
}
include!("generated/upstream.rs");

#[test]
fn lattice_linearity_cache_and_gain() -> RegResult<()> {
    for d in [2,3] {
        let f=cloud(41,d);let v=cloud(41,2);
        let no=FixedNoBlurLattice::new(&f,&v)?;
        let count=no.lattice_size();
        let expected=no.slice(&f)?;
        assert_eq!(no.lattice_size(),count);
        let p=Permutohedral::new(&f,false)?;
        let gain=1.0/(1.0+2.0_f64.powi(-(d as i32)));
        close(&p.filter(&v,0,false)?,&(&expected*gain),1e-13,1e-13);
        for blur in [true,false] {
            let p=Permutohedral::new(&f,blur)?;
            close(&p.filter(&(&v*2.0),0,false)?,&(p.filter(&v,0,false)?*2.0),1e-12,1e-12);
            assert_eq!(p.filter(&v,41,false)?.norm(),0.0);
        }
    }Ok(())
}
#[test]
fn posterior_normalization_and_variance_dimension() -> RegResult<()> {
    for d in [2,3] {
        let x=cloud(29,d);let y=cloud(37,d)*0.95;
        let cpd=posterior_statistics(&x,&y,0.7,0.0,false,Backend::Direct,None,None,&FgtOptions::default())?;
        assert!((cpd.mass-x.nrows() as f64).abs()<1e-12);
        let inv=posterior_statistics(&x,&y,0.7,0.0,true,Backend::Direct,None,None,&FgtOptions::default())?;
        assert!((&inv.rho-Vector::from_element(y.nrows(),1.0)).amax()<1e-12);
        let z=&y*1.1;
        let direct_sse:f64=(0..y.nrows()).map(|i| cpd.rho[i]*z.row(i).norm_squared()
            -2.0*z.row(i).dot(&cpd.px.row(i))+cpd.x2[i]).sum();
        assert!((variance_from_statistics(&z,&cpd,1e-12)?-direct_sse/(cpd.mass*d as f64)).abs()<1e-12);
        assert!(initial_variance(&x,&y)?>0.0);
    }Ok(())
}
#[test]
fn twist_finite_difference_point_and_plane() -> RegResult<()> {
    for d in [2,3] {
        let k=if d==2 {3}else{6};let y=cloud(81,d);
        let point=y.row(4).transpose().into_owned();let j=twist_jacobian(&point)?;
        for a in 0..k {
            let mut delta=Vector::zeros(k);delta[a]=1e-7;
            let plus=twist_pose(&delta,d)?;let minus=twist_pose(&(-delta),d)?;
            let derivative=(plus.rotation*&point+plus.translation-minus.rotation*&point-minus.translation)/2e-7;
            assert!((derivative-j.column(a)).amax()<3e-9);
        }
        let pose=twist_pose(&Vector::from_element(k,0.02),d)?;
        let z=apply_pose(&y,&pose.rotation,&pose.translation)?;
        let weights=Vector::from_fn(y.nrows(),|i,_|0.2+(i%7) as f64/10.0);
        let mut normals=cloud(81,d)*0.73;
        for i in 0..normals.nrows() {
            for a in 0..d {normals[(i,a)]+=(0.31*i as f64+1.7*a as f64).cos();}
            let norm=normals.row(i).norm();normals.row_mut(i).scale_mut(1.0/norm);
        }
        for plane in [false,true] {
            let stats=targets(&z,&weights,if plane {Some(&normals)}else{None});
            let options=FilterOptions {inner_iterations:5,objective:if plane {"point_to_plane"}else{"point_to_point"}.into(),..Default::default()};
            close(&fit_rigid(&y,&stats,&options)?.next,&z,2e-9,2e-9);
        }
    }Ok(())
}
#[test]
fn no_ridge_and_degree_feasibility() -> RegResult<()> {
    for d in [2,3] {
        let y=cloud(d+2,d);let z=&y*2.0+Matrix::from_element(d+2,d,3.0);
        let stats=targets(&z,&Vector::from_element(d+2,1.0),None);
        let fit=fit_analytic(&y,&stats,10,&AnalyticOptions::default())?;
        assert_eq!(fit.step.degree,1);close(&fit.next,&z,2e-11,2e-11);
        assert_eq!(exponents(d,10)?.len(),if d==2 {66}else{286});
    }Ok(())
}
#[test]
fn frozen_rigid_pose_and_composition() -> RegResult<()> {
    for d in [2,3] {
        let y=cloud(80,d);let mut x=y.clone();
        for i in 0..80 {x[(i,0)]+=0.03*y[(i,1)].powi(2)+0.04;}
        let mut options=Options::default();options.backend=Backend::Direct;
        options.rigid.sigma2=0.03;options.rigid.max_iterations=20;
        options.analytic.max_iterations=12;options.analytic.max_degree=3;
        let r=Matrix::identity(d,d);let t=Vector::zeros(d);
        options.method=Method::Rigid;let rigid=registration(&x,&y,&options,&r,&t,None)?;
        options.method=Method::Nonrigid;let non=registration(&x,&y,&options,&r,&t,None)?;
        close(&rigid.rotation,&non.rotation,0.0,0.0);
        assert_eq!(rigid.translation,non.translation);
        close(&non.apply(&y)?,&non.transformed,2e-10,2e-10);
        assert_eq!(non.steps.len(),non.analytic_stage.best_iteration);
        let history_min=non.analytic_stage.history.iter().fold(non.analytic_stage.initial_sigma2,|m,h|m.min(h.sigma2));
        assert!((history_min-non.analytic_stage.final_sigma2).abs()<1e-12);
        options.method=Method::Rigid;options.backend=Backend::PermutohedralNoBlur;
        options.rigid.update_sigma2=false;
        let cached=registration(&x,&y,&options,&r,&t,None)?;
        assert_eq!(cached.rigid_stage.index_builds,1);
    }Ok(())
}
#[test]
fn invalid_inputs_are_errors_not_fallbacks() {
    assert!(Backend::parse("grid").is_err());assert!(exponents(4,2).is_err());
    assert!(exponents(2,11).is_err());assert!(degree_schedule(0,1,10).is_err());
    assert!(AnalyticOptions {max_degree:11,..Default::default()}.validate().is_err());
    assert!(FilterOptions {solver:"kabsch".into(),objective:"point_to_plane".into(),..Default::default()}.validate().is_err());
    let invalid=Matrix::from_element(2,2,f64::NAN);
    assert!(Permutohedral::new(&invalid,true).is_err());
    assert!(basis(&invalid,1).is_err());
}
