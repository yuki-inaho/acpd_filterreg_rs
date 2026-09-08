//! Native Rust example; no C++ or Python dependency. Build is unverified here.
use acpd_core::*;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d=std::env::args().nth(1).unwrap_or_else(||"3".into()).parse::<usize>()?;
    if ![2,3].contains(&d) {return Err("dimension must be 2 or 3".into());}
    let moving=Matrix::from_fn(80,d,|i,a| (0.7*i as f64+0.9*a as f64).sin()
        +0.3*(i as f64*(a+1) as f64*0.17).cos());
    let mut fixed=moving.clone();
    for i in 0..fixed.nrows() {fixed[(i,0)]+=0.04+0.02*moving[(i,1)].powi(2);}
    let mut options=Options::default();
    options.rigid.sigma2=0.08;
    let result=registration(&fixed,&moving,&options,&Matrix::identity(d,d),&Vector::zeros(d),None)?;
    println!("returned analytic steps: {}",result.steps.len());
    println!("analytic stop: {}",result.analytic_stage.stop_reason);
    println!("map reproduction error: {}",(result.apply(&moving)?-&result.transformed).norm());
    Ok(())
}
