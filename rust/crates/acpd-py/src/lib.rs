//! Thin PyO3 boundary. NumPy inputs are copied into owned core matrices before detach.
use acpd_core as core;
use ndarray::{
    Array1,Array2
};
use numpy::{
    IntoPyArray,PyArray1,PyArray2,PyReadonlyArray1,PyReadonlyArray2,PyUntypedArrayMethods
};
use pyo3::exceptions::{
    PyRuntimeError,PyValueError
};
use pyo3::prelude::*;
use pyo3::types::{
    PyDict,PyList
};
fn error(e:core::Error)->PyErr {
    match e {
        core::Error::Invalid(s)=>PyValueError::new_err(s),core::Error::Numerical(s)=>PyRuntimeError::new_err(s)
    }
}
fn matrix(a:&PyReadonlyArray2<'_,f64>)->PyResult<core::Matrix> {
    if !a.is_c_contiguous() {
        return Err(PyValueError::new_err("native matrices must be C-contiguous"));
    }
    if !a.is_aligned() {
        return Err(PyValueError::new_err("native matrices must be aligned for float64"));
    }
    let shape=a.shape();
    Ok(core::Matrix::from_row_slice(shape[0],shape[1],a.as_slice()?))
}
fn vector(a:&PyReadonlyArray1<'_,f64>)->PyResult<core::Vector> {
    if !a.is_c_contiguous() {
        return Err(PyValueError::new_err("native vectors must be C-contiguous"));
    }
    if !a.is_aligned() {
        return Err(PyValueError::new_err("native vectors must be aligned for float64"));
    }
    Ok(core::Vector::from_column_slice(a.as_slice()?))
}
fn array<'py>(py:Python<'py>,a:&core::Matrix)->PyResult<Bound<'py,PyArray2<f64>>> {
    let mut data=Vec::with_capacity(a.len());
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            data.push(a[(i,j)]);
        }
    }
    Ok(Array2::from_shape_vec((a.nrows(),a.ncols()),data)
    .map_err(|e|PyRuntimeError::new_err(e.to_string()))?.into_pyarray(py))
}
fn array1<'py>(py:Python<'py>,a:&core::Vector)->Bound<'py,PyArray1<f64>> {
    Array1::from_vec(a.iter().copied().collect()).into_pyarray(py)
}
fn item<'py>(d:&Bound<'py,PyDict>,key:&str)->PyResult<Bound<'py,PyAny>> {
    d.get_item(key)?.ok_or_else(||PyValueError::new_err(format!("missing native option: {key}")))
}
fn number(d:&Bound<'_,PyDict>,key:&str)->PyResult<f64> {
    item(d,key)?.extract()
}
fn integer(d:&Bound<'_,PyDict>,key:&str)->PyResult<usize> {
    item(d,key)?.extract()
}
fn boolean(d:&Bound<'_,PyDict>,key:&str)->PyResult<bool> {
    item(d,key)?.extract()
}
fn options(d:&Bound<'_,PyDict>)->PyResult<core::Options> {
    let method:String=item(d,"method")?.extract()?;
    let backend:String=item(d,"backend")?.extract()?;
    let out=core::Options {
        method:core::Method::parse(&method).map_err(error)?,backend:core::Backend::parse(&backend).map_err(error)?,
        rigid:core::FilterOptions {
            max_iterations:integer(d,"rigid_max_iterations")?,tolerance:number(d,"rigid_tolerance")?,
            w:number(d,"rigid_w")?,sigma2:number(d,"rigid_sigma2")?,min_sigma2:number(d,"rigid_min_sigma2")?,
            update_sigma2:boolean(d,"rigid_update_sigma2")?,
            solver:item(d,"rigid_solver")?.extract()?, objective:item(d,"rigid_objective")?.extract()?,
            inner_iterations:integer(d,"rigid_inner_iterations")?,
        },
        analytic:core::AnalyticOptions {
            max_iterations:integer(d,"analytic_max_iterations")?,tolerance:number(d,"analytic_tolerance")?,
            w:number(d,"analytic_w")?,sigma2:number(d,"analytic_sigma2")?,min_sigma2:number(d,"analytic_min_sigma2")?,
            min_degree:integer(d,"analytic_min_degree")?,max_degree:integer(d,"analytic_max_degree")?,
            rank_tolerance:number(d,"analytic_rank_tolerance")?,
            min_mass:number(d,"analytic_min_mass")?,
            initialization:item(d,"analytic_initialization")?.extract()?,
            stable_patience:integer(d,"analytic_stable_patience")?,
            no_improve_patience:integer(d,"analytic_no_improve_patience")?,
            min_iterations:integer(d,"analytic_min_iterations")?,
            improvement_relative:number(d,"analytic_improvement_relative")?,
            rebound_relative:number(d,"analytic_rebound_relative")?,
            divergence_radius:number(d,"analytic_divergence_radius")?,
        },
    };
    out.validate().map_err(error)?;
    Ok(out)
}
fn stage<'py>(py:Python<'py>,s:&core::Stage)->PyResult<Bound<'py,PyDict>> {
    let out=PyDict::new(py);
    let history=PyList::empty(py);
    for h in &s.history {
        let row=PyDict::new(py);
        row.set_item("iteration",h.iteration)?;
        row.set_item("degree",h.degree)?;
        row.set_item("active",h.active)?;
        row.set_item("rank",h.rank)?;
        row.set_item("sigma2",h.sigma2)?;
        row.set_item("nll_before",if h.nll_before.is_finite() {
            Some(h.nll_before)
        }else {
            None
        })?;
        row.set_item("step_rms",h.step_rms)?;
        row.set_item("fit_rms",h.fit_rms)?;
        row.set_item("raw_degree",h.raw_degree)?;
        row.set_item("lattice_vertices",h.lattice_vertices)?;
        row.set_item("lattice_mode",&h.lattice_mode)?;
        history.append(row)?;
    }
    out.set_item("history",history)?;
    out.set_item("initial_sigma2",s.initial_sigma2)?;
    out.set_item("final_sigma2",s.final_sigma2)?;
    out.set_item("converged",s.converged)?;
    out.set_item("stop_reason",&s.stop_reason)?;
    out.set_item("best_iteration",s.best_iteration)?;
    out.set_item("index_builds",s.index_builds)?;
    Ok(out)
}
fn result<'py>(py:Python<'py>,r:&core::RegistrationResult)->PyResult<Bound<'py,PyDict>> {
    let out=PyDict::new(py);
    out.set_item("rotation",array(py,&r.rotation)?)?;
    out.set_item("translation",array1(py,&r.translation))?;
    out.set_item("center",array1(py,&r.center))?;
    out.set_item("normalization_scale",r.normalization_scale)?;
    out.set_item("transformed",array(py,&r.transformed)?)?;
    out.set_item("rigid_transformed",array(py,&r.rigid_transformed)?)?;
    out.set_item("sigma2",r.sigma2)?;
    out.set_item("method",r.method.name())?;
    out.set_item("backend",r.backend.name())?;
    out.set_item("rigid_stage",stage(py,&r.rigid_stage)?)?;
    out.set_item("analytic_stage",stage(py,&r.analytic_stage)?)?;
    let steps=PyList::empty(py);
    for s in &r.steps {
        let item=PyDict::new(py);
        item.set_item("degree",s.degree)?;
        item.set_item("coefficients",array(py,&s.coefficients)?)?;
        steps.append(item)?;
    }
    out.set_item("steps",steps)?;
    Ok(out)
}
#[pyfunction]
fn registration<'py>(py:Python<'py>,fixed:PyReadonlyArray2<'py,f64>,moving:PyReadonlyArray2<'py,f64>,
config:Bound<'py,PyDict>,rotation:PyReadonlyArray2<'py,f64>,translation:PyReadonlyArray1<'py,f64>,target_normals:PyReadonlyArray2<'py,f64>)
->PyResult<Bound<'py,PyDict>> {
    let x=matrix(&fixed)?;
    let y=matrix(&moving)?;
    let r=matrix(&rotation)?;
    let t=vector(&translation)?;
    let nv=matrix(&target_normals)?;
    let o=options(&config)?;
    let output=py.detach(move||core::registration(&x,&y,&o,&r,&t,if nv.nrows()==0 {
        None
    }else {
        Some(&nv)
    })).map_err(error)?;
    result(py,&output)
}
#[pyfunction]
fn gaussian_sum<'py>(py:Python<'py>,sources:PyReadonlyArray2<'py,f64>,queries:PyReadonlyArray2<'py,f64>,values:PyReadonlyArray2<'py,f64>,
sigma2:f64,backend:&str)->PyResult<Bound<'py,PyArray2<f64>>> {
    let s=matrix(&sources)?;
    let q=matrix(&queries)?;
    let v=matrix(&values)?;
    let b=core::Backend::parse(backend).map_err(error)?;
    let out=py.detach(move||core::gaussian_sum(&s,&q,&v,sigma2,b)).map_err(error)?;
    array(py,&out)
}
#[pyfunction]
fn posterior_stats<'py>(py:Python<'py>,fixed:PyReadonlyArray2<'py,f64>,moving:PyReadonlyArray2<'py,f64>,sigma2:f64,w:f64,
filterreg:bool,backend:&str)->PyResult<Bound<'py,PyDict>> {
    let x=matrix(&fixed)?;
    let y=matrix(&moving)?;
    let b=core::Backend::parse(backend).map_err(error)?;
    let s=py.detach(move||core::posterior_statistics(&x,&y,sigma2,w,filterreg,b,None,None)).map_err(error)?;
    let out=PyDict::new(py);
    out.set_item("rho",array1(py,&s.rho))?;
    out.set_item("px",array(py,&s.px)?)?;
    out.set_item("x2",array1(py,&s.x2))?;
    out.set_item("mass",s.mass)?;
    out.set_item("nll",if s.nll.is_finite() {
        Some(s.nll)
    }else {
        None
    })?;
    out.set_item("vertices",s.vertices)?;
    out.set_item("unsupported",s.unsupported)?;
    out.set_item("lattice_mode",s.lattice_mode)?;
    Ok(out)
}
#[pyfunction]
fn basis<'py>(py:Python<'py>,points:PyReadonlyArray2<'py,f64>,degree:usize)->PyResult<Bound<'py,PyArray2<f64>>> {
    let p=matrix(&points)?;
    let out=py.detach(move||core::basis(&p,degree)).map_err(error)?;
    array(py,&out)
}
#[pyfunction]
fn permutohedral_filter<'py>(py:Python<'py>,features:PyReadonlyArray2<'py,f64>,values:PyReadonlyArray2<'py,f64>,
with_blur:bool,start:usize,reverse:bool)->PyResult<Bound<'py,PyDict>> {
    let f=matrix(&features)?;
    let v=matrix(&values)?;
    let (values,vertices)=py.detach(move|| -> core::RegResult<(core::Matrix,usize)> {
        let lattice=core::Permutohedral::new(&f,with_blur)?;
        Ok((lattice.filter(&v,start,reverse)?,lattice.lattice_size()))
    }).map_err(error)?;
    let out=PyDict::new(py);
    out.set_item("values",array(py,&values)?)?;
    out.set_item("vertices",vertices)?;
    Ok(out)
}
#[pymodule]
fn _native(m:&Bound<'_,PyModule>)->PyResult<()> {
    m.add_function(wrap_pyfunction!(registration,m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_sum,m)?)?;
    m.add_function(wrap_pyfunction!(posterior_stats,m)?)?;
    m.add_function(wrap_pyfunction!(basis,m)?)?;
    m.add_function(wrap_pyfunction!(permutohedral_filter,m)?)?;
    Ok(())
}
