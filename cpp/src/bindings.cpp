#include "acpd/analytic.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <memory>
#include <cstdint>
#include <cmath>
namespace nb=nanobind;
using namespace nb::literals;
namespace {
    using Input = nb::ndarray<nb::numpy,const double,nb::ndim<2>,nb::c_contig,nb::device::cpu>;
    using InputVector = nb::ndarray<nb::numpy,const double,nb::ndim<1>,nb::c_contig,nb::device::cpu>;
    acpd::Matrix matrix(const Input& a) {
        if (reinterpret_cast<std::uintptr_t>(a.data()) % alignof(double) != 0)
        throw std::invalid_argument("native matrices must be aligned for float64");
        acpd::Matrix out(a.shape(0),a.shape(1));
        for(std::size_t i=0;i<a.shape(0);++i) for(std::size_t j=0;j<a.shape(1);++j) out(i,j)=a(i,j);
        return out;
    }
    acpd::Vector vector(const InputVector& a) {
        if (reinterpret_cast<std::uintptr_t>(a.data()) % alignof(double) != 0)
        throw std::invalid_argument("native vectors must be aligned for float64");
        acpd::Vector out(a.shape(0));
        for(std::size_t i=0;i<a.shape(0);++i) out[i]=a(i);
        return out;
    }
    nb::object array(const acpd::Matrix& a) {
        const auto n=static_cast<std::size_t>(a.rows()), d=static_cast<std::size_t>(a.cols());
        auto data=std::make_unique<double[]>(n*d);
        for(std::size_t i=0;i<n;++i) for(std::size_t j=0;j<d;++j) data[i*d+j]=a(i,j);
        nb::capsule owner(data.get(),[](void* p) noexcept {
            delete[] static_cast<double*>(p);
        });
        auto* pointer=data.release();
        return nb::ndarray<nb::numpy,double,nb::ndim<2>>(pointer, {
            n,d
        },owner).cast();
    }
    nb::object array1(const acpd::Vector& a) {
        auto data=std::make_unique<double[]>(a.size());
        for(int i=0;i<a.size();++i) data[i]=a[i];
        nb::capsule owner(data.get(),[](void* p) noexcept {
            delete[] static_cast<double*>(p);
        });
        auto* pointer=data.release();
        return nb::ndarray<nb::numpy,double,nb::ndim<1>>(pointer, {
            static_cast<std::size_t>(a.size())
        },owner).cast();
    }
    template<typename T> T get(const nb::dict& d,const char* key) {
        if(!d.contains(key)) throw std::invalid_argument(std::string("missing native option: ")+key);
        return nb::cast<T>(d[key]);
    }
    acpd::Options options(const nb::dict& p) {
        acpd::Options o;
        o.method=acpd::method_from_string(get<std::string>(p,"method"));
        o.backend=acpd::backend_from_string(get<std::string>(p,"backend"));
        o.rigid.max_iterations=get<int>(p,"rigid_max_iterations");
        o.rigid.tolerance=get<double>(p,"rigid_tolerance");
        o.rigid.w=get<double>(p,"rigid_w");
        o.rigid.sigma2=get<double>(p,"rigid_sigma2");
        o.rigid.min_sigma2=get<double>(p,"rigid_min_sigma2");
        o.rigid.update_sigma2=get<bool>(p,"rigid_update_sigma2");
        o.analytic.max_iterations=get<int>(p,"analytic_max_iterations");
        o.analytic.tolerance=get<double>(p,"analytic_tolerance");
        o.analytic.w=get<double>(p,"analytic_w");
        o.analytic.sigma2=get<double>(p,"analytic_sigma2");
        o.analytic.min_sigma2=get<double>(p,"analytic_min_sigma2");
        o.analytic.min_degree=get<int>(p,"analytic_min_degree");
        o.analytic.max_degree=get<int>(p,"analytic_max_degree");
        o.analytic.rank_tolerance=get<double>(p,"analytic_rank_tolerance");
        o.analytic.min_mass=get<double>(p,"analytic_min_mass");
        o.rigid.solver=get<std::string>(p,"rigid_solver");
        o.rigid.objective=get<std::string>(p,"rigid_objective");
        o.rigid.inner_iterations=get<int>(p,"rigid_inner_iterations");
        o.analytic.initialization=get<std::string>(p,"analytic_initialization");
        o.analytic.stable_patience=get<int>(p,"analytic_stable_patience");
        o.analytic.no_improve_patience=get<int>(p,"analytic_no_improve_patience");
        o.analytic.min_iterations=get<int>(p,"analytic_min_iterations");
        o.analytic.improvement_relative=get<double>(p,"analytic_improvement_relative");
        o.analytic.rebound_relative=get<double>(p,"analytic_rebound_relative");
        o.validate();
        return o;
    }
    nb::dict stage(const acpd::Stage& s) {
        nb::dict out;
        nb::list history;
        for(const auto& h:s.history) {
            nb::dict r;
            r["iteration"]=h.iteration;
            r["degree"]=h.degree;
            r["active"]=h.active;
            r["rank"]=h.rank;
            r["sigma2"]=h.sigma2;
            r["nll_before"]=std::isfinite(h.nll_before)?nb::cast(h.nll_before):nb::none();
            r["step_rms"]=h.step_rms;
            r["fit_rms"]=h.fit_rms;
            r["raw_degree"]=h.raw_degree;
            r["lattice_vertices"]=h.lattice_vertices;
            r["lattice_mode"]=h.lattice_mode;
            history.append(r);
        }
        out["history"]=history;
        out["initial_sigma2"]=s.initial_sigma2;
        out["final_sigma2"]=s.final_sigma2;
        out["converged"]=s.converged;
        out["stop_reason"]=s.stop_reason;
        out["best_iteration"]=s.best_iteration;
        out["index_builds"]=s.index_builds;
        return out;
    }
    nb::dict result(const acpd::Result& r) {
        nb::dict out;
        out["rotation"]=array(r.rotation);
        out["translation"]=array1(r.translation);
        out["center"]=array1(r.center);
        out["normalization_scale"]=r.normalization_scale;
        out["transformed"]=array(r.transformed);
        out["rigid_transformed"]=array(r.rigid_transformed);
        out["sigma2"]=r.sigma2;
        out["method"]=acpd::name(r.method);
        out["backend"]=acpd::name(r.backend);
        out["rigid_stage"]=stage(r.rigid_stage);
        out["analytic_stage"]=stage(r.analytic_stage);
        nb::list steps;
        for(const auto& s:r.steps) {
            nb::dict item;
            item["degree"]=s.degree;
            item["coefficients"]=array(s.coefficients);
            steps.append(item);
        }
        out["steps"]=steps;
        return out;
    }
}
NB_MODULE(_native,m) {
    m.doc()="2D/3D FilterReg and compositional Analytic-CPD: independent C++ core";
    nb::exception<acpd::NumericalError>(m,"NumericalError",PyExc_RuntimeError);
    m.def("registration",[](Input x,Input y,nb::dict p,Input r,InputVector t,Input normals) {
        // Own numerical inputs before releasing the GIL. copy=False at the public
        // boundary prohibits dtype/layout conversion, not algorithm work buffers.
        auto fixed=matrix(x), moving=matrix(y), rotation=matrix(r), nv=matrix(normals); auto translation=vector(t); auto o=options(p);
        acpd::Result out;
        {
            nb::gil_scoped_release release; out=acpd::registration(fixed,moving,o,rotation,translation,nv);
        }
        return result(out);
    },"fixed"_a.noconvert(),"moving"_a.noconvert(),"options"_a,"rotation"_a.noconvert(),"translation"_a.noconvert(),"target_normals"_a.noconvert());
    m.def("gaussian_sum",[](Input s,Input q,Input v,double sigma2,const std::string& backend) {
        auto sources=matrix(s), queries=matrix(q), values=matrix(v); acpd::Matrix out; auto b=acpd::backend_from_string(backend);
        {
            nb::gil_scoped_release release; out=acpd::gaussian_sum(sources,queries,values,sigma2,b);
        }
        return array(out);
    },"sources"_a.noconvert(),"queries"_a.noconvert(),"values"_a.noconvert(),"sigma2"_a,"backend"_a);
    m.def("posterior_stats",[](Input x,Input y,double sigma2,double w,bool inverse,const std::string& backend) {
        auto fixed=matrix(x), moving=matrix(y); acpd::Statistics s; auto b=acpd::backend_from_string(backend);
        {
            nb::gil_scoped_release release; s=acpd::posterior_statistics(fixed,moving,sigma2,w,inverse,b);
        }
        nb::dict out; out["rho"]=array1(s.rho); out["px"]=array(s.px); out["x2"]=array1(s.x2); out["mass"]=s.mass; out["nll"]=std::isfinite(s.nll)?nb::cast(s.nll):nb::none(); out["vertices"]=s.vertices; out["unsupported"]=s.unsupported; out["lattice_mode"]=s.lattice_mode; return out;
    },"fixed"_a.noconvert(),"moving"_a.noconvert(),"sigma2"_a,"w"_a,"filterreg"_a,"backend"_a);
    m.def("permutohedral_filter",[](Input f,Input v,bool with_blur,int start,bool reverse) {
        auto features=matrix(f),values=matrix(v);acpd::Matrix out;std::size_t vertices=0;
        {
            nb::gil_scoped_release release; acpd::Permutohedral lattice(features,with_blur);
            out=lattice.filter(values,start,reverse);vertices=lattice.lattice_size();
        }
        nb::dict result;result["values"]=array(out);result["vertices"]=vertices;return result;
    },"features"_a.noconvert(),"values"_a.noconvert(),"with_blur"_a,"start"_a,"reverse"_a);
    m.def("basis",[](Input p,int degree) {
        auto points=matrix(p);
        acpd::Matrix out;
        {
            nb::gil_scoped_release release;
            out=acpd::basis(points,degree);
        }
        return array(out);
    },"points"_a.noconvert(),"degree"_a);
}
