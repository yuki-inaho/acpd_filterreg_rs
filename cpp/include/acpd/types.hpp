#pragma once
#include <Eigen/Core>
#include <stdexcept>
#include <string>
#include <vector>
namespace acpd {
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
    /// All accelerated backends below are genuine permutohedral lattices.
    enum class Backend {
        Direct, Permutohedral, PermutohedralNoBlur, Probreg
    };
    enum class Method {
        Rigid, Analytic, Nonrigid
    };
    struct NumericalError : std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    Backend backend_from_string(const std::string& value);
    Method method_from_string(const std::string& value);
    std::string name(Backend value);
    std::string name(Method value);
    struct FilterOptions {
        int max_iterations = 60;
        double tolerance = 1e-7;
        double w = 0.1;
        double sigma2 = -1.0;
        // -1: CPD-style initialization; otherwise world units squared
        double min_sigma2 = 1e-8;
        // normalized units squared
        bool update_sigma2 = true;
        std::string solver = "twist";
        // twist or kabsch (latter is point-to-point only)
        std::string objective = "point_to_point";
        // or point_to_plane (point-to-line in 2D)
        int inner_iterations = 1;
        // Gauss-Newton iterations with E-step statistics frozen
        void validate() const;
    };
    struct AnalyticOptions {
        int max_iterations = 220;
        // Upper bound, not a target: degree continuation ends the stage when the highest
        // scheduled degree converges. 55 is one DegreeScheduleDecreasingStages unit for
        // degrees 1..10 and leaves a single iteration at degree 10, too few for the 2D
        // anneal; 220 is four units with the same decreasing shape.
        int min_degree = 1;
        int max_degree = 10;
        double tolerance = 1e-7;
        double w = 0.1;
        double sigma2 = -1.0;
        double min_sigma2 = 1e-12;
        double rank_tolerance = 1e-12;
        double min_mass = 1e-12;
        std::string initialization = "auto";
        // "auto" resolves by method: nonrigid -> "filterreg", analytic -> "cpd".
        // Deterministic and declared, not a quality-triggered fallback.
        // The "filterreg" variance handoff is NOT in the ACPD paper.
        int stable_patience = 5;
        int no_improve_patience = 8;
        int min_iterations = 6;
        // source iter>=5, with zero-based source indexing
        double improvement_relative = 1e-6;
        double rebound_relative = 1e-3;
        void validate() const;
    };
    struct Options {
        Method method = Method::Nonrigid;
        Backend backend = Backend::Permutohedral;
        // FilterReg ONLY; ACPD always exact/direct
        FilterOptions rigid;
        AnalyticOptions analytic;
        void validate() const;
    };
    struct Iteration {
        int iteration = 0, raw_degree = 0, degree = 0, active = 0, rank = 0;
        double sigma2 = 0, nll_before = 0, step_rms = 0, fit_rms = 0;
        int lattice_vertices = 0;
        std::string lattice_mode = "direct";
    };
    struct Stage {
        std::vector<Iteration> history;
        double initial_sigma2 = 0, final_sigma2 = 0;
        bool converged = false;
        std::string stop_reason = "not_run";
        int best_iteration = 0, index_builds = 0;
    };
    struct AnalyticStep {
        int degree = 1;
        Matrix coefficients;
        // (K,d) correction coefficients: A(p)=p+Phi(p)C
    };
    struct Result {
        Matrix rotation;
        Vector translation;
        Vector center;
        double normalization_scale = 1;
        Matrix transformed, rigid_transformed;
        std::vector<AnalyticStep> steps;
        Stage rigid_stage, analytic_stage;
        double sigma2 = 0;
        Method method = Method::Nonrigid;
        Backend backend = Backend::Permutohedral;
        Matrix apply(const Matrix& points) const;
    };
    void validate_cloud(const Matrix& points, const std::string& label);
    void validate_pair(const Matrix& fixed, const Matrix& moving, bool registration = false);
    void validate_pose(const Matrix& rotation, const Vector& translation, int dimension);
    void validate_normals(const Matrix& normals, const Matrix& fixed);
    void require_finite(const Matrix& value, const char* context);
    Matrix apply_pose(const Matrix& points, const Matrix& rotation, const Vector& translation);
    Result registration(const Matrix& fixed, const Matrix& moving, const Options& options,
    const Matrix& initial_rotation, const Vector& initial_translation,
    const Matrix& target_normals = Matrix());
}
// namespace acpd
