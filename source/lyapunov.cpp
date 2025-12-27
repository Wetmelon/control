#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <complex>

#include "solver.hpp"

namespace control {

// Primary solver: Schur-based Bartels–Stewart using complex Schur decomposition.
// This transforms A -> U*T*U^H, Q -> U^H * Q * U and solves T*X + X*T^H = -Qhat
// by vectorizing the triangular system. For moderate sizes this is robust.
Matrix lyap(const Matrix& A, const Matrix& Q) {
    const int n = static_cast<int>(A.rows());
    if (n == 0) return Matrix::Zero(0, 0);

    // Quick check: ensure A is square and Q matches
    if (A.rows() != A.cols() || Q.rows() != n || Q.cols() != n) {
        throw std::invalid_argument("lyap: dimension mismatch");
    }

    // If A is very small, fall back to direct Kronecker solve for simplicity
    if (n <= 4) {
        // Direct vectorized solve (real arithmetic)
        const int       N = n * n;
        Eigen::MatrixXd K = Eigen::MatrixXd::Zero(N, N);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                int row = i + j * n;
                for (int r = 0; r < n; ++r) {
                    for (int s = 0; s < n; ++s) {
                        int    col = r + s * n;
                        double val = 0.0;
                        if (j == s) val += A(i, r);
                        if (i == r) val += A(j, s);
                        K(row, col) = val;
                    }
                }
            }
        }
        Eigen::VectorXd vecQ(N);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) vecQ(i + j * n) = Q(i, j);
        Eigen::VectorXd                             rhs = -vecQ;
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver(K);
        Eigen::VectorXd                             vecX = solver.solve(rhs);
        Matrix                                      X    = Matrix::Zero(n, n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) X(i, j) = vecX(i + j * n);
        return (X + X.transpose()) * 0.5;
    }

    // Use complex Schur decomposition for robust triangulation
    Eigen::MatrixXcd                      A_c = A.cast<std::complex<double>>();
    Eigen::ComplexSchur<Eigen::MatrixXcd> schur;
    schur.compute(A_c);
    if (schur.info() != Eigen::Success) {
        // Fallback to numerical integral approximation
        // Approximate X = integral_0^inf exp(A t) Q exp(A^T t) dt
        const double dt0  = 1e-2;
        const double tol  = 1e-12;
        Matrix       X    = Matrix::Zero(n, n);
        Matrix       term = Q;
        double       t    = 0.0;
        double       dt   = dt0;
        for (int iter = 0; iter < 100000; ++iter) {
            // midpoint approximation step
            Matrix E1        = (A * (t + dt * 0.5)).exp();
            Matrix integrand = E1 * Q * E1.transpose();
            Matrix delta     = integrand * dt;
            X += delta;
            if (delta.norm() < tol * X.norm() + 1e-18) break;
            t += dt;
            dt = std::min(dt * 1.5, 1.0);
        }
        return (X + X.transpose()) * 0.5;
    }

    Eigen::MatrixXcd U = schur.matrixU();
    Eigen::MatrixXcd T = schur.matrixT();

    // Transform Q into Schur basis: Qhat = U^H * Q * U
    Eigen::MatrixXcd Qc = U.adjoint() * Q.cast<std::complex<double>>() * U;

    // Build Kronecker system on complex triangular T: (I kron T + conj(T) kron I) vec(Xc) = -vec(Qc)
    const int        N   = n * n;
    Eigen::MatrixXcd K   = Eigen::MatrixXcd::Zero(N, N);
    auto             idx = [&](int i, int j) { return i + j * n; };
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int row = idx(i, j);
            for (int r = 0; r < n; ++r) {
                for (int s = 0; s < n; ++s) {
                    int col = idx(r, s);
                    // (I kron T) contributes T(i,r) when j==s
                    std::complex<double> val = std::complex<double>(0.0, 0.0);
                    if (j == s) val += T(i, r);
                    // (conj(T) kron I) contributes conj(T(j,s)) when i==r
                    if (i == r) val += std::conj(T(j, s));
                    K(row, col) = val;
                }
            }
        }
    }

    // rhs = -vec(Qc)
    Eigen::MatrixXcd vecQ(N, 1);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) vecQ(idx(i, j), 0) = Qc(i, j);
    Eigen::MatrixXcd rhs = -vecQ;

    // Solve the linear system (complex)
    Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> solverK(K);
    Eigen::MatrixXcd                             vecXc = solverK.solve(rhs);

    Eigen::MatrixXcd Xc = Eigen::MatrixXcd::Zero(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) Xc(i, j) = vecXc(idx(i, j), 0);

    // Transform back: X = U * Xc * U^H
    Eigen::MatrixXcd Xfull = U * Xc * U.adjoint();
    Matrix           X     = Xfull.real();

    // Symmetrize to reduce numerical asymmetry
    return (X + X.transpose()) * 0.5;
}

}  // namespace control
