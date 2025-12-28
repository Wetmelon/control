#include "solver.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <complex>

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

// Discrete Lyapunov equation solver: A^T * X * A - X + Q = 0
Matrix dlyap(const Matrix& A, const Matrix& Q) {
    const int n = static_cast<int>(A.rows());
    if (n == 0) return Matrix::Zero(0, 0);

    // Quick check: ensure A is square and Q matches
    if (A.rows() != A.cols() || Q.rows() != n || Q.cols() != n) {
        throw std::invalid_argument("dlyap: dimension mismatch");
    }

    // For discrete systems, we can use a similar approach but with the discrete form
    // A^T * X * A - X = -Q
    // This can be solved using vectorization or iterative methods

    // Use iterative method for discrete Lyapunov equation
    Matrix       X        = Matrix::Zero(n, n);
    const double tol      = 1e-12;
    const int    max_iter = 1000;

    // Simple fixed-point iteration: X_{k+1} = A^T * X_k * A + Q
    for (int iter = 0; iter < max_iter; ++iter) {
        Matrix X_new = A.transpose() * X * A + Q;
        Matrix diff  = X_new - X;
        X            = X_new;

        if (diff.norm() < tol * X.norm()) {
            break;
        }
    }

    return X;
}

// Continuous Algebraic Riccati Equation solver: A^T*X + X*A - X*B*R^(-1)*B^T*X + Q = 0
Matrix care(const Matrix& A, const Matrix& B, const Matrix& Q, const Matrix& R) {
    const int n = static_cast<int>(A.rows());
    const int m = static_cast<int>(B.cols());

    if (A.rows() != A.cols() || B.rows() != n || Q.rows() != n || Q.cols() != n ||
        R.rows() != m || R.cols() != m) {
        throw std::invalid_argument("care: dimension mismatch");
    }

    // Check positive definiteness
    Eigen::SelfAdjointEigenSolver<Matrix> R_eigen(R);
    if (R_eigen.eigenvalues().minCoeff() <= 0) {
        throw std::invalid_argument("care: R must be positive definite");
    }

    Eigen::SelfAdjointEigenSolver<Matrix> Q_eigen(Q);
    if (Q_eigen.eigenvalues().minCoeff() < -1e-12) {
        throw std::invalid_argument("care: Q must be positive semidefinite");
    }

    // Use Schur method for CARE
    Matrix H                  = Matrix::Zero(2 * n, 2 * n);
    H.topLeftCorner(n, n)     = A;
    H.topRightCorner(n, n)    = -B * R.inverse() * B.transpose();
    H.bottomLeftCorner(n, n)  = -Q;
    H.bottomRightCorner(n, n) = -A.transpose();

    // Compute Schur decomposition
    Eigen::RealSchur<Matrix> schur(H);
    if (schur.info() != Eigen::Success) {
        throw std::runtime_error("care: Schur decomposition failed");
    }

    const Matrix& U = schur.matrixU();
    const Matrix& T = schur.matrixT();

    // Extract stable part (negative real eigenvalues)
    Matrix X = Matrix::Zero(n, n);
    for (int i = 0; i < 2 * n; ++i) {
        if (T(i, i) < 0) {
            // This is a stable eigenvalue, extract corresponding X
            Matrix v = U.block(0, i, n, 1);
            Matrix w = U.block(n, i, n, 1);
            if (w.norm() > 1e-12) {
                X += (v * w.transpose()) / w.squaredNorm();
            }
        }
    }

    return X;
}

// Discrete Algebraic Riccati Equation solver: A^T*X*A - X - A^T*X*B*(R + B^T*X*B)^(-1)*B^T*X*A + Q = 0
Matrix dare(const Matrix& A, const Matrix& B, const Matrix& Q, const Matrix& R) {
    const int n = static_cast<int>(A.rows());
    const int m = static_cast<int>(B.cols());

    if (A.rows() != A.cols() || B.rows() != n || Q.rows() != n || Q.cols() != n ||
        R.rows() != m || R.cols() != m) {
        throw std::invalid_argument("dare: dimension mismatch");
    }

    // Check positive definiteness
    Eigen::SelfAdjointEigenSolver<Matrix> R_eigen(R);
    if (R_eigen.eigenvalues().minCoeff() <= 0) {
        throw std::invalid_argument("dare: R must be positive definite");
    }

    Eigen::SelfAdjointEigenSolver<Matrix> Q_eigen(Q);
    if (Q_eigen.eigenvalues().minCoeff() < -1e-12) {
        throw std::invalid_argument("dare: Q must be positive semidefinite");
    }

    // Use iterative method for DARE
    Matrix       X        = Matrix::Zero(n, n);
    Matrix       Rinv     = R.inverse();
    const double tol      = 1e-10;
    const int    max_iter = 100;

    for (int iter = 0; iter < max_iter; ++iter) {
        Matrix X_prev = X;

        // Compute F = A^T*X*A - X - A^T*X*B*(R + B^T*X*B)^(-1)*B^T*X*A + Q
        Matrix BRB      = B.transpose() * X * B;
        Matrix temp     = R + BRB;
        Matrix temp_inv = temp.inverse();
        Matrix F        = A.transpose() * X * A - X - A.transpose() * X * B * temp_inv * B.transpose() * X * A + Q;

        // Solve A^T*DX*A - DX = -F
        Matrix DX = dlyap(A.transpose(), -F);

        X = X_prev + DX;

        // Check convergence
        if (DX.norm() < tol * X.norm()) {
            break;
        }
    }

    return X;
}

// Controllability matrix
Matrix ctrb(const Matrix& A, const Matrix& B) {
    const int n = static_cast<int>(A.rows());
    const int m = static_cast<int>(B.cols());

    if (A.rows() != A.cols() || B.rows() != n) {
        throw std::invalid_argument("ctrb: dimension mismatch");
    }

    Matrix Ctrb = Matrix::Zero(n, n * m);
    Matrix Ak   = Matrix::Identity(n, n);

    for (int k = 0; k < n; ++k) {
        Ctrb.block(0, k * m, n, m) = Ak * B;
        Ak                         = A * Ak;
    }

    return Ctrb;
}

// Observability matrix
Matrix obsv(const Matrix& C, const Matrix& A) {
    const int n = static_cast<int>(A.rows());
    const int p = static_cast<int>(C.rows());

    if (A.rows() != A.cols() || C.cols() != n) {
        throw std::invalid_argument("obsv: dimension mismatch");
    }

    Matrix Obsv = Matrix::Zero(n * p, n);
    Matrix Ak   = Matrix::Identity(n, n);

    for (int k = 0; k < n; ++k) {
        Obsv.block(k * p, 0, p, n) = C * Ak;
        Ak                         = A * Ak;
    }

    return Obsv;
}

// System norm
double norm(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D, const std::string& type) {
    const int n = static_cast<int>(A.rows());
    const int m = static_cast<int>(B.cols());
    const int p = static_cast<int>(C.rows());

    if (A.rows() != A.cols() || B.rows() != n || C.cols() != n || D.rows() != p || D.cols() != m) {
        throw std::invalid_argument("norm: dimension mismatch");
    }

    if (type == "inf" || type == "infinity") {
        // H-infinity norm using bisection method
        double       lower    = 0.0;
        double       upper    = 1e6;
        const double tol      = 1e-6;
        const int    max_iter = 100;

        for (int iter = 0; iter < max_iter; ++iter) {
            double gamma = (lower + upper) / 2.0;

            // Check if ||G||_inf < gamma by solving H-infinity ARE
            Matrix Q = C.transpose() * C;
            Matrix R = D.transpose() * D - gamma * gamma * Matrix::Identity(m, m);

            if (R.eigenvalues().real().minCoeff() > 0) {
                try {
                    Matrix X = care(A, B, Q, R);
                    if (X.eigenvalues().real().maxCoeff() < 0) {
                        upper = gamma;
                    } else {
                        lower = gamma;
                    }
                } catch (...) {
                    lower = gamma;
                }
            } else {
                lower = gamma;
            }

            if (upper - lower < tol) break;
        }

        return (lower + upper) / 2.0;
    } else if (type == "2") {
        // H-2 norm
        // For stable systems, ||G||_2^2 = trace(C*W_c*C^T) where W_c solves A*W_c + W_c*A^T + B*B^T = 0
        Matrix Wc = lyap(A, B * B.transpose());
        return std::sqrt((C * Wc * C.transpose()).trace());
    } else {
        throw std::invalid_argument("norm: unsupported norm type '" + type + "'");
    }
}

}  // namespace control
