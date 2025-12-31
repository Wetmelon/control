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

    // Use complex Schur decomposition (Bartels–Stewart) for robust triangular solve.
    Eigen::MatrixXcd                      A_c = A.cast<std::complex<double>>();
    Eigen::ComplexSchur<Eigen::MatrixXcd> schur;
    schur.compute(A_c);
    if (schur.info() != Eigen::Success) {
        // fallback to vectorized solver (rare)
        const int       N   = n * n;
        Eigen::MatrixXd K   = Eigen::MatrixXd::Zero(N, N);
        auto            idx = [&](int i, int j) { return i + j * n; };
        Matrix          At  = A.transpose();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                int row = idx(i, j);
                for (int r = 0; r < n; ++r) {
                    for (int s = 0; s < n; ++s) {
                        int    col = idx(r, s);
                        double val = 0.0;
                        if (i == r) val += At(j, s);
                        if (j == s) val += At(i, r);
                        K(row, col) = val;
                    }
                }
            }
        }

        Eigen::VectorXd vecQ(N);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) vecQ(idx(i, j)) = Q(i, j);

        Eigen::VectorXd                             rhs = -vecQ;
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solverK(K);
        Eigen::VectorXd                             vecX = solverK.solve(rhs);

        Matrix X = Matrix::Zero(n, n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) X(i, j) = vecX(idx(i, j));

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
                    int                  col = idx(r, s);
                    std::complex<double> val = std::complex<double>(0.0, 0.0);
                    if (j == s) val += T(i, r);
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

    // Use complex Schur decomposition for robust solution of T^H Y T - Y = -Qhat
    Eigen::MatrixXcd                      A_c = A.cast<std::complex<double>>();
    Eigen::ComplexSchur<Eigen::MatrixXcd> schur;
    schur.compute(A_c);
    if (schur.info() != Eigen::Success) {
        // fallback to previous vectorized real solver
        const int       N   = n * n;
        Eigen::MatrixXd K   = Eigen::MatrixXd::Zero(N, N);
        auto            idx = [&](int i, int j) { return i + j * n; };
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                int row = idx(i, j);
                for (int r = 0; r < n; ++r) {
                    for (int s = 0; s < n; ++s) {
                        int    col = idx(r, s);
                        double val = A(i, r) * A(s, j);
                        if (row == col) val -= 1.0;
                        K(row, col) = val;
                    }
                }
            }
        }

        Eigen::VectorXd vecQ(N);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) vecQ(idx(i, j)) = Q(i, j);

        Eigen::VectorXd                             rhs = -vecQ;
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver(K);
        Eigen::VectorXd                             vecX = solver.solve(rhs);

        Matrix X = Matrix::Zero(n, n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) X(i, j) = vecX(idx(i, j));

        return (X + X.transpose()) * 0.5;
    }

    Eigen::MatrixXcd U = schur.matrixU();
    Eigen::MatrixXcd T = schur.matrixT();

    // Transform Q into Schur basis: Qhat = U^H * Q * U
    Eigen::MatrixXcd Qc = U.adjoint() * Q.cast<std::complex<double>>() * U;

    // Build Kronecker system: (T^T kron T^H - I) vec(Y) = -vec(Qc)
    const int        N   = n * n;
    Eigen::MatrixXcd Kc  = Eigen::MatrixXcd::Zero(N, N);
    auto             idx = [&](int i, int j) { return i + j * n; };
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int row = idx(i, j);
            for (int r = 0; r < n; ++r) {
                for (int s = 0; s < n; ++s) {
                    int col = idx(r, s);
                    // T^T(i,r) = T(r,i); T^H(j,s) = conj(T(s,j))
                    std::complex<double> val = T(r, i) * std::conj(T(s, j));
                    if (row == col) val -= std::complex<double>(1.0, 0.0);
                    Kc(row, col) = val;
                }
            }
        }
    }

    Eigen::MatrixXcd vecQ(N, 1);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) vecQ(idx(i, j), 0) = Qc(i, j);

    Eigen::MatrixXcd                             rhs = -vecQ;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> solverK(Kc);
    Eigen::MatrixXcd                             vecYc = solverK.solve(rhs);

    Eigen::MatrixXcd Yc = Eigen::MatrixXcd::Zero(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) Yc(i, j) = vecYc(idx(i, j), 0);

    Eigen::MatrixXcd Xc = U * Yc * U.adjoint();
    Matrix           X  = Xc.real();
    return (X + X.transpose()) * 0.5;
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

    // For 1x1 case, solve quadratic directly
    if (n == 1) {
        double a = A(0, 0), b = B(0, 0), q = Q(0, 0), r = R(0, 0);
        double rinv = 1.0 / r;
        // Equation: a*p + p*a - p*b*rinv*b*p + q = 0
        // -2*a*p - b*b*rinv*p*p + q = 0
        // b*b*rinv*p*p + 2*a*p - q = 0
        double aa   = b * b * rinv;
        double bb   = -2 * a;
        double cc   = -q;
        double disc = bb * bb - 4 * aa * cc;
        double p1   = (-bb + sqrt(disc)) / (2 * aa);
        double p2   = (-bb - sqrt(disc)) / (2 * aa);
        // Choose the positive definite one
        double p = p1 > 0 ? p1 : p2;
        // values computed above are used to select positive root `p`
        return Matrix::Constant(1, 1, p);
    }

    // Robust Hamiltonian-based solver (invariant subspace method)
    Matrix Rinv = R.inverse();

    // Build complex Hamiltonian matrix H = [A, -B R^{-1} B^T; -Q, -A^T]
    Eigen::MatrixXcd A_c    = A.cast<std::complex<double>>();
    Eigen::MatrixXcd B_c    = B.cast<std::complex<double>>();
    Eigen::MatrixXcd Q_c    = Q.cast<std::complex<double>>();
    Eigen::MatrixXcd Rinv_c = Rinv.cast<std::complex<double>>();

    Eigen::MatrixXcd BRB = B_c * Rinv_c * B_c.adjoint();
    Eigen::MatrixXcd H(2 * n, 2 * n);
    H.setZero();
    H.topLeftCorner(n, n)     = A_c;
    H.topRightCorner(n, n)    = -BRB;
    H.bottomLeftCorner(n, n)  = -Q_c;
    H.bottomRightCorner(n, n) = -A_c.adjoint();

    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces(H);
    if (ces.info() == Eigen::Success) {
        auto evals = ces.eigenvalues();
        auto evecs = ces.eigenvectors();

        // collect indices with negative real part (stable eigenvalues)
        std::vector<int> stable;
        for (int i = 0; i < evals.size(); ++i) {
            if (evals(i).real() < 0.0) stable.push_back(i);
        }

        if ((int)stable.size() == n) {
            Eigen::MatrixXcd V(2 * n, n);
            for (int k = 0; k < n; ++k) V.col(k) = evecs.col(stable[k]);

            Eigen::MatrixXcd V1 = V.topRows(n);
            Eigen::MatrixXcd V2 = V.bottomRows(n);

            Eigen::FullPivLU<Eigen::MatrixXcd> lu(V1);
            if (lu.isInvertible()) {
                Eigen::MatrixXcd Xc = V2 * lu.inverse();
                Matrix           Xr = Xc.real();
                Matrix           Xs = (Xr + Xr.transpose()) * 0.5;
                return Xs;
            }
        }
    }

    // Fall back to a damped Newton iteration if eigen-method fails
    Matrix       X        = Matrix::Identity(n, n) * 1e-6;
    const double tol      = 1e-10;
    const int    max_iter = 1000;
    const double alpha    = 0.25;

    for (int iter = 0; iter < max_iter; ++iter) {
        Matrix X_prev = X;
        Matrix F      = A.transpose() * X + X * A - X * B * Rinv * B.transpose() * X + Q;
        Matrix DX     = lyap(A.transpose(), -F);
        if (!DX.allFinite()) break;
        X = X_prev + alpha * DX;
        if (DX.norm() < tol) break;
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

    // Use simple fixed-point (Kleinman) iteration for DARE:
    // X_{k+1} = A^T X_k A - A^T X_k B (R + B^T X_k B)^{-1} B^T X_k A + Q
    Matrix       X        = Q;  // initial guess
    const double tol      = 1e-9;
    const int    max_iter = 1000;

    for (int iter = 0; iter < max_iter; ++iter) {
        Matrix X_prev = X;

        Matrix BRB      = B.transpose() * X * B;
        Matrix temp     = R + BRB;
        Matrix temp_inv = temp.inverse();

        Matrix X_new = A.transpose() * X * A - A.transpose() * X * B * temp_inv * B.transpose() * X * A + Q;

        Matrix diff = X_new - X_prev;
        X           = X_new;

        if (diff.norm() < tol * std::max(1.0, X.norm())) break;
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
        double lower = 0.0;
        double upper = 1e6;
        // Quick numeric fallback for scalar systems (robust and simple)
        if (m == 1 && p == 1) {
            // sweep frequency on logarithmic scale
            const int Nw     = 10001;
            double    maxmag = 0.0;
            for (int k = 0; k < Nw; ++k) {
                double               w = std::pow(10.0, -3.0 + 6.0 * (double)k / (Nw - 1));
                std::complex<double> s(0.0, w);
                Eigen::MatrixXcd     M   = (s * Eigen::MatrixXcd::Identity(n, n) - A.cast<std::complex<double>>()).inverse();
                std::complex<double> G   = (C.cast<std::complex<double>>() * M * B.cast<std::complex<double>>() + D.cast<std::complex<double>>())(0, 0);
                double               mag = std::abs(G);
                if (mag > maxmag) maxmag = mag;
            }
            return maxmag;
        }
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
