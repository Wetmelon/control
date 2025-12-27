#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "LTI.hpp"
#include "solver.hpp"
#include "ss.hpp"
#include "types.hpp"

namespace control {

StateSpace StateSpace::balred(size_t r) const {
    if (isDiscrete()) {
        throw std::runtime_error("balred: discrete-time systems are not yet supported");
    }

    const int n = static_cast<int>(A.rows());
    if (n == 0 || r == 0) {
        // Truncate to a pure D matrix
        return StateSpace(Matrix::Zero(0, 0), Matrix::Zero(0, B.cols()), Matrix::Zero(C.rows(), 0), D, Ts);
    }
    if (r >= static_cast<size_t>(n)) {
        return *this;  // nothing to do
    }

    // Compute controllability and observability gramians
    Matrix BBt = B * B.transpose();
    Matrix CtC = C.transpose() * C;

    Matrix P = solve_continuous_lyap(A, BBt);
    Matrix Q = solve_continuous_lyap(A.transpose(), CtC);

    // Ensure symmetry
    P = (P + P.transpose()) * 0.5;
    Q = (Q + Q.transpose()) * 0.5;

    // Cholesky (fallback to eigen if not positive definite)
    Eigen::LLT<Matrix> lltP(P);
    Eigen::LLT<Matrix> lltQ(Q);
    Matrix             Rp, Rq;
    if (lltP.info() == Eigen::Success && lltQ.info() == Eigen::Success) {
        // P = Lp * Lp^T where Lp is lower-triangular
        Rp = lltP.matrixL();
        Rq = lltQ.matrixL();
    } else {
        // Fallback: symmetric eigen decomposition to get sqrt
        Eigen::SelfAdjointEigenSolver<Matrix> esP(P);
        Eigen::SelfAdjointEigenSolver<Matrix> esQ(Q);
        if (esP.info() != Eigen::Success || esQ.info() != Eigen::Success) {
            throw std::runtime_error("balred: failed to factor gramians");
        }
        Matrix sqrtP = esP.operatorSqrt();
        Matrix sqrtQ = esQ.operatorSqrt();
        Rp           = sqrtP;
        Rq           = sqrtQ;
    }

    // Compute SVD of Rp^T * Rq
    Matrix                   M = Rp.transpose() * Rq;
    Eigen::JacobiSVD<Matrix> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Matrix                   U = svd.matrixU();
    Matrix                   V = svd.matrixV();
    Eigen::VectorXd          s = svd.singularValues();

    // Form balancing transform T and its inverse
    // Use Sigma = diag(s)
    Eigen::VectorXd sqrt_s     = s.array().sqrt();
    Eigen::VectorXd inv_sqrt_s = sqrt_s.array().inverse();

    // Compute T = Rp.inverse() * U * sqrt(S)
    // For triangular Rp, solving is better than inverting
    Matrix U_sqrt = U * sqrt_s.asDiagonal();
    Matrix T      = Rp.triangularView<Eigen::Lower>().solve(U_sqrt);

    // Compute Tinv = (inv_sqrt(S) * V.transpose()) * Rq.transpose().inverse()
    Matrix Vt        = V.transpose();
    Matrix invsqrtVt = inv_sqrt_s.asDiagonal() * Vt;
    Matrix Tinv      = Rq.triangularView<Eigen::Lower>().transpose().solve(invsqrtVt.transpose()).transpose();

    // Transform system to balanced coordinates
    Matrix A_bal = Tinv * A * T;
    Matrix B_bal = Tinv * B;
    Matrix C_bal = C * T;

    // Partition and truncate
    int    rint = static_cast<int>(r);
    Matrix A11  = A_bal.topLeftCorner(rint, rint);
    Matrix A12  = A_bal.topRightCorner(rint, n - rint);
    Matrix A21  = A_bal.bottomLeftCorner(n - rint, rint);
    Matrix A22  = A_bal.bottomRightCorner(n - rint, n - rint);

    Matrix B1 = B_bal.topRows(rint);
    Matrix B2 = B_bal.bottomRows(n - rint);

    Matrix C1 = C_bal.leftCols(rint);
    Matrix C2 = C_bal.rightCols(n - rint);

    // Truncated reduced-order model
    StateSpace red(A11, B1, C1, D, Ts);
    return red;
}

StateSpace StateSpace::minreal(double tol) const {
    if (isDiscrete()) {
        throw std::runtime_error("minreal: discrete-time systems are not yet supported");
    }

    const int n = static_cast<int>(A.rows());
    if (n == 0) {
        return *this;  // nothing to do
    }

    // Helper: compute orthonormal basis for column space of M using SVD
    auto colspace_basis = [&](const Matrix& M, double atol) -> Matrix {
        if (M.size() == 0) return Matrix::Zero(M.rows(), 0);
        Eigen::JacobiSVD<Matrix> svd(M, Eigen::ComputeThinU);
        Eigen::VectorXd          s    = svd.singularValues();
        int                      rank = 0;
        double                   smax = s.size() ? s(0) : 0.0;
        for (int i = 0; i < s.size(); ++i) {
            if (s(i) > atol * std::max(1.0, smax)) ++rank;
        }
        if (rank == 0) return Matrix::Zero(M.rows(), 0);
        return svd.matrixU().leftCols(rank);
    };

    double atol = tol;

    // Build controllability matrix [B, A*B, A^2*B, ...]
    int m = static_cast<int>(B.cols());
    if (m == 0) {
        // no inputs -> no controllable states
        return StateSpace(Matrix::Zero(0, 0), Matrix::Zero(0, B.cols()), Matrix::Zero(C.rows(), 0), D, Ts);
    }
    Matrix Ctrb = Matrix::Zero(n, n * m);
    Matrix Ak   = Matrix::Identity(n, n);
    for (int k = 0; k < n; ++k) {
        Matrix block               = Ak * B;  // n x m
        Ctrb.block(0, k * m, n, m) = block;
        Ak                         = A * Ak;
    }

    Matrix Uc = colspace_basis(Ctrb, atol);
    int    rc = static_cast<int>(Uc.cols());

    if (rc == 0) {
        // No controllable dynamics -> return D-only
        return StateSpace(Matrix::Zero(0, 0), Matrix::Zero(0, B.cols()), Matrix::Zero(C.rows(), 0), D, Ts);
    }

    // Build orthonormal complement of Uc via SVD of (I - Uc*Uc^T)
    Matrix P       = Uc * Uc.transpose();
    Matrix I       = Matrix::Identity(n, n);
    Matrix Mcomp   = I - P;
    Matrix Uc_perp = colspace_basis(Mcomp, atol);

    // Form orthonormal transform Q = [Uc, Uc_perp]
    const int nc2 = static_cast<int>(Uc_perp.cols());
    Matrix    Q   = Matrix::Zero(n, rc + nc2);
    if (rc > 0) Q.leftCols(rc) = Uc;
    if (nc2 > 0) Q.rightCols(nc2) = Uc_perp;

    // Transform coordinates (Q is orthonormal if Uc and Uc_perp are orthonormal and orthogonal)
    Matrix Qt  = Q.transpose();
    Matrix A_t = Qt * A * Q;
    Matrix B_t = Qt * B;
    Matrix C_t = C * Q;

    // Keep controllable block (first rc states)
    Matrix A_c = A_t.topLeftCorner(rc, rc);
    Matrix B_c = B_t.topRows(rc);
    Matrix C_c = C_t.leftCols(rc);

    // Now remove unobservable states from the controllable subsystem
    // Build observability matrix for (A_c, C_c): [C_c; C_c*A_c; ...]
    Matrix Ob  = Matrix::Zero(rc * C_c.rows(), rc);
    Matrix Ak2 = Matrix::Identity(rc, rc);
    for (int k = 0; k < rc; ++k) {
        Matrix rowblock                             = C_c * Ak2;  // p x rc
        Ob.block(k * C_c.rows(), 0, C_c.rows(), rc) = rowblock;
        Ak2                                         = A_c * Ak2;
    }

    Matrix Uo = colspace_basis(Ob.transpose(), atol);
    int    ro = static_cast<int>(Uo.cols());

    if (ro == 0) {
        // No observable states -> return D-only
        return StateSpace(Matrix::Zero(0, 0), Matrix::Zero(0, B.cols()), Matrix::Zero(C.rows(), 0), D, Ts);
    }

    // Form final transform for controllable subsystem using observable basis
    // Uo is rc x ro (basis in controllable coordinates). We need a selector S of
    // size (rc + nc2) x ro where the top rc rows are Uo and bottom nc2 rows are zero.
    int    rfinal = ro;
    Matrix S      = Matrix::Zero(rc + nc2, std::max(1, rfinal));
    if (rfinal > 0) S.topRows(rc).leftCols(rfinal) = Uo.leftCols(rfinal);

    // Build final transform from original coordinates: n x ro
    Matrix T_final = Q * S;
    Matrix Tfin_t  = T_final.transpose();

    Matrix A_f = Tfin_t * A * T_final;
    Matrix B_f = Tfin_t * B;
    Matrix C_f = C * T_final;

    // Keep top-left ro x ro
    Matrix A_min = A_f.topLeftCorner(ro, ro);
    Matrix B_min = B_f.topRows(ro);
    Matrix C_min = C_f.leftCols(ro);

    StateSpace red(A_min, B_min, C_min, D, Ts);
    return red;
}

}  // namespace control
