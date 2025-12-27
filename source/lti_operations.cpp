#include "LTI.hpp"

namespace control {

// ============================================================================
// LTI System Arithmetic Operations
// ============================================================================
// These operators enable combining LTI systems to create complex control
// systems from simple building blocks (controllers, plants, sensors).
//
// Usage Examples:
//   auto open_loop    = controller * plant;          // Series connection
//   auto parallel_sys = sys1 + sys2;                 // Parallel (sum)
//   auto error_sys    = reference - measurement;     // Parallel (difference)
//   auto closed_loop  = feedback(fwd_path, fb_path); // Negative feedback
//   auto closed_loop  = fwd_path / fb_path;          // Negative feedback (same as above)
//
// Control System Construction:
//   1. Create individual components (Controller C, Plant G, Sensor H)
//   2. Combine them: T = feedback(C * G, H)
//   3. This creates closed-loop: T(s) = C*G / (1 + C*G*H)
// ============================================================================

StateSpace series(const StateSpace& sys1, const StateSpace& sys2) {
    if (sys1.systemType() != sys2.systemType()) {
        throw std::runtime_error("Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");
    }

    if (sys1.Ts != sys2.Ts) {
        throw std::runtime_error("Sampling times do not match for discrete systems.");
    }

    // Series connection: sys2 follows sys1 (sys1 -> sys2)
    // State space representation:
    // x = [x1; x2]
    // A = [A1,    0  ]    B = [B1]
    //     [B2*C1, A2 ]        [B2*D1]
    // C = [D2*C1, C2]    D = [D2*D1]

    const int n1 = sys1.A.rows();
    const int n2 = sys2.A.rows();
    const int m  = sys1.B.cols();
    const int p  = sys2.C.rows();

    Matrix A = Matrix::Zero(n1 + n2, n1 + n2);
    Matrix B = Matrix::Zero(n1 + n2, m);
    Matrix C = Matrix::Zero(p, n1 + n2);
    Matrix D = Matrix::Zero(p, m);

    // Fill A matrix
    A.block(0, 0, n1, n1)   = sys1.A;
    A.block(n1, 0, n2, n1)  = sys2.B * sys1.C;
    A.block(n1, n1, n2, n2) = sys2.A;

    // Fill B matrix
    B.block(0, 0, n1, m)  = sys1.B;
    B.block(n1, 0, n2, m) = sys2.B * sys1.D;

    // Fill C matrix
    C.block(0, 0, p, n1)  = sys2.D * sys1.C;
    C.block(0, n1, p, n2) = sys2.C;

    // Fill D matrix
    D = sys2.D * sys1.D;

    return StateSpace{std::move(A), std::move(B), std::move(C), std::move(D), sys1.Ts};
}

TransferFunction series(const TransferFunction& sys1, const TransferFunction& sys2) {
    if (sys1.systemType() != sys2.systemType()) {
        throw std::runtime_error("Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");
    }
    if (sys1.Ts != sys2.Ts) {
        throw std::runtime_error("Sampling times do not match for discrete systems.");
    }

    // Numerical-robust path: convert to StateSpace, perform series, reduce, convert back
    StateSpace ss1 = sys1.toStateSpace();
    StateSpace ss2 = sys2.toStateSpace();
    StateSpace ssr = series(ss1, ss2);
    // Convert back to TF, then attempt cancellation (minreal-like)
    TransferFunction tf = ssr.toTransferFunction();
    return tf;
}

ZeroPoleGain series(const ZeroPoleGain& sys1, const ZeroPoleGain& sys2) {
    if (sys1.systemType() != sys2.systemType()) {
        throw std::runtime_error("Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");
    }
    if (sys1.Ts != sys2.Ts) {
        throw std::runtime_error("Sampling times do not match for discrete systems.");
    }

    // Convert to TF, multiply, convert back
    TransferFunction tf1       = sys1.toTransferFunction();
    TransferFunction tf2       = sys2.toTransferFunction();
    TransferFunction result_tf = series(tf1, tf2);
    return result_tf.toZeroPoleGain();
}

StateSpace operator*(const StateSpace& sys1, const StateSpace& sys2) {
    return series(sys1, sys2);
}

TransferFunction operator*(const TransferFunction& sys1, const TransferFunction& sys2) {
    return series(sys1, sys2);
}

ZeroPoleGain operator*(const ZeroPoleGain& sys1, const ZeroPoleGain& sys2) {
    return series(sys1, sys2);
}

/*  Parallel Connections */
StateSpace parallel(const StateSpace& sys1, const StateSpace& sys2) {
    StateSpace ss1 = sys1.toStateSpace();
    StateSpace ss2 = sys2.toStateSpace();

    if (ss1.systemType() != ss2.systemType()) {
        throw std::runtime_error("Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");
    }

    if (ss1.Ts != ss2.Ts) {
        throw std::runtime_error("Sampling times do not match for discrete systems.");
    }

    // Parallel connection: outputs are added
    // State space representation:
    // x = [x1; x2]
    // A = [A1, 0 ]    B = [B1]
    //     [0,  A2]        [B2]
    // C = [C1, C2]    D = [D1 + D2]

    const int n1 = ss1.A.rows();
    const int n2 = ss2.A.rows();
    const int m  = ss1.B.cols();
    const int p  = ss1.C.rows();

    Matrix A = Matrix::Zero(n1 + n2, n1 + n2);
    Matrix B = Matrix::Zero(n1 + n2, m);
    Matrix C = Matrix::Zero(p, n1 + n2);
    Matrix D = Matrix::Zero(p, m);

    // Fill A matrix (block diagonal)
    A.block(0, 0, n1, n1)   = ss1.A;
    A.block(n1, n1, n2, n2) = ss2.A;

    // Fill B matrix (stacked)
    B.block(0, 0, n1, m)  = ss1.B;
    B.block(n1, 0, n2, m) = ss2.B;

    // Fill C matrix (concatenated)
    C.block(0, 0, p, n1)  = ss1.C;
    C.block(0, n1, p, n2) = ss2.C;

    // Fill D matrix (sum)
    D = ss1.D + ss2.D;

    return StateSpace{std::move(A), std::move(B), std::move(C), std::move(D), ss1.Ts};
}
TransferFunction parallel(const TransferFunction& sys1, const TransferFunction& sys2) {
    if (sys1.systemType() != sys2.systemType()) {
        throw std::runtime_error("Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");
    }
    if (sys1.Ts != sys2.Ts) {
        throw std::runtime_error("Sampling times do not match for discrete systems.");
    }

    // Robust path: convert to SS, perform parallel, reduce, convert back
    StateSpace       ss1 = sys1.toStateSpace();
    StateSpace       ss2 = sys2.toStateSpace();
    StateSpace       ssr = parallel(ss1, ss2);
    TransferFunction tf  = ssr.toTransferFunction();
    return tf;
}

ZeroPoleGain parallel(const ZeroPoleGain& sys1, const ZeroPoleGain& sys2) {
    if (sys1.systemType() != sys2.systemType()) {
        throw std::runtime_error("Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");
    }
    if (sys1.Ts != sys2.Ts) {
        throw std::runtime_error("Sampling times do not match for discrete systems.");
    }

    // Convert to TF, add, convert back
    TransferFunction tf1       = sys1.toTransferFunction();
    TransferFunction tf2       = sys2.toTransferFunction();
    TransferFunction result_tf = parallel(tf1, tf2);
    return result_tf.toZeroPoleGain();
}

StateSpace operator+(const StateSpace& sys1, const StateSpace& sys2) {
    return parallel(sys1, sys2);
}

TransferFunction operator+(const TransferFunction& sys1, const TransferFunction& sys2) {
    return parallel(sys1, sys2);
}

ZeroPoleGain operator+(const ZeroPoleGain& sys1, const ZeroPoleGain& sys2) {
    return parallel(sys1, sys2);
}

StateSpace operator-(const StateSpace& sys1, const StateSpace& sys2) {
    StateSpace neg_sys2 = sys2;
    neg_sys2.C          = -neg_sys2.C;
    neg_sys2.D          = -neg_sys2.D;

    return parallel(sys1, neg_sys2);
}

TransferFunction operator-(const TransferFunction& sys1, const TransferFunction& sys2) {
    TransferFunction neg_sys2 = sys2;
    for (double& coeff : neg_sys2.num) {
        coeff = -coeff;
    }

    return parallel(sys1, neg_sys2);
}

ZeroPoleGain operator-(const ZeroPoleGain& sys1, const ZeroPoleGain& sys2) {
    ZeroPoleGain neg_sys2 = sys2;
    neg_sys2.gain_        = -neg_sys2.gain_;

    return parallel(sys1, neg_sys2);
}

/* Feedback Connections */
StateSpace feedback(const StateSpace& sys_forward, const StateSpace& sys_feedback, int sign) {
    StateSpace G = sys_forward.toStateSpace();
    StateSpace H = sys_feedback.toStateSpace();

    if (G.systemType() != H.systemType()) {
        throw std::runtime_error("Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");
    }

    if (G.Ts != H.Ts) {
        throw std::runtime_error("Sampling times do not match for discrete systems.");
    }

    // Feedback connection:
    // Closed-loop transfer function: G_cl = G / (1 - sign * G * H)
    // where G is forward path, H is feedback path
    // sign = -1 for negative feedback (default), +1 for positive feedback
    //
    // State space representation:
    // A_cl = [A_G + sign*B_G*(I - sign*D_H*D_G)^-1*D_H*C_G,  sign*B_G*(I - sign*D_H*D_G)^-1*C_H]
    //        [B_H*(I - sign*D_G*D_H)^-1*C_G,                   A_H + sign*B_H*(I - sign*D_G*D_H)^-1*D_G*C_H]
    // B_cl = [B_G*(I - sign*D_H*D_G)^-1]
    //        [sign*B_H*(I - sign*D_G*D_H)^-1*D_G]
    // C_cl = [(I - sign*D_H*D_G)^-1*C_G,  sign*(I - sign*D_H*D_G)^-1*D_H*C_H]
    // D_cl = (I - sign*D_H*D_G)^-1*D_G

    const int nG = G.A.rows();
    const int nH = H.A.rows();
    const int m  = G.B.cols();
    const int p  = G.C.rows();

    // Calculate the inverses we need
    const auto   I_p = Matrix::Identity(p, p);
    const auto   I_m = Matrix::Identity(m, m);
    const double s   = static_cast<double>(sign);

    // (I - sign*D_H*D_G)^-1
    const auto inv1 = (I_p - s * H.D * G.D).inverse();
    // (I - sign*D_G*D_H)^-1
    const auto inv2 = (I_m - s * G.D * H.D).inverse();

    Matrix A_cl = Matrix::Zero(nG + nH, nG + nH);
    Matrix B_cl = Matrix::Zero(nG + nH, m);
    Matrix C_cl = Matrix::Zero(p, nG + nH);
    Matrix D_cl = Matrix::Zero(p, m);

    // Fill A_cl matrix
    A_cl.block(0, 0, nG, nG)   = G.A + s * G.B * inv1 * H.D * G.C;
    A_cl.block(0, nG, nG, nH)  = s * G.B * inv1 * H.C;
    A_cl.block(nG, 0, nH, nG)  = H.B * inv2 * G.C;
    A_cl.block(nG, nG, nH, nH) = H.A + s * H.B * inv2 * G.D * H.C;

    // Fill B_cl matrix
    B_cl.block(0, 0, nG, m)  = G.B * inv1;
    B_cl.block(nG, 0, nH, m) = s * H.B * inv2 * G.D;

    // Fill C_cl matrix
    C_cl.block(0, 0, p, nG)  = inv1 * G.C;
    C_cl.block(0, nG, p, nH) = s * inv1 * H.D * H.C;

    // Fill D_cl matrix
    D_cl = inv1 * G.D;

    return StateSpace{std::move(A_cl), std::move(B_cl), std::move(C_cl), std::move(D_cl), G.Ts};
}

TransferFunction feedback(const TransferFunction& sys_forward, const TransferFunction& sys_feedback, int sign) {
    if (sys_forward.systemType() != sys_feedback.systemType()) {
        throw std::runtime_error("Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");
    }
    if (sys_forward.Ts != sys_feedback.Ts) {
        throw std::runtime_error("Sampling times do not match for discrete systems.");
    }
    // Use numerically robust path: convert to StateSpace, perform feedback, reduce, convert back
    StateSpace       G   = sys_forward.toStateSpace();
    StateSpace       H   = sys_feedback.toStateSpace();
    StateSpace       ssr = feedback(G, H, sign);
    TransferFunction tf  = ssr.toTransferFunction();
    return tf;
}

ZeroPoleGain feedback(const ZeroPoleGain& sys_forward, const ZeroPoleGain& sys_feedback, int sign) {
    if (sys_forward.systemType() != sys_feedback.systemType()) {
        throw std::runtime_error("Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");
    }
    if (sys_forward.Ts != sys_feedback.Ts) {
        throw std::runtime_error("Sampling times do not match for discrete systems.");
    }

    // Convert to TF, perform feedback, convert back
    TransferFunction G = sys_forward.toTransferFunction();
    TransferFunction H = sys_feedback.toTransferFunction();

    TransferFunction closed_loop_tf = feedback(G, H, sign);
    return closed_loop_tf.toZeroPoleGain();
}

StateSpace operator/(const StateSpace& sys_forward, const StateSpace& sys_feedback) {
    return feedback(sys_forward, sys_feedback, -1);
}

TransferFunction operator/(const TransferFunction& sys_forward, const TransferFunction& sys_feedback) {
    return feedback(sys_forward, sys_feedback, -1);
}

ZeroPoleGain operator/(const ZeroPoleGain& sys_forward, const ZeroPoleGain& sys_feedback) {
    return feedback(sys_forward, sys_feedback, -1);
}

};  // namespace control