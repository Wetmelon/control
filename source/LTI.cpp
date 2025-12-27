#include "LTI.hpp"

#include <optional>

#include "ss.hpp"
#include "tf.hpp"
#include "types.hpp"
#include "zpk.hpp"

namespace control {

/* ZPK Free Functions */
ZeroPoleGain zpk(const std::vector<Zero>& zeros,
                 const std::vector<Pole>& poles,
                 double                   gain,
                 std::optional<double>    Ts) {
    return ZeroPoleGain{zeros, poles, gain, Ts};
}

ZeroPoleGain zpk(const TransferFunction& tf) {
    return tf.toZeroPoleGain();
}

ZeroPoleGain zpk(TransferFunction&& tf) {
    return tf.toZeroPoleGain();  // Method handles const ref, move not needed
}

ZeroPoleGain zpk(const StateSpace& sys) {
    return sys.toZeroPoleGain();
}

ZeroPoleGain zpk(StateSpace&& sys) {
    return sys.toZeroPoleGain();  // Method handles const ref, move not needed
}

/* SS Free Functions */
StateSpace ss(const TransferFunction& tf) {
    return tf.toStateSpace();
}

StateSpace ss(TransferFunction&& tf) {
    return tf.toStateSpace();  // Method handles const ref, move not needed
}

StateSpace ss(const ZeroPoleGain& zpk_sys) {
    return zpk_sys.toStateSpace();
}

StateSpace ss(ZeroPoleGain&& zpk_sys) {
    return zpk_sys.toStateSpace();  // Method handles const ref, move not needed
}

/* TF Free Functions */
TransferFunction tf(const StateSpace& sys) {
    return sys.toTransferFunction();
}

TransferFunction tf(StateSpace&& sys) {
    return sys.toTransferFunction();  // Method handles const ref, move not needed
}

TransferFunction tf(const ZeroPoleGain& zpk_sys) {
    return zpk_sys.toTransferFunction();
}

TransferFunction tf(ZeroPoleGain&& zpk_sys) {
    return zpk_sys.toTransferFunction();  // Method handles const ref, move not needed
}

/**
 * @brief Extract a SISO transfer function from a MIMO StateSpace system.
 *
 * For MIMO systems, extracts the transfer function from a specific input to a specific output.
 * This creates a SISO subsystem: G_ij(s) = C_i(sI-A)^(-1)B_j + D_ij
 * where i is the output index and j is the input index (0-based).
 *
 * @param sys          StateSpace system (can be SISO or MIMO)
 * @param output_idx   Output index (0-based, must be < number of outputs)
 * @param input_idx    Input index (0-based, must be < number of inputs)
 * @return TransferFunction  Transfer function from input_idx to output_idx
 * @throws std::out_of_range if indices are out of bounds
 */
TransferFunction tf(const StateSpace& sys, int output_idx, int input_idx) {
    // Validate indices
    int num_outputs = sys.C.rows();
    int num_inputs  = sys.B.cols();

    if (output_idx < 0 || output_idx >= num_outputs) {
        throw std::out_of_range("Output index " + std::to_string(output_idx) +
                                " is out of range [0, " + std::to_string(num_outputs - 1) + "]");
    }

    if (input_idx < 0 || input_idx >= num_inputs) {
        throw std::out_of_range("Input index " + std::to_string(input_idx) +
                                " is out of range [0, " + std::to_string(num_inputs - 1) + "]");
    }

    int n = sys.A.rows();  // State dimension

    // Extract SISO subsystem: C_row(i), B_col(j), D(i,j)
    RowVec C_i  = sys.C.row(output_idx);
    ColVec B_j  = sys.B.col(input_idx);
    double D_ij = sys.D(output_idx, input_idx);

    // For very simple cases, handle directly
    if (n == 0) {
        // Pure gain system (no states)
        return TransferFunction({D_ij}, {1.0}, sys.Ts);
    }

    // Compute transfer function using Faddeev-LeVerrier algorithm
    // This is numerically stable for high-order systems

    // Step 1: Compute characteristic polynomial using Faddeev-LeVerrier
    // det(sI - A) = s^n + a_{n-1}s^{n-1} + ... + a_0
    std::vector<double> p(n + 1, 0.0);  // p[0] = 0
    std::vector<Matrix> H(n + 1);
    H[0] = Matrix::Identity(n, n);

    for (int k = 1; k <= n; ++k) {
        H[k] = sys.A * H[k - 1];
        p[k] = H[k].trace() / static_cast<double>(k);
    }

    // Build denominator polynomial: s^n + a_{n-1}s^{n-1} + ... + a_0
    std::vector<double> den(n + 1);
    den[0] = 1.0;  // Leading coefficient s^n
    for (int i = 1; i <= n; ++i) {
        den[i] = (i % 2 == 1 ? -p[i] : p[i]);
    }

    // Step 2: Compute numerator coefficients
    // For the transfer function G(s) = [C*adj(sI-A)*B + D*det(sI-A)] / det(sI-A)
    std::vector<double> num(n + 1, 0.0);

    // Add D*det(sI-A) term
    for (int i = 0; i <= n; ++i) {
        num[i] += D_ij * den[i];
    }

    // Add C*adj(sI-A)*B term using the Faddeev-LeVerrier matrices
    // The adjoint matrix adj(sI-A) = sum_{k=0}^{n-1} s^{n-1-k} * F_k
    // where F_k satisfy a similar recurrence
    if (n > 0) {
        std::vector<Matrix> F(n);
        F[0] = Matrix::Identity(n, n);

        for (int k = 1; k < n; ++k) {
            F[k] = sys.A * F[k - 1] - p[k] * Matrix::Identity(n, n);
        }

        // Compute C * F_k * B for each k and add to appropriate power of s
        for (int k = 0; k < n; ++k) {
            double coeff = C_i.dot(F[k] * B_j);
            num[k + 1] += coeff;
        }
    }

    // Normalize denominator to have leading coefficient 1
    double den_leading = den[0];
    if (std::abs(den_leading) > 1e-10) {
        for (double& coeff : den) {
            coeff /= den_leading;
        }
        for (double& coeff : num) {
            coeff /= den_leading;
        }
    }

    // Remove leading zeros from numerator (but keep at least one coefficient)
    while (num.size() > 1 && std::abs(num[0]) < 1e-10) {
        num.erase(num.begin());
    }

    // Create and return transfer function
    TransferFunction result(num, den, sys.Ts);
    return result;
}

TransferFunction tf(StateSpace&& sys, int output_idx, int input_idx) {
    // Forward to const& implementation to avoid duplicating logic
    return tf(static_cast<const StateSpace&>(sys), output_idx, input_idx);
}

/**
 * @brief Convert a continuous-time LTI system to discrete-time using specified method.
 *
 * @param sys           Continuous-time LTI system
 * @param Ts            Sampling time
 * @param method        Discretization method (default: ZOH)
 * @param prewarp       Optional pre-warp frequency for Tustin method
 * @return StateSpace   Discrete-time StateSpace system
 */
StateSpace c2d(const LTI& sys, double Ts, DiscretizationMethod method, std::optional<double> prewarp) {
    return sys.discretize(Ts, method, prewarp);
}

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
}  // namespace control
