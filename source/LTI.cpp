#include "LTI.hpp"

#include <cstddef>
#include <optional>

#include "types.hpp"
#include "unsupported/Eigen/MatrixFunctions"

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

}  // namespace control
