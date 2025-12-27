#pragma once

#include "LTI.hpp"         // IWYU pragma: keep
#include "integrator.hpp"  // IWYU pragma: keep
#include "solver.hpp"      // IWYU pragma: keep
#include "ss.hpp"          // IWYU pragma: keep
#include "tf.hpp"          // IWYU pragma: keep
#include "types.hpp"       // IWYU pragma: keep
#include "utility.hpp"     // IWYU pragma: keep
#include "zpk.hpp"         // IWYU pragma: keep

namespace control {

// ---------------------------------------------------------------------------
// LTI operations for mixed types always return StateSpace representation
// ---------------------------------------------------------------------------

// Compute Gramian matrices (continuous-time iterative method)

template <class T>
concept SSConvertible = requires(const T& t) { { t.toStateSpace() }; };

template <SSConvertible T>
Matrix gramian(const T& sys, GramianType type) {
    return sys.toStateSpace().gramian(type);
}

template <SSConvertible T>
StateSpace minreal(const T& sys, double tol = 1e-9) {
    return sys.toStateSpace().minreal(tol);
}

template <SSConvertible T>
StateSpace balred(const T& sys, size_t r) {
    return sys.toStateSpace().balred(r);
}

template <SSConvertible A, SSConvertible B>
StateSpace series(const A& a, const B& b) {
    return series(a.toStateSpace(), b.toStateSpace());
}

template <SSConvertible A, SSConvertible B>
StateSpace parallel(const A& a, const B& b) {
    return parallel(a.toStateSpace(), b.toStateSpace());
}

template <SSConvertible A, SSConvertible B>
StateSpace feedback(const A& a, const B& b, int sign = -1) {
    return feedback(a.toStateSpace(), b.toStateSpace(), sign);
}

template <SSConvertible A, SSConvertible B>
StateSpace operator*(const A& a, const B& b) {
    return series(a.toStateSpace(), b.toStateSpace());
}

template <SSConvertible A, SSConvertible B>
StateSpace operator+(const A& a, const B& b) {
    return parallel(a.toStateSpace(), b.toStateSpace());
}

template <SSConvertible A, SSConvertible B>
StateSpace operator-(const A& a, const B& b) {
    StateSpace neg_b = b.toStateSpace();
    neg_b.C          = -neg_b.C;
    neg_b.D          = -neg_b.D;

    return parallel(a.toStateSpace(), neg_b);
}

template <SSConvertible A, SSConvertible B>
StateSpace operator/(const A& a, const B& b) {
    return feedback(a.toStateSpace(), b.toStateSpace(), -1);
}

}  // namespace control
// ============================================================================
// State-Space Formatting for fmt::format
#include "fmt/core.h"

template <typename SystemType>
std::string formatStateSpaceMatrices(const SystemType& sys) {
    std::string result = "A = \n";
    for (int i = 0; i < sys.A.rows(); ++i) {
        for (int j = 0; j < sys.A.cols(); ++j) {
            result += fmt::format("{:>10.4f}", sys.A(i, j));
        }
        result += "\n";
    }
    result += "\nB = \n";
    for (int i = 0; i < sys.B.rows(); ++i) {
        for (int j = 0; j < sys.B.cols(); ++j) {
            result += fmt::format("{:>10.4f}", sys.B(i, j));
        }
        result += "\n";
    }
    result += "\nC = \n";
    for (int i = 0; i < sys.C.rows(); ++i) {
        for (int j = 0; j < sys.C.cols(); ++j) {
            result += fmt::format("{:>10.4f}", sys.C(i, j));
        }
        result += "\n";
    }
    result += "\nD = \n";
    for (int i = 0; i < sys.D.rows(); ++i) {
        for (int j = 0; j < sys.D.cols(); ++j) {
            result += fmt::format("{:>10.4f}", sys.D(i, j));
        }
        result += "\n";
    }
    return result;
}

template <>
struct fmt::formatter<control::StateSpace> {
    constexpr auto parse(fmt::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const control::StateSpace& sys, fmt::format_context& ctx) const {
        return fmt::format_to(ctx.out(), "{}", formatStateSpaceMatrices(sys));
    }
};