#pragma once

#include "ss.hpp"
#include "tf.hpp"
#include "zpk.hpp"

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