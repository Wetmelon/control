#pragma once

#include <vector>

#include "constexpr_math.hpp"

namespace wetmelon::control {
constexpr std::vector<double> linspace(double start, double end, size_t num) {
    std::vector<double> result;
    result.reserve(num);
    if (num == 1) {
        result.push_back(start);
    } else {
        double step = (end - start) / static_cast<double>(num - 1);
        for (size_t i = 0; i < num; ++i) {
            result.push_back(start + i * step);
        }
    }
    return result;
}

constexpr std::vector<double> linspace(const std::pair<double, double>& span, size_t num) {
    return linspace(span.first, span.second, num);
}

constexpr std::vector<double> logspace(double start, double end, size_t num, double base = 10) {
    std::vector<double> result;
    result.reserve(num);
    if (num == 1) {
        result.push_back(start);
    } else {
        // Compute logarithms in the requested base
        const double log_base = wet::log(base);
        double       log_start = wet::log(start) / log_base;
        double       log_end = wet::log(end) / log_base;
        double       step = (log_end - log_start) / static_cast<double>(num - 1);
        for (size_t i = 0; i < num; ++i) {
            result.push_back(wet::pow(base, log_start + i * step));
        }
    }
    return result;
}

constexpr std::vector<double> logspace(const std::pair<double, double>& span, size_t num, double base = 10) {
    return logspace(span.first, span.second, num, base);
}

constexpr double mag2db(double mag) {
    return 20.0 * wet::log10(mag);
}
constexpr double db2mag(double db) {
    return wet::pow(10.0, db / 20.0);
}

constexpr double rad2deg(double rad) {
    return rad * (180.0 / std::numbers::pi);
}

constexpr double deg2rad(double deg) {
    return deg * (std::numbers::pi / 180.0);
}

constexpr double wrap(double x, double min, double max) {
    return x - (max - min) * wet::floor((x - min) / (max - min));
}
}; // namespace wetmelon::control