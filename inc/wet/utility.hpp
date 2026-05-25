#pragma once

#include <numbers>
#include <type_traits>

#include "wet/math/wetmelon_math.hpp"

namespace wetmelon::control {

template<typename T = double>
    requires std::is_floating_point_v<T>
constexpr T mag2db(T mag) {
    return T{20} * wet::log10(mag);
}

template<typename T = double>
    requires std::is_floating_point_v<T>
constexpr T db2mag(T db) {
    return wet::pow(T{10}, db / T{20});
}

template<typename T = double>
    requires std::is_floating_point_v<T>
constexpr T rad2deg(T rad) {
    return rad * (T{180} / std::numbers::pi_v<T>);
}

template<typename T = double>
    requires std::is_floating_point_v<T>
constexpr T deg2rad(T deg) {
    return deg * (std::numbers::pi_v<T> / T{180});
}

template<typename T>
    requires std::is_floating_point_v<T>
constexpr T wrap(T x, T min, T max) {
    return x - ((max - min) * wet::floor((x - min) / (max - min)));
}
} // namespace wetmelon::control