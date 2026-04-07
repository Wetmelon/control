#pragma once
#include <type_traits>

#include "constexpr_complex.hpp"

namespace wetmelon::control {

// True if T is wet::complex<U> for some U
template<typename T>
struct is_complex : std::false_type {};

template<typename T>
struct is_complex<wet::complex<T>> : std::true_type {};

template<typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

// True if T is float, double, wet::complex<float>, wet::complex<double>, or const-qualified versions
// Usage: static_assert(is_matrix_element_v<T>);
template<typename T>
struct is_matrix_element : std::bool_constant<
                               std::is_same_v<std::remove_const_t<T>, float> || std::is_same_v<std::remove_const_t<T>, double> || is_complex<std::remove_const_t<T>>::value> {};

template<typename T>
inline constexpr bool is_matrix_element_v = is_matrix_element<T>::value;

// Extracts the underlying scalar type (float/double) from T or wet::complex<T>
template<typename T>
struct scalar_type {
    using type = T;
};
template<typename T>
struct scalar_type<wet::complex<T>> {
    using type = T;
};
template<typename T>
using scalar_type_t = typename scalar_type<T>::type;

/**
 * @brief Concept for any type that provides 2D matrix-like element access
 *
 * A MatrixLike type must provide:
 * - Static `rows()` and `cols()` returning dimensions
 * - `operator()(size_t, size_t)` for element access
 * - A `value_type` typedef
 */
template<typename M>
concept MatrixLike = requires(const M& m, size_t i, size_t j) {
    { M::rows() } -> std::convertible_to<size_t>;
    { M::cols() } -> std::convertible_to<size_t>;
    { m(i, j) } -> std::convertible_to<typename M::value_type>;
};

/**
 * @brief Concept for a MatrixLike type with specific dimensions
 */
template<typename M, size_t R, size_t C>
concept MatrixLikeOf = MatrixLike<M> && (M::rows() == R) && (M::cols() == C);

} // namespace wetmelon::control
