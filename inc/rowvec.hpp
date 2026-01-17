#pragma once

#include <cstddef>

#include "matrix.hpp"

namespace wetmelon::control {

/**
 * @brief Row vector specialization of Matrix<1, N, T>
 * @ingroup linear_algebra
 * @tparam N Vector dimension
 * @tparam T Element type
 */
template<size_t N, typename T = double>
struct RowVec : public Matrix<1, N, T> {
    /**
     * @brief Default constructor
     */
    constexpr RowVec() = default;
    /**
     * @brief Copy constructor and assignment
     */
    constexpr RowVec(const RowVec&) = default;
    constexpr RowVec& operator=(const RowVec&) = default;
    /**
     * @brief Move constructor and assignment
     */
    constexpr RowVec(RowVec&&) = default;
    constexpr RowVec& operator=(RowVec&&) = default;
    /**
     * @brief Destructor
     */
    constexpr ~RowVec() = default;

    /**
     * @brief Constructor from initializer list
     */
    constexpr RowVec(std::initializer_list<T> values) : Matrix<1, N, T>() {
        size_t i = 0;
        for (const auto& val : values) {
            if (i < N) {
                this->data_[0][i] = val;
            }
            ++i;
        }
    }

    // Intentionally implicit conversion constructors
    template<typename U>
    constexpr RowVec(const RowVec<N, U>& other) : Matrix<1, N, T>(other) {}

    // Intentionally implicit conversion constructors
    template<typename U>
    constexpr RowVec(const Matrix<1, N, U>& other) : Matrix<1, N, T>(other) {}

    template<typename U>
    constexpr RowVec& operator=(const Matrix<1, N, U>& other) {
        Matrix<1, N, T>::operator=(other);
        return *this;
    }

    /**
     * @brief Element access operator
     * @param idx Vector index
     * @return Const reference to element
     */
    constexpr const T& operator[](size_t idx) const { return this->data_[0][idx]; }
    /**
     * @brief Element access operator
     * @param idx Vector index
     * @return Reference to element
     */
    constexpr T& operator[](size_t idx) { return this->data_[0][idx]; }

    /**
     * @brief Dot product (inner product)
     * @param vec1 First vector
     * @param vec2 Second vector
     * @return Scalar dot product
     */
    [[nodiscard]] friend constexpr T dot(const RowVec<N, T>& vec1, const RowVec<N, T>& vec2) {
        T result = 0;
        for (std::size_t i = 0; i < N; ++i) {
            result += vec1.data_[0][i] * vec2.data_[0][i];
        }
        return result;
    }

    // Cross product for 3D vectors
    template<size_t D = N>
    /**
     * @brief Cross product for 3D vectors
     * @param other Vector to cross with
     * @return 3D cross product result
     */
    [[nodiscard]] constexpr RowVec<3, T> cross(const RowVec<3, T>& other) const
        requires(D == 3)
    {
        return RowVec<3, T>{
            this->data_[0][1] * other.data_[0][2] - this->data_[0][2] * other.data_[0][1],
            this->data_[0][2] * other.data_[0][0] - this->data_[0][0] * other.data_[0][2],
            this->data_[0][0] * other.data_[0][1] - this->data_[0][1] * other.data_[0][0]
        };
    }

    /**
     * @brief Euclidean norm (magnitude) of the vector
     * @return Vector magnitude
     */
    [[nodiscard]] constexpr T norm() const { return wet::sqrt(dot(*this, *this)); }

    /**
     * @brief Normalized vector (unit length)
     * @return Normalized vector
     */
    [[nodiscard]] constexpr RowVec normalized() const {
        T n = norm();
        if (n == 0) {
            return *this; // Avoid division by zero
        }
        return *this * (T(1) / n);
    }
};

// Binary operators for RowVec
template<typename T, size_t N>
[[nodiscard]] constexpr RowVec<N, T> operator+(const RowVec<N, T>& lhs, const RowVec<N, T>& rhs) {
    return RowVec<N, T>(lhs + static_cast<const Matrix<1, N, T>&>(rhs));
}

template<typename T, size_t N>
[[nodiscard]] constexpr RowVec<N, T> operator-(const RowVec<N, T>& lhs, const RowVec<N, T>& rhs) {
    return RowVec<N, T>(lhs - static_cast<const Matrix<1, N, T>&>(rhs));
}

template<typename T, size_t N, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
[[nodiscard]] constexpr RowVec<N, T> operator*(const RowVec<N, T>& vec, Scalar scalar) {
    return RowVec<N, T>(static_cast<const Matrix<1, N, T>&>(vec) * scalar);
}

template<typename T, size_t N, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
[[nodiscard]] constexpr RowVec<N, T> operator*(Scalar scalar, const RowVec<N, T>& vec) {
    return vec * scalar;
}

template<typename T, size_t N, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
[[nodiscard]] constexpr RowVec<N, T> operator+(const RowVec<N, T>& vec, Scalar scalar) {
    return RowVec<N, T>(static_cast<const Matrix<1, N, T>&>(vec) + scalar);
}

template<typename T, size_t N, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
[[nodiscard]] constexpr RowVec<N, T> operator+(Scalar scalar, const RowVec<N, T>& vec) {
    return vec + scalar;
}

template<typename T, size_t N, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
[[nodiscard]] constexpr RowVec<N, T> operator-(const RowVec<N, T>& vec, Scalar scalar) {
    return RowVec<N, T>(static_cast<const Matrix<1, N, T>&>(vec) - scalar);
}

template<typename T, size_t N>
[[nodiscard]] constexpr RowVec<N, T> operator/(const RowVec<N, T>& vec, T scalar) {
    return RowVec<N, T>(static_cast<const Matrix<1, N, T>&>(vec) / scalar);
}

template<typename T, size_t N>
[[nodiscard]] constexpr RowVec<N, T> operator-(const RowVec<N, T>& vec) {
    Matrix<1, N, T> base = vec;
    return RowVec<N, T>(-base);
}

// Deduce RowVec<N, T> from variadic constructor arguments, e.g. RowVec vec{1.0f, 2.0f, 3.0f};
// Integer literals deduce to double
template<typename T, typename... Args>
    requires(std::is_integral_v<T> && (std::is_integral_v<Args> && ...))
RowVec(T, Args...) -> RowVec<1 + sizeof...(Args), double>;

template<typename T, typename... Args>
    requires(!std::is_integral_v<T> || !(std::is_integral_v<Args> && ...))
RowVec(T, Args...) -> RowVec<1 + sizeof...(Args), T>;

template<typename U, size_t M>
RowVec(Matrix<1, M, U>) -> RowVec<M, U>;

template<typename T, size_t N>
RowVec(const std::array<T, N>&) -> RowVec<N, T>;

template<typename T>
using RowVec2 = RowVec<2, T>;

template<typename T>
using RowVec3 = RowVec<3, T>;

template<typename T>
using RowVec4 = RowVec<4, T>;

}; // namespace wetmelon::control