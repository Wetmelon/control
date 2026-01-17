#pragma once

#include "matrix.hpp"

namespace wetmelon::control {

/**
 * @brief Concrete Column vector specialization of Matrix<N, 1, T>
 * @ingroup linear_algebra
 *
 * @tparam N Vector dimension
 * @tparam T Element type
 */
template<size_t N, typename T = double>
struct ColVec : public Matrix<N, 1, T> {
    /**
     * @brief Default constructor
     */
    constexpr ColVec() = default;
    /**
     * @brief Copy constructor and assignment
     */
    constexpr ColVec(const ColVec&) = default;
    constexpr ColVec& operator=(const ColVec&) = default;
    /**
     * @brief Move constructor and assignment
     */
    constexpr ColVec(ColVec&&) = default;
    constexpr ColVec& operator=(ColVec&&) = default;
    /**
     * @brief Destructor
     */
    constexpr ~ColVec() = default;

    /**
     * @brief Constructor from initializer list
     */
    constexpr ColVec(std::initializer_list<T> values) : Matrix<N, 1, T>() {
        size_t i = 0;
        for (const auto& val : values) {
            if (i < N) {
                this->data_[i][0] = val;
            }
            ++i;
        }
    }

    // Constructor from std::array<T, N> - enables CTAD
    constexpr ColVec(const std::array<T, N>& arr) : Matrix<N, 1, T>() {
        for (size_t i = 0; i < N; ++i) {
            this->data_[i][0] = arr[i];
        }
    }

    template<typename U>
    constexpr ColVec(const ColVec<N, U>& other) : Matrix<N, 1, T>(other) {}

    template<typename U>
    constexpr ColVec(const Matrix<N, 1, U>& other) : Matrix<N, 1, T>(other) {}

    template<typename U>
    constexpr ColVec& operator=(const Matrix<N, 1, U>& other) {
        Matrix<N, 1, T>::operator=(other);
        return *this;
    }

    /**
     * @brief Element access operator
     * @param idx Vector index
     * @return Const reference to element
     */
    constexpr const T& operator[](size_t idx) const { return this->data_[idx][0]; }
    constexpr const T& operator()(size_t idx) const { return this->data_[idx][0]; }

    /**
     * @brief Element access operator
     * @param idx Vector index
     * @return Reference to element
     */
    constexpr T& operator[](size_t idx) { return this->data_[idx][0]; }
    constexpr T& operator()(size_t idx) { return this->data_[idx][0]; }

    /**
     * @brief Dot product (inner product)
     * @param vec1 First vector
     * @param vec2 Second vector
     * @return Scalar dot product
     */
    [[nodiscard]] friend constexpr T dot(const ColVec<N, T>& vec1, const ColVec<N, T>& vec2) {
        T result = 0;
        for (std::size_t i = 0; i < N; ++i) {
            result += vec1.data_[i][0] * vec2.data_[i][0];
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
    [[nodiscard]] constexpr ColVec<3, T> cross(const ColVec<3, T>& other) const
        requires(D == 3)
    {
        return ColVec<3, T>{
            this->data_[1][0] * other.data_[2][0] - this->data_[2][0] * other.data_[1][0],
            this->data_[2][0] * other.data_[0][0] - this->data_[0][0] * other.data_[2][0],
            this->data_[0][0] * other.data_[1][0] - this->data_[1][0] * other.data_[0][0]
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
    [[nodiscard]] constexpr ColVec normalized() const {
        T n = norm();
        if (n == 0) {
            return *this; // Avoid division by zero
        }
        return *this * (T(1) / n);
    }

    [[nodiscard]] constexpr size_t size() const { return N; }

    template<typename U>
    [[nodiscard]] constexpr ColVec<N, U> as() const {
        return ColVec<N, U>(*this);
    }
};

// Binary operators for ColVec
template<typename T, size_t N>
[[nodiscard]] constexpr ColVec<N, T> operator+(const ColVec<N, T>& lhs, const ColVec<N, T>& rhs) {
    return ColVec<N, T>(lhs + static_cast<const Matrix<N, 1, T>&>(rhs));
}

template<typename T, size_t N>
[[nodiscard]] constexpr ColVec<N, T> operator-(const ColVec<N, T>& lhs, const ColVec<N, T>& rhs) {
    return ColVec<N, T>(lhs - static_cast<const Matrix<N, 1, T>&>(rhs));
}

template<typename T, size_t N, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
[[nodiscard]] constexpr ColVec<N, T> operator*(const ColVec<N, T>& vec, Scalar scalar) {
    return ColVec<N, T>(static_cast<const Matrix<N, 1, T>&>(vec) * scalar);
}

template<typename T, size_t N, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
[[nodiscard]] constexpr ColVec<N, T> operator*(Scalar scalar, const ColVec<N, T>& vec) {
    return vec * scalar;
}

template<typename T, size_t N, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
[[nodiscard]] constexpr ColVec<N, T> operator+(const ColVec<N, T>& vec, Scalar scalar) {
    return ColVec<N, T>(static_cast<const Matrix<N, 1, T>&>(vec) + scalar);
}

template<typename T, size_t N, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
[[nodiscard]] constexpr ColVec<N, T> operator+(Scalar scalar, const ColVec<N, T>& vec) {
    return vec + scalar;
}

template<typename T, size_t N, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
[[nodiscard]] constexpr ColVec<N, T> operator-(const ColVec<N, T>& vec, Scalar scalar) {
    return ColVec<N, T>(static_cast<const Matrix<N, 1, T>&>(vec) - scalar);
}

template<typename T, size_t N>
[[nodiscard]] constexpr ColVec<N, T> operator/(const ColVec<N, T>& vec, T scalar) {
    return ColVec<N, T>(static_cast<const Matrix<N, 1, T>&>(vec) / scalar);
}

template<typename T, size_t N>
[[nodiscard]] constexpr ColVec<N, T> operator-(const ColVec<N, T>& vec) {
    Matrix<N, 1, T> base = vec;
    return ColVec<N, T>(-base);
}

// Deduce ColVec<N, T> from variadic constructor arguments, e.g. ColVec vec{1.0f, 2.0f, 3.0f};
// Integer literals deduce to double
template<typename T, typename... Args>
    requires(std::is_integral_v<T> && (std::is_integral_v<Args> && ...))
ColVec(T, Args...) -> ColVec<1 + sizeof...(Args), double>;

template<typename T, typename... Args>
    requires(!std::is_integral_v<T> || !(std::is_integral_v<Args> && ...))
ColVec(T, Args...) -> ColVec<1 + sizeof...(Args), T>;

template<typename U, size_t M>
ColVec(Matrix<M, 1, U>) -> ColVec<M, U>;

template<typename T, size_t N>
ColVec(const std::array<T, N>&) -> ColVec<N, T>;

// Common aliases
template<typename T>
using Vec2 = ColVec<2, T>;

template<typename T>
using Vec3 = ColVec<3, T>;

template<typename T>
using Vec4 = ColVec<4, T>;

}; // namespace wetmelon::control