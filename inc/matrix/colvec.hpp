#pragma once

#include <cstddef>

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
                this->data_[i] = val;
            }
            ++i;
        }
    }

    // Constructor from std::array<T, N> - enables CTAD
    constexpr ColVec(const std::array<T, N>& arr) : Matrix<N, 1, T>() {
        for (size_t i = 0; i < N; ++i) {
            this->data_[i] = arr[i];
        }
    }

    template<typename U>
    explicit constexpr ColVec(const ColVec<N, U>& other) : Matrix<N, 1, T>(other) {}

    template<typename U>
    explicit constexpr ColVec(const Matrix<N, 1, U>& other) : Matrix<N, 1, T>(other) {}

    template<typename U>
    constexpr ColVec& operator=(const Matrix<N, 1, U>& other) {
        Matrix<N, 1, T>::operator=(other);
        return *this;
    }

    // Bring base class operator() into scope (single-arg overload hides it otherwise)
    using Matrix<N, 1, T>::operator();

    /**
     * @brief Element access operator
     * @param idx Vector index
     * @return Const reference to element
     */
    constexpr const T& operator[](size_t idx) const { return this->data_[idx]; }
    constexpr const T& operator()(size_t idx) const { return this->data_[idx]; }

    /**
     * @brief Element access operator
     * @param idx Vector index
     * @return Reference to element
     */
    constexpr T& operator[](size_t idx) { return this->data_[idx]; }
    constexpr T& operator()(size_t idx) { return this->data_[idx]; }

    /**
     * @brief Dot product (inner product)
     * @param vec1 First vector
     * @param vec2 Second vector
     * @return Scalar dot product
     */
    [[nodiscard]] friend constexpr T dot(const ColVec<N, T>& vec1, const ColVec<N, T>& vec2) {
        T result = 0;
        for (std::size_t i = 0; i < N; ++i) {
            result += vec1.data_[i] * vec2.data_[i];
        }
        return result;
    }

    // Cross product for 3D vectors
    template<size_t D = N>
    /**
     * @brief 3D cross product with another 3D column vector
     * @param other The other 3D column vector
     * @return 3D cross product result
     */
    [[nodiscard]] constexpr ColVec<3, T> cross(const ColVec<3, T>& other) const
        requires(D == 3)
    {
        return ColVec<3, T>{
            this->data_[1] * other.data_[2] - this->data_[2] * other.data_[1],
            this->data_[2] * other.data_[0] - this->data_[0] * other.data_[2],
            this->data_[0] * other.data_[1] - this->data_[1] * other.data_[0]
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

    // Type-preserving compound assignment (base returns Matrix&)
    template<MatrixLike M>
    constexpr ColVec& operator+=(const M& other) {
        Matrix<N, 1, T>::operator+=(other);
        return *this;
    }

    template<MatrixLike M>
    constexpr ColVec& operator-=(const M& other) {
        Matrix<N, 1, T>::operator-=(other);
        return *this;
    }

    constexpr ColVec& operator+=(T scalar) {
        Matrix<N, 1, T>::operator+=(scalar);
        return *this;
    }

    constexpr ColVec& operator-=(T scalar) {
        Matrix<N, 1, T>::operator-=(scalar);
        return *this;
    }

    constexpr ColVec& operator*=(T scalar) {
        Matrix<N, 1, T>::operator*=(scalar);
        return *this;
    }

    constexpr ColVec& operator/=(T scalar) {
        Matrix<N, 1, T>::operator/=(scalar);
        return *this;
    }
};

/**
 * @brief Matrix * ColVec returns ColVec (not Matrix<N,1>)
 */
template<size_t Rows, size_t Cols, typename T, typename U>
[[nodiscard]] constexpr ColVec<Rows, T> operator*(const Matrix<Rows, Cols, T>& mat, const ColVec<Cols, U>& vec) {
    ColVec<Rows, T> result;
    for (size_t i = 0; i < Rows; ++i) {
        T sum = 0;
        for (size_t k = 0; k < Cols; ++k) {
            sum += mat(i, k) * static_cast<T>(vec[k]);
        }
        result[i] = sum;
    }
    return result;
}

// Type-preserving operators: ColVec op ColVec → ColVec
template<size_t N, typename T>
[[nodiscard]] constexpr ColVec<N, T> operator+(const ColVec<N, T>& a, const ColVec<N, T>& b) {
    ColVec<N, T> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

template<size_t N, typename T>
[[nodiscard]] constexpr ColVec<N, T> operator-(const ColVec<N, T>& a, const ColVec<N, T>& b) {
    ColVec<N, T> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

template<size_t N, typename T>
[[nodiscard]] constexpr ColVec<N, T> operator-(const ColVec<N, T>& a) {
    ColVec<N, T> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = -a[i];
    }
    return result;
}

template<size_t N, typename T, typename Scalar>
    requires(std::is_arithmetic_v<Scalar> || is_matrix_element_v<Scalar>)
[[nodiscard]] constexpr ColVec<N, T> operator*(const ColVec<N, T>& a, Scalar s) {
    ColVec<N, T> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i] * static_cast<T>(s);
    }
    return result;
}

template<typename Scalar, size_t N, typename T>
    requires(std::is_arithmetic_v<Scalar> || is_matrix_element_v<Scalar>)
[[nodiscard]] constexpr ColVec<N, T> operator*(Scalar s, const ColVec<N, T>& a) {
    return a * s;
}

template<size_t N, typename T, typename Scalar>
    requires(std::is_arithmetic_v<Scalar> || is_matrix_element_v<Scalar>)
[[nodiscard]] constexpr ColVec<N, T> operator/(const ColVec<N, T>& a, Scalar s) {
    ColVec<N, T> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i] / static_cast<T>(s);
    }
    return result;
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