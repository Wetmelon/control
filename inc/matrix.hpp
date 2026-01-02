#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <type_traits>

#include "constexpr_math.hpp"

template<typename T>
struct is_matrix_type : std::false_type {};

// Helper to check if type is arithmetic (floating-point or complex)
template<typename T>
struct is_matrix_element : std::bool_constant<std::is_floating_point_v<T>> {};

template<typename T>
struct is_matrix_element<std::complex<T>> : std::bool_constant<std::is_floating_point_v<T>> {};

template<typename T>
inline constexpr bool is_matrix_element_v = is_matrix_element<T>::value;

template<size_t Rows, size_t Cols, typename T = double>
struct Matrix {
protected:
    std::array<std::array<T, Cols>, Rows> data_{};

    template<size_t, size_t, typename>
    friend struct Matrix;

public:
    static_assert(is_matrix_element_v<T>, "Matrix element type must be floating-point or complex<floating-point>");

    // Default constructor
    constexpr Matrix() = default;

    // Copy constructor and assignment
    constexpr Matrix(const Matrix&) = default;
    constexpr Matrix& operator=(const Matrix&) = default;

    // Move constructor and assignment
    constexpr Matrix(Matrix&&) = default;
    constexpr Matrix& operator=(Matrix&&) = default;

    // Destructor
    constexpr ~Matrix() = default;

    // Conversion operator= for type conversion from another Matrix (copy)
    template<typename U>
    constexpr Matrix& operator=(const Matrix<Rows, Cols, U>& other) {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                data_[r][c] = static_cast<T>(other.data_[r][c]);
            }
        }
        return *this;
    }

    // Conversion operator= for type conversion from another Matrix (rvalue)
    template<typename U>
    constexpr Matrix& operator=(Matrix<Rows, Cols, U>&& other) {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                data_[r][c] = static_cast<T>(other.data_[r][c]);
            }
        }
        return *this;
    }

    // Assignment from braced initializer list
    constexpr Matrix& operator=(std::initializer_list<std::initializer_list<T>> init) {
        size_t r = 0;
        for (const auto& row : init) {
            size_t c = 0;
            for (const auto& val : row) {
                if (r < Rows && c < Cols) {
                    data_[r][c] = val;
                }
                ++c;
            }
            ++r;
        }
        return *this;
    }

    // Conversion constructor for type conversion from another Matrix
    template<typename U>
    constexpr Matrix(const Matrix<Rows, Cols, U>& other) : Matrix() {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                data_[r][c] = static_cast<T>(other.data_[r][c]);
            }
        }
    }

    // Constructor from braced initializer list for matrices
    constexpr Matrix(std::initializer_list<std::initializer_list<T>> init) : Matrix() {
        size_t r = 0;
        for (const auto& row : init) {
            size_t c = 0;
            for (const auto& val : row) {
                if (r < Rows && c < Cols) {
                    data_[r][c] = val;
                }
                ++c;
            }
            ++r;
        }
    }

    // Constructor from std::array<std::array<T, M>, N> - enables proper CTAD
    constexpr Matrix(const std::array<std::array<T, Cols>, Rows>& arr) : Matrix() {
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                data_[i][j] = arr[i][j];
            }
        }
    }

    // Static factory function for zero matrix
    [[nodiscard]] static constexpr Matrix zeros() { return Matrix{}; }

    // Static factory function for identity matrix (only for square matrices)
    [[nodiscard]] static constexpr Matrix identity()
        requires(Rows == Cols)
    {
        Matrix result;
        for (size_t r = 0; r < Rows; ++r) {
            result.data_[r][r] = T{1};
        }
        return result;
    }

    // Bulk fill
    constexpr void fill(const T& value) {
        for (auto& row : data_) {
            row.fill(value);
        }
    }

    // Introspection
    [[nodiscard]] static constexpr size_t rows() { return Rows; }
    [[nodiscard]] static constexpr size_t cols() { return Cols; }

    // Raw contiguous access (row-major)
    [[nodiscard]] constexpr const T* data() const { return &data_[0][0]; }
    [[nodiscard]] constexpr T*       data() { return &data_[0][0]; }

    // Range helpers over the flattened storage
    [[nodiscard]] constexpr const T* begin() const { return data(); }
    [[nodiscard]] constexpr const T* end() const { return data() + (Rows * Cols); }
    [[nodiscard]] constexpr T*       begin() { return data(); }
    [[nodiscard]] constexpr T*       end() { return data() + (Rows * Cols); }

    // Access operator
    constexpr T&       operator()(size_t row, size_t col) { return data_[row][col]; }
    constexpr const T& operator()(size_t row, size_t col) const { return data_[row][col]; }

    // Compound addition
    constexpr Matrix& operator+=(const Matrix& other) {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                data_[r][c] += other.data_[r][c];
            }
        }
        return *this;
    }

    // Compound subtraction
    constexpr Matrix& operator-=(const Matrix& other) {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                data_[r][c] -= other.data_[r][c];
            }
        }
        return *this;
    }

    // Scalar multiplication
    constexpr Matrix& operator*=(T scalar) {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                data_[r][c] *= scalar;
            }
        }
        return *this;
    }

    // Scalar division
    constexpr Matrix& operator/=(T scalar) {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                data_[r][c] /= scalar;
            }
        }
        return *this;
    }

    // Equality
    [[nodiscard]] constexpr bool operator==(const Matrix& other) const {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                if (data_[r][c] != other.data_[r][c]) {
                    return false;
                }
            }
        }
        return true;
    }

    // Inequality
    [[nodiscard]] constexpr bool operator!=(const Matrix& other) const { return !(*this == other); }

    // Unary negation
    [[nodiscard]] constexpr Matrix operator-() const {
        Matrix result;
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                result.data_[r][c] = -data_[r][c];
            }
        }
        return result;
    }

    // Binary addition
    [[nodiscard]] constexpr Matrix operator+(const Matrix& other) const {
        Matrix result = *this;
        result += other;
        return result;
    }

    // Binary subtraction
    [[nodiscard]] constexpr Matrix operator-(const Matrix& other) const {
        Matrix result = *this;
        result -= other;
        return result;
    }

    // Matrix multiplication
    template<size_t P>
    [[nodiscard]] constexpr Matrix<Rows, P, T> operator*(const Matrix<Cols, P, T>& rhs) const {
        Matrix<Rows, P, T> result;
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < P; ++c) {
                T accum = T{0};
                for (size_t k = 0; k < Cols; ++k) {
                    accum += data_[r][k] * rhs.data_[k][c];
                }
                result.data_[r][c] = accum;
            }
        }
        return result;
    }

    [[nodiscard]] constexpr auto eigenvalues() const {
        // Use QR algorithm to compute eigenvalues
        return compute_eigenvalues_qr(*this);
    }

    // Transpose
    [[nodiscard]] constexpr Matrix<Cols, Rows, T> transpose() const {
        Matrix<Cols, Rows, T> result;
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                result.data_[c][r] = data_[r][c];
            }
        }
        return result;
    }

    // Trace (square only)
    [[nodiscard]] constexpr T trace() const
        requires(Rows == Cols)
    {
        T accum = T{0};
        for (size_t i = 0; i < Rows; ++i) {
            accum += data_[i][i];
        }
        return accum;
    }

    // Matrix inverse (Gauss-Jordan with partial pivoting). Supports sizes up to 6x6.
    [[nodiscard]] constexpr std::optional<Matrix> inverse() const
        requires(Rows == Cols)
    {
        if constexpr (Rows == 1) {
            if (data_[0][0] == T{0}) {
                return std::nullopt;
            }
            Matrix result;
            result.data_[0][0] = T{1} / data_[0][0];
            return result;
        } else if constexpr (Rows == 2) {
            T det = (data_[0][0] * data_[1][1]) - (data_[0][1] * data_[1][0]);
            if (det == T{0}) {
                return std::nullopt;
            }

            T      inv_det = T{1} / det;
            Matrix result;
            result.data_[0][0] = data_[1][1] * inv_det;
            result.data_[0][1] = -data_[0][1] * inv_det;
            result.data_[1][0] = -data_[1][0] * inv_det;
            result.data_[1][1] = data_[0][0] * inv_det;
            return result;
        } else {
            static_assert(Rows <= 6, "Matrix inversion only implemented up to 6x6");

            Matrix<Rows, Rows, T> aug{};
            Matrix<Rows, Rows, T> inv = Matrix::identity();
            aug = *this;

            for (size_t i = 0; i < Rows; ++i) {
                // Pivot selection
                size_t pivot = i;
                T      max_abs = std::abs(aug.data_[i][i]);
                for (size_t r = i + 1; r < Rows; ++r) {
                    T val = std::abs(aug.data_[r][i]);
                    if (val > max_abs) {
                        max_abs = val;
                        pivot = r;
                    }
                }
                if (max_abs == T{0}) {
                    return std::nullopt;
                }
                if (pivot != i) {
                    std::swap(aug.data_[i], aug.data_[pivot]);
                    std::swap(inv.data_[i], inv.data_[pivot]);
                }

                T diag = aug.data_[i][i];
                for (size_t c = 0; c < Rows; ++c) {
                    aug.data_[i][c] /= diag;
                    inv.data_[i][c] /= diag;
                }

                for (size_t r = 0; r < Rows; ++r) {
                    if (r == i)
                        continue;
                    T factor = aug.data_[r][i];
                    if (factor == T{0})
                        continue;
                    for (size_t c = 0; c < Rows; ++c) {
                        aug.data_[r][c] -= factor * aug.data_[i][c];
                        inv.data_[r][c] -= factor * inv.data_[i][c];
                    }
                }
            }
            return inv;
        }
    }

    [[nodiscard]] constexpr Matrix<1, Cols, T> row(size_t r_idx) const {
        Matrix<1, Cols, T> result;
        for (size_t c = 0; c < Cols; ++c) {
            result.data_[0][c] = data_[r_idx][c];
        }
        return result;
    }

    [[nodiscard]] constexpr Matrix<Rows, 1, T> col(size_t c_idx) const {
        Matrix<Rows, 1, T> result;
        for (size_t r = 0; r < Rows; ++r) {
            result.data_[r][0] = data_[r][c_idx];
        }
        return result;
    }
};

template<typename T, size_t N, size_t M>
struct is_matrix_type<Matrix<N, M, T>> : std::true_type {};

// Common small-matrix aliases
template<typename T>
using Mat2 = Matrix<2, 2, T>;

template<typename T>
using Mat3 = Matrix<3, 3, T>;

template<typename T>
using Mat4 = Matrix<4, 4, T>;

template<typename T>
using Mat3x4 = Matrix<3, 4, T>;

template<typename T>
using Mat4x3 = Matrix<4, 3, T>;

// Scalar multiplication (scalar * matrix)
template<typename T, size_t N, size_t M, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
[[nodiscard]] constexpr Matrix<N, M, T> operator*(Scalar scalar, const Matrix<N, M, T>& mat) {
    Matrix<N, M, T> result = mat;
    result *= scalar;
    return result;
}

// Scalar multiplication (matrix * scalar)
template<typename T, size_t N, size_t M, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
[[nodiscard]] constexpr Matrix<N, M, T> operator*(const Matrix<N, M, T>& mat, Scalar scalar) {
    return scalar * mat;
}

// Scalar division
template<typename T, size_t N, size_t M>
    requires std::is_arithmetic_v<decltype(std::declval<T>() / std::declval<T>())>
[[nodiscard]] constexpr Matrix<N, M, T> operator/(const Matrix<N, M, T>& mat, auto scalar) {
    Matrix<N, M, T> result = mat;
    result /= scalar;
    return result;
}

template<size_t N, typename T = double>
struct ColVec : public Matrix<N, 1, T> {
    // Default constructor
    constexpr ColVec() = default;
    constexpr ColVec(const ColVec&) = default;
    constexpr ColVec& operator=(const ColVec&) = default;
    constexpr ColVec(ColVec&&) = default;
    constexpr ColVec& operator=(ColVec&&) = default;
    constexpr ~ColVec() = default;

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

    // Element access
    constexpr const T& operator[](size_t idx) const { return this->data_[idx][0]; }
    constexpr T&       operator[](size_t idx) { return this->data_[idx][0]; }

    [[nodiscard]] friend constexpr T dot(const ColVec<N, T>& vec1, const ColVec<N, T>& vec2) {
        T result = 0;
        for (std::size_t i = 0; i < N; ++i) {
            result += vec1.data_[i][0] * vec2.data_[i][0];
        }
        return result;
    }

    // Cross product for 3D vectors
    template<size_t D = N>
    [[nodiscard]] constexpr ColVec<3, T> cross(const ColVec<3, T>& other) const
        requires(D == 3)
    {
        return ColVec<3, T>{
            this->data_[1][0] * other.data_[2][0] - this->data_[2][0] * other.data_[1][0],
            this->data_[2][0] * other.data_[0][0] - this->data_[0][0] * other.data_[2][0],
            this->data_[0][0] * other.data_[1][0] - this->data_[1][0] * other.data_[0][0]
        };
    }

    // Euclidean norm (magnitude)
    [[nodiscard]] constexpr T norm() const { return wet::sqrt(dot(*this, *this)); }

    // Normalized vector
    [[nodiscard]] constexpr ColVec normalized() const {
        T n = norm();
        if (n == 0) {
            return *this; // Avoid division by zero
        }
        return *this * (T(1) / n);
    }
};

template<size_t N, typename T = double>
struct RowVec : public Matrix<1, N, T> {
    // Default constructor
    constexpr RowVec() = default;
    constexpr RowVec(const RowVec&) = default;
    constexpr RowVec& operator=(const RowVec&) = default;
    constexpr RowVec(RowVec&&) = default;
    constexpr RowVec& operator=(RowVec&&) = default;
    constexpr ~RowVec() = default;

    // Constructor from braced initializer list, e.g. RowVec<int, 3> vec = {1, 2, 3};
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

    // Element access
    constexpr const T& operator[](size_t idx) const { return this->data_[0][idx]; }
    constexpr T&       operator[](size_t idx) { return this->data_[0][idx]; }

    [[nodiscard]] friend constexpr T dot(const RowVec<N, T>& vec1, const RowVec<N, T>& vec2) {
        T result = 0;
        for (std::size_t i = 0; i < N; ++i) {
            result += vec1.data_[0][i] * vec2.data_[0][i];
        }
        return result;
    }

    // Cross product for 3D vectors
    template<size_t D = N>
    [[nodiscard]] constexpr RowVec<3, T> cross(const RowVec<3, T>& other) const
        requires(D == 3)
    {
        return RowVec<3, T>{
            this->data_[0][1] * other.data_[0][2] - this->data_[0][2] * other.data_[0][1],
            this->data_[0][2] * other.data_[0][0] - this->data_[0][0] * other.data_[0][2],
            this->data_[0][0] * other.data_[0][1] - this->data_[0][1] * other.data_[0][0]
        };
    }

    // Euclidean norm (magnitude)
    [[nodiscard]] constexpr T norm() const { return wet::sqrt(dot(*this, *this)); }

    // Normalized vector
    [[nodiscard]] constexpr RowVec normalized() const {
        T n = norm();
        if (n == 0) {
            return *this; // Avoid division by zero
        }
        return *this * (T(1) / n);
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

template<typename T, size_t N>
[[nodiscard]] constexpr ColVec<N, T> operator/(const ColVec<N, T>& vec, T scalar) {
    return ColVec<N, T>(static_cast<const Matrix<N, 1, T>&>(vec) / scalar);
}

template<typename T, size_t N>
[[nodiscard]] constexpr ColVec<N, T> operator-(const ColVec<N, T>& vec) {
    Matrix<N, 1, T> base = vec;
    return ColVec<N, T>(-base);
}

// Mixed Matrix/ColVec operators (implicit upcast from ColVec to Matrix)
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, 1, T> operator+(const Matrix<N, 1, T>& lhs, const ColVec<N, T>& rhs) {
    return lhs + static_cast<const Matrix<N, 1, T>&>(rhs);
}

template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, 1, T> operator+(const ColVec<N, T>& lhs, const Matrix<N, 1, T>& rhs) {
    return static_cast<const Matrix<N, 1, T>&>(lhs) + rhs;
}

template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, 1, T> operator-(const Matrix<N, 1, T>& lhs, const ColVec<N, T>& rhs) {
    return lhs - static_cast<const Matrix<N, 1, T>&>(rhs);
}

template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, 1, T> operator-(const ColVec<N, T>& lhs, const Matrix<N, 1, T>& rhs) {
    return static_cast<const Matrix<N, 1, T>&>(lhs) - rhs;
}

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

template<typename T, size_t N>
[[nodiscard]] constexpr RowVec<N, T> operator/(const RowVec<N, T>& vec, T scalar) {
    return RowVec<N, T>(static_cast<const Matrix<1, N, T>&>(vec) / scalar);
}

template<typename T, size_t N>
[[nodiscard]] constexpr RowVec<N, T> operator-(const RowVec<N, T>& vec) {
    Matrix<1, N, T> base = vec;
    return RowVec<N, T>(-base);
}

// Mixed Matrix/RowVec operators (implicit upcast from RowVec to Matrix)
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<1, N, T> operator+(const Matrix<1, N, T>& lhs, const RowVec<N, T>& rhs) {
    return lhs + static_cast<const Matrix<1, N, T>&>(rhs);
}

template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<1, N, T> operator+(const RowVec<N, T>& lhs, const Matrix<1, N, T>& rhs) {
    return static_cast<const Matrix<1, N, T>&>(lhs) + rhs;
}

template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<1, N, T> operator-(const Matrix<1, N, T>& lhs, const RowVec<N, T>& rhs) {
    return lhs - static_cast<const Matrix<1, N, T>&>(rhs);
}

template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<1, N, T> operator-(const RowVec<N, T>& lhs, const Matrix<1, N, T>& rhs) {
    return static_cast<const Matrix<1, N, T>&>(lhs) - rhs;
}

// Template Deduction Guides
// Deduce Matrix<N, M, T> from std::array<std::array<T, M>, N>
// This enables proper CTAD since both N and M are part of the array type
template<typename T, size_t N, size_t M>
Matrix(const std::array<std::array<T, M>, N>&) -> Matrix<N, M, T>;

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

// Common aliases
template<typename T>
using Vec2 = ColVec<2, T>;

template<typename T>
using Vec3 = ColVec<3, T>;

template<typename T>
using Vec4 = ColVec<4, T>;

template<typename T>
using RowVec2 = RowVec<2, T>;

template<typename T>
using RowVec3 = RowVec<3, T>;

template<typename T>
using RowVec4 = RowVec<4, T>;
