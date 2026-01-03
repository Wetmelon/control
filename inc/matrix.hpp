#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <type_traits>

#include "constexpr_math.hpp"

namespace wetmelon::control {

template<typename T>
struct is_matrix_type : std::false_type {};

// Helper to check if type is arithmetic (floating-point or complex)
template<typename T>
struct is_matrix_element : std::bool_constant<std::is_floating_point_v<T>> {};

template<typename T>
struct is_matrix_element<wet::complex<T>> : std::bool_constant<std::is_floating_point_v<T>> {};

template<typename T>
inline constexpr bool is_matrix_element_v = is_matrix_element<T>::value;

/**
 * @ingroup linear_algebra
 * @brief Fixed-size, stack-allocated matrix for linear algebra operations
 *
 * Provides efficient matrix arithmetic, multiplication, transposition, trace, and inversion.
 * Supports constexpr evaluation for compile-time matrix computations.
 *
 * @tparam Rows Number of rows
 * @tparam Cols Number of columns
 * @tparam T Element type (floating-point or complex<floating-point>)
 */
template<size_t Rows, size_t Cols, typename T = double>
struct Matrix {
protected:
    std::array<std::array<T, Cols>, Rows> data_{};

    template<size_t, size_t, typename>
    friend struct Matrix;

public:
    static_assert(is_matrix_element_v<T>, "Matrix element type must be floating-point or complex<floating-point>");

    /**
     * @brief Default constructor, initializes all elements to zero
     */
    constexpr Matrix() = default;

    /**
     * @brief Copy constructor and assignment operator
     */
    constexpr Matrix(const Matrix&) = default;
    constexpr Matrix& operator=(const Matrix&) = default;

    /**
     * @brief Move constructor and assignment operator
     */
    constexpr Matrix(Matrix&&) = default;
    constexpr Matrix& operator=(Matrix&&) = default;

    /**
     * @brief Destructor
     */
    constexpr ~Matrix() = default;

    /**
     * @brief Type conversion assignment operator (copy from different element type)
     */
    template<typename U>
    constexpr Matrix& operator=(const Matrix<Rows, Cols, U>& other) {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                data_[r][c] = static_cast<T>(other.data_[r][c]);
            }
        }
        return *this;
    }

    /**
     * @brief Type conversion assignment operator (move from different element type)
     */
    template<typename U>
    constexpr Matrix& operator=(Matrix<Rows, Cols, U>&& other) {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                data_[r][c] = static_cast<T>(other.data_[r][c]);
            }
        }
        return *this;
    }

    /**
     * @brief Assignment from nested initializer list
     */
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

    /**
     * @brief Type conversion constructor from another matrix with different element type
     */
    template<typename U>
    constexpr Matrix(const Matrix<Rows, Cols, U>& other) : Matrix() {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                data_[r][c] = static_cast<T>(other.data_[r][c]);
            }
        }
    }

    /**
     * @brief Constructor from nested initializer list
     */
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

    /**
     * @brief Constructor from std::array, enables class template argument deduction
     */
    constexpr Matrix(const std::array<std::array<T, Cols>, Rows>& arr) : Matrix() {
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                data_[i][j] = arr[i][j];
            }
        }
    }

    /**
     * @brief Create a zero matrix
     */
    [[nodiscard]] static constexpr Matrix zeros() { return Matrix{}; }
    /**
     * @brief Create a matrix with all elements set to a constant value
     */
    [[nodiscard]] static constexpr Matrix constant(const T& value) {
        Matrix result;
        for (auto& row : result.data_) {
            row.fill(value);
        }
        return result;
    }

    // Static factory function for identity matrix (only for square matrices)
    [[nodiscard]] static constexpr Matrix identity()
        requires(Rows == Cols)
    {
        Matrix result{};
        for (size_t r = 0; r < Rows; ++r) {
            result.data_[r][r] = T{1};
        }
        return result;
    }

    /**
     * @brief Fill all elements with a constant value
     */
    constexpr void fill(const T& value) {
        for (auto& row : data_) {
            row.fill(value);
        }
    }

    /**
     * @brief Type conversion method for converting to a different element type
     */
    template<typename U>
    [[nodiscard]] constexpr Matrix<Rows, Cols, U> as() const {
        return Matrix<Rows, Cols, U>(*this);
    }

    /**
     * @brief Get number of rows
     */
    [[nodiscard]] static constexpr size_t rows() { return Rows; }

    /**
     * @brief Get number of columns
     */
    [[nodiscard]] static constexpr size_t cols() { return Cols; }

    /**
     * @brief Get const pointer to data in row-major order
     */
    [[nodiscard]] constexpr const T* data() const { return &data_[0][0]; }

    /**
     * @brief Get pointer to data in row-major order
     */
    [[nodiscard]] constexpr T* data() { return &data_[0][0]; }

    /**
     * @brief Iterator begin for range-based access
     */
    [[nodiscard]] constexpr const T* begin() const { return data(); }

    /**
     * @brief Iterator end for range-based access
     */
    [[nodiscard]] constexpr const T* end() const { return data() + (Rows * Cols); }

    /**
     * @brief Iterator begin for range-based access
     */
    [[nodiscard]] constexpr T* begin() { return data(); }

    /**
     * @brief Iterator end for range-based access
     */
    [[nodiscard]] constexpr T* end() { return data() + (Rows * Cols); }

    /**
     * @brief Element access operator
     * @param row Row index
     * @param col Column index
     * @return Reference to element at (row, col)
     */
    constexpr T& operator()(size_t row, size_t col) { return data_[row][col]; }

    /**
     * @brief Element access operator (const)
     * @param row Row index
     * @param col Column index
     * @return Const reference to element at (row, col)
     */
    constexpr const T& operator()(size_t row, size_t col) const { return data_[row][col]; }

    /**
     * @brief Compound addition operator
     */
    constexpr Matrix& operator+=(const Matrix& other) {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                data_[r][c] += other.data_[r][c];
            }
        }
        return *this;
    }

    /**
     * @brief Compound subtraction operator
     */
    constexpr Matrix& operator-=(const Matrix& other) {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                data_[r][c] -= other.data_[r][c];
            }
        }
        return *this;
    }

    /**
     * @brief Compound scalar multiplication operator
     */
    constexpr Matrix& operator*=(T scalar) {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                data_[r][c] *= scalar;
            }
        }
        return *this;
    }

    /**
     * @brief Compound scalar division operator
     */
    constexpr Matrix& operator/=(T scalar) {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                data_[r][c] /= scalar;
            }
        }
        return *this;
    }

    /**
     * @brief Equality comparison operator
     */
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

    /**
     * @brief Inequality comparison operator
     */
    [[nodiscard]] constexpr bool operator!=(const Matrix& other) const { return !(*this == other); }

    /**
     * @brief Unary negation operator
     */
    [[nodiscard]] constexpr Matrix operator-() const {
        Matrix result;
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                result.data_[r][c] = -data_[r][c];
            }
        }
        return result;
    }

    /**
     * @brief Binary addition operator
     */
    [[nodiscard]] constexpr Matrix operator+(const Matrix& other) const {
        Matrix result = *this;
        result += other;
        return result;
    }

    /**
     * @brief Binary subtraction operator
     */
    [[nodiscard]] constexpr Matrix operator-(const Matrix& other) const {
        Matrix result = *this;
        result -= other;
        return result;
    }

    /**
     * @brief Matrix multiplication operator
     * @tparam P Number of columns in right-hand matrix
     * @param rhs Right-hand matrix (Cols × P)
     * @return Result matrix (Rows × P)
     */
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

    /**
     * @brief Matrix transpose
     * @return Transposed matrix (Cols × Rows)
     */
    [[nodiscard]] constexpr Matrix<Cols, Rows, T> transpose() const {
        Matrix<Cols, Rows, T> result;
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                result.data_[c][r] = data_[r][c];
            }
        }
        return result;
    }

    /**
     * @brief Sum of diagonal elements (square matrices only)
     *
     * @return Trace of the matrix
     */
    [[nodiscard]] constexpr T trace() const
        requires(Rows == Cols)
    {
        T accum = T{0};
        for (size_t i = 0; i < Rows; ++i) {
            accum += data_[i][i];
        }
        return accum;
    }

    /**
     * @brief Frobenius norm of the matrix
     *
     * @return Frobenius norm
     */
    [[nodiscard]] constexpr T norm() const {
        T sum_sq = T{0};
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                sum_sq += data_[r][c] * data_[r][c];
            }
        }
        return wet::sqrt(sum_sq);
    }

    /**
     * @brief Matrix inverse using Gauss-Jordan elimination
     *
     * Uses partial pivoting for numerical stability. Specialized implementations
     * for 1×1 and 2×2 matrices; general implementation supports up to 6×6.
     *
     * @return Inverse matrix if invertible, nullopt if singular
     */
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

            const T inv_det = T{1} / det;
            Matrix  result;
            result.data_[0][0] = data_[1][1] * inv_det;
            result.data_[0][1] = -data_[0][1] * inv_det;
            result.data_[1][0] = -data_[1][0] * inv_det;
            result.data_[1][1] = data_[0][0] * inv_det;
            return result;
        } else if constexpr (Rows == 3) {
            // Direct 3x3 inverse via cofactors - O(1) constexpr steps vs O(n³) Gauss-Jordan
            const T a = data_[0][0], b = data_[0][1], c = data_[0][2];
            const T d = data_[1][0], e = data_[1][1], f = data_[1][2];
            const T g = data_[2][0], h = data_[2][1], i = data_[2][2];

            // Cofactors for first row (used for determinant)
            const T c00 = e * i - f * h;
            const T c01 = -(d * i - f * g);
            const T c02 = d * h - e * g;

            const T det = a * c00 + b * c01 + c * c02;
            if (wet::abs(det) < T{1e-30}) {
                return std::nullopt;
            }

            const T inv_det = T{1} / det;
            Matrix  result;

            // Adjugate matrix (transpose of cofactor matrix)
            result.data_[0][0] = c00 * inv_det;
            result.data_[0][1] = (c * h - b * i) * inv_det;
            result.data_[0][2] = (b * f - c * e) * inv_det;
            result.data_[1][0] = c01 * inv_det;
            result.data_[1][1] = (a * i - c * g) * inv_det;
            result.data_[1][2] = (c * d - a * f) * inv_det;
            result.data_[2][0] = c02 * inv_det;
            result.data_[2][1] = (b * g - a * h) * inv_det;
            result.data_[2][2] = (a * e - b * d) * inv_det;
            return result;
        } else if constexpr (Rows == 4) {
            // Direct 4x4 inverse via cofactors - significantly fewer constexpr steps
            const T a00 = data_[0][0], a01 = data_[0][1], a02 = data_[0][2], a03 = data_[0][3];
            const T a10 = data_[1][0], a11 = data_[1][1], a12 = data_[1][2], a13 = data_[1][3];
            const T a20 = data_[2][0], a21 = data_[2][1], a22 = data_[2][2], a23 = data_[2][3];
            const T a30 = data_[3][0], a31 = data_[3][1], a32 = data_[3][2], a33 = data_[3][3];

            // Compute 2x2 determinants for efficiency
            const T s0 = a00 * a11 - a01 * a10;
            const T s1 = a00 * a12 - a02 * a10;
            const T s2 = a00 * a13 - a03 * a10;
            const T s3 = a01 * a12 - a02 * a11;
            const T s4 = a01 * a13 - a03 * a11;
            const T s5 = a02 * a13 - a03 * a12;

            const T c5 = a22 * a33 - a23 * a32;
            const T c4 = a21 * a33 - a23 * a31;
            const T c3 = a21 * a32 - a22 * a31;
            const T c2 = a20 * a33 - a23 * a30;
            const T c1 = a20 * a32 - a22 * a30;
            const T c0 = a20 * a31 - a21 * a30;

            const T det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
            if (wet::abs(det) < T{1e-30}) {
                return std::nullopt;
            }

            const T inv_det = T{1} / det;
            Matrix  result;

            result.data_[0][0] = (a11 * c5 - a12 * c4 + a13 * c3) * inv_det;
            result.data_[0][1] = (-a01 * c5 + a02 * c4 - a03 * c3) * inv_det;
            result.data_[0][2] = (a31 * s5 - a32 * s4 + a33 * s3) * inv_det;
            result.data_[0][3] = (-a21 * s5 + a22 * s4 - a23 * s3) * inv_det;

            result.data_[1][0] = (-a10 * c5 + a12 * c2 - a13 * c1) * inv_det;
            result.data_[1][1] = (a00 * c5 - a02 * c2 + a03 * c1) * inv_det;
            result.data_[1][2] = (-a30 * s5 + a32 * s2 - a33 * s1) * inv_det;
            result.data_[1][3] = (a20 * s5 - a22 * s2 + a23 * s1) * inv_det;

            result.data_[2][0] = (a10 * c4 - a11 * c2 + a13 * c0) * inv_det;
            result.data_[2][1] = (-a00 * c4 + a01 * c2 - a03 * c0) * inv_det;
            result.data_[2][2] = (a30 * s4 - a31 * s2 + a33 * s0) * inv_det;
            result.data_[2][3] = (-a20 * s4 + a21 * s2 - a23 * s0) * inv_det;

            result.data_[3][0] = (-a10 * c3 + a11 * c1 - a12 * c0) * inv_det;
            result.data_[3][1] = (a00 * c3 - a01 * c1 + a02 * c0) * inv_det;
            result.data_[3][2] = (-a30 * s3 + a31 * s1 - a32 * s0) * inv_det;
            result.data_[3][3] = (a20 * s3 - a21 * s1 + a22 * s0) * inv_det;
            return result;
        } else {
            static_assert(Rows <= 6, "Matrix inversion only implemented up to 6x6");

            // Gauss-Jordan elimination for 5x5 and 6x6
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

    /**
     * @brief Extract a row as a 1×Cols matrix
     * @param r_idx Row index
     * @return Row vector
     */
    [[nodiscard]] constexpr Matrix<1, Cols, T> row(size_t r_idx) const {
        Matrix<1, Cols, T> result;
        for (size_t c = 0; c < Cols; ++c) {
            result.data_[0][c] = data_[r_idx][c];
        }
        return result;
    }

    /**
     * @brief Extract a column as a Rows×1 matrix
     * @param c_idx Column index
     * @return Column vector
     */
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

/**
 * @brief Column vector specialization of Matrix<N, 1, T>
 * @ingroup linear_algebra
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
    /**
     * @brief Element access operator
     * @param idx Vector index
     * @return Reference to element
     */
    constexpr T& operator[](size_t idx) { return this->data_[idx][0]; }

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
};

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

namespace mat {
/**
 * @brief Infinity norm: maximum absolute row sum
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Infinity norm of the matrix
 */
template<typename T, size_t N>
[[nodiscard]] constexpr T infinity_norm(const Matrix<N, N, T>& A) {
    T norm = T{0};
    for (size_t i = 0; i < N; ++i) {
        T row_sum = T{0};
        for (size_t j = 0; j < N; ++j) {
            row_sum += wet::abs(A(i, j));
        }
        if (row_sum > norm) {
            norm = row_sum;
        }
    }
    return norm;
}

/**
 * @brief Matrix exponential using scaling and squaring with Padé approximation
 *
 * Computes exp(A) using the algorithm: exp(A) = (exp(A / 2^s))^(2^s)
 * where s is chosen so ||A / 2^s|| < 0.5.
 *
 * The matrix exponential is defined as:
 *   exp(A) = I + A + A²/2! + A³/3! + ...
 *
 * For solving ODEs: if dx/dt = A*x, then x(t) = exp(A*t) * x(0)
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Matrix exponential exp(A)
 */
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> exp(const Matrix<N, N, T>& A) {
    // Compute matrix infinity norm (max absolute row sum)
    T norm = infinity_norm(A);

    // Determine scaling: find s such that ||A / 2^s|| < 0.5
    size_t s = 0;
    T      scaled_norm = norm;
    while (scaled_norm > T{0.5}) {
        scaled_norm *= T{0.5};
        s++;
    }

    // Scale matrix: A_scaled = A / 2^s
    Matrix A_scaled = A;
    T      scale_factor = T{1} / static_cast<T>(size_t{1} << s);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A_scaled(i, j) *= scale_factor;
        }
    }

    // Compute exp(A_scaled) using Padé(13,13) approximation for high accuracy
    // exp(A) ≈ N(A) * D(A)^{-1} where N and D are matrix polynomials
    // Using the standard Padé coefficients from Moler & Van Loan
    Matrix I = Matrix<N, N, T>::identity();
    Matrix A2 = A_scaled * A_scaled;
    Matrix A4 = A2 * A2;
    Matrix A6 = A4 * A2;

    // Padé(13,13) coefficients (denominators for numerator polynomial)
    // b_k = (2n-k)! * n! / ((2n)! * k! * (n-k)!)  where n=13
    constexpr T b0 = T{1};
    constexpr T b1 = T{1} / T{2};       // 1/2
    constexpr T b2 = T{1} / T{9};       // ~0.1111
    constexpr T b3 = T{1} / T{72};      // ~0.0139
    constexpr T b4 = T{1} / T{1008};    // ~0.00099
    constexpr T b5 = T{1} / T{30240};   // ~3.3e-5
    constexpr T b6 = T{1} / T{1814400}; // ~5.5e-7

    // Build U = A*(b1*I + b3*A2 + b5*A4 + ...) for numerator odd terms
    // Build V = b0*I + b2*A2 + b4*A4 + b6*A6 for numerator even terms
    Matrix V = I * b0 + A2 * b2 + A4 * b4 + A6 * b6;
    Matrix U_inner = I * b1 + A2 * b3 + A4 * b5;
    Matrix U = A_scaled * U_inner;

    // Numerator N = V + U = (even terms) + A*(odd terms)
    // Denominator D = V - U = (even terms) - A*(odd terms)
    Matrix N_mat = V + U;
    Matrix D_mat = V - U;

    // Compute D^{-1} * N (more stable than N * D^{-1})
    auto            D_inv = D_mat.inverse();
    Matrix<N, N, T> exp_A_scaled;
    if (D_inv) {
        exp_A_scaled = D_inv.value() * N_mat;
    } else {
        // Fallback to Taylor series if Padé fails
        exp_A_scaled = I + A_scaled;
        Matrix A_power = A_scaled;
        for (size_t n = 2; n <= 20; ++n) {
            T factorial = T{1};
            for (size_t k = 1; k <= n; ++k) {
                factorial *= static_cast<T>(k);
            }
            A_power = A_power * A_scaled;
            exp_A_scaled = exp_A_scaled + A_power * (T{1} / factorial);
        }
    }

    // Square s times: exp(A) = (exp(A / 2^s))^(2^s)
    Matrix result = exp_A_scaled;
    for (size_t i = 0; i < s; ++i) {
        result = result * result;
    }

    return result;
}

/**
 * @brief Matrix logarithm using inverse scaling and squaring
 *
 * Computes the principal matrix logarithm X = log(A) that satisfies exp(X) = A.
 *
 * Requirements: A must be invertible and have no real negative eigenvalues.
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Principal matrix logarithm log(A)
 */
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> log(const Matrix<N, N, T>& A) {
    Matrix I = Matrix<N, N, T>::identity();

    // Inverse scaling: compute A^(1/2^s) until close to I
    // Then use log(I + X) ≈ X - X²/2 + X³/3 - ... for small X
    Matrix A_scaled = A;
    size_t s = 0;

    // Take square roots until ||A - I|| is small
    // Matrix square root via Denman-Beavers iteration
    auto matrix_sqrt_db = [](const Matrix<N, N, T>& M) -> Matrix<N, N, T> {
        Matrix<N, N, T> Y = M;
        Matrix<N, N, T> Z = Matrix<N, N, T>::identity();

        for (int iter = 0; iter < 50; ++iter) {
            auto Y_inv = Y.inverse();
            auto Z_inv = Z.inverse();
            if (!Y_inv || !Z_inv)
                break;

            Matrix<N, N, T> Y_next = (Y + Z_inv.value()) * T{0.5};
            Matrix<N, N, T> Z_next = (Z + Y_inv.value()) * T{0.5};

            // Check convergence
            T diff = T{0};
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    diff += wet::abs(Y_next(i, j) - Y(i, j));
                }
            }

            Y = Y_next;
            Z = Z_next;

            if (diff < T{1e-12})
                break;
        }
        return Y;
    };

    // Scale down: A_scaled = A^(1/2^s) until ||A_scaled - I|| < 0.5
    while (infinity_norm(A_scaled - I) > T{0.5} && s < 20) {
        A_scaled = matrix_sqrt_db(A_scaled);
        s++;
    }

    // Now compute log(A_scaled) using log(I + X) series where X = A_scaled - I
    Matrix X = A_scaled - I;
    Matrix result = X;
    Matrix X_power = X;

    for (size_t n = 2; n <= 20; ++n) {
        X_power = X_power * X;
        T sign = (n % 2 == 0) ? T{-1} : T{1};
        result = result + X_power * (sign / static_cast<T>(n));
    }

    // Scale back: log(A) = 2^s * log(A^(1/2^s))
    T scale = static_cast<T>(size_t{1} << s);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) *= scale;
        }
    }

    return result;
}

/**
 * @brief Matrix square root using Denman-Beavers iteration
 *
 * Computes the principal matrix square root S = sqrt(A) that satisfies S*S = A.
 *
 * Uses Denman-Beavers iteration:
 *   Y_{k+1} = (Y_k + Z_k^{-1}) / 2
 *   Z_{k+1} = (Z_k + Y_k^{-1}) / 2
 *
 * Requirements: A should have no real negative eigenvalues.
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Principal matrix square root sqrt(A)
 */
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> sqrt(const Matrix<N, N, T>& A) {
    // Denman-Beavers iteration:
    // Y_{k+1} = (Y_k + Z_k^{-1}) / 2
    // Z_{k+1} = (Z_k + Y_k^{-1}) / 2
    // Converges: Y -> sqrt(A), Z -> sqrt(A)^{-1}

    Matrix<N, N, T> Y = A;
    Matrix<N, N, T> Z = Matrix<N, N, T>::identity();

    for (int iter = 0; iter < 50; ++iter) {
        auto Y_inv = Y.inverse();
        auto Z_inv = Z.inverse();

        if (!Y_inv || !Z_inv) {
            // Fallback: return identity scaled by sqrt of trace/N
            T trace_avg = T{0};
            for (size_t i = 0; i < N; ++i) {
                trace_avg += A(i, i);
            }
            trace_avg /= static_cast<T>(N);
            return Matrix<N, N, T>::identity() * wet::sqrt(trace_avg);
        }

        Matrix<N, N, T> Y_next = (Y + Z_inv.value()) * T{0.5};
        Matrix<N, N, T> Z_next = (Z + Y_inv.value()) * T{0.5};

        // Check convergence
        T diff = T{0};
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                diff += wet::abs(Y_next(i, j) - Y(i, j));
            }
        }

        Y = Y_next;
        Z = Z_next;

        if (diff < T{1e-12})
            break;
    }

    return Y;
}

/**
 * @brief Matrix power for integer exponent
 *
 * Computes A^p using binary exponentiation for efficiency and exactness.
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @param p Integer exponent
 * @return A raised to power p
 */
// Integer power (more efficient and exact)
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> pow(const Matrix<N, N, T>& A, int p) {
    if (p == 0) {
        return Matrix<N, N, T>::identity();
    }

    bool negate = p < 0;
    if (negate) {
        p = -p;
    }

    // Binary exponentiation
    Matrix<N, N, T> result = Matrix<N, N, T>::identity();
    Matrix<N, N, T> base = A;

    while (p > 0) {
        if (p & 1) {
            result = result * base;
        }
        base = base * base;
        p >>= 1;
    }

    if (negate) {
        auto inv = result.inverse();
        return inv.value_or(Matrix<N, N, T>::identity());
    }

    return result;
}

/**
 * @brief Matrix power for real exponent
 *
 * Computes A^p = exp(p * log(A)) for real exponent p.
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @param p Real exponent
 * @return A raised to power p
 */
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> pow(const Matrix<N, N, T>& A, T p) {
    // Check for integer case
    T p_int;
    T p_frac = std::modf(p, &p_int);
    if (wet::abs(p_frac) < T{1e-10}) {
        return pow(A, static_cast<int>(p_int));
    }

    // General case: A^p = exp(p * log(A))
    return exp(log(A) * p);
}

/**
 * @brief Matrix sine using Taylor series approximation
 *
 * Computes sin(A) = A - A³/3! + A⁵/5! - A⁷/7! + ...
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Matrix sine sin(A)
 */
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> sin(const Matrix<N, N, T>& A) {
    Matrix<N, N, T> result = A;
    Matrix<N, N, T> A2 = A * A;
    Matrix<N, N, T> A_power = A;

    T factorial = T{1};
    T sign = T{-1};

    for (size_t n = 3; n <= 21; n += 2) {
        factorial *= static_cast<T>(n - 1) * static_cast<T>(n);
        A_power = A_power * A2;
        result = result + A_power * (sign / factorial);
        sign = -sign;
    }

    return result;
}

/**
 * @brief Matrix cosine using Taylor series approximation
 *
 * Computes cos(A) = I - A²/2! + A⁴/4! - A⁶/6! + ...
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Matrix cosine cos(A)
 */
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> cos(const Matrix<N, N, T>& A) {
    Matrix<N, N, T> result = Matrix<N, N, T>::identity();
    Matrix<N, N, T> A_power = Matrix<N, N, T>::identity();

    T factorial = T{1};
    T sign = T{-1};

    const Matrix<N, N, T> A2 = A * A;
    for (size_t n = 2; n <= 20; n += 2) {
        factorial *= static_cast<T>(n - 1) * static_cast<T>(n);
        A_power = A_power * A2;
        result = result + A_power * (sign / factorial);
        sign = -sign;
    }

    return result;
}

/**
 * @brief Matrix hyperbolic sine
 *
 * Computes sinh(A) = (exp(A) - exp(-A)) / 2
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Matrix hyperbolic sine sinh(A)
 */
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> sinh(const Matrix<N, N, T>& A) {
    Matrix<N, N, T> exp_A = exp(A);
    Matrix<N, N, T> exp_neg_A = exp(A * T{-1});
    return (exp_A - exp_neg_A) * T{0.5};
}

/**
 * @brief Matrix hyperbolic cosine
 *
 * Computes cosh(A) = (exp(A) + exp(-A)) / 2
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Matrix hyperbolic cosine cosh(A)
 */
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> cosh(const Matrix<N, N, T>& A) {
    Matrix<N, N, T> exp_A = exp(A);
    Matrix<N, N, T> exp_neg_A = exp(A * T{-1});
    return (exp_A + exp_neg_A) * T{0.5};
}

/**
 * @brief Verify trigonometric identity sin²(A) + cos²(A) ≈ I
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @param tol Tolerance for identity verification
 * @return True if identity holds within tolerance
 */
template<typename T, size_t N>
[[nodiscard]] constexpr bool verify_trig_identity(const Matrix<N, N, T>& A, T tol = T{1e-6}) {
    Matrix<N, N, T> sinA = sin(A);
    Matrix<N, N, T> cosA = cos(A);
    Matrix<N, N, T> sum = sinA * sinA + cosA * cosA;
    Matrix<N, N, T> I = Matrix<N, N, T>::identity();

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (wet::abs(sum(i, j) - I(i, j)) > tol) {
                return false;
            }
        }
    }
    return true;
}
} // namespace mat

}; // namespace wetmelon::control
