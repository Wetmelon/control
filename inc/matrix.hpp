#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <type_traits>

#include "constexpr_complex.hpp"
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

template<size_t Rows, size_t Cols, typename T>
struct Matrix;

template<size_t N, typename T>
struct ColVec;

template<size_t N, typename T>
struct RowVec;

template<size_t Rows, size_t Cols, size_t ParentCols, typename T>
struct Block;

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
    typedef T value_type;

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
     * @brief Returns a view over a sub-block of the matrix
     *
     * @param start_row   Starting row index of the block
     * @param start_col   Starting column index of the block
     * @param block_rows  Number of rows in the block
     * @param block_cols  Number of columns in the block
     *
     * @return a Matrix<block_rows, block_cols, T> view of the specified block
     */
    template<size_t Brows, size_t Bcols>
    constexpr Block<Brows, Bcols, Cols, T> block(size_t start_row, size_t start_col) {
        if constexpr (Brows == 0 || Bcols == 0) {
            return Block<Brows, Bcols, Cols, T>{nullptr};
        } else {
            return Block<Brows, Bcols, Cols, T>{&data_[start_row][start_col]};
        }
    }

    template<size_t Brows, size_t Bcols>
    constexpr Block<Brows, Bcols, Cols, const T> block(size_t start_row, size_t start_col) const {
        if constexpr (Brows == 0 || Bcols == 0) {
            return Block<Brows, Bcols, Cols, const T>{nullptr};
        } else {
            return Block<Brows, Bcols, Cols, const T>{&data_[start_row][start_col]};
        }
    }

    /**
     * @brief Constructor from a compatible Block (copies data)
     */
    template<size_t ParentCols, typename U>
    constexpr Matrix(const Block<Rows, Cols, ParentCols, U>& block) : Matrix() {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                data_[r][c] = static_cast<T>(block(r, c));
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

    /**
     * @brief  Create an identity matrix (square matrix with ones on the diagonal)
     */
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
     * @brief Create a diagonal matrix from an array of diagonal elements
     *
     * @param diag Array containing the diagonal elements
     * @return Diagonal matrix with the specified diagonal elements
     */
    [[nodiscard]] static constexpr Matrix diagonal(const std::array<T, Rows>& diag)
        requires(Rows == Cols)
    {
        Matrix result{};
        for (size_t r = 0; r < Rows; ++r) {
            result.data_[r][r] = diag[r];
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
     * @brief Compound scalar addition operator
     */
    constexpr Matrix& operator+=(T scalar) {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                data_[r][c] += scalar;
            }
        }
        return *this;
    }

    /**
     * @brief Compound scalar subtraction operator
     */
    constexpr Matrix& operator-=(T scalar) {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                data_[r][c] -= scalar;
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
     * @brief Binary addition operator with broadcasting
     */
    template<size_t OtherRows, size_t OtherCols>
    [[nodiscard]] constexpr Matrix<std::max(Rows, OtherRows), std::max(Cols, OtherCols), T> operator+(const Matrix<OtherRows, OtherCols, T>& other) const {
        constexpr size_t            ResRows = std::max(Rows, OtherRows);
        constexpr size_t            ResCols = std::max(Cols, OtherCols);
        auto                        bcast_this = broadcast<ResRows, ResCols>(*this);
        auto                        bcast_other = broadcast<ResRows, ResCols>(other);
        Matrix<ResRows, ResCols, T> result;
        for (size_t r = 0; r < ResRows; ++r) {
            for (size_t c = 0; c < ResCols; ++c) {
                result(r, c) = bcast_this(r, c) + bcast_other(r, c);
            }
        }
        return result;
    }

    /**
     * @brief Binary subtraction operator with broadcasting
     */
    template<size_t OtherRows, size_t OtherCols>
    [[nodiscard]] constexpr Matrix<std::max(Rows, OtherRows), std::max(Cols, OtherCols), T> operator-(const Matrix<OtherRows, OtherCols, T>& other) const {
        constexpr size_t            ResRows = std::max(Rows, OtherRows);
        constexpr size_t            ResCols = std::max(Cols, OtherCols);
        auto                        bcast_this = broadcast<ResRows, ResCols>(*this);
        auto                        bcast_other = broadcast<ResRows, ResCols>(other);
        Matrix<ResRows, ResCols, T> result;
        for (size_t r = 0; r < ResRows; ++r) {
            for (size_t c = 0; c < ResCols; ++c) {
                result(r, c) = bcast_this(r, c) - bcast_other(r, c);
            }
        }
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

    /**
     * @brief Matrix multiplication with a Block (right-hand side)
     * @tparam P Number of columns in the right-hand block
     * @tparam ParentCols Number of columns in the block's parent matrix
     */
    template<size_t P, size_t ParentCols, typename U>
    [[nodiscard]] constexpr Matrix<Rows, P, T> operator*(const Block<Cols, P, ParentCols, U>& rhs) const {
        Matrix<Rows, P, T> result = Matrix<Rows, P, T>::zeros();
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < P; ++c) {
                T accum = T{0};
                for (size_t k = 0; k < Cols; ++k) {
                    accum += data_[r][k] * static_cast<T>(rhs(k, c));
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
     * @brief Reshape the matrix to new dimensions (total elements must match)
     * @tparam NewRows New number of rows
     * @tparam NewCols New number of columns
     * @return Reshaped matrix
     */
    template<size_t NewRows, size_t NewCols>
    [[nodiscard]] constexpr Matrix<NewRows, NewCols, T> reshape() const
        requires(NewRows* NewCols == Rows * Cols)
    {
        Matrix<NewRows, NewCols, T> result;
        for (size_t i = 0; i < Rows * Cols; ++i) {
            size_t r_new = i / NewCols;
            size_t c_new = i % NewCols;
            size_t r_old = i / Cols;
            size_t c_old = i % Cols;
            result(r_new, c_new) = (*this)(r_old, c_old);
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
     * @brief Dot product (Frobenius inner product) with another matrix
     *
     * Computes the sum of element-wise products between this matrix and another matrix
     * of the same dimensions.
     *
     * @param other Matrix to compute dot product with
     * @return Scalar dot product
     */
    [[nodiscard]] constexpr T dot(const Matrix& other) const {
        T result = T{0};
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                result += data_[r][c] * other.data_[r][c];
            }
        }
        return result;
    }

    /**
     * @brief Sum of all elements in the matrix
     * @return Sum of all elements
     */
    [[nodiscard]] constexpr T sum() const {
        T result = T{0};
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                result += data_[r][c];
            }
        }
        return result;
    }

    /**
     * @brief Mean (average) of all elements in the matrix
     * @return Mean of all elements
     */
    [[nodiscard]] constexpr T mean() const {
        return sum() / static_cast<T>(Rows * Cols);
    }

    /**
     * @brief Matrix inverse using Gauss-Jordan elimination
     *
     * Uses partial pivoting for numerical stability. Specialized implementations
     * for 1×1 and 2×2 matrices; general implementation supports arbitrary sizes.
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
            // Gauss-Jordan elimination with partial pivoting for general NxN matrices
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

    /**
     * @brief Extract the first NewRows rows as a thin view (Block)
     * @tparam NewRows Number of rows to extract (must be <= Rows)
     * @return Block view with the first NewRows rows
     */
    template<size_t NewRows>
    [[nodiscard]] constexpr Block<NewRows, Cols, Cols, T> head() {
        static_assert(NewRows <= Rows, "head<NewRows> called with NewRows > Rows");
        return Block<NewRows, Cols, Cols, T>(&data_[0][0]);
    }

    /**
     * @brief Extract the first NewRows rows as a thin const view (Block)
     * @tparam NewRows Number of rows to extract (must be <= Rows)
     * @return Const Block view with the first NewRows rows
     */
    template<size_t NewRows>
    [[nodiscard]] constexpr Block<NewRows, Cols, Cols, const T> head() const {
        static_assert(NewRows <= Rows, "head<NewRows> called with NewRows > Rows");
        return Block<NewRows, Cols, Cols, const T>(&data_[0][0]);
    }

    /**
     * @brief Extract the last NewRows rows as a thin view (Block)
     * @tparam NewRows Number of rows to extract (must be <= Rows)
     * @return Block view with the last NewRows rows
     */
    template<size_t NewRows>
    [[nodiscard]] constexpr Block<NewRows, Cols, Cols, T> tail() {
        static_assert(NewRows <= Rows, "tail<NewRows> called with NewRows > Rows");
        return Block<NewRows, Cols, Cols, T>(&data_[Rows - NewRows][0]);
    }

    /**
     * @brief Extract the last NewRows rows as a thin const view (Block)
     * @tparam NewRows Number of rows to extract (must be <= Rows)
     * @return Const Block view with the last NewRows rows
     */
    template<size_t NewRows>
    [[nodiscard]] constexpr Block<NewRows, Cols, Cols, const T> tail() const {
        static_assert(NewRows <= Rows, "tail<NewRows> called with NewRows > Rows");
        return Block<NewRows, Cols, Cols, const T>(&data_[Rows - NewRows][0]);
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

// Broadcasting helper for compatible shapes (e.g., 1xN to MxN)
template<size_t TargetRows, size_t TargetCols, size_t SrcRows, size_t SrcCols, typename T>
[[nodiscard]] constexpr Matrix<TargetRows, TargetCols, T> broadcast(const Matrix<SrcRows, SrcCols, T>& src) {
    static_assert((SrcRows == 1 || SrcRows == TargetRows) && (SrcCols == 1 || SrcCols == TargetCols), "Broadcasting requires compatible dimensions (dimensions must be equal or 1)");
    Matrix<TargetRows, TargetCols, T> result;
    for (size_t r = 0; r < TargetRows; ++r) {
        for (size_t c = 0; c < TargetCols; ++c) {
            size_t r_src = (SrcRows == 1) ? 0 : r;
            size_t c_src = (SrcCols == 1) ? 0 : c;
            result(r, c) = src(r_src, c_src);
        }
    }
    return result;
}

// Scalar multiplication (scalar * matrix)
template<typename T, size_t N, size_t M, typename Scalar>
    requires(std::is_arithmetic_v<Scalar> || is_matrix_element_v<Scalar>)
[[nodiscard]] constexpr Matrix<N, M, T> operator*(Scalar scalar, const Matrix<N, M, T>& mat) {
    Matrix<N, M, T> result = mat;
    result *= scalar;
    return result;
}

// Scalar multiplication (matrix * scalar)
template<typename T, size_t N, size_t M, typename Scalar>
    requires(std::is_arithmetic_v<Scalar> || is_matrix_element_v<Scalar>)
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

// Scalar addition (scalar + matrix)
template<typename T, size_t N, size_t M, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
[[nodiscard]] constexpr Matrix<N, M, T> operator+(Scalar scalar, const Matrix<N, M, T>& mat) {
    Matrix<N, M, T> result = mat;
    result += scalar;
    return result;
}

// Scalar addition (matrix + scalar)
template<typename T, size_t N, size_t M, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
[[nodiscard]] constexpr Matrix<N, M, T> operator+(const Matrix<N, M, T>& mat, Scalar scalar) {
    return scalar + mat;
}

// Scalar subtraction (matrix - scalar)
template<typename T, size_t N, size_t M, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
[[nodiscard]] constexpr Matrix<N, M, T> operator-(const Matrix<N, M, T>& mat, Scalar scalar) {
    Matrix<N, M, T> result = mat;
    result -= scalar;
    return result;
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

}; // namespace wetmelon::control

#include "block.hpp"            // IWYU pragma: keep
#include "colvec.hpp"           // IWYU pragma: keep
#include "matrix_functions.hpp" // IWYU pragma: keep
#include "rowvec.hpp"           // IWYU pragma: keep
