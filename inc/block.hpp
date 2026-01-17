#pragma once

#include "matrix.hpp"

namespace wetmelon::control {

/**
 * @brief  Block view (non-owning) into a parent matrix
 * @ingroup linear_algebra
 *
 * @tparam Rows        Number of rows in the block
 * @tparam Cols        Number of columns in the block
 * @tparam ParentCols  Number of columns in the parent matrix
 * @tparam T           Element type
 *
 */
template<size_t Rows, size_t Cols, size_t ParentCols, typename T>
struct Block {
private:
    T* const start_ptr;

public:
    // Create a block from raw pointer
    constexpr explicit Block(T* data_ptr) : start_ptr(data_ptr) {}

    // Create a block from a Matrix (non-owning)
    constexpr explicit Block(Matrix<Rows, ParentCols, T>&& mat) : start_ptr(const_cast<T*>(mat.data())) {}
    constexpr explicit Block(const Matrix<Rows, ParentCols, T>& mat) : start_ptr(mat.data()) {}
    constexpr explicit Block(Matrix<Rows, ParentCols, T>& mat) : start_ptr(const_cast<T*>(mat.data())) {}

    constexpr Block(const Block& other) : start_ptr(other.start_ptr) {}
    constexpr Block& operator=(const Block& other) {
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                (*this)(i, j) = other(i, j);
            }
        }
        return *this;
    }

    constexpr Block(Block&& other) noexcept : start_ptr(other.start_ptr) {}
    constexpr Block& operator=(Block&& other) noexcept {
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                (*this)(i, j) = other(i, j);
            }
        }
        return *this;
    }

    [[nodiscard]] constexpr T& operator()(size_t r, size_t c) {
        return start_ptr[r * ParentCols + c];
    }

    [[nodiscard]] constexpr const T& operator()(size_t r, size_t c) const {
        return start_ptr[r * ParentCols + c];
    }

    constexpr Block& operator=(const Matrix<Rows, Cols, T>& mat) {
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                (*this)(i, j) = mat(i, j);
            }
        }
        return *this;
    }

    constexpr Block& operator=(T scalar) {
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                (*this)(i, j) = scalar;
            }
        }
        return *this;
    }

    [[nodiscard]] constexpr T*            data() { return start_ptr; }
    [[nodiscard]] constexpr const T*      data() const { return start_ptr; }
    [[nodiscard]] static constexpr size_t size() { return Rows * Cols; }
    [[nodiscard]] static constexpr size_t rows() { return Rows; }
    [[nodiscard]] static constexpr size_t cols() { return Cols; }

    // Arithmetic operators returning Matrix
    [[nodiscard]] constexpr Matrix<Rows, Cols, T> operator+(const Block& other) const {
        Matrix<Rows, Cols, T> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) + other(i, j);
            }
        }
        return result;
    }

    [[nodiscard]] constexpr Matrix<Rows, Cols, T> operator-(const Block& other) const {
        Matrix<Rows, Cols, T> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) - other(i, j);
            }
        }
        return result;
    }

    // Matrix multiplication
    template<size_t OtherCols>
    [[nodiscard]] constexpr Matrix<Rows, OtherCols, T> operator*(const Block<Cols, OtherCols, ParentCols, T>& other) const {
        Matrix<Rows, OtherCols, T> result = Matrix<Rows, OtherCols, T>::zeros();
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < OtherCols; ++j) {
                for (size_t k = 0; k < Cols; ++k) {
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }
        return result;
    }

    // Multiply Block by a full Matrix on the right
    template<size_t OtherCols, typename U>
    [[nodiscard]] constexpr Matrix<Rows, OtherCols, T> operator*(const Matrix<Cols, OtherCols, U>& other) const {
        Matrix<Rows, OtherCols, T> result = Matrix<Rows, OtherCols, T>::zeros();
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < OtherCols; ++j) {
                for (size_t k = 0; k < Cols; ++k) {
                    result(i, j) += (*this)(i, k) * static_cast<T>(other(k, j));
                }
            }
        }
        return result;
    }

    // Scalar operations
    [[nodiscard]] constexpr Matrix<Rows, Cols, T> operator+(T scalar) const {
        Matrix<Rows, Cols, T> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) + scalar;
            }
        }
        return result;
    }

    [[nodiscard]] constexpr Matrix<Rows, Cols, T> operator*(T scalar) const {
        Matrix<Rows, Cols, T> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) * scalar;
            }
        }
        return result;
    }

    /**
     * @brief In-place addition with compatible Matrix
     */
    template<typename U>
    Block& operator+=(const Matrix<Rows, Cols, U>& other) {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                (*this)(r, c) += other(r, c);
            }
        }
        return *this;
    }

    /**
     * @brief In-place addition with compatible Block
     */
    template<size_t OtherParentCols, typename U>
    Block& operator+=(const Block<Rows, Cols, OtherParentCols, U>& other) {
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                (*this)(r, c) += other(r, c);
            }
        }
        return *this;
    }

    // Compound assignment operators
    constexpr Block& operator+=(const Block& other) {
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                (*this)(i, j) += other(i, j);
            }
        }
        return *this;
    }

    constexpr Block& operator-=(const Block& other) {
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                (*this)(i, j) -= other(i, j);
            }
        }
        return *this;
    }

    constexpr Block& operator*=(const Block& other) {
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                (*this)(i, j) *= other(i, j);
            }
        }
        return *this;
    }

    constexpr Block& operator+=(T scalar) {
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                (*this)(i, j) += scalar;
            }
        }
        return *this;
    }

    constexpr Block& operator-=(T scalar) {
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                (*this)(i, j) -= scalar;
            }
        }
        return *this;
    }

    constexpr Block& operator*=(T scalar) {
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                (*this)(i, j) *= scalar;
            }
        }
        return *this;
    }

    constexpr Block& operator/=(T scalar) {
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                (*this)(i, j) /= scalar;
            }
        }
        return *this;
    }

    // Sub-block access
    template<size_t Brows, size_t Bcols>
    constexpr Block<Brows, Bcols, ParentCols, T> block(size_t start_row, size_t start_col) {
        return Block<Brows, Bcols, ParentCols, T>(start_ptr + start_row * ParentCols + start_col);
    }

    template<size_t Brows, size_t Bcols>
    constexpr Block<Brows, Bcols, ParentCols, const T> block(size_t start_row, size_t start_col) const {
        return Block<Brows, Bcols, ParentCols, const T>(start_ptr + start_row * ParentCols + start_col);
    }
};

// CTAD for Block from Matrix
template<size_t Rows, size_t Cols, typename T>
Block(const Matrix<Rows, Cols, T>&) -> Block<Rows, Cols, Cols, T>;

template<size_t Rows, size_t Cols, typename T>
Block(Matrix<Rows, Cols, T>&) -> Block<Rows, Cols, Cols, T>;

} // namespace wetmelon::control