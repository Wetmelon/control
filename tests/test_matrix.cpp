#include <cstddef>
#include <random>
#include <vector>

#include "doctest.h"
#include "fmt/core.h"
#include "matlab.hpp"
#include "matrix.hpp"

using namespace wetmelon::control;

TEST_CASE("Matrix basic construction and access") {
    Matrix<2, 3, float> mat;
    // Default initialized to 0
    CHECK(mat(0, 0) == 0.0f);
    CHECK(mat(1, 1) == 0.0f);
    CHECK(mat(1, 2) == 0.0f);

    // Set values
    mat(0, 0) = 1.0f;
    mat(0, 1) = 2.0f;
    mat(0, 2) = 3.0f;
    mat(1, 0) = 4.0f;
    mat(1, 1) = 5.0f;
    mat(1, 2) = 6.0f;

    CHECK(mat(0, 0) == 1.0f);
    CHECK(mat(0, 1) == 2.0f);
    CHECK(mat(0, 2) == 3.0f);
    CHECK(mat(1, 0) == 4.0f);
    CHECK(mat(1, 1) == 5.0f);
    CHECK(mat(1, 2) == 6.0f);
}

TEST_CASE("Matrix initializer list constructor") {
    Matrix<2, 2> mat = {{1, 2}, {3, 4}};

    CHECK(mat(0, 0) == 1);
    CHECK(mat(0, 1) == 2);
    CHECK(mat(1, 0) == 3);
    CHECK(mat(1, 1) == 4);
}

TEST_CASE("Matrix copy and assignment") {
    Matrix<2, 2, float> mat1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};

    Matrix mat2 = mat1;
    CHECK(mat2(0, 0) == 1.0f);
    CHECK(mat2(1, 1) == 4.0f);

    Matrix<2, 2, float> mat3;
    mat3 = mat1;
    CHECK(mat3(0, 1) == 2.0f);
}

TEST_CASE("Matrix type conversion") {
    Matrix<2, 2> mat_int = {{1, 2}, {3, 4}};

    Matrix<2, 2, float> mat_float(mat_int);
    CHECK(mat_float(0, 0) == 1.0f);
    CHECK(mat_float(1, 1) == 4.0f);

    Matrix<2, 2, float> mat_assign;
    mat_assign = mat_int;
    CHECK(mat_assign(0, 1) == 2.0f);
}

TEST_CASE("Matrix addition and subtraction") {
    Matrix<2, 2> mat1 = {{1, 2}, {3, 4}};

    Matrix<2, 2> mat2 = {{5, 6}, {7, 8}};

    auto sum = mat1 + mat2;
    CHECK(sum(0, 0) == 6);
    CHECK(sum(1, 1) == 12);

    auto diff = mat2 - mat1;
    CHECK(diff(0, 0) == 4);
    CHECK(diff(1, 1) == 4);

    mat1 += mat2;
    CHECK(mat1(0, 0) == 6);

    mat1 -= mat2;
    CHECK(mat1(0, 0) == 1);
}

TEST_CASE("Matrix scalar operations") {
    Matrix<2, 2> mat = {{1, 2}, {3, 4}};

    Matrix<2, 2> scaled = mat * 2;
    CHECK(scaled(0, 0) == 2);
    CHECK(scaled(1, 1) == 8);

    scaled = 3 * mat;
    CHECK(scaled(0, 1) == 6);

    scaled = mat / 2;
    CHECK(scaled(0, 0) == doctest::Approx(0.5));
    CHECK(scaled(1, 0) == doctest::Approx(1.5));

    mat *= 2;
    CHECK(mat(0, 0) == 2);

    mat /= 2;
    CHECK(mat(0, 0) == 1);
}

TEST_CASE("Matrix equality") {
    Matrix<2, 2> mat1 = {{1, 2}, {3, 4}};

    Matrix<2, 2> mat2 = {{1, 2}, {3, 4}};

    Matrix<2, 2> mat3 = {{1, 2}, {3, 5}};

    CHECK(mat1 == mat2);
    CHECK(mat1 != mat3);
}

TEST_CASE("Matrix unary negation") {
    Matrix<2, 2> mat = {{1, -2}, {3, 4}};

    auto neg = -mat;
    CHECK(neg(0, 0) == -1);
    CHECK(neg(0, 1) == 2);
    CHECK(neg(1, 0) == -3);
    CHECK(neg(1, 1) == -4);
}

TEST_CASE("Matrix transpose") {
    Matrix<2, 3> mat = {{1, 2, 3}, {4, 5, 6}};

    auto trans = mat.transpose();
    CHECK(trans(0, 0) == 1);
    CHECK(trans(0, 1) == 4);
    CHECK(trans(1, 0) == 2);
    CHECK(trans(1, 1) == 5);
    CHECK(trans(2, 0) == 3);
    CHECK(trans(2, 1) == 6);
}

TEST_CASE("Matrix multiplication") {
    Matrix<2, 2> mat1 = {{1, 2}, {3, 4}};

    Matrix<2, 2> mat2 = {{5, 6}, {7, 8}};

    auto prod = mat1 * mat2;
    CHECK(prod(0, 0) == 19); // 1*5 + 2*7
    CHECK(prod(0, 1) == 22); // 1*6 + 2*8
    CHECK(prod(1, 0) == 43); // 3*5 + 4*7
    CHECK(prod(1, 1) == 50); // 3*6 + 4*8
}

TEST_CASE("Matrix constexpr") {
    constexpr Matrix<2, 2> mat = {{1, 2}, {3, 4}};

    CHECK(mat(0, 0) == 1);
    CHECK(mat(1, 1) == 4);
}

TEST_CASE("Matrix vector operations") {
    // Using Matrix<N, 1, T> as column vectors
    Matrix<3, 1> vec1 = {{1}, {2}, {3}};

    Matrix<3, 1> vec2 = {{4}, {5}, {6}};

    // Vector addition
    auto vec_sum = vec1 + vec2;
    CHECK(vec_sum(0, 0) == 5);
    CHECK(vec_sum(1, 0) == 7);
    CHECK(vec_sum(2, 0) == 9);

    // Scalar multiplication
    auto vec_scaled = vec1 * 2;
    CHECK(vec_scaled(0, 0) == 2);
    CHECK(vec_scaled(1, 0) == 4);
    CHECK(vec_scaled(2, 0) == 6);

    // Vector subtraction
    auto vec_diff = vec2 - vec1;
    CHECK(vec_diff(0, 0) == 3);
    CHECK(vec_diff(1, 0) == 3);
    CHECK(vec_diff(2, 0) == 3);

    // Unary negation
    auto vec_neg = -vec1;
    CHECK(vec_neg(0, 0) == -1);
    CHECK(vec_neg(1, 0) == -2);
    CHECK(vec_neg(2, 0) == -3);

    // Dot product via matrix multiplication (vec1^T * vec2)
    Matrix<1, 3> vec1_T = vec1.transpose();
    auto         dot_product = vec1_T * vec2;
    CHECK(dot_product(0, 0) == 32); // 1*4 + 2*5 + 3*6

    // Test Matrix::dot member function
    auto direct_dot = vec1.dot(vec2);
    CHECK(direct_dot == 32); // 1*4 + 2*5 + 3*6
}

TEST_CASE("Matrix zeros() static method") {
    auto zero_mat = Matrix<3, 2, float>::zeros();
    CHECK(zero_mat(0, 0) == 0.0f);
    CHECK(zero_mat(1, 0) == 0.0f);
    CHECK(zero_mat(2, 0) == 0.0f);
    CHECK(zero_mat(0, 1) == 0.0f);
    CHECK(zero_mat(1, 1) == 0.0f);
    CHECK(zero_mat(2, 1) == 0.0f);

    // Test with different types
    auto zero_int = Matrix<2, 2, double>::zeros();
    CHECK(zero_int(0, 0) == 0);
    CHECK(zero_int(1, 1) == 0);
}

TEST_CASE("Matrix identity() static method") {
    auto identity_mat = Matrix<3, 3, float>::identity();
    CHECK(identity_mat(0, 0) == 1.0f);
    CHECK(identity_mat(1, 1) == 1.0f);
    CHECK(identity_mat(2, 2) == 1.0f);
    CHECK(identity_mat(0, 1) == 0.0f);
    CHECK(identity_mat(0, 2) == 0.0f);
    CHECK(identity_mat(1, 0) == 0.0f);
    CHECK(identity_mat(1, 2) == 0.0f);
    CHECK(identity_mat(2, 0) == 0.0f);
    CHECK(identity_mat(2, 1) == 0.0f);

    // Test with different types
    auto identity_int = Matrix<2, 2, double>::identity();
    CHECK(identity_int(0, 0) == 1);
    CHECK(identity_int(1, 1) == 1);
    CHECK(identity_int(0, 1) == 0);
    CHECK(identity_int(1, 0) == 0);
}

TEST_CASE("Matrix diagonal() static method") {
    auto diag_mat = Matrix<3, 3, float>::diagonal({1.0f, 2.0f, 3.0f});
    CHECK(diag_mat(0, 0) == 1.0f);
    CHECK(diag_mat(1, 1) == 2.0f);
    CHECK(diag_mat(2, 2) == 3.0f);
    CHECK(diag_mat(0, 1) == 0.0f);
    CHECK(diag_mat(0, 2) == 0.0f);
    CHECK(diag_mat(1, 0) == 0.0f);
    CHECK(diag_mat(1, 2) == 0.0f);
    CHECK(diag_mat(2, 0) == 0.0f);
    CHECK(diag_mat(2, 1) == 0.0f);

    // Test with different types
    auto diag_int = Matrix<2, 2, double>::diagonal({5.0, 7.0});
    CHECK(diag_int(0, 0) == 5.0);
    CHECK(diag_int(1, 1) == 7.0);
    CHECK(diag_int(0, 1) == 0.0);
    CHECK(diag_int(1, 0) == 0.0);

    // Test compile-time usage
    static_assert(Matrix<2, 2, double>::diagonal({1.0, 2.0})(0, 0) == 1.0);
    static_assert(Matrix<2, 2, double>::diagonal({1.0, 2.0})(1, 1) == 2.0);
    static_assert(Matrix<2, 2, double>::diagonal({1.0, 2.0})(0, 1) == 0.0);
}

TEST_CASE("Matrix fill, data, dims") {
    Matrix<2, 3, double> mat;
    mat.fill(7);
    CHECK(mat(0, 0) == 7);
    CHECK(mat(1, 2) == 7);

    auto* ptr = mat.data();
    CHECK(ptr[0] == 7);
    CHECK(ptr[5] == 7);

    static_assert(Matrix<2, 3, double>::rows() == 2);
    static_assert(Matrix<2, 3, double>::cols() == 3);
}

TEST_CASE("Matrix inverse() - 1x1 matrix") {
    Matrix<1, 1, float> mat = {{2.0f}};
    auto                inv = mat.inverse();
    REQUIRE(inv.has_value());
    CHECK(inv->operator()(0, 0) == 0.5f);

    // Test identity property: mat * inv = identity
    auto product = mat * inv.value();
    CHECK(product(0, 0) == doctest::Approx(1.0));
}

TEST_CASE("Matrix inverse() - 2x2 matrix") {
    Matrix<2, 2, float> mat = {{4.0f, 2.0f}, {7.0f, 6.0f}};
    auto                inv = mat.inverse();
    REQUIRE(inv.has_value());

    // Expected inverse of [[4, 2], [7, 6]] is [[0.6, -0.2], [-0.7, 0.4]]
    CHECK(inv->operator()(0, 0) == doctest::Approx(0.6));
    CHECK(inv->operator()(0, 1) == doctest::Approx(-0.2));
    CHECK(inv->operator()(1, 0) == doctest::Approx(-0.7));
    CHECK(inv->operator()(1, 1) == doctest::Approx(0.4));

    // Test identity property: mat * inv = identity
    auto product = mat * inv.value();
    CHECK(product(0, 0) == doctest::Approx(1.0));
    CHECK(product(0, 1) == doctest::Approx(0.0));
    CHECK(product(1, 0) == doctest::Approx(0.0));
    CHECK(product(1, 1) == doctest::Approx(1.0));
}

TEST_CASE("Matrix inverse() - singular matrix returns nullopt") {
    Matrix<2, 2, float> singular = {
        {1.0f, 2.0f},
        {2.0f, 4.0f} // determinant = 0
    };
    auto inv = singular.inverse();
    CHECK_FALSE(inv.has_value());
}

TEST_CASE("Matrix-matrix multiplication - various dimensions") {
    // 2x3 * 3x2 = 2x2
    Matrix<2, 3> mat1 = {{1, 2, 3}, {4, 5, 6}};
    Matrix<3, 2> mat2 = {{7, 8}, {9, 10}, {11, 12}};
    auto         result = mat1 * mat2;
    CHECK(result(0, 0) == 58);  // 1*7 + 2*9 + 3*11
    CHECK(result(0, 1) == 64);  // 1*8 + 2*10 + 3*12
    CHECK(result(1, 0) == 139); // 4*7 + 5*9 + 6*11
    CHECK(result(1, 1) == 154); // 4*8 + 5*10 + 6*12

    // 3x1 * 1x3 = 3x3 (outer product)
    Matrix<3, 1> vec1 = {{1}, {2}, {3}};
    Matrix<1, 3> vec2 = {{4, 5, 6}};
    auto         outer = vec1 * vec2;
    CHECK(outer(0, 0) == 4);
    CHECK(outer(0, 1) == 5);
    CHECK(outer(0, 2) == 6);
    CHECK(outer(1, 0) == 8);
    CHECK(outer(1, 1) == 10);
    CHECK(outer(1, 2) == 12);
    CHECK(outer(2, 0) == 12);
    CHECK(outer(2, 1) == 15);
    CHECK(outer(2, 2) == 18);
}

TEST_CASE("is_matrix_type type trait") {
    // Test that Matrix types are detected
    CHECK(is_matrix_type<Matrix<2, 2, double>>::value == true);
    CHECK(is_matrix_type<Matrix<3, 1, float>>::value == true);
    CHECK(is_matrix_type<Matrix<1, 5, double>>::value == true);

    // Test that non-Matrix types are not detected
    CHECK(is_matrix_type<int>::value == false);
    CHECK(is_matrix_type<float>::value == false);
    CHECK(is_matrix_type<std::vector<int>>::value == false);

    // Test with typedefs (ColVec, RowVec)
    using ColVec2f = Matrix<2, 1, float>;
    using RowVec2f = Matrix<1, 2, float>;
    CHECK(is_matrix_type<ColVec2f>::value == true);
    CHECK(is_matrix_type<RowVec2f>::value == true);
}

TEST_CASE("Matrix operations with floating point precision") {
    // Test inverse with floating point precision
    Matrix<2, 2> mat = {{1.0, 0.5}, {0.5, 1.0}};
    auto         inv = mat.inverse();
    CHECK(inv.has_value());

    // Expected inverse of [[1, 0.5], [0.5, 1]] is [[1.333..., -0.666...], [-0.666..., 1.333...]]
    CHECK(inv->operator()(0, 0) == doctest::Approx(1.333333333).epsilon(1e-6));
    CHECK(inv->operator()(0, 1) == doctest::Approx(-0.666666667).epsilon(1e-6));
    CHECK(inv->operator()(1, 0) == doctest::Approx(-0.666666667).epsilon(1e-6));
    CHECK(inv->operator()(1, 1) == doctest::Approx(1.333333333).epsilon(1e-6));

    // Test identity property
    auto product = mat * inv.value();
    CHECK(product(0, 0) == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(product(0, 1) == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(product(1, 0) == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(product(1, 1) == doctest::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("Matrix trace") {
    Matrix<2, 2> mat = {{3, 1}, {2, 5}};
    CHECK(mat.trace() == 8);
}

TEST_CASE("Block Assignment") {
    Matrix<4, 4> mat = Matrix<4, 4>::zeros();
    Matrix<2, 2> block = {{1, 2}, {3, 4}};

    mat.block<2, 2>(0, 0) = block;

    CHECK(mat(0, 0) == 1);
    CHECK(mat(0, 1) == 2);
    CHECK(mat(1, 0) == 3);
    CHECK(mat(1, 1) == 4);
}

TEST_CASE("Block view operations") {
    Matrix<4, 4, double> mat = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

    // Create a 2x2 block starting at (1,1)
    auto block = mat.block<2, 2>(1, 1);

    // Check access
    CHECK(block(0, 0) == 6);
    CHECK(block(0, 1) == 7);
    CHECK(block(1, 0) == 10);
    CHECK(block(1, 1) == 11);

    // Modify through block
    block(0, 0) = 99;
    CHECK(mat(1, 1) == 99);

    // Assign a matrix to block
    Matrix<2, 2, double> sub = {{100, 101}, {102, 103}};
    block = sub;
    CHECK(mat(1, 1) == 100);
    CHECK(mat(1, 2) == 101);
    CHECK(mat(2, 1) == 102);
    CHECK(mat(2, 2) == 103);

    // Test copy construction
    auto block2 = block;
    block2(0, 0) = 200;
    CHECK(mat(1, 1) == 200);

    // Test copy assignment
    Matrix<4, 4, double> mat2 = {{21, 22, 23, 24}, {25, 26, 27, 28}, {29, 30, 31, 32}, {33, 34, 35, 36}};
    auto                 block3 = mat2.block<2, 2>(1, 1);
    block3 = block; // Assign block to block3
    CHECK(mat2(1, 1) == 200);
    CHECK(mat2(1, 2) == 101);
    CHECK(mat2(2, 1) == 102);
    CHECK(mat2(2, 2) == 103);

    // Test const block
    const auto& const_mat = mat;
    auto        const_block = const_mat.block<2, 2>(1, 1);
    CHECK(const_block(0, 0) == 200);
    // const_block(0, 0) = 300; // This should not compile if const-correct

    // Test move operations (basic check)
    auto block4 = std::move(block3);
    CHECK(block4(0, 0) == 200);
}

TEST_SUITE("Matrix Inversion Tests") {
    /**
     * @brief Helper function that generates a random matrix of given size
     *
     * @param rng Random number generator
     *
     * @return Randomly generated matrix (rows x cols) [-10.0, 10.0]
     */
    template<size_t rows, size_t cols>
    auto make_random_matrix(std::mt19937 & rng) {
        Matrix<rows, cols> mat;
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                mat(r, c) = std::uniform_real_distribution<double>(-10.0, 10.0)(rng);
            }
        }
        return mat;
    };

    template<size_t N>
    void test_inverse_for_size() {
        std::mt19937 rng(0x1234); // Fixed seed for reproducibility

        constexpr size_t num_tests = 10;

        for (size_t i = 0; i < num_tests; ++i) {
            // Generate a random matrix
            auto mat = make_random_matrix<N, N>(rng);

            // Take its inverse
            auto inv = mat.inverse();
            REQUIRE(inv.has_value());

            // Verify that mat * inv = identity
            auto ident = mat * inv.value();
            for (size_t r = 0; r < N; ++r) {
                for (size_t c = 0; c < N; ++c) {
                    if (r == c) {
                        CHECK(ident(r, c) == doctest::Approx(1.0).epsilon(1e-10));
                    } else {
                        CHECK(ident(r, c) == doctest::Approx(0.0).epsilon(1e-10));
                    }
                }
            }
        }
    }

    template<size_t... I>
    void test_all_sizes(std::index_sequence<I...>) {
        // I = 0..8  →  N = 1..9
        (test_inverse_for_size<I + 1>(), ...);
    }

    TEST_CASE("Matrix Inversion for sizes 1 to 9") {
        test_all_sizes(std::make_index_sequence<9>{});
    }
}

TEST_CASE("Matrix sum and mean") {
    Matrix<2, 3> mat = {{1, 2, 3}, {4, 5, 6}};

    CHECK(mat.sum() == 21);
    CHECK(mat.mean() == doctest::Approx(3.5));

    Matrix<1, 1> scalar_mat = {{5.0}};
    CHECK(scalar_mat.sum() == 5.0);
    CHECK(scalar_mat.mean() == 5.0);
}

TEST_CASE("Block scalar assignment") {
    Matrix<3, 3> mat = Matrix<3, 3>::zeros();

    auto block = mat.block<2, 2>(0, 0);
    block = 7.0;

    CHECK(mat(0, 0) == 7);
    CHECK(mat(0, 1) == 7);
    CHECK(mat(1, 0) == 7);
    CHECK(mat(1, 1) == 7);
    CHECK(mat(2, 2) == 0); // Unchanged
}
TEST_CASE("Broadcasting addition") {
    Matrix<1, 3> row = {{1, 2, 3}};
    Matrix<2, 3> mat = {{4, 5, 6}, {7, 8, 9}};

    auto result = mat + row; // Should broadcast row to both rows of mat

    CHECK(result.rows() == 2);
    CHECK(result.cols() == 3);
    CHECK(result(0, 0) == 5);  // 4 + 1
    CHECK(result(0, 1) == 7);  // 5 + 2
    CHECK(result(0, 2) == 9);  // 6 + 3
    CHECK(result(1, 0) == 8);  // 7 + 1
    CHECK(result(1, 1) == 10); // 8 + 2
    CHECK(result(1, 2) == 12); // 9 + 3
}

TEST_CASE("Matrix reshape") {
    Matrix<2, 3> mat = {{1, 2, 3}, {4, 5, 6}};

    auto reshaped = mat.reshape<3, 2>();

    CHECK(reshaped.rows() == 3);
    CHECK(reshaped.cols() == 2);
    CHECK(reshaped(0, 0) == 1);
    CHECK(reshaped(0, 1) == 2);
    CHECK(reshaped(1, 0) == 3);
    CHECK(reshaped(1, 1) == 4);
    CHECK(reshaped(2, 0) == 5);
    CHECK(reshaped(2, 1) == 6);
}

TEST_CASE("Block span-like methods") {
    Matrix<3, 3> mat = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    auto block = mat.block<2, 2>(1, 1);

    CHECK(block.size() == 4);
    CHECK(block.rows() == 2);
    CHECK(block.cols() == 2);
    CHECK(block.data() == &mat(1, 1));
}

TEST_CASE("Block arithmetic operators") {
    Matrix<3, 3> mat1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    Matrix<3, 3> mat2 = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};

    auto block1 = mat1.block<2, 2>(0, 0);
    auto block2 = mat2.block<2, 2>(0, 0);

    auto sum = block1 + block2;
    CHECK(sum.rows() == 2);
    CHECK(sum.cols() == 2);
    CHECK(sum(0, 0) == 1 + 9);
    CHECK(sum(0, 1) == 2 + 8);
    CHECK(sum(1, 0) == 4 + 6);
    CHECK(sum(1, 1) == 5 + 5);

    auto diff = block1 - block2;
    CHECK(diff(0, 0) == 1 - 9);
    CHECK(diff(1, 1) == 5 - 5);

    auto prod = block1 * block2; // Matrix multiplication
    CHECK(prod(0, 0) == 21);     // 1*9 + 2*6
    CHECK(prod(1, 1) == 57);     // 4*8 + 5*5

    auto scalar_add = block1 + 10.0;
    CHECK(scalar_add(0, 0) == 1 + 10);
    CHECK(scalar_add(1, 1) == 5 + 10);

    auto scalar_mul = block1 * 2.0;
    CHECK(scalar_mul(0, 0) == 1 * 2);
    CHECK(scalar_mul(1, 1) == 5 * 2);
}

TEST_CASE("Block from block") {
    Matrix<4, 4> mat = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

    auto outer_block = mat.block<3, 3>(0, 0);
    auto inner_block = outer_block.block<2, 2>(1, 1);

    CHECK(inner_block.rows() == 2);
    CHECK(inner_block.cols() == 2);
    CHECK(inner_block(0, 0) == 6);  // mat(1,1)
    CHECK(inner_block(0, 1) == 7);  // mat(1,2)
    CHECK(inner_block(1, 0) == 10); // mat(2,1)
    CHECK(inner_block(1, 1) == 11); // mat(2,2)

    // Modify through inner block
    inner_block = 99.0;
    CHECK(mat(1, 1) == 99);
    CHECK(mat(2, 2) == 99);
}

TEST_CASE("Block compound assignments") {
    Matrix<3, 3> mat1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    Matrix<3, 3> mat2 = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}};

    auto block1 = mat1.block<2, 2>(0, 0);
    auto block2 = mat2.block<2, 2>(0, 0);

    // Test += with another block
    block1 += block2;
    CHECK(mat1(0, 0) == 1 + 1);
    CHECK(mat1(0, 1) == 2 + 1);
    CHECK(mat1(1, 0) == 4 + 2);
    CHECK(mat1(1, 1) == 5 + 2);

    // Test -= with scalar
    block1 -= 1.0;
    CHECK(mat1(0, 0) == 2 - 1);
    CHECK(mat1(1, 1) == 7 - 1);

    // Test *= with scalar
    block1 *= 2.0;
    CHECK(mat1(0, 0) == 1 * 2);
    CHECK(mat1(1, 1) == 6 * 2);

    // Test /= with scalar
    block1 /= 2.0;
    CHECK(mat1(0, 0) == 1);
    CHECK(mat1(1, 1) == 6);

    // Test *= with another block (element-wise)
    block1 *= block2;
    CHECK(mat1(0, 0) == 1 * 1);
    CHECK(mat1(0, 1) == 2 * 1);
    CHECK(mat1(1, 0) == 5 * 2);
    CHECK(mat1(1, 1) == 6 * 2);
}
