#include <vector>

#include "doctest.h"
#include "matrix.hpp"

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
    CHECK(product(0, 0) == doctest::Approx(1.0f));
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
