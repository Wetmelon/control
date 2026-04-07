
#include "doctest.h"
#include "matrix.hpp"

using namespace wetmelon::control;

TEST_SUITE("ColVec") {
    TEST_CASE("ColVec basic construction and access") {
        ColVec<3, float> vec;
        // Default initialized to 0
        CHECK(vec[0] == 0.0f);
        CHECK(vec[1] == 0.0f);
        CHECK(vec[2] == 0.0f);

        // Set values
        vec[0] = 1.0f;
        vec[1] = 2.0f;
        vec[2] = 3.0f;

        CHECK(vec[0] == 1.0f);
        CHECK(vec[1] == 2.0f);
        CHECK(vec[2] == 3.0f);
    }

    TEST_CASE("ColVec initializer list constructor") {
        ColVec<3> vec = {1, 2, 3};

        CHECK(vec[0] == 1);
        CHECK(vec[1] == 2);
        CHECK(vec[2] == 3);
    }

    TEST_CASE("ColVec copy and assignment") {
        ColVec vec1 = {1.0f, 2.0f, 3.0f};

        ColVec vec2 = vec1;
        CHECK(vec2[0] == 1.0f);
        CHECK(vec2[2] == 3.0f);

        ColVec<3, float> vec3;
        vec3 = vec1;
        CHECK(vec3[1] == 2.0f);
    }

    TEST_CASE("ColVec addition and subtraction") {
        ColVec vec1 = {1, 2, 3};
        ColVec vec2 = {4, 5, 6};

        auto vec_sum = vec1 + vec2;
        CHECK(vec_sum[0] == 5);
        CHECK(vec_sum[1] == 7);
        CHECK(vec_sum[2] == 9);

        auto vec_diff = vec2 - vec1;
        CHECK(vec_diff[0] == 3);
        CHECK(vec_diff[1] == 3);
        CHECK(vec_diff[2] == 3);

        vec1 += vec2;
        CHECK(vec1[0] == 5);

        vec1 -= vec2;
        CHECK(vec1[0] == 1);
    }

    TEST_CASE("ColVec scalar operations") {
        ColVec<3> vec = {1, 2, 3};

        auto vec_scaled = vec * 2;
        CHECK(vec_scaled[0] == 2);
        CHECK(vec_scaled[1] == 4);
        CHECK(vec_scaled[2] == 6);

        vec_scaled = 3 * vec;
        CHECK(vec_scaled[0] == 3);
        CHECK(vec_scaled[1] == 6);
        CHECK(vec_scaled[2] == 9);

        vec_scaled = vec / 2;
        CHECK(vec_scaled[0] == doctest::Approx(0.5));
        CHECK(vec_scaled[1] == doctest::Approx(1.0));
        CHECK(vec_scaled[2] == doctest::Approx(1.5));

        vec *= 2;
        CHECK(vec[0] == 2);

        vec /= 2;
        CHECK(vec[0] == 1);
    }

    TEST_CASE("ColVec equality") {
        ColVec<3> vec1 = {1, 2, 3};
        ColVec<3> vec2 = {1, 2, 3};
        ColVec<3> vec3 = {1, 2, 4};

        CHECK(vec1 == vec2);
        CHECK(vec1 != vec3);
    }

    TEST_CASE("ColVec unary negation") {
        ColVec<3> vec = {1, -2, 3};

        auto vec_neg = -vec;
        CHECK(vec_neg[0] == -1);
        CHECK(vec_neg[1] == 2);
        CHECK(vec_neg[2] == -3);
    }

    TEST_CASE("ColVec dot product") {
        ColVec<3> vec1 = {1, 2, 3};
        ColVec<3> vec2 = {4, 5, 6};

        auto dot_prod = dot(vec1, vec2);
        CHECK(dot_prod == 32); // 1*4 + 2*5 + 3*6
    }

    TEST_CASE("ColVec cross product") {
        ColVec vec1 = {1, 2, 3};
        ColVec vec2 = {4, 5, 6};

        auto vec_cross = vec1.cross(vec2);
        CHECK(vec_cross[0] == -3); // 2*6 - 3*5
        CHECK(vec_cross[1] == 6);  // 3*4 - 1*6
        CHECK(vec_cross[2] == -3); // 1*5 - 2*4
    }

    TEST_CASE("ColVec norm and normalized") {
        ColVec vec = {3.0f, 4.0f, 0.0f};

        auto n = vec.norm();
        CHECK(n == 5.0f); // sqrt(9 + 16 + 0)

        auto vec_norm = vec.normalized();
        CHECK(vec_norm[0] == 0.6f); // 3/5
        CHECK(vec_norm[1] == 0.8f); // 4/5
        CHECK(vec_norm[2] == 0.0f);
    }

    TEST_CASE("ColVec constexpr") {
        constexpr ColVec<3> vec = {1, 2, 3};

        static_assert(vec[0] == 1);
        static_assert(vec[2] == 3);
    }
}

TEST_SUITE("RowVec") {
    TEST_CASE("RowVec basic construction and access") {
        RowVec<3, float> vec;
        CHECK(vec[0] == 0.0f);
        CHECK(vec[1] == 0.0f);
        CHECK(vec[2] == 0.0f);

        vec[0] = 1.0f;
        vec[1] = 2.0f;
        vec[2] = 3.0f;

        CHECK(vec[0] == 1.0f);
        CHECK(vec[1] == 2.0f);
        CHECK(vec[2] == 3.0f);
    }

    TEST_CASE("RowVec initializer list constructor") {
        RowVec<3> vec = {1, 2, 3};

        CHECK(vec[0] == 1);
        CHECK(vec[1] == 2);
        CHECK(vec[2] == 3);
    }

    TEST_CASE("RowVec dot product") {
        RowVec<3> vec1 = {1, 2, 3};
        RowVec<3> vec2 = {4, 5, 6};

        auto dot_prod = dot(vec1, vec2);
        CHECK(dot_prod == 32);
    }

    TEST_CASE("RowVec cross product") {
        RowVec<3> vec1 = {1, 2, 3};
        RowVec<3> vec2 = {4, 5, 6};

        auto vec_cross = vec1.cross(vec2);
        CHECK(vec_cross[0] == -3);
        CHECK(vec_cross[1] == 6);
        CHECK(vec_cross[2] == -3);
    }

    TEST_CASE("RowVec norm and normalized") {
        RowVec vec = {3.0f, 4.0f, 0.0f};

        auto n = vec.norm();
        CHECK(n == 5.0f);

        auto vec_norm = vec.normalized();
        CHECK(vec_norm[0] == 0.6f);
        CHECK(vec_norm[1] == 0.8f);
        CHECK(vec_norm[2] == 0.0f);
    }
}
