#include "../inc/transfer_function.hpp"
#include "doctest.h"

using namespace wetmelon::control;

TEST_CASE("TransferFunction to StateSpace conversion") {
    // Test 1/(s+1) transfer function
    TransferFunction<1, 2, double> tf{
        .num = {1.0},
        .den = {1.0, 1.0}
    };

    auto ss = tf.to_state_space();

    // Check dimensions
    CHECK(ss.A.rows() == 1);
    CHECK(ss.A.cols() == 1);
    CHECK(ss.B.rows() == 1);
    CHECK(ss.B.cols() == 1);
    CHECK(ss.C.rows() == 1);
    CHECK(ss.C.cols() == 1);
    CHECK(ss.D.rows() == 1);
    CHECK(ss.D.cols() == 1);

    // Check matrices
    CHECK(ss.A(0, 0) == doctest::Approx(-1.0));
    CHECK(ss.B(0, 0) == doctest::Approx(1.0));
    CHECK(ss.C(0, 0) == doctest::Approx(1.0));
    CHECK(ss.D(0, 0) == doctest::Approx(0.0));
}

TEST_CASE("TransferFunction with numerator degree equal to denominator") {
    // Test (s+1)/(s^2 + s + 1)
    TransferFunction<2, 3, double> tf{
        .num = {1.0, 1.0},
        .den = {1.0, 1.0, 1.0}
    };

    auto ss = tf.to_state_space();

    // Check dimensions
    CHECK(ss.A.rows() == 2);
    CHECK(ss.A.cols() == 2);
    CHECK(ss.B.rows() == 2);
    CHECK(ss.B.cols() == 1);
    CHECK(ss.C.rows() == 1);
    CHECK(ss.C.cols() == 2);
    CHECK(ss.D.rows() == 1);
    CHECK(ss.D.cols() == 1);

    // Check A matrix (companion form)
    CHECK(ss.A(0, 0) == doctest::Approx(0.0));
    CHECK(ss.A(0, 1) == doctest::Approx(1.0));
    CHECK(ss.A(1, 0) == doctest::Approx(-1.0));
    CHECK(ss.A(1, 1) == doctest::Approx(-1.0));

    // Check B matrix
    CHECK(ss.B(0, 0) == doctest::Approx(0.0));
    CHECK(ss.B(1, 0) == doctest::Approx(1.0));

    // Check C matrix
    CHECK(ss.C(0, 0) == doctest::Approx(1.0));
    CHECK(ss.C(0, 1) == doctest::Approx(1.0));

    // Check D matrix
    CHECK(ss.D(0, 0) == doctest::Approx(0.0));
}

TEST_CASE("TransferFunction multiplication") {
    // Test 1/(s+1) * 1/(s+2) = 1/((s+1)(s+2))
    TransferFunction<1, 2, double> tf1{
        .num = {1.0},
        .den = {1.0, 1.0}
    };

    TransferFunction<1, 2, double> tf2{
        .num = {1.0},
        .den = {1.0, 2.0}
    };

    auto result = tf1 * tf2;

    // Result should be TransferFunction<1, 3, double>
    CHECK(result.num.size() == 1);
    CHECK(result.den.size() == 3);

    // Numerator: 1 * 1 = 1
    CHECK(result.num[0] == doctest::Approx(1.0));

    // Denominator: (1,1) * (1,2) = (1, 1+2, 1*2) = (1, 3, 2)
    CHECK(result.den[0] == doctest::Approx(1.0));
    CHECK(result.den[1] == doctest::Approx(3.0));
    CHECK(result.den[2] == doctest::Approx(2.0));
}

TEST_CASE("ZPK to TransferFunction conversion") {
    // Test 1/(s+1) in ZPK form
    ZPK<0, 1, double> zpk{
        .poles = {wet::complex<double>{-1.0, 0.0}},
        .gain = 1.0
    };

    auto tf = zpk.to_transfer_function();

    CHECK(tf.num.size() == 1);
    CHECK(tf.den.size() == 2);

    CHECK(tf.num[0] == doctest::Approx(1.0));
    CHECK(tf.den[0] == doctest::Approx(1.0));
    CHECK(tf.den[1] == doctest::Approx(1.0));
}

TEST_CASE("ZPK with zeros to TransferFunction") {
    // Test (s-1)/(s+1) in ZPK form
    ZPK<1, 1, double> zpk{
        .zeros = {wet::complex<double>{1.0, 0.0}},
        .poles = {wet::complex<double>{-1.0, 0.0}},
        .gain = 1.0
    };

    auto tf = zpk.to_transfer_function();

    CHECK(tf.num.size() == 2);
    CHECK(tf.den.size() == 2);

    // Numerator: gain * (s - zero) = 1 * (s - 1) = s - 1
    CHECK(tf.num[0] == doctest::Approx(1.0));
    CHECK(tf.num[1] == doctest::Approx(-1.0));

    // Denominator: (s - pole) = s + 1
    CHECK(tf.den[0] == doctest::Approx(1.0));
    CHECK(tf.den[1] == doctest::Approx(1.0));
}

TEST_CASE("TransferFunction as() conversion") {
    TransferFunction<1, 2, double> tf_double{
        .num = {1.0},
        .den = {1.0, 1.0}
    };

    auto tf_float = tf_double.as<float>();

    CHECK(tf_float.num[0] == doctest::Approx(1.0f));
    CHECK(tf_float.den[0] == doctest::Approx(1.0f));
    CHECK(tf_float.den[1] == doctest::Approx(1.0f));
}

TEST_CASE("ZPK as() conversion") {
    ZPK<1, 1, double> zpk_double{
        .zeros = {wet::complex<double>{1.0, 0.0}},
        .poles = {wet::complex<double>{-1.0, 0.0}},
        .gain = 1.0
    };

    auto zpk_float = zpk_double.as<float>();

    CHECK(zpk_float.zeros[0].real() == doctest::Approx(1.0f));
    CHECK(zpk_float.poles[0].real() == doctest::Approx(-1.0f));
    CHECK(zpk_float.gain == doctest::Approx(1.0f));
}

TEST_CASE("TransferFunction constexpr compilation") {
    // Test that to_state_space() works at compile time
    constexpr TransferFunction<1, 2, double> tf{
        .num = {1.0},
        .den = {1.0, 1.0}
    };

    constexpr auto ss = tf.to_state_space();

    // Just check that it compiles and runs
    CHECK(ss.A(0, 0) == doctest::Approx(-1.0));
    CHECK(ss.B(0, 0) == doctest::Approx(1.0));
    CHECK(ss.C(0, 0) == doctest::Approx(1.0));
    CHECK(ss.D(0, 0) == doctest::Approx(0.0));
}
