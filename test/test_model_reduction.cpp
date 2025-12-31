#include "control.hpp"
#include "doctest.h"

using namespace control;

TEST_CASE("minreal removes uncontrollable states") {
    // A = diag(0, -1), B = [1;0] (second state uncontrollable), C = [1 0]
    Matrix A = Matrix::Zero(2, 2);
    A(0, 0)  = 0.0;
    A(1, 1)  = -1.0;
    Matrix B = Matrix::Zero(2, 1);
    B(0, 0)  = 1.0;
    Matrix C = Matrix::Zero(1, 2);
    C(0, 0)  = 1.0;
    Matrix D = Matrix::Zero(1, 1);

    StateSpace sys(A, B, C, D);
    StateSpace red = sys.minreal();

    CHECK(red.A.rows() == 1);
    CHECK(red.B.rows() == 1);
    CHECK(red.C.cols() == 1);
    CHECK(red.A(0, 0) == doctest::Approx(0.0));
    CHECK(red.B(0, 0) == doctest::Approx(1.0));
    CHECK(red.C(0, 0) == doctest::Approx(1.0));
}

TEST_CASE("minreal removes unobservable states") {
    // A = diag(0, -1), B = [1;1] (controllable), C = [1 0] (second state unobservable)
    Matrix A = Matrix::Zero(2, 2);
    A(0, 0)  = 0.0;
    A(1, 1)  = -1.0;
    Matrix B = Matrix::Zero(2, 1);
    B(0, 0)  = 1.0;
    B(1, 0)  = 1.0;
    Matrix C = Matrix::Zero(1, 2);
    C(0, 0)  = 1.0;
    Matrix D = Matrix::Zero(1, 1);

    StateSpace sys(A, B, C, D);
    StateSpace red = sys.minreal();

    CHECK(red.A.rows() == 1);
    CHECK(red.B.rows() == 1);
    CHECK(red.C.cols() == 1);
    // A reduced should be approx 0 (dominant retained mode)
    CHECK(red.A(0, 0) == doctest::Approx(0.0));
}

TEST_CASE("minreal no-op on minimal system") {
    // Simple minimal system: A = [ -1 ], B = [1], C = [1]
    Matrix A = Matrix::Constant(1, 1, -1.0);
    Matrix B = Matrix::Constant(1, 1, 1.0);
    Matrix C = Matrix::Constant(1, 1, 1.0);
    Matrix D = Matrix::Zero(1, 1);

    StateSpace sys(A, B, C, D);
    StateSpace red = sys.minreal();

    CHECK(red.A.rows() == 1);
    CHECK(red.A(0, 0) == doctest::Approx(-1.0));
    CHECK(red.B(0, 0) == doctest::Approx(1.0));
    CHECK(red.C(0, 0) == doctest::Approx(1.0));
}
