#include <cstddef>

#include "doctest.h"
#include "wet/backend.hpp"
#include "wet/controllers/lqr.hpp"
#include "wet/matlab.hpp"
#include "wet/matrix/matrix.hpp"

using namespace wet;

namespace {

// Classic CTMS "Inverted Pendulum: State-Space Methods" plant (continuous-time).
// https://ctms.engin.umich.edu/CTMS/  (Control: State-Space section)
//   M=0.5 m=0.2 b=0.1 I=0.006 g=9.8 l=0.3,  p = I(M+m)+Mml^2 = 0.0132
struct CtmsPendulum {
    Matrix<4, 4> A{
        {0.0, 1.0, 0.0, 0.0},
        {0.0, -0.1818181818, 2.6727272727, 0.0},
        {0.0, 0.0, 0.0, 1.0},
        {0.0, -0.4545454545, 31.1818181818, 0.0},
    };
    ColVec<4> B{0.0, 1.8181818182, 0.0, 4.5454545455};
};

} // namespace

TEST_CASE("continuous_lqr reproduces the CTMS inverted-pendulum gain") {
    const CtmsPendulum sys;

    // CTMS weighting: Q = C'C with the cart-position and pendulum-angle weights
    // bumped to 5000 and 100; R = 1.
    const Matrix<4, 4> Q{
        {5000.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 100.0, 0.0},
        {0.0, 0.0, 0.0, 0.0},
    };
    const Matrix<1, 1> R{{1.0}};

    const auto res = design::continuous_lqr(sys.A, sys.B, Q, R);
    REQUIRE(res.success);

    // MATLAB's K = lqr(A,B,Q,R) for this problem: [-70.7107 -37.8345 105.5298 20.9238].
    CHECK(res.K(0, 0) == doctest::Approx(-70.7107).epsilon(1e-3));
    CHECK(res.K(0, 1) == doctest::Approx(-37.8345).epsilon(1e-3));
    CHECK(res.K(0, 2) == doctest::Approx(105.5298).epsilon(1e-3));
    CHECK(res.K(0, 3) == doctest::Approx(20.9238).epsilon(1e-3));

    // Continuous-time stability: every closed-loop pole has negative real part.
    for (size_t i = 0; i < 4; ++i) {
        CHECK(res.e[i].real() < 0.0);
    }
}

TEST_CASE("matlab::lqr alias returns the same gain") {
    const CtmsPendulum sys;
    const Matrix<4, 4> Q{
        {5000.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 100.0, 0.0},
        {0.0, 0.0, 0.0, 0.0},
    };
    const Matrix<1, 1> R{{1.0}};

    const auto K = matlab::lqr(sys.A, sys.B, Q, R);
    const auto K2 = design::continuous_lqr(sys.A, sys.B, Q, R).K;
    for (size_t j = 0; j < 4; ++j) {
        CHECK(K(0, j) == doctest::Approx(K2(0, j)));
    }
}
