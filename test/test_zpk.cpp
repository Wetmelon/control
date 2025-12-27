#include "control.hpp"
#include "doctest.h"

using namespace control;

TEST_CASE("ZPK to TF and SS conversions are non-circular and correct") {
    // G(s) = 3 * (s + 1) / (s + 2)
    std::vector<Zero> zeros = {Zero(-1.0)};
    std::vector<Pole> poles = {Pole(-2.0)};
    double            gain  = 3.0;

    ZeroPoleGain zpk_sys(zeros, poles, gain);

    // Direct transfer function from ZPK
    TransferFunction tf_from_zpk = tf(zpk_sys);
    CHECK(tf_from_zpk.num.size() == 2);
    CHECK(tf_from_zpk.den.size() == 2);
    CHECK(doctest::Approx(tf_from_zpk.num[0]) == 3.0);
    CHECK(doctest::Approx(tf_from_zpk.num[1]) == 3.0);
    CHECK(doctest::Approx(tf_from_zpk.den[0]) == 1.0);
    CHECK(doctest::Approx(tf_from_zpk.den[1]) == 2.0);

    // StateSpace from ZPK then back to TF
    StateSpace       ss_from_zpk = ss(zpk_sys);
    TransferFunction tf_from_ss  = tf(ss_from_zpk);

    // Numerators and denominators should match (within tolerance)
    REQUIRE(tf_from_ss.num.size() == tf_from_zpk.num.size());
    REQUIRE(tf_from_ss.den.size() == tf_from_zpk.den.size());

    for (size_t i = 0; i < tf_from_zpk.num.size(); ++i) {
        CHECK(doctest::Approx(tf_from_ss.num[i]) == tf_from_zpk.num[i]);
    }
    for (size_t i = 0; i < tf_from_zpk.den.size(); ++i) {
        CHECK(doctest::Approx(tf_from_ss.den[i]) == tf_from_zpk.den[i]);
    }
}
