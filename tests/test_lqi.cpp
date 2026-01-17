#include <fmt/format.h>
#include <fmt/ostream.h>

#include "doctest.h"
#include "lqi.hpp"
#include "state_space.hpp"

using namespace wetmelon::control;

TEST_CASE("LQIResult::as<U>() conversion") {
    constexpr auto lqi_d = design::lqi(
        StateSpace<1, 1, 1, 1, 1>{Matrix<1, 1>{{1.0}}, Matrix<1, 1>{{1.0}}, Matrix<1, 1>{{1.0}}, Matrix<1, 1>::zeros()},
        Matrix<2, 2>{{1.0, 0.0}, {0.0, 1.0}}, Matrix<1, 1>{{1.0}}
    );

    constexpr auto lqi_f = lqi_d.as<float>();

    static_assert(lqi_f.success);
    CHECK(lqi_f.success);
    CHECK(lqi_f.K(0, 0) != 0.0f);
    CHECK(lqi_f.K(0, 1) != 0.0f);
}
