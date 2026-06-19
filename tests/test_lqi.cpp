#include <fmt/format.h>
#include <fmt/ostream.h>

#include "wet/controllers/lqi.hpp"
#include "wet/systems/state_space.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

TEST_CASE("LQIResult::as<U>() conversion") {
    constexpr auto lqi_d = design::discrete_lqi(
        StateSpace<1, 1, 1, 1, 1>{Matrix<1, 1>{{1.0}}, Matrix<1, 1>{{1.0}}, Matrix<1, 1>{{1.0}}, Matrix<1, 1>::zeros()},
        Matrix<2, 2>{{1.0, 0.0}, {0.0, 1.0}}, Matrix<1, 1>{{1.0}}
    );

    constexpr auto lqi_f = lqi_d.as<float>();

    static_assert(lqi_f.success);
    CHECK(lqi_f.success);
    CHECK(lqi_f.K(0, 0) != 0.0f);
    CHECK(lqi_f.K(0, 1) != 0.0f);
}
