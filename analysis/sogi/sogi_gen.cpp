#include <numbers>

#include "wet/filters/sogi.hpp"

using namespace wet;

namespace {
SOGI my_sogi{};
MSTOGI my_togi{};
};

auto update_sogi(float in) {
    return my_sogi(in, 60.0f, std::numbers::sqrt2_v<float>, 1.0f / 5000.0f);
}

auto update_togi(float in) {
    return my_togi(in, 60.0f, std::numbers::sqrt2_v<float>, 1.0f / 5000.0f);
}
