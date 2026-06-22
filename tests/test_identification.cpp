#include "wet/estimation/identification.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet::estimation;

// The only logic in identification.hpp is the IdentifiedModelLike concept; the
// rest are POD model/handoff structs. Pin the concept's accept/reject behavior.
static_assert(IdentifiedModelLike<FOPDTModel<double>>);
static_assert(IdentifiedModelLike<SOPDTModel<double>>);
static_assert(IdentifiedModelLike<ARXModel<2, 2, double>>);
static_assert(IdentifiedModelLike<GreyBoxIdentificationResult<float>>);

// Missing the floating-point scalar_type / success contract → rejected.
struct NotAModel {};
static_assert(!IdentifiedModelLike<NotAModel>);

struct IntModel {
    using scalar_type = int;
    bool success{};
};
static_assert(!IdentifiedModelLike<IntModel>); // scalar_type must be floating point

TEST_CASE("Identification model structs default to unsuccessful") {
    CHECK_FALSE(FOPDTModel<double>{}.success);
    CHECK_FALSE(SOPDTModel<double>{}.success);
    CHECK(GreyBoxIdentificationResult<double>{}.kind == IdentificationModelKind::unknown);
}
