
#include "wet_trig.hpp"

// Force emission of wet:: constexpr functions for instruction-count analysis.
// constexpr implies inline linkage, which means they'd be optimized away with
// no extern caller.  These wrappers live in a separate namespace to avoid
// colliding with wet_trig.hpp's `namespace wet`.
namespace wet_emit {

float sin(float x) { return wet::sin(x); }
float cos(float x) { return wet::cos(x); }
float asin(float x) { return wet::asin(x); }
float acos(float x) { return wet::acos(x); }
float atan(float x) { return wet::atan(x); }

float atan2(float y, float x) { return wet::atan2(y, x); }

wet::SinCosResult sincos(float x) { return wet::sincos(x); }

} // namespace wet_emit
