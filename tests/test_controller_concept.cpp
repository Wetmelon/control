#include "wet/controllers/adrc.hpp"
#include "wet/controllers/cascade.hpp"
#include "wet/controllers/controller_concept.hpp"
#include "wet/controllers/lead_lag.hpp"
#include "wet/controllers/pid.hpp"
#include "wet/controllers/pr.hpp"
#include "wet/controllers/smc.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wetmelon::control;

/**
 * @file test_controller_concept.cpp
 * @brief Compile-time audit that every SISO runtime controller in the library
 *        satisfies the SISOController concept, with an additional check that
 *        stateful controllers (PI, PID, PR, ADRC) also satisfy the
 *        SISOControllerWithBackCalculation refinement.
 *
 * These static_asserts exist to catch future regressions: if a controller's
 * runtime signature drifts away from `T control(R r, Y y); void reset();`, the
 * concept fails and the test stops compiling -- well before the controller
 * silently stops working inside `Cascade<Outer, Inner>` or in autotuning
 * harnesses.
 *
 * Out-of-scope controllers (different protocol by design):
 * - SinglePhasePLL: tracker, `step(input, Ts)` returns void
 * - RelayAutotuner: experiment driver, `step(y)` returns a struct
 * - LQR/LQI/LQG/LQGI: state-feedback (consume ColVec<NX>, not (r, y))
 *
 * Placeholders (full implementation lands later; concept compliance is part
 * of the implementation, not retrofit):
 * - STSMCController, DOBController, InputShaper, RepetitiveController,
 *   HarmonicSuppressor
 */

namespace {

// PID family ----------------------------------------------------------------
static_assert(SISOController<PIDController<float, PIDMode::P>, float, float>);
static_assert(SISOController<PIDController<float, PIDMode::PI>, float, float>);
static_assert(SISOController<PIDController<float, PIDMode::PID>, float, float>);
static_assert(SISOController<PIDController<double, PIDMode::PID>, double, double>);

// Stateful PID modes expose back_calculate; stateless P does not.
static_assert(!SISOControllerWithBackCalculation<PIDController<float, PIDMode::P>, float, float, float>);
static_assert(SISOControllerWithBackCalculation<PIDController<float, PIDMode::PI>, float, float, float>);
static_assert(SISOControllerWithBackCalculation<PIDController<float, PIDMode::PID>, float, float, float>);

// All PIDController specializations expose enable/disable/is_enabled for
// bumpless transfer and operator manual mode.
static_assert(SISOControllerWithModeControl<PIDController<float, PIDMode::P>, float>);
static_assert(SISOControllerWithModeControl<PIDController<float, PIDMode::PI>, float>);
static_assert(SISOControllerWithModeControl<PIDController<float, PIDMode::PID>, float>);

// PR family -----------------------------------------------------------------
static_assert(SISOController<PRController<float>, float, float>);
static_assert(SISOController<MultiPRController<3, float>, float, float>);

// The single-resonant PR has back_calculate; the multi-resonant aggregate
// does not (deliberately deferred -- distributing the unwind across N
// resonants at different frequencies is a v2 design question).
static_assert(SISOControllerWithBackCalculation<PRController<float>, float, float, float>);
static_assert(!SISOControllerWithBackCalculation<MultiPRController<3, float>, float, float, float>);

// Lead-lag compensator ------------------------------------------------------
static_assert(SISOController<LeadLagController<float>, float, float>);
// Lead-lag is a structural filter -- no integral / resonant state to wind
// down, so it intentionally does not implement the hook.
static_assert(!SISOControllerWithBackCalculation<LeadLagController<float>, float, float, float>);

// ADRC ----------------------------------------------------------------------
static_assert(SISOController<ADRCController<1, float>, float, float>);
static_assert(SISOController<ADRCController<2, float>, float, float>);
static_assert(SISOControllerWithBackCalculation<ADRCController<1, float>, float, float, float>);
static_assert(SISOControllerWithBackCalculation<ADRCController<2, float>, float, float, float>);

// SMC -----------------------------------------------------------------------
// SMC's `control(r, y, phi=0)` has a trailing default argument, so it
// satisfies the two-arg concept signature without modification.
static_assert(SISOController<SMCController<float>, float, float>);

// Cascade itself --------------------------------------------------------------
// A composed Cascade is again a SISOController, so cascades-of-cascades work
// (Cascade3 etc.).
static_assert(SISOController<CascadePPI<float>, float, float>);

} // anonymous namespace

TEST_CASE("controller_concept: static audit compiled successfully") {
    // All compile-time. This case exists so doctest reports the file as
    // exercised and gives the audit a place to live in the test output.
    CHECK(true);
}
