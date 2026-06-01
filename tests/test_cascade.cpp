#include "wet/controllers/cascade.hpp"
#include "wet/matrix/matrix.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wetmelon::control;

namespace {

template<typename T = float>
struct StatefulOuter {
    T gain{T{1}};
    T acc{T{0}};

    [[nodiscard]] constexpr T control(T r, T y) {
        acc += (r - y);
        return gain * acc;
    }

    constexpr void reset() {
        acc = T{0};
    }
};

template<typename T = float>
struct StatefulInner {
    T gain{T{1}};
    T state{T{0}};

    [[nodiscard]] constexpr T control(T r, T y) {
        state = (state * T{0.5}) + (r - y);
        return gain * state;
    }

    constexpr void reset() {
        state = T{0};
    }
};

} // namespace

TEST_SUITE("Cascade Design") {
    TEST_CASE("controller concept accepts P PI and PID mode specializations") {
        static_assert(SISOController<PIDController<float, PIDMode::P>, float, float>);
        static_assert(SISOController<PIDController<float, PIDMode::PI>, float, float>);
        static_assert(SISOController<PIDController<float, PIDMode::PID>, float, float>);

        // PI and PID expose the back-calculation hook; P does not (stateless).
        static_assert(!SISOControllerWithBackCalculation<PIDController<float, PIDMode::P>, float, float, float>);
        static_assert(SISOControllerWithBackCalculation<PIDController<float, PIDMode::PI>, float, float, float>);
        static_assert(SISOControllerWithBackCalculation<PIDController<float, PIDMode::PID>, float, float, float>);
    }

    TEST_CASE("P-mode specialization computes proportional output") {
        PIDController<float, PIDMode::P> p{design::pid(
            2.0f,
            0.0f,
            0.0f,
            0.1f,
            -10.0f,
            10.0f,
            -1.0f,
            1.0f
        )};
        CHECK(p.control(1.0f, 0.25f) == doctest::Approx(1.5f));
    }

    TEST_CASE("PI-mode specialization integrates error") {
        const PIDController<float, PIDMode::PI> pi{design::pid(
            1.0f,
            2.0f,
            0.0f,
            0.1f,
            -10.0f,
            10.0f,
            -10.0f,
            10.0f
        )};
        auto                                    runtime = pi;

        const float u0 = runtime.control(1.0f, 0.0f);
        const float u1 = runtime.control(1.0f, 0.0f);
        CHECK(u1 > u0);
    }
}

TEST_SUITE("Cascade Runtime") {
    TEST_CASE("CascadePPI matches hand-written cascade composition on 2-state plant") {
        constexpr float Ts = 0.01f;

        const Matrix<2, 2, float> A{{1.0f, 0.01f}, {0.0f, 0.98f}};
        const Matrix<2, 1, float> B{{0.0f}, {0.05f}};
        const Matrix<1, 2, float> C{{1.0f, 0.0f}};

        const PIDController<float, PIDMode::P>  outer{design::pid(
            1.5f,
            0.0f,
            0.0f,
            Ts,
            -10.0f,
            10.0f,
            -1.0f,
            1.0f
        )};
        const PIDController<float, PIDMode::PI> inner{design::pid(
            0.8f,
            0.5f,
            0.0f,
            Ts,
            -10.0f,
            10.0f,
            -5.0f,
            5.0f
        )};

        PIDController<float, PIDMode::P>  outer_manual = outer;
        PIDController<float, PIDMode::PI> inner_manual = inner;
        CascadePPI<float>                 cascade(outer, inner);

        ColVec<2, float> x{};
        constexpr float  r = 1.0f;

        for (int k = 0; k < 400; ++k) {
            const float y = (C * x)[0];

            const float r_inner_manual = outer_manual.control(r, y);
            const float u_manual = inner_manual.control(r_inner_manual, y);
            const float u_cascade = cascade.control(r, y);

            CHECK(u_cascade == doctest::Approx(u_manual).epsilon(1e-6));

            const ColVec<1, float> u{u_cascade};
            x = ColVec<2, float>(A * x + B * u);
        }
    }

    TEST_CASE("reset propagates to outer and inner state") {
        StatefulOuter<float>                                outer{2.0f, 0.0f};
        StatefulInner<float>                                inner{3.0f, 0.0f};
        Cascade<StatefulOuter<float>, StatefulInner<float>> cascade(outer, inner);

        (void)cascade.control(1.0f, 0.0f);
        (void)cascade.control(1.0f, 0.2f);

        CHECK(cascade.outer().acc != doctest::Approx(0.0f));
        CHECK(cascade.inner().state != doctest::Approx(0.0f));

        cascade.reset();

        CHECK(cascade.outer().acc == doctest::Approx(0.0f));
        CHECK(cascade.inner().state == doctest::Approx(0.0f));
    }

    TEST_CASE("cascade accepts separate measurements for outer and inner loops") {
        constexpr float Ts = 0.01f;

        PIDController<float, PIDMode::P>  outer{design::pid(
            2.0f,
            0.0f,
            0.0f,
            Ts,
            -100.0f,
            100.0f,
            -1.0f,
            1.0f
        )};
        PIDController<float, PIDMode::PI> inner{design::pid(
            1.0f,
            0.5f,
            0.0f,
            Ts,
            -100.0f,
            100.0f,
            -10.0f,
            10.0f
        )};

        PIDController<float, PIDMode::P>          outer_manual = outer;
        PIDController<float, PIDMode::PI>         inner_manual = inner;
        Cascade<decltype(outer), decltype(inner)> cascade(outer, inner);

        const float r = 1.0f;
        const float y_outer = 0.25f;
        const float y_inner = -0.1f;

        const float r_inner_manual = outer_manual.control(r, y_outer);
        const float u_manual = inner_manual.control(r_inner_manual, y_inner);
        const float u_cascade = cascade.control(r, y_outer, 0.0f, y_inner);

        CHECK(u_cascade == doctest::Approx(u_manual).epsilon(1e-6));
    }

    TEST_CASE("P-PI pos-vel cascade supports additive velocity feedforward") {
        constexpr float Ts = 0.01f;

        PIDController<float, PIDMode::P>  pos_outer{design::pid(
            2.0f,
            0.0f,
            0.0f,
            Ts,
            -100.0f,
            100.0f,
            -1.0f,
            1.0f
        )};
        PIDController<float, PIDMode::PI> vel_inner{design::pid(
            0.8f,
            0.4f,
            0.0f,
            Ts,
            -100.0f,
            100.0f,
            -20.0f,
            20.0f
        )};

        PIDController<float, PIDMode::P>                  pos_outer_manual = pos_outer;
        PIDController<float, PIDMode::PI>                 vel_inner_manual = vel_inner;
        Cascade<decltype(pos_outer), decltype(vel_inner)> cascade(pos_outer, vel_inner);

        const float pos_ref = 1.2f;
        const float pos_meas = 0.9f;
        const float vel_meas = -0.15f;
        const float vel_ff = 0.25f;

        const float vel_ref_manual = vel_ff + pos_outer_manual.control(pos_ref, pos_meas);
        const float u_manual = vel_inner_manual.control(vel_ref_manual, vel_meas);

        const float u_cascade = cascade.control(pos_ref, pos_meas, vel_ff, vel_meas);
        CHECK(u_cascade == doctest::Approx(u_manual).epsilon(1e-6));
    }

    TEST_CASE("Cascade3 forwards three measurement signals through nested loops") {
        constexpr float Ts = 0.01f;

        using POuter = PIDController<float, PIDMode::P>;
        using PIMid = PIDController<float, PIDMode::PI>;
        using PIDInner = PIDController<float, PIDMode::PID>;
        using MidInner = Cascade<PIMid, PIDInner>;
        using Loop3 = Cascade3<POuter, PIMid, PIDInner>;

        POuter   outer{design::pid(1.2f, 0.0f, 0.0f, Ts, -50.0f, 50.0f, -1.0f, 1.0f)};
        PIMid    middle{design::pid(0.8f, 0.4f, 0.0f, Ts, -50.0f, 50.0f, -10.0f, 10.0f)};
        PIDInner inner{design::pid(0.6f, 0.2f, 0.05f, Ts, -50.0f, 50.0f, -10.0f, 10.0f)};

        POuter   outer_manual = outer;
        PIMid    middle_manual = middle;
        PIDInner inner_manual = inner;

        MidInner mid_inner(middle, inner);
        Loop3    loop(outer, mid_inner);

        const float r = 2.0f;
        const float y_outer = 1.6f;
        const float y_middle = 0.5f;
        const float y_inner = 0.2f;

        const float r_mid = outer_manual.control(r, y_outer);
        const float r_inner = middle_manual.control(r_mid, y_middle);
        const float u_manual = inner_manual.control(r_inner, y_inner);

        const float u_loop = loop.control(r, y_outer, 0.0f, y_middle, 0.0f, y_inner);
        CHECK(u_loop == doctest::Approx(u_manual).epsilon(1e-6));
    }
} // TEST_SUITE

namespace {

/// Stateful non-PID controller used to prove the anti-windup protocol works
/// for user-defined types. Acts as a simple discrete integrator with optional
/// freeze-on-saturation: when downstream rejects the commanded value, the
/// integrator unwinds by the saturation amount on the next tick.
struct StatefulIntegrator {
    float Ki{1.0f};
    float Ts{0.01f};
    float integral{0.0f};
    int   back_calculate_calls{0};

    [[nodiscard]] constexpr float control(float r, float y) {
        integral += (r - y) * Ts;
        return Ki * integral;
    }

    constexpr void reset() {
        integral = 0.0f;
        back_calculate_calls = 0;
    }

    constexpr void back_calculate(float u_unsat, float u_sat) {
        ++back_calculate_calls;
        // Unwind the integrator by the saturation amount so we don't push
        // further into the rail on the next tick.
        if (Ki != 0.0f) {
            integral += (u_sat - u_unsat) / Ki;
        }
    }
};

} // namespace

TEST_SUITE("Cascade Anti-windup") {
    TEST_CASE("PID back_calculate unwinds the integrator by Ts*(u_sat-u_unsat)/Kbc") {
        constexpr float                   Ts = 0.01f;
        constexpr float                   Kbc = 0.5f;
        PIDController<float, PIDMode::PI> pi{design::pid(
            2.0f,  // Kp
            10.0f, // Ki
            0.0f,  // Kd (unused for PI)
            Ts,
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            Kbc
        )};

        // Drive the integrator up.
        for (int i = 0; i < 50; ++i) {
            (void)pi.control(1.0f, 0.0f);
        }
        const float integral_before = pi.integral;
        REQUIRE(integral_before > 0.0f);

        // Tell the PI that "we wanted 5.0 but only got 1.0" -- a 4-unit clip.
        // Expected delta: Ts * (u_sat - u_unsat) / Kbc = 0.01 * (-4) / 0.5 = -0.08.
        pi.back_calculate(5.0f, 1.0f);
        CHECK((pi.integral - integral_before) == doctest::Approx(-0.08f).epsilon(1e-6));
    }

    TEST_CASE("PID back_calculate is a no-op when u_unsat == u_sat") {
        PIDController<float, PIDMode::PID> pid{design::pid(1.0f, 1.0f, 0.0f, 0.01f)};
        pid.integral = 3.0f;
        pid.back_calculate(2.5f, 2.5f);
        CHECK(pid.integral == doctest::Approx(3.0f));
    }

    TEST_CASE("Cascade clamps r_inner and back-calculates a stateful outer") {
        StatefulIntegrator               outer{.Ki = 2.0f, .Ts = 0.01f};
        PIDController<float, PIDMode::P> inner{design::pid(
            1.0f, // Kp
            0.0f, 0.0f,
            0.01f,
            -100.0f, 100.0f
        )};

        Cascade<StatefulIntegrator, PIDController<float, PIDMode::P>> cascade(outer, inner);
        cascade.r_inner_min = -0.5f;
        cascade.r_inner_max = 0.5f;

        // Large error -> integrator winds up -> r_inner_unsat exceeds the clamp
        // -> Cascade should call outer.back_calculate.
        for (int i = 0; i < 100; ++i) {
            (void)cascade.control(10.0f, 0.0f);
        }
        CHECK(cascade.outer().back_calculate_calls > 0);

        // Without back-calculation the integrator would have grown unbounded.
        // With it, the integrator is bounded around the value that produces
        // r_inner == r_inner_max (since the unwind exactly cancels each
        // over-the-rail step).
        const float u_steady = cascade.control(10.0f, 0.0f);
        const float r_inner_implied = u_steady; // P-inner with Kp=1, b=1, r-y=u
        CHECK(r_inner_implied <= cascade.r_inner_max + 1e-4f);
        CHECK(r_inner_implied >= cascade.r_inner_min - 1e-4f);
    }

    TEST_CASE("Cascade does not call back_calculate on a stateless outer") {
        // P controller does not satisfy SISOControllerWithBackCalculation;
        // the cascade must still compile and run, just without the hook.
        PIDController<float, PIDMode::P>  outer{design::pid(10.0f, 0.0f, 0.0f, 0.01f, -100.0f, 100.0f)};
        PIDController<float, PIDMode::PI> inner{design::pid(1.0f, 1.0f, 0.0f, 0.01f)};

        Cascade<PIDController<float, PIDMode::P>, PIDController<float, PIDMode::PI>> cascade(outer, inner);
        cascade.r_inner_min = -1.0f;
        cascade.r_inner_max = 1.0f;

        // Should run cleanly; r_inner gets clamped, but there's no hook on
        // the P-outer to call.
        const float u = cascade.control(5.0f, 0.0f);
        CHECK(std::isfinite(u));
    }

    TEST_CASE("infinity defaults make the clamp a no-op") {
        // Default-constructed cascade should not affect r_inner.
        StatefulOuter<float>                                outer{.gain = 2.0f};
        StatefulInner<float>                                inner{.gain = 0.5f};
        Cascade<StatefulOuter<float>, StatefulInner<float>> cascade(outer, inner);

        const float u = cascade.control(3.0f, 1.0f);
        // outer: acc += 3-1=2 -> 2*2 = 4 ; inner: state += 4-1=3 -> 0.5*3 = 1.5
        CHECK(u == doctest::Approx(1.5f).epsilon(1e-6f));
    }
} // TEST_SUITE
