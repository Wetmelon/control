#include <array>

#include "wet/controllers/excitation.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wetmelon::control;

TEST_SUITE("Excitation Design") {
    TEST_CASE("chirp config validates and converts") {
        constexpr design::ChirpConfig<double> cfg{
            .amplitude = 2.0,
            .f_start_hz = 1.0,
            .f_end_hz = 10.0,
            .duration_s = 3.0,
            .mode = design::ChirpMode::Linear,
        };
        static_assert(cfg.valid());

        constexpr auto chirp = design::synthesize_chirp(cfg);
        static_assert(chirp.success);

        constexpr auto chirp_f = chirp.template as<float>();
        static_assert(chirp_f.success);
        static_assert(chirp_f.config.amplitude == 2.0f);
        static_assert(chirp_f.config.mode == design::ChirpMode::Linear);
    }

    TEST_CASE("chirp log mode rejects non-positive endpoints") {
        constexpr design::ChirpConfig<double> cfg{
            .amplitude = 1.0,
            .f_start_hz = 0.0,
            .f_end_hz = 10.0,
            .duration_s = 1.0,
            .mode = design::ChirpMode::Log,
        };
        static_assert(!cfg.valid());

        constexpr auto chirp = design::synthesize_chirp(cfg);
        static_assert(!chirp.success);
    }

    TEST_CASE("prbs config reports period") {
        constexpr design::PRBSConfig<double> cfg{
            .amplitude = 1.0,
            .lfsr_order = 5,
            .clock_period_s = 0.1,
            .seed = 0x1u,
        };
        static_assert(cfg.valid());

        constexpr auto prbs = design::synthesize_prbs(cfg);
        static_assert(prbs.success);
        static_assert(prbs.period_bits == 31);
    }

    TEST_CASE("step train config validates") {
        constexpr design::StepTrainConfig<double> good{
            .amplitude = 1.0,
            .hold_s = 0.1,
            .cycles = 2,
        };
        static_assert(good.valid());

        constexpr design::StepTrainConfig<double> bad{
            .amplitude = 1.0,
            .hold_s = 0.1,
            .cycles = 0,
        };
        static_assert(!bad.valid());
    }

    TEST_CASE("ramp config validates") {
        constexpr design::RampConfig<double> good{
            .target = -1.0,
            .rate = 2.0,
            .hold_at_end_s = 0.2,
        };
        static_assert(good.valid());

        constexpr design::RampConfig<double> bad{
            .target = 1.0,
            .rate = 0.0,
            .hold_at_end_s = 0.2,
        };
        static_assert(!bad.valid());
    }

    TEST_CASE("multi-sine validates and converts") {
        constexpr design::MultiSineConfig<2, double> cfg{
            .tones = std::array{
                design::Tone<double>{1.0, 1.0, 0.0},
                design::Tone<double>{0.5, 2.0, 0.0},
            },
        };
        static_assert(cfg.valid());

        constexpr auto ms = design::synthesize_multi_sine(cfg);
        static_assert(ms.success);

        constexpr auto ms_f = ms.template as<float>();
        static_assert(ms_f.config.tones[0].amplitude == 1.0f);
        static_assert(ms_f.config.tones[1].freq_hz == 2.0f);
    }
} // TEST_SUITE

TEST_SUITE("Excitation Runtime") {
    TEST_CASE("linear chirp with constant frequency matches sine samples") {
        constexpr design::ChirpConfig<double> cfg{
            .amplitude = 1.0,
            .f_start_hz = 1.0,
            .f_end_hz = 1.0,
            .duration_s = 1.0,
            .mode = design::ChirpMode::Linear,
        };
        constexpr auto design = design::synthesize_chirp(cfg);
        Chirp<double>  chirp(design, 0.1);

        CHECK(chirp.step(0.0) == doctest::Approx(0.0).epsilon(1e-12));
        CHECK(chirp.step(0.25) == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(chirp.done(1.0));

        (void)chirp.step();
        CHECK(chirp.time() == doctest::Approx(0.1).epsilon(1e-12));
        chirp.reset();
        CHECK(chirp.time() == doctest::Approx(0.0).epsilon(1e-12));
    }

    TEST_CASE("step train alternates sign and ends after configured cycles") {
        constexpr design::StepTrainConfig<double> cfg{
            .amplitude = 2.0,
            .hold_s = 0.1,
            .cycles = 2,
        };
        constexpr auto    design = design::synthesize_step_train(cfg);
        StepTrain<double> st(design, 0.1);

        CHECK(st.step(0.05) == doctest::Approx(2.0));
        CHECK(st.step(0.15) == doctest::Approx(-2.0));
        CHECK(st.step(0.25) == doctest::Approx(2.0));
        CHECK(st.step(0.35) == doctest::Approx(-2.0));
        CHECK(st.done(0.4));
        CHECK(st.step(0.41) == doctest::Approx(0.0));
    }

    TEST_CASE("ramp follows slew rate and then holds target") {
        constexpr design::RampConfig<double> cfg{
            .target = -1.0,
            .rate = 2.0,
            .hold_at_end_s = 0.2,
        };
        constexpr auto design = design::synthesize_ramp(cfg);
        Ramp<double>   ramp(design, 0.1);

        CHECK(ramp.ramp_duration() == doctest::Approx(0.5).epsilon(1e-12));
        CHECK(ramp.step(0.25) == doctest::Approx(-0.5).epsilon(1e-12));
        CHECK(ramp.step(0.75) == doctest::Approx(-1.0).epsilon(1e-12));
        CHECK(ramp.done(0.7));
    }

    TEST_CASE("multi-sine sums tones") {
        constexpr design::MultiSineConfig<2, double> cfg{
            .tones = std::array{
                design::Tone<double>{1.0, 1.0, 0.0},
                design::Tone<double>{0.5, 2.0, 0.0},
            },
        };
        constexpr auto       design = design::synthesize_multi_sine(cfg);
        MultiSine<2, double> ms(design, 0.1);

        CHECK(ms.step(0.0) == doctest::Approx(0.0).epsilon(1e-12));
        CHECK(ms.step(0.25) == doctest::Approx(1.0).epsilon(1e-6));
        CHECK_FALSE(ms.done());
    }

    TEST_CASE("prbs is reproducible and supports time-based sampling") {
        constexpr design::PRBSConfig<float> cfg{
            .amplitude = 1.0f,
            .lfsr_order = 5,
            .clock_period_s = 0.1f,
            .seed = 0x1u,
        };
        constexpr auto design = design::synthesize_prbs(cfg);

        PRBS<float> seq_a(design, 0.1f);
        PRBS<float> seq_b(design, 0.1f);

        for (int i = 0; i < 40; ++i) {
            CHECK(seq_a.step() == doctest::Approx(seq_b.step()));
        }

        PRBS<float> seq_t(design, 0.1f);
        for (int i = 0; i < 10; ++i) {
            const float t = static_cast<float>(i) * cfg.clock_period_s;
            CHECK(seq_t.step(t) == doctest::Approx(seq_b.step(t)));
        }

        CHECK(seq_t.done(static_cast<float>(design.period_bits) * cfg.clock_period_s));
    }

    TEST_CASE("prbs order-5 covers all non-zero states exactly once per period") {
        constexpr design::PRBSConfig<float> cfg{
            .amplitude = 1.0f,
            .lfsr_order = 5,
            .clock_period_s = 0.1f,
            .seed = 0x1u,
        };
        constexpr auto design = design::synthesize_prbs(cfg);

        PRBS<float>          seq(design, cfg.clock_period_s);
        std::array<bool, 32> seen{};

        for (std::size_t i = 0; i < design.period_bits; ++i) {
            const std::uint32_t s = seq.state();
            REQUIRE(s > 0u);
            REQUIRE(s < seen.size());
            CHECK_FALSE(seen[s]);
            seen[s] = true;
            (void)seq.step();
        }

        CHECK(seq.done());
        CHECK(seq.state() == (cfg.seed & 0x1Fu));
    }
} // TEST_SUITE
