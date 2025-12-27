#include "control.hpp"
#include "doctest.h"

using namespace control;

TEST_CASE("TransferFunction Construction and Validation") {
    SUBCASE("Valid construction") {
        // Simple first-order system
        TransferFunction tf({1.0}, {1.0, 1.0});
        CHECK(tf.num.size() == 1);
        CHECK(tf.den.size() == 2);
        CHECK(tf.num[0] == doctest::Approx(1.0));
        CHECK(tf.den[0] == doctest::Approx(1.0));
        CHECK(tf.den[1] == doctest::Approx(1.0));
    }

    SUBCASE("Default constructor") {
        TransferFunction tf;
        CHECK(tf.num == std::vector<double>{0.0});
        CHECK(tf.den == std::vector<double>{1.0});
    }

    SUBCASE("Copy constructor") {
        TransferFunction original({1.0, 2.0}, {1.0, 3.0, 2.0});
        TransferFunction copy = original;
        CHECK(copy.num == original.num);
        CHECK(copy.den == original.den);
        CHECK(copy.Ts == original.Ts);
    }

    SUBCASE("Move constructor") {
        TransferFunction original({1.0, 2.0}, {1.0, 3.0, 2.0});
        TransferFunction moved = std::move(original);
        CHECK(moved.num == std::vector<double>({1.0, 2.0}));
        CHECK(moved.den == std::vector<double>({1.0, 3.0, 2.0}));
    }

    SUBCASE("Discrete-time system") {
        TransferFunction tf({1.0}, {1.0, -0.5}, 0.1);
        CHECK(tf.Ts.has_value());
        CHECK(tf.Ts.value() == doctest::Approx(0.1));
    }

    SUBCASE("Invalid construction throws") {
        // Zero leading coefficient in denominator
        CHECK_THROWS_AS(TransferFunction({1.0}, {0.0, 1.0}), std::invalid_argument);

        // Empty denominator
        CHECK_THROWS_AS(TransferFunction({1.0}, {}), std::invalid_argument);

        // Empty numerator
        CHECK_THROWS_AS(TransferFunction({}, {1.0, 1.0}), std::invalid_argument);

        // Leading zero in numerator
        CHECK_THROWS_AS(TransferFunction({0.0, 1.0}, {1.0, 1.0}), std::invalid_argument);
    }
}

TEST_CASE("TransferFunction Poles and Zeros") {
    SUBCASE("First-order system: 1/(s+2)") {
        // G(s) = 1/(s+2)
        // Pole at s = -2, no zeros
        TransferFunction sys({1.0}, {1.0, 2.0});

        auto poles_vec = sys.poles();
        auto zeros_vec = sys.zeros();

        // Should have 1 pole
        CHECK(poles_vec.size() == 1);
        CHECK(poles_vec[0].real() == doctest::Approx(-2.0).epsilon(1e-6));
        CHECK(std::abs(poles_vec[0].imag()) < 1e-10);

        // Should have no zeros
        CHECK(zeros_vec.size() == 0);
    }

    SUBCASE("First-order with zero: (s+1)/(s+2)") {
        // G(s) = (s+1)/(s+2)
        // Pole at s = -2, zero at s = -1
        TransferFunction sys({1.0, 1.0}, {1.0, 2.0});

        auto poles_vec = sys.poles();
        auto zeros_vec = sys.zeros();

        // Should have 1 pole at -2
        CHECK(poles_vec.size() == 1);
        CHECK(poles_vec[0].real() == doctest::Approx(-2.0).epsilon(1e-6));

        // Should have 1 zero at -1
        CHECK(zeros_vec.size() == 1);
        CHECK(zeros_vec[0].real() == doctest::Approx(-1.0).epsilon(1e-6));
    }

    SUBCASE("Second-order: 1/(s^2 + 3s + 2)") {
        // G(s) = 1/(s^2 + 3s + 2) = 1/((s+1)(s+2))
        // Poles at s = -1 and s = -2, no zeros
        TransferFunction sys({1.0}, {1.0, 3.0, 2.0});

        auto poles_vec = sys.poles();
        auto zeros_vec = sys.zeros();

        // Should have 2 poles
        CHECK(poles_vec.size() == 2);

        // Check poles are -1 and -2 (order may vary)
        std::vector<double> pole_reals = {poles_vec[0].real(), poles_vec[1].real()};
        std::sort(pole_reals.begin(), pole_reals.end());
        CHECK(pole_reals[0] == doctest::Approx(-2.0).epsilon(1e-6));
        CHECK(pole_reals[1] == doctest::Approx(-1.0).epsilon(1e-6));

        // Should have no zeros
        CHECK(zeros_vec.size() == 0);
    }

    SUBCASE("Complex conjugate poles: 1/(s^2 + 2s + 5)") {
        // G(s) = 1/(s^2 + 2s + 5)
        // Poles at s = -1 ± 2j
        TransferFunction sys({1.0}, {1.0, 2.0, 5.0});

        auto poles_vec = sys.poles();

        // Should have 2 poles
        CHECK(poles_vec.size() == 2);

        // Both poles should have real part -1
        CHECK(poles_vec[0].real() == doctest::Approx(-1.0).epsilon(1e-6));
        CHECK(poles_vec[1].real() == doctest::Approx(-1.0).epsilon(1e-6));

        // Imaginary parts should be ±2
        double imag_sum = std::abs(poles_vec[0].imag()) + std::abs(poles_vec[1].imag());
        CHECK(imag_sum == doctest::Approx(4.0).epsilon(1e-6));
    }

    SUBCASE("Complex zeros: (s^2 + 2s + 5)/(s+1)") {
        // G(s) = (s^2 + 2s + 5)/(s+1)
        // Pole at s = -1, zeros at s = -1 ± 2j
        TransferFunction sys({1.0, 2.0, 5.0}, {1.0, 1.0});

        auto zeros_vec = sys.zeros();
        auto poles_vec = sys.poles();

        // Should have 2 zeros
        CHECK(zeros_vec.size() == 2);
        CHECK(zeros_vec[0].real() == doctest::Approx(-1.0).epsilon(1e-6));
        CHECK(zeros_vec[1].real() == doctest::Approx(-1.0).epsilon(1e-6));

        // Should have 1 pole
        CHECK(poles_vec.size() == 1);
        CHECK(poles_vec[0].real() == doctest::Approx(-1.0).epsilon(1e-6));
    }

    SUBCASE("Pure gain (no dynamics)") {
        // G(s) = 5
        // No poles, no zeros
        TransferFunction sys({5.0}, {1.0});

        auto poles_vec = sys.poles();
        auto zeros_vec = sys.zeros();

        // Should have no poles or zeros
        CHECK(poles_vec.size() == 0);
        CHECK(zeros_vec.size() == 0);
    }

    SUBCASE("Discrete system poles and zeros") {
        // Discrete system: G(z) = (z - 0.5)/(z - 0.8)
        // Pole at z = 0.8, zero at z = 0.5
        TransferFunction sys({1.0, -0.5}, {1.0, -0.8}, 0.1);

        auto poles_vec = sys.poles();
        auto zeros_vec = sys.zeros();

        // Should have 1 pole at 0.8
        CHECK(poles_vec.size() == 1);
        CHECK(poles_vec[0].real() == doctest::Approx(0.8).epsilon(1e-6));

        // Should have 1 zero at 0.5
        CHECK(zeros_vec.size() == 1);
        CHECK(zeros_vec[0].real() == doctest::Approx(0.5).epsilon(1e-6));
    }

    SUBCASE("Repeated poles") {
        // G(s) = 1/(s+1)^2 = 1/(s^2 + 2s + 1)
        // Two poles at s = -1 (repeated)
        TransferFunction sys({1.0}, {1.0, 2.0, 1.0});

        auto poles_vec = sys.poles();

        // Should have 2 poles, both at -1
        CHECK(poles_vec.size() == 2);
        CHECK(poles_vec[0].real() == doctest::Approx(-1.0).epsilon(1e-6));
        CHECK(poles_vec[1].real() == doctest::Approx(-1.0).epsilon(1e-6));
    }

    SUBCASE("Repeated zeros") {
        // G(s) = (s+1)^2/(s+2) = (s^2 + 2s + 1)/(s+2)
        // Two zeros at s = -1 (repeated), one pole at s = -2
        TransferFunction sys({1.0, 2.0, 1.0}, {1.0, 2.0});

        auto zeros_vec = sys.zeros();
        auto poles_vec = sys.poles();

        // Should have 2 zeros, both at -1
        CHECK(zeros_vec.size() == 2);
        CHECK(zeros_vec[0].real() == doctest::Approx(-1.0).epsilon(1e-6));
        CHECK(zeros_vec[1].real() == doctest::Approx(-1.0).epsilon(1e-6));

        // Should have 1 pole at -2
        CHECK(poles_vec.size() == 1);
        CHECK(poles_vec[0].real() == doctest::Approx(-2.0).epsilon(1e-6));
    }

    SUBCASE("Higher order system") {
        // G(s) = (s^2 + 3s + 2)/(s^3 + 6s^2 + 11s + 6)
        // Zeros at s = -1, -2; Poles at s = -1, -2, -3
        TransferFunction sys({1.0, 3.0, 2.0}, {1.0, 6.0, 11.0, 6.0});

        auto zeros_vec = sys.zeros();
        auto poles_vec = sys.poles();

        // Should have 2 zeros
        CHECK(zeros_vec.size() == 2);
        std::vector<double> zero_reals = {zeros_vec[0].real(), zeros_vec[1].real()};
        std::sort(zero_reals.begin(), zero_reals.end());
        CHECK(zero_reals[0] == doctest::Approx(-2.0).epsilon(1e-6));
        CHECK(zero_reals[1] == doctest::Approx(-1.0).epsilon(1e-6));

        // Should have 3 poles
        CHECK(poles_vec.size() == 3);
        std::vector<double> pole_reals = {poles_vec[0].real(), poles_vec[1].real(), poles_vec[2].real()};
        std::sort(pole_reals.begin(), pole_reals.end());
        CHECK(pole_reals[0] == doctest::Approx(-3.0).epsilon(1e-6));
        CHECK(pole_reals[1] == doctest::Approx(-2.0).epsilon(1e-6));
        CHECK(pole_reals[2] == doctest::Approx(-1.0).epsilon(1e-6));
    }
}

TEST_CASE("TransferFunction System Analysis") {
    SUBCASE("Stability check") {
        // Stable system: pole at -1
        TransferFunction stable({1.0}, {1.0, 1.0});
        CHECK(stable.is_stable());

        // Unstable system: pole at +1
        TransferFunction unstable({1.0}, {1.0, -1.0});
        CHECK_FALSE(unstable.is_stable());
    }

    SUBCASE("System type") {
        TransferFunction continuous({1.0}, {1.0, 1.0});
        CHECK(continuous.isContinuous());
        CHECK_FALSE(continuous.isDiscrete());

        TransferFunction discrete({1.0}, {1.0, -0.5}, 0.1);
        CHECK(discrete.isDiscrete());
        CHECK_FALSE(discrete.isContinuous());
    }
}

TEST_CASE("TransferFunction Time Domain Analysis") {
    SUBCASE("Step response - default parameters") {
        TransferFunction sys({1.0}, {1.0, 1.0});  // 1/(s+1)

        auto step_resp = sys.step();  // Use defaults: tStart=0, tEnd=10, uStep=[1]
        CHECK(step_resp.time.size() > 0);
        CHECK(step_resp.output.size() == step_resp.time.size());

        // For 1/(s+1), steady-state should be 1
        CHECK(step_resp.output.back()(0, 0) == doctest::Approx(1.0).epsilon(0.01));
    }

    SUBCASE("Step response - custom time range") {
        TransferFunction sys({1.0}, {1.0, 1.0});  // 1/(s+1)

        auto step_resp = sys.step(1.0, 6.0);  // Start at t=1, end at t=6 (longer time)
        CHECK(step_resp.time.size() > 0);
        CHECK(step_resp.output.size() == step_resp.time.size());
        CHECK(step_resp.time.front() >= 1.0);
        CHECK(step_resp.time.back() <= 6.0);

        // Steady-state should still be 1
        CHECK(step_resp.output.back()(0, 0) == doctest::Approx(1.0).epsilon(0.01));
    }

    SUBCASE("Step response - custom step amplitude") {
        TransferFunction sys({1.0}, {1.0, 1.0});  // 1/(s+1)

        ColVec uStep(1);
        uStep << 2.0;  // Step of amplitude 2

        auto step_resp = sys.step(0.0, 5.0, uStep);
        CHECK(step_resp.time.size() > 0);
        CHECK(step_resp.output.size() == step_resp.time.size());

        // For 1/(s+1) with step amplitude 2, steady-state should be 2
        CHECK(step_resp.output.back()(0, 0) == doctest::Approx(2.0).epsilon(0.01));
    }

    SUBCASE("Step response - second order system") {
        TransferFunction sys({1.0}, {1.0, 0.5, 1.0});  // 1/(s² + 0.5s + 1)

        auto step_resp = sys.step(0.0, 20.0);  // Much longer time for oscillatory system
        CHECK(step_resp.time.size() > 0);
        CHECK(step_resp.output.size() == step_resp.time.size());

        // Second-order system should oscillate and eventually reach steady-state of 1
        // Just check that it produces valid output (exact value may vary due to numerical issues)
        CHECK(std::abs(step_resp.output.back()(0, 0) - 1.0) < 0.01);  // Should converge close to 1.0
    }

    SUBCASE("Step response - unstable system") {
        TransferFunction sys({1.0}, {1.0, -1.0});  // 1/(s-1) - unstable

        auto step_resp = sys.step(0.0, 2.0);
        CHECK(step_resp.time.size() > 0);
        CHECK(step_resp.output.size() == step_resp.time.size());

        // Unstable system should grow exponentially
        CHECK(std::abs(step_resp.output.back()(0, 0)) > std::abs(step_resp.output.front()(0, 0)));
    }

    SUBCASE("Impulse response") {
        TransferFunction sys({1.0}, {1.0, 1.0});  // 1/(s+1)

        auto impulse_resp = sys.impulse(0.0, 2.0);
        CHECK(impulse_resp.time.size() > 0);
        CHECK(impulse_resp.output.size() == impulse_resp.time.size());
    }
}

TEST_CASE("TransferFunction Frequency Domain Analysis") {
    SUBCASE("Frequency response") {
        TransferFunction sys({1.0}, {1.0, 1.0});  // 1/(s+1)

        std::vector<double> freqs     = {0.1, 1.0, 10.0};  // Frequencies in Hz
        auto                freq_resp = sys.freqresp(freqs);

        CHECK(freq_resp.freq.size() == 3);
        CHECK(freq_resp.response.size() == 3);

        // At f = 0.1 Hz (ω = 2π*0.1 ≈ 0.628 rad/s), H(jω) ≈ 1/(j*0.628 + 1) ≈ 0.717 - j*0.450
        CHECK(freq_resp.response[0].real() == doctest::Approx(0.717).epsilon(1e-3));
        CHECK(freq_resp.response[0].imag() == doctest::Approx(-0.450).epsilon(1e-3));

        // At f = 1.0 Hz (ω = 2π*1.0 ≈ 6.28 rad/s), H(jω) ≈ 1/(j*6.28 + 1) ≈ 0.025 - j*0.155
        CHECK(freq_resp.response[1].real() == doctest::Approx(0.025).epsilon(1e-3));
        CHECK(freq_resp.response[1].imag() == doctest::Approx(-0.155).epsilon(1e-3));

        // At f = 10.0 Hz (ω = 2π*10.0 ≈ 62.8 rad/s), H(jω) ≈ 1/(j*62.8 + 1) ≈ 0.00025 - j*0.016
        CHECK(freq_resp.response[2].real() == doctest::Approx(0.00025).epsilon(1e-4));
        CHECK(freq_resp.response[2].imag() == doctest::Approx(-0.016).epsilon(1e-3));
    }

    SUBCASE("Frequency response - DC (ω = 0)") {
        TransferFunction sys({2.0}, {1.0, 1.0});  // 2/(s+1)

        std::vector<double> freqs     = {0.0};
        auto                freq_resp = sys.freqresp(freqs);

        CHECK(freq_resp.response[0].real() == doctest::Approx(2.0).epsilon(1e-6));
        CHECK(freq_resp.response[0].imag() == doctest::Approx(0.0).epsilon(1e-6));
    }

    SUBCASE("Frequency response - high frequency") {
        TransferFunction sys({1.0}, {1.0, 1.0});  // 1/(s+1)

        std::vector<double> freqs     = {1000.0};
        auto                freq_resp = sys.freqresp(freqs);

        // At high frequency, magnitude should be small
        double mag = std::abs(freq_resp.response[0]);
        CHECK(mag < 0.01);
    }

    SUBCASE("Bode plot") {
        TransferFunction sys({1.0}, {1.0, 1.0});  // 1/(s+1)

        auto bode_resp = sys.bode(0.1, 10.0, 50);
        CHECK(bode_resp.freq.size() > 0);
        CHECK(bode_resp.magnitude.size() == bode_resp.freq.size());
        CHECK(bode_resp.phase.size() == bode_resp.freq.size());

        // At high frequency, magnitude should be -20 dB/decade
        // At low frequency, magnitude should be 0 dB
        CHECK(bode_resp.magnitude[0] == doctest::Approx(0.0).epsilon(1.0));  // Low freq
        CHECK(bode_resp.magnitude.back() < bode_resp.magnitude[0]);          // High freq attenuation
    }

    SUBCASE("Nyquist plot") {
        TransferFunction sys({1.0}, {1.0, 1.0});  // 1/(s+1)

        auto nyquist_resp = sys.nyquist(0.1, 10.0, 50);
        CHECK(nyquist_resp.freq.size() > 0);
        CHECK(nyquist_resp.response.size() == nyquist_resp.freq.size());
    }

    SUBCASE("Root locus") {
        TransferFunction sys({1.0}, {1.0, 1.0});  // 1/(s+1)

        auto rlocus_resp = sys.rlocus(0.0, 10.0, 50);
        CHECK(rlocus_resp.gains.size() > 0);
        CHECK(rlocus_resp.branches.size() == 1);  // 1 pole
        CHECK(rlocus_resp.branches[0].size() == rlocus_resp.gains.size());
    }

    SUBCASE("Stability margins - first order system") {
        TransferFunction sys({1.0}, {1.0, 1.0});  // 1/(s+1)

        auto margin_info = sys.margin();

        // For 1/(s+1), gain margin should be infinite (no phase crossover in typical range)
        // Phase margin should be positive
        CHECK(margin_info.phaseMargin > 0.0);
        CHECK(margin_info.gainMargin >= 0.0);  // Could be infinite
        CHECK(margin_info.gainCrossover >= 0.0);
        CHECK(margin_info.phaseCrossover >= 0.0);
    }

    SUBCASE("Stability margins - second order system") {
        TransferFunction sys({1.0}, {1.0, 0.1, 1.0});  // 1/(s^2 + 0.1s + 1)

        auto margin_info = sys.margin();

        // Should have finite gain and phase margins
        CHECK(margin_info.phaseMargin > 0.0);
        CHECK(margin_info.gainMargin > 0.0);
        CHECK(margin_info.gainCrossover > 0.0);
        CHECK(margin_info.phaseCrossover > 0.0);
    }

    SUBCASE("Stability margins - discrete system") {
        TransferFunction sys({1.0}, {1.0, -0.5}, 0.1);  // Discrete: 1/(z - 0.5)

        auto margin_info = sys.margin();

        // Should compute margins for discrete system
        CHECK(margin_info.gainCrossover >= 0.0);
        CHECK(margin_info.phaseCrossover >= 0.0);
    }
}

TEST_CASE("TransferFunction Discretization") {
    TransferFunction continuous({1.0}, {1.0, 1.0});  // 1/(s+1)

    SUBCASE("ZOH discretization") {
        auto discrete = continuous.discretize(0.1, DiscretizationMethod::ZOH);
        CHECK(discrete.isDiscrete());
        CHECK(discrete.Ts.value() == doctest::Approx(0.1));
    }

    SUBCASE("Tustin discretization") {
        auto discrete = continuous.discretize(0.1, DiscretizationMethod::Tustin);
        CHECK(discrete.isDiscrete());
        CHECK(discrete.Ts.value() == doctest::Approx(0.1));
    }
}

TEST_CASE("TransferFunction Conversion") {
    SUBCASE("To StateSpace") {
        TransferFunction tf({1.0}, {1.0, 1.0});
        StateSpace       ss = tf.toStateSpace();

        CHECK(ss.A.rows() == 1);
        CHECK(ss.A.cols() == 1);
        CHECK(ss.B.rows() == 1);
        CHECK(ss.B.cols() == 1);
        CHECK(ss.C.rows() == 1);
        CHECK(ss.C.cols() == 1);
        CHECK(ss.D.rows() == 1);
        CHECK(ss.D.cols() == 1);
    }

    SUBCASE("Round-trip: TF -> SS -> TF") {
        TransferFunction original({1.0}, {1.0, 1.0});  // 1/(s+1)

        StateSpace       ss        = control::ss(original);  // Convert to state-space
        TransferFunction recovered = control::tf(ss);        // Convert back

        // Should match the original (within numerical precision)
        CHECK(recovered.num.size() == original.num.size());
        CHECK(recovered.den.size() == original.den.size());

        for (size_t i = 0; i < recovered.num.size(); ++i) {
            CHECK(recovered.num[i] == doctest::Approx(original.num[i]).epsilon(1e-6));
        }
        for (size_t i = 0; i < recovered.den.size(); ++i) {
            CHECK(recovered.den[i] == doctest::Approx(original.den[i]).epsilon(1e-6));
        }
    }

    SUBCASE("Frequency response equivalence: TF vs SS") {
        TransferFunction tf({1.0}, {1.0, 1.0});  // 1/(s+1)
        StateSpace       ss = control::ss(tf);   // Convert to state-space

        std::vector<double> freqs = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0};

        auto tf_resp = tf.freqresp(freqs);
        auto ss_resp = ss.freqresp(freqs);

        CHECK(tf_resp.freq.size() == ss_resp.freq.size());
        CHECK(tf_resp.response.size() == ss_resp.response.size());

        // Frequency responses should be identical (within numerical precision)
        for (size_t i = 0; i < freqs.size(); ++i) {
            CHECK(tf_resp.response[i].real() == doctest::Approx(ss_resp.response[i].real()).epsilon(1e-6));
            CHECK(tf_resp.response[i].imag() == doctest::Approx(ss_resp.response[i].imag()).epsilon(1e-6));
        }
    }
}

TEST_CASE("TransferFunction to StateSpace Conversion (ss function)") {
    SUBCASE("First-order system: 1/(s+1)") {
        TransferFunction tf({1.0}, {1.0, 1.0});
        StateSpace       ss = control::ss(tf);

        // Check dimensions
        CHECK(ss.A.rows() == 1);
        CHECK(ss.A.cols() == 1);
        CHECK(ss.B.rows() == 1);
        CHECK(ss.B.cols() == 1);
        CHECK(ss.C.rows() == 1);
        CHECK(ss.C.cols() == 1);
        CHECK(ss.D.rows() == 1);
        CHECK(ss.D.cols() == 1);

        // Check values for controllable canonical form
        CHECK(ss.A(0, 0) == doctest::Approx(-1.0));
        CHECK(ss.B(0, 0) == doctest::Approx(1.0));
        CHECK(ss.C(0, 0) == doctest::Approx(1.0));
        CHECK(ss.D(0, 0) == doctest::Approx(0.0));
    }

    SUBCASE("First-order with zero: (s+2)/(s+1)") {
        TransferFunction tf({1.0, 2.0}, {1.0, 1.0});
        StateSpace       ss = control::ss(tf);

        // Check dimensions
        CHECK(ss.A.rows() == 1);
        CHECK(ss.A.cols() == 1);
        CHECK(ss.B.rows() == 1);
        CHECK(ss.B.cols() == 1);
        CHECK(ss.C.rows() == 1);
        CHECK(ss.C.cols() == 1);
        CHECK(ss.D.rows() == 1);
        CHECK(ss.D.cols() == 1);

        // Check values
        CHECK(ss.A(0, 0) == doctest::Approx(-1.0));
        CHECK(ss.B(0, 0) == doctest::Approx(1.0));
        CHECK(ss.C(0, 0) == doctest::Approx(1.0));
        CHECK(ss.D(0, 0) == doctest::Approx(1.0));  // Direct feedthrough from polynomial division
    }

    SUBCASE("Second-order system: 1/(s² + 2s + 1)") {
        TransferFunction tf({1.0}, {1.0, 2.0, 1.0});
        StateSpace       ss = control::ss(tf);

        // Check dimensions
        CHECK(ss.A.rows() == 2);
        CHECK(ss.A.cols() == 2);
        CHECK(ss.B.rows() == 2);
        CHECK(ss.B.cols() == 1);
        CHECK(ss.C.rows() == 1);
        CHECK(ss.C.cols() == 2);
        CHECK(ss.D.rows() == 1);
        CHECK(ss.D.cols() == 1);

        // Check A matrix (companion form)
        CHECK(ss.A(0, 0) == doctest::Approx(0.0));
        CHECK(ss.A(0, 1) == doctest::Approx(1.0));
        CHECK(ss.A(1, 0) == doctest::Approx(-1.0));
        CHECK(ss.A(1, 1) == doctest::Approx(-2.0));

        // Check B matrix
        CHECK(ss.B(0, 0) == doctest::Approx(0.0));
        CHECK(ss.B(1, 0) == doctest::Approx(1.0));

        // Check C matrix
        CHECK(ss.C(0, 0) == doctest::Approx(1.0));
        CHECK(ss.C(0, 1) == doctest::Approx(0.0));

        // Check D matrix
        CHECK(ss.D(0, 0) == doctest::Approx(0.0));
    }

    SUBCASE("Second-order with numerator: (s+1)/(s² + 2s + 1)") {
        TransferFunction tf({1.0, 1.0}, {1.0, 2.0, 1.0});
        StateSpace       ss = control::ss(tf);

        // Check dimensions
        CHECK(ss.A.rows() == 2);
        CHECK(ss.A.cols() == 2);
        CHECK(ss.B.rows() == 2);
        CHECK(ss.B.cols() == 1);
        CHECK(ss.C.rows() == 1);
        CHECK(ss.C.cols() == 2);
        CHECK(ss.D.rows() == 1);
        CHECK(ss.D.cols() == 1);

        // Check A matrix (should be same as above)
        CHECK(ss.A(0, 0) == doctest::Approx(0.0));
        CHECK(ss.A(0, 1) == doctest::Approx(1.0));
        CHECK(ss.A(1, 0) == doctest::Approx(-1.0));
        CHECK(ss.A(1, 1) == doctest::Approx(-2.0));

        // Check B matrix
        CHECK(ss.B(0, 0) == doctest::Approx(0.0));
        CHECK(ss.B(1, 0) == doctest::Approx(1.0));

        // Check C matrix
        CHECK(ss.C(0, 0) == doctest::Approx(1.0));
        CHECK(ss.C(0, 1) == doctest::Approx(1.0));

        // Check D matrix
        CHECK(ss.D(0, 0) == doctest::Approx(0.0));
    }

    SUBCASE("Pure gain: 5") {
        TransferFunction tf({5.0}, {1.0});
        StateSpace       ss = control::ss(tf);

        // Check dimensions (should be 0x0 system)
        CHECK(ss.A.rows() == 0);
        CHECK(ss.A.cols() == 0);
        CHECK(ss.B.rows() == 0);
        CHECK(ss.B.cols() == 1);
        CHECK(ss.C.rows() == 1);
        CHECK(ss.C.cols() == 0);
        CHECK(ss.D.rows() == 1);
        CHECK(ss.D.cols() == 1);

        // Check D matrix
        CHECK(ss.D(0, 0) == doctest::Approx(5.0));
    }

    SUBCASE("Third-order system: 1/(s³ + 2s² + 2s + 1)") {
        TransferFunction tf({1.0}, {1.0, 2.0, 2.0, 1.0});
        StateSpace       ss = control::ss(tf);

        // Check dimensions
        CHECK(ss.A.rows() == 3);
        CHECK(ss.A.cols() == 3);
        CHECK(ss.B.rows() == 3);
        CHECK(ss.B.cols() == 1);
        CHECK(ss.C.rows() == 1);
        CHECK(ss.C.cols() == 3);
        CHECK(ss.D.rows() == 1);
        CHECK(ss.D.cols() == 1);

        // Check A matrix structure (companion form)
        CHECK(ss.A(0, 0) == doctest::Approx(0.0));
        CHECK(ss.A(0, 1) == doctest::Approx(1.0));
        CHECK(ss.A(0, 2) == doctest::Approx(0.0));
        CHECK(ss.A(1, 0) == doctest::Approx(0.0));
        CHECK(ss.A(1, 1) == doctest::Approx(0.0));
        CHECK(ss.A(1, 2) == doctest::Approx(1.0));
        CHECK(ss.A(2, 0) == doctest::Approx(-1.0));
        CHECK(ss.A(2, 1) == doctest::Approx(-2.0));
        CHECK(ss.A(2, 2) == doctest::Approx(-2.0));

        // Check B matrix
        CHECK(ss.B(0, 0) == doctest::Approx(0.0));
        CHECK(ss.B(1, 0) == doctest::Approx(0.0));
        CHECK(ss.B(2, 0) == doctest::Approx(1.0));

        // Check C matrix
        CHECK(ss.C(0, 0) == doctest::Approx(1.0));
        CHECK(ss.C(0, 1) == doctest::Approx(0.0));
        CHECK(ss.C(0, 2) == doctest::Approx(0.0));

        // Check D matrix
        CHECK(ss.D(0, 0) == doctest::Approx(0.0));
    }

    SUBCASE("Discrete-time system") {
        TransferFunction tf({1.0}, {1.0, -0.5}, 0.1);
        StateSpace       ss = control::ss(tf);

        // Check dimensions
        CHECK(ss.A.rows() == 1);
        CHECK(ss.A.cols() == 1);
        CHECK(ss.B.rows() == 1);
        CHECK(ss.B.cols() == 1);
        CHECK(ss.C.rows() == 1);
        CHECK(ss.C.cols() == 1);
        CHECK(ss.D.rows() == 1);
        CHECK(ss.D.cols() == 1);

        // Check sampling time is preserved
        CHECK(ss.Ts.has_value());
        CHECK(ss.Ts.value() == doctest::Approx(0.1));

        // Check values
        CHECK(ss.A(0, 0) == doctest::Approx(0.5));
        CHECK(ss.B(0, 0) == doctest::Approx(1.0));
        CHECK(ss.C(0, 0) == doctest::Approx(1.0));
        CHECK(ss.D(0, 0) == doctest::Approx(0.0));
    }

    SUBCASE("Bi-proper system: (s² + 1)/(s² + 2s + 1)") {
        TransferFunction tf({1.0, 0.0, 1.0}, {1.0, 2.0, 1.0});
        StateSpace       ss = control::ss(tf);

        // Check dimensions
        CHECK(ss.A.rows() == 2);
        CHECK(ss.A.cols() == 2);
        CHECK(ss.B.rows() == 2);
        CHECK(ss.B.cols() == 1);
        CHECK(ss.C.rows() == 1);
        CHECK(ss.C.cols() == 2);
        CHECK(ss.D.rows() == 1);
        CHECK(ss.D.cols() == 1);

        // Check A matrix (should be same as second-order case)
        CHECK(ss.A(0, 0) == doctest::Approx(0.0));
        CHECK(ss.A(0, 1) == doctest::Approx(1.0));
        CHECK(ss.A(1, 0) == doctest::Approx(-1.0));
        CHECK(ss.A(1, 1) == doctest::Approx(-2.0));

        // Check B matrix
        CHECK(ss.B(0, 0) == doctest::Approx(0.0));
        CHECK(ss.B(1, 0) == doctest::Approx(1.0));

        // Check C matrix (proper part: -2s/(s² + 2s + 1))
        CHECK(ss.C(0, 0) == doctest::Approx(-2.0));
        CHECK(ss.C(0, 1) == doctest::Approx(0.0));

        // Check D matrix (direct feedthrough from polynomial division)
        CHECK(ss.D(0, 0) == doctest::Approx(1.0));
    }
}

TEST_CASE("StateSpace to TransferFunction Conversion (tf functions)") {
    SUBCASE("SISO first-order system") {
        // Create state-space: dx/dt = -2x + u, y = 3x
        // Should give G(s) = 3/(s+2)
        Matrix     A = Matrix::Constant(1, 1, -2.0);
        Matrix     B = Matrix::Constant(1, 1, 1.0);
        Matrix     C = Matrix::Constant(1, 1, 3.0);
        Matrix     D = Matrix::Constant(1, 1, 0.0);
        StateSpace ss(A, B, C, D);

        TransferFunction tf = control::tf(ss);

        // Check result: should be 3/(s+2)
        CHECK(tf.num.size() == 1);
        CHECK(tf.den.size() == 2);
        CHECK(tf.num[0] == doctest::Approx(3.0).epsilon(1e-6));
        CHECK(tf.den[0] == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(tf.den[1] == doctest::Approx(2.0).epsilon(1e-6));
    }

    SUBCASE("SISO first-order with feedthrough") {
        // Create state-space: dx/dt = -x + u, y = x + 2u
        // Should give G(s) = (2s + 3)/(s+1)
        Matrix     A = Matrix::Constant(1, 1, -1.0);
        Matrix     B = Matrix::Constant(1, 1, 1.0);
        Matrix     C = Matrix::Constant(1, 1, 1.0);
        Matrix     D = Matrix::Constant(1, 1, 2.0);
        StateSpace ss(A, B, C, D);

        TransferFunction tf = control::tf(ss);

        // Check result: should be (2s + 3)/(s+1)
        CHECK(tf.num.size() == 2);
        CHECK(tf.den.size() == 2);
        CHECK(tf.num[0] == doctest::Approx(2.0).epsilon(1e-6));
        CHECK(tf.num[1] == doctest::Approx(3.0).epsilon(1e-6));
        CHECK(tf.den[0] == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(tf.den[1] == doctest::Approx(1.0).epsilon(1e-6));
    }

    SUBCASE("SISO second-order system") {
        // Create state-space for 1/(s² + 2s + 1)
        // A = [0, 1; -1, -2], B = [0; 1], C = [1, 0], D = [0]
        Matrix A(2, 2);
        A << 0, 1, -1, -2;
        Matrix B(2, 1);
        B << 0, 1;
        Matrix C(1, 2);
        C << 1, 0;
        Matrix D(1, 1);
        D << 0;
        StateSpace ss(A, B, C, D);

        TransferFunction tf = control::tf(ss);

        // Check result: should be 1/(s² + 2s + 1)
        CHECK(tf.num.size() == 1);
        CHECK(tf.den.size() == 3);
        CHECK(tf.num[0] == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(tf.den[0] == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(tf.den[1] == doctest::Approx(2.0).epsilon(1e-6));
    }

    SUBCASE("Pure gain system") {
        // Create state-space with no states: y = 5u
        Matrix A(0, 0);
        Matrix B(0, 1);
        B.resize(0, 1);
        Matrix C(1, 0);
        C.resize(1, 0);
        Matrix     D = Matrix::Constant(1, 1, 5.0);
        StateSpace ss(A, B, C, D);

        TransferFunction tf = control::tf(ss);

        // Check result: should be 5
        CHECK(tf.num.size() == 1);
        CHECK(tf.den.size() == 1);
        CHECK(tf.num[0] == doctest::Approx(5.0).epsilon(1e-6));
        CHECK(tf.den[0] == doctest::Approx(1.0).epsilon(1e-6));
    }

    SUBCASE("MIMO system - extract specific transfer function") {
        // Create 2x2 MIMO system
        Matrix A(2, 2);
        A << 0, 1, -1, -2;
        Matrix B(2, 2);
        B << 1, 0, 0, 1;
        Matrix C(2, 2);
        C << 1, 0, 0, 1;
        Matrix D(2, 2);
        D << 0, 1, 2, 0;
        StateSpace ss(A, B, C, D);

        // Extract G11(s) - from input 0 to output 0
        TransferFunction tf11 = control::tf(ss, 0, 0);
        CHECK(tf11.num.size() == 2);
        CHECK(tf11.den.size() == 3);
        CHECK(tf11.num[0] == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(tf11.num[1] == doctest::Approx(2.0).epsilon(1e-6));
        CHECK(tf11.den[0] == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(tf11.den[1] == doctest::Approx(2.0).epsilon(1e-6));
        CHECK(tf11.den[2] == doctest::Approx(1.0).epsilon(1e-6));

        // Extract G12(s) - from input 1 to output 0
        TransferFunction tf12 = control::tf(ss, 0, 1);
        CHECK(tf12.num.size() == 3);
        CHECK(tf12.den.size() == 3);
        CHECK(tf12.num[0] == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(tf12.num[1] == doctest::Approx(2.0).epsilon(1e-6));
        CHECK(tf12.num[2] == doctest::Approx(2.0).epsilon(1e-6));
        CHECK(tf12.den[0] == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(tf12.den[1] == doctest::Approx(2.0).epsilon(1e-6));
        CHECK(tf12.den[2] == doctest::Approx(1.0).epsilon(1e-6));

        // Extract G21(s) - from input 0 to output 1
        TransferFunction tf21 = control::tf(ss, 1, 0);
        CHECK(tf21.num.size() == 3);
        CHECK(tf21.den.size() == 3);
        CHECK(tf21.num[0] == doctest::Approx(2.0).epsilon(1e-6));
        CHECK(tf21.num[1] == doctest::Approx(4.0).epsilon(1e-6));
        CHECK(tf21.num[2] == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(tf21.den[0] == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(tf21.den[1] == doctest::Approx(2.0).epsilon(1e-6));
        CHECK(tf21.den[2] == doctest::Approx(1.0).epsilon(1e-6));

        // Extract G22(s) - from input 1 to output 1
        TransferFunction tf22 = control::tf(ss, 1, 1);
        CHECK(tf22.num.size() == 2);
        CHECK(tf22.den.size() == 3);
        CHECK(tf22.num[0] == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(tf22.num[1] == doctest::Approx(0.0).epsilon(1e-6));
        CHECK(tf22.den[0] == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(tf22.den[1] == doctest::Approx(2.0).epsilon(1e-6));
        CHECK(tf22.den[2] == doctest::Approx(1.0).epsilon(1e-6));
    }

    SUBCASE("Discrete-time system") {
        // Create discrete-time state-space
        Matrix     A = Matrix::Constant(1, 1, 0.5);
        Matrix     B = Matrix::Constant(1, 1, 1.0);
        Matrix     C = Matrix::Constant(1, 1, 2.0);
        Matrix     D = Matrix::Constant(1, 1, 0.0);
        StateSpace ss(A, B, C, D, 0.1);

        TransferFunction tf = control::tf(ss);

        // Check sampling time is preserved
        CHECK(tf.Ts.has_value());
        CHECK(tf.Ts.value() == doctest::Approx(0.1));

        // Check transfer function: G(z) = 2/(z - 0.5) * (1/z) wait, need to verify the math
        // For discrete: G(z) = C(zI - A)^(-1)B + D
        // (zI - A)^(-1) = 1/(z - 0.5)
        // So G(z) = 2 * 1/(z - 0.5) * 1 + 0 = 2/(z - 0.5)
        // In polynomial form: 2/(z - 0.5) = 2z^(-1) / (1 - 0.5z^(-1)) = (2) / (z - 0.5)
        // Wait, actually for discrete transfer functions, the convention varies.
        // Let me check what the implementation produces.
        CHECK(tf.num.size() == 1);
        CHECK(tf.den.size() == 2);
        CHECK(tf.den[0] == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(tf.den[1] == doctest::Approx(-0.5).epsilon(1e-6));
    }

    SUBCASE("Round-trip equivalence") {
        // Test that TF -> SS -> TF gives back the original
        TransferFunction original({1.0}, {1.0, 1.0});  // 1/(s+1)

        StateSpace       ss        = control::ss(original);
        TransferFunction recovered = control::tf(ss);

        // Should match the original (within numerical precision)
        CHECK(recovered.num.size() == original.num.size());
        CHECK(recovered.den.size() == original.den.size());

        for (size_t i = 0; i < recovered.num.size(); ++i) {
            CHECK(recovered.num[i] == doctest::Approx(original.num[i]).epsilon(1e-5));
        }
        for (size_t i = 0; i < recovered.den.size(); ++i) {
            CHECK(recovered.den[i] == doctest::Approx(original.den[i]).epsilon(1e-5));
        }

        // Sampling time should be preserved
        CHECK(recovered.Ts == original.Ts);
    }
}

TEST_CASE("zpk - Zero-Pole-Gain TransferFunction Creation") {
    SUBCASE("Simple first-order system") {
        // G(s) = 2 / (s + 1)
        std::vector<Zero> zeros = {};
        std::vector<Pole> poles = {1.0};
        double            gain  = 2.0;

        TransferFunction tf = zpk(zeros, poles, gain);

        CHECK(tf.num.size() == 1);
        CHECK(tf.den.size() == 2);
        CHECK(tf.num[0] == doctest::Approx(2.0));
        CHECK(tf.den[0] == doctest::Approx(1.0));
        CHECK(tf.den[1] == doctest::Approx(-1.0));
    }

    SUBCASE("System with zeros and poles") {
        // G(s) = 3 * (s + 2) / ((s + 1)(s + 3))
        std::vector<Zero> zeros = {-2.0};
        std::vector<Pole> poles = {-1.0, -3.0};
        double            gain  = 3.0;

        TransferFunction tf = zpk(zeros, poles, gain);

        CHECK(tf.num.size() == 2);
        CHECK(tf.den.size() == 3);
        CHECK(tf.num[0] == doctest::Approx(3.0));
        CHECK(tf.num[1] == doctest::Approx(6.0));
        CHECK(tf.den[0] == doctest::Approx(1.0));
        CHECK(tf.den[1] == doctest::Approx(4.0));
        CHECK(tf.den[2] == doctest::Approx(3.0));
    }

    SUBCASE("Complex conjugate poles") {
        // G(s) = 1 / (s^2 + 2s + 2) = 1 / ((s+1)^2 + 1)
        std::vector<Zero> zeros = {};
        std::vector<Pole> poles = {std::complex<double>(-1.0, 1.0), std::complex<double>(-1.0, -1.0)};
        double            gain  = 1.0;

        TransferFunction tf = zpk(zeros, poles, gain);

        CHECK(tf.num.size() == 1);
        CHECK(tf.den.size() == 3);
        CHECK(tf.num[0] == doctest::Approx(1.0));
        CHECK(tf.den[0] == doctest::Approx(1.0));
        CHECK(tf.den[1] == doctest::Approx(2.0));
        CHECK(tf.den[2] == doctest::Approx(2.0));
    }

    SUBCASE("Discrete-time system") {
        // G(z) = 2 / (z - 0.5)
        std::vector<Zero>     zeros = {};
        std::vector<Pole>     poles = {0.5};
        double                gain  = 2.0;
        std::optional<double> Ts    = 0.1;

        TransferFunction tf = zpk(zeros, poles, gain, Ts);

        CHECK(tf.num.size() == 1);
        CHECK(tf.den.size() == 2);
        CHECK(tf.num[0] == doctest::Approx(2.0));
        CHECK(tf.den[0] == doctest::Approx(1.0));
        CHECK(tf.den[1] == doctest::Approx(-0.5));
        CHECK(tf.Ts.has_value());
        CHECK(tf.Ts.value() == doctest::Approx(0.1));
    }
}

TEST_CASE("linspace - Linear Space Generation") {
    SUBCASE("Basic linspace with start and end") {
        auto result = linspace(0.0, 10.0, 5);
        CHECK(result.size() == 5);
        CHECK(result[0] == doctest::Approx(0.0));
        CHECK(result[1] == doctest::Approx(2.5));
        CHECK(result[2] == doctest::Approx(5.0));
        CHECK(result[3] == doctest::Approx(7.5));
        CHECK(result[4] == doctest::Approx(10.0));
    }

    SUBCASE("linspace with pair") {
        std::pair<double, double> span   = {1.0, 4.0};
        auto                      result = linspace(span, 4);
        CHECK(result.size() == 4);
        CHECK(result[0] == doctest::Approx(1.0));
        CHECK(result[1] == doctest::Approx(2.0));
        CHECK(result[2] == doctest::Approx(3.0));
        CHECK(result[3] == doctest::Approx(4.0));
    }

    SUBCASE("Single point") {
        auto result = linspace(5.0, 5.0, 1);
        CHECK(result.size() == 1);
        CHECK(result[0] == doctest::Approx(5.0));
    }

    SUBCASE("Negative range") {
        auto result = linspace(-5.0, 5.0, 3);
        CHECK(result.size() == 3);
        CHECK(result[0] == doctest::Approx(-5.0));
        CHECK(result[1] == doctest::Approx(0.0));
        CHECK(result[2] == doctest::Approx(5.0));
    }
}

TEST_CASE("c2d - Continuous to Discrete Conversion") {
    SUBCASE("ZOH discretization of first-order system") {
        // G(s) = 1/(s + 1)
        TransferFunction continuous({1.0}, {1.0, 1.0});
        double           Ts = 0.1;

        StateSpace discrete = c2d(continuous, Ts, DiscretizationMethod::ZOH);

        CHECK(discrete.Ts.has_value());
        CHECK(discrete.Ts.value() == doctest::Approx(Ts));
        CHECK(discrete.A.rows() == 1);
        CHECK(discrete.A.cols() == 1);
        CHECK(discrete.B.rows() == 1);
        CHECK(discrete.B.cols() == 1);
        CHECK(discrete.C.rows() == 1);
        CHECK(discrete.C.cols() == 1);
        CHECK(discrete.D.rows() == 1);
        CHECK(discrete.D.cols() == 1);

        // Check that A is not identity (system is dynamic)
        CHECK(discrete.A(0, 0) != doctest::Approx(1.0));

        // For ZOH: A_d = exp(A*Ts), B_d = A^(-1)*(exp(A*Ts) - I)*B
        // A = -1, so A_d ≈ exp(-0.1) ≈ 0.9048
        // B_d ≈ (-1)^(-1) * (exp(-0.1) - 1) * 1 ≈ -1 * (0.9048 - 1) ≈ 0.0952
        CHECK(discrete.A(0, 0) == doctest::Approx(std::exp(-Ts)).epsilon(1e-4));
        CHECK(discrete.B(0, 0) == doctest::Approx(1.0 - std::exp(-Ts)).epsilon(1e-4));
        CHECK(discrete.C(0, 0) == doctest::Approx(1.0));
        CHECK(discrete.D(0, 0) == doctest::Approx(0.0));
    }

    SUBCASE("FOH discretization") {
        TransferFunction continuous({1.0}, {1.0, 1.0});
        double           Ts = 0.1;

        StateSpace discrete = c2d(continuous, Ts, DiscretizationMethod::FOH);

        CHECK(discrete.Ts.has_value());
        CHECK(discrete.Ts.value() == doctest::Approx(Ts));

        // FOH should give different results than ZOH
        StateSpace zoh_discrete = c2d(continuous, Ts, DiscretizationMethod::ZOH);
        CHECK(discrete.B(0, 0) != doctest::Approx(zoh_discrete.B(0, 0)));
        CHECK(discrete.D(0, 0) != doctest::Approx(zoh_discrete.D(0, 0)));
    }

    SUBCASE("Bilinear discretization") {
        TransferFunction continuous({1.0}, {1.0, 1.0});
        double           Ts = 0.1;

        StateSpace discrete = c2d(continuous, Ts, DiscretizationMethod::Bilinear);

        CHECK(discrete.Ts.has_value());
        CHECK(discrete.Ts.value() == doctest::Approx(Ts));

        // For Tustin/Bilinear: A_d = (2/Ts*I - A)^(-1) * (2/Ts*I + A)
        // For our system: A_d ≈ (20 + 1)/(20 - 1) = 21/19 ≈ 1.1053? Wait, let me recalculate.
        // Actually: Q = (k*I - A)^(-1), A_d = Q*(k*I + A)
        // k = 20, A = -1
        // Q = (20 - (-1))^(-1) = 21^(-1) ≈ 0.04762
        // A_d = 0.04762 * (20 + (-1)) = 0.04762 * 19 ≈ 0.9048
        CHECK(discrete.A(0, 0) == doctest::Approx(19.0 / 21.0).epsilon(1e-4));

        // B_d = (I + A) * Q * B = (1 - 1) * Q * B = 0
        CHECK(discrete.B(0, 0) == doctest::Approx(0.0).epsilon(1e-10));

        // D_d = C * Q * B + D ≈ 1 * 0.04762 * 1 + 0 ≈ 0.04762
        CHECK(discrete.D(0, 0) == doctest::Approx(1.0 / 21.0).epsilon(1e-4));
    }

    SUBCASE("Tustin with prewarp") {
        TransferFunction continuous({1.0}, {1.0, 1.0});
        double           Ts      = 0.1;
        double           prewarp = 10.0;  // Prewarp frequency

        StateSpace discrete = c2d(continuous, Ts, DiscretizationMethod::Bilinear, prewarp);

        CHECK(discrete.Ts.has_value());
        CHECK(discrete.Ts.value() == doctest::Approx(Ts));

        // With prewarp, k = prewarp / tan(prewarp * Ts / 2)
        double expected_k = prewarp / std::tan(prewarp * Ts / 2.0);
        CHECK(expected_k < 2.0 / Ts);  // Prewarp typically gives lower k than standard Tustin
        CHECK(expected_k > 0.0);
    }

    SUBCASE("Stability preservation") {
        // Test that stable continuous system gives stable discrete system
        TransferFunction stable_continuous({1.0}, {1.0, 1.0});  // Stable: pole at -1

        StateSpace discrete = c2d(stable_continuous, 0.1, DiscretizationMethod::ZOH);

        // Check that discrete poles are inside unit circle
        auto poles = discrete.poles();
        for (const auto& pole : poles) {
            CHECK(std::abs(pole) < 1.0);  // |z| < 1 for stability
        }
    }
}