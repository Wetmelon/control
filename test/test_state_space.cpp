#include "control.hpp"
#include "doctest.h"

using namespace control;

TEST_CASE("StateSpace Construction and Validation") {
    SUBCASE("Valid construction") {
        Matrix A = Matrix::Constant(2, 2, -1.0);
        Matrix B = Matrix::Constant(2, 1, 1.0);
        Matrix C = Matrix::Constant(1, 2, 1.0);
        Matrix D = Matrix::Constant(1, 1, 0.0);

        StateSpace ss(A, B, C, D);
        CHECK(ss.A.rows() == 2);
        CHECK(ss.A.cols() == 2);
        CHECK(ss.B.rows() == 2);
        CHECK(ss.B.cols() == 1);
        CHECK(ss.C.rows() == 1);
        CHECK(ss.C.cols() == 2);
        CHECK(ss.D.rows() == 1);
        CHECK(ss.D.cols() == 1);
    }

    SUBCASE("Move construction") {
        Matrix A = Matrix::Constant(1, 1, -1.0);
        Matrix B = Matrix::Constant(1, 1, 1.0);
        Matrix C = Matrix::Constant(1, 1, 1.0);
        Matrix D = Matrix::Constant(1, 1, 0.0);

        StateSpace ss1(A, B, C, D);
        StateSpace ss2(std::move(ss1));

        CHECK(ss2.A.rows() == 1);
        CHECK(ss2.A.cols() == 1);
        CHECK(ss2.B.rows() == 1);
        CHECK(ss2.B.cols() == 1);
    }

    SUBCASE("Copy construction") {
        StateSpace ss1{
            Matrix::Constant(1, 1, -1.0),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Constant(1, 1, 0.0),
        };
        StateSpace ss2 = ss1;

        CHECK(ss2.A == ss1.A);
        CHECK(ss2.B == ss1.B);
        CHECK(ss2.C == ss1.C);
        CHECK(ss2.D == ss1.D);
    }

    SUBCASE("Discrete-time system") {
        StateSpace ss{
            Matrix::Constant(1, 1, 0.9),
            Matrix::Constant(1, 1, 0.1),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Constant(1, 1, 0.0),
            0.1  // Ts
        };
        CHECK(ss.isDiscrete());
        CHECK_FALSE(ss.isContinuous());
        CHECK(ss.Ts.value() == doctest::Approx(0.1));
    }

    SUBCASE("Invalid matrix dimensions throw") {
        Matrix A = Matrix::Constant(2, 2, -1.0);
        Matrix B = Matrix::Constant(1, 1, 1.0);  // Wrong size
        Matrix C = Matrix::Constant(1, 2, 1.0);
        Matrix D = Matrix::Constant(1, 1, 0.0);

        CHECK_THROWS_AS(StateSpace(A, B, C, D), std::invalid_argument);
    }
}

TEST_CASE("StateSpace Poles and Zeros") {
    SUBCASE("Simple SISO system poles") {
        // System with A = -2 has pole at s = -2
        StateSpace sys{
            Matrix::Constant(1, 1, -2.0),  // A
            Matrix::Constant(1, 1, 1.0),   // B
            Matrix::Constant(1, 1, 1.0),   // C
            Matrix::Constant(1, 1, 0.0)    // D
        };

        auto poles_vec = sys.poles();

        CHECK(poles_vec.size() == 1);
        CHECK(poles_vec[0].real() == doctest::Approx(-2.0).epsilon(1e-6));
    }

    SUBCASE("Second-order system poles") {
        // System with diagonal A = [-1, 0; 0, -2]
        StateSpace sys{
            (Matrix(2, 2) << -1.0, 0.0, 0.0, -2.0).finished(),  // A
            Matrix::Constant(2, 1, 1.0),                        // B
            Matrix::Constant(1, 2, 1.0),                        // C
            Matrix::Zero(1, 1)                                  // D
        };

        auto poles_vec = sys.poles();

        CHECK(poles_vec.size() == 2);

        // Sort poles by real part
        std::vector<double> pole_reals = {poles_vec[0].real(), poles_vec[1].real()};
        std::sort(pole_reals.begin(), pole_reals.end());

        CHECK(pole_reals[0] == doctest::Approx(-2.0).epsilon(1e-6));
        CHECK(pole_reals[1] == doctest::Approx(-1.0).epsilon(1e-6));
    }

    SUBCASE("SISO system zeros (converts to TF internally)") {
        // System G(s) = C(sI-A)^(-1)B + D
        // With A=-2, B=1, C=-1, D=1: G(s) = -1/(s+2) + 1 = (s+1)/(s+2)
        // This has zero at s = -1, pole at s = -2
        StateSpace sys{
            Matrix::Constant(1, 1, -2.0),  // A = -2
            Matrix::Constant(1, 1, 1.0),   // B = 1
            Matrix::Constant(1, 1, -1.0),  // C = -1 (creates zero at -1)
            Matrix::Constant(1, 1, 1.0)    // D = 1
        };

        auto zeros_vec = sys.zeros();
        auto poles_vec = sys.poles();

        // Should have 1 zero at -1
        CHECK(zeros_vec.size() == 1);
        CHECK(zeros_vec[0].real() == doctest::Approx(-1.0).epsilon(1e-6));

        // Should have 1 pole at -2
        CHECK(poles_vec.size() == 1);
        CHECK(poles_vec[0].real() == doctest::Approx(-2.0).epsilon(1e-6));
    }

    SUBCASE("MIMO system throws on zeros()") {
        // 2x2 MIMO system
        StateSpace mimo_sys{
            Matrix::Identity(2, 2),  // A
            Matrix::Identity(2, 2),  // B (2 inputs)
            Matrix::Identity(2, 2),  // C (2 outputs)
            Matrix::Zero(2, 2)       // D
        };

        // zeros() only works for SISO
        CHECK_THROWS_AS(mimo_sys.zeros(), std::invalid_argument);
    }

    SUBCASE("Oscillator poles (complex conjugate)") {
        // Simple harmonic oscillator: x'' + ω²x = 0
        // A = [0, 1; -ω², 0], poles at ±jω
        const double omega = 2.0;
        StateSpace   sys{
            (Matrix(2, 2) << 0.0, 1.0, -omega * omega, 0.0).finished(),
            Matrix::Constant(2, 1, 0.0),
            Matrix::Constant(1, 2, 1.0),
            Matrix::Zero(1, 1),
        };

        auto poles_vec = sys.poles();

        CHECK(poles_vec.size() == 2);

        // Poles should be purely imaginary (±2j)
        CHECK(std::abs(poles_vec[0].real()) < 1e-10);
        CHECK(std::abs(poles_vec[1].real()) < 1e-10);

        // Magnitude of imaginary parts should sum to 2*omega
        double imag_sum = std::abs(poles_vec[0].imag()) + std::abs(poles_vec[1].imag());
        CHECK(imag_sum == doctest::Approx(2.0 * omega).epsilon(1e-6));
    }

    SUBCASE("Stability check using poles") {
        // Stable system: all poles have negative real parts
        StateSpace stable_sys{
            Matrix::Constant(1, 1, -1.0),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Zero(1, 1),
        };

        CHECK(stable_sys.is_stable());

        auto poles = stable_sys.poles();
        for (const auto& pole : poles) {
            CHECK(pole.real() < 0.0);
        }
    }

    SUBCASE("Discrete system poles") {
        // Discrete system with pole at z = 0.8
        StateSpace sys{
            Matrix::Constant(1, 1, 0.8),  // A = 0.8
            Matrix::Constant(1, 1, 1.0),  // B = 1
            Matrix::Constant(1, 1, 1.0),  // C = 1
            Matrix::Zero(1, 1),           // D = 0
            0.1                           // Ts = 0.1
        };

        auto poles_vec = sys.poles();

        CHECK(poles_vec.size() == 1);
        CHECK(poles_vec[0].real() == doctest::Approx(0.8).epsilon(1e-6));
        CHECK(sys.isDiscrete());
    }

    SUBCASE("Discrete system zeros") {
        // Discrete SISO system: equivalent to G(z) = (z - 0.5)/(z - 0.8)
        // A = 0.8, B = 0.2, C = 1, D = 0.5
        // This gives G(z) = [0.5 + 0.2/(z-0.8)] = (0.5z - 0.4 + 0.2)/(z - 0.8) = (0.5z - 0.2)/(z - 0.8)
        // Wait, let me recalculate properly...
        // Actually, let's use a simpler example
        StateSpace sys{
            Matrix::Constant(1, 1, 0.8),  // A = 0.8
            Matrix::Constant(1, 1, 0.2),  // B = 0.2
            Matrix::Constant(1, 1, 1.0),  // C = 1
            Matrix::Constant(1, 1, 0.5),  // D = 0.5
            0.1                           // Ts = 0.1
        };

        auto zeros_vec = sys.zeros();
        auto poles_vec = sys.poles();

        // Should have 1 pole at 0.8
        CHECK(poles_vec.size() == 1);
        CHECK(poles_vec[0].real() == doctest::Approx(0.8).epsilon(1e-6));

        // Should have 1 zero (computed from transfer function)
        CHECK(zeros_vec.size() == 1);
    }

    SUBCASE("Second-order system with repeated poles") {
        // System with repeated eigenvalue at -1
        // A = [-1, 1; 0, -1], eigenvalues are -1, -1
        StateSpace sys{
            (Matrix(2, 2) << -1.0, 1.0, 0.0, -1.0).finished(),
            Matrix::Constant(2, 1, 1.0),
            Matrix::Constant(1, 2, 1.0),
            Matrix::Zero(1, 1),
        };

        auto poles_vec = sys.poles();

        CHECK(poles_vec.size() == 2);
        CHECK(poles_vec[0].real() == doctest::Approx(-1.0).epsilon(1e-6));
        CHECK(poles_vec[1].real() == doctest::Approx(-1.0).epsilon(1e-6));
    }

    SUBCASE("Unstable system poles") {
        // Unstable system: at least one pole with non-negative real part
        StateSpace unstable_sys{
            Matrix::Constant(1, 1, 1.0),  // Positive eigenvalue
            Matrix::Constant(1, 1, 1.0),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Zero(1, 1),
        };

        CHECK_FALSE(unstable_sys.is_stable());

        auto poles = unstable_sys.poles();
        CHECK(poles[0].real() > 0.0);
    }
}

TEST_CASE("StateSpace Time Domain Analysis") {
    SUBCASE("Step response - continuous, default parameters") {
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),  // A = -1
            Matrix::Constant(1, 1, 1.0),   // B = 1
            Matrix::Constant(1, 1, 1.0),   // C = 1
            Matrix::Constant(1, 1, 0.0)    // D = 0
        };

        auto step_resp = sys.step();  // Use defaults
        CHECK(step_resp.time.size() > 0);
        CHECK(step_resp.output.size() == step_resp.time.size());

        // For 1/(s+1), steady-state should be 1
        CHECK(step_resp.output.back()(0, 0) == doctest::Approx(1.0).epsilon(0.01));
    }

    SUBCASE("Step response - continuous, custom time range") {
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),  // A = -1
            Matrix::Constant(1, 1, 1.0),   // B = 1
            Matrix::Constant(1, 1, 1.0),   // C = 1
            Matrix::Constant(1, 1, 0.0)    // D = 0
        };

        auto step_resp = sys.step(2.0, 8.0);  // Start at t=2, end at t=8 (longer time)
        CHECK(step_resp.time.size() > 0);
        CHECK(step_resp.output.size() == step_resp.time.size());
        CHECK(step_resp.time.front() >= 2.0);
        CHECK(step_resp.time.back() <= 8.0);

        // Steady-state should be 1
        CHECK(step_resp.output.back()(0, 0) == doctest::Approx(1.0).epsilon(0.01));
    }

    SUBCASE("Step response - continuous, custom step amplitude") {
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),  // A = -1
            Matrix::Constant(1, 1, 1.0),   // B = 1
            Matrix::Constant(1, 1, 1.0),   // C = 1
            Matrix::Constant(1, 1, 0.0)    // D = 0
        };

        ColVec uStep(1);
        uStep << 3.0;  // Step of amplitude 3

        auto step_resp = sys.step(0.0, 5.0, uStep);
        CHECK(step_resp.time.size() > 0);
        CHECK(step_resp.output.size() == step_resp.time.size());

        // For 1/(s+1) with step amplitude 3, steady-state should be 3
        CHECK(step_resp.output.back()(0, 0) == doctest::Approx(3.0).epsilon(0.01));
    }

    SUBCASE("Step response - continuous, MIMO system") {
        // 2x2 MIMO system
        StateSpace mimo_sys{
            (Matrix(2, 2) << -1.0, 0.0, 0.0, -2.0).finished(),  // A (diagonal)
            (Matrix(2, 2) << 1.0, 0.0, 0.0, 1.0).finished(),    // B
            (Matrix(2, 2) << 1.0, 0.0, 0.0, 2.0).finished(),    // C
            Matrix::Zero(2, 2)                                  // D
        };

        ColVec uStep(2);
        uStep << 1.0, 2.0;  // Step inputs [1, 2]

        auto step_resp = mimo_sys.step(0.0, 5.0, uStep);
        CHECK(step_resp.time.size() > 0);
        CHECK(step_resp.output.size() == step_resp.time.size());
        CHECK(step_resp.output[0].rows() == 2);  // 2 outputs

        // Check steady-state values
        auto final_output = step_resp.output.back();
        CHECK(final_output.rows() == 2);
        CHECK(final_output.cols() == 1);
        // Output 1 should reach steady-state of 1 (from input 1 through first subsystem)
        // Output 2 should reach steady-state of 2 (from input 2 through second subsystem with gain 2)
        CHECK(final_output(0, 0) == doctest::Approx(1.0).epsilon(0.05));
        CHECK(final_output(1, 0) == doctest::Approx(2.0).epsilon(0.05));
    }

    SUBCASE("Step response - continuous, system with D term") {
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),  // A = -1
            Matrix::Constant(1, 1, 1.0),   // B = 1
            Matrix::Constant(1, 1, 1.0),   // C = 1
            Matrix::Constant(1, 1, 0.5)    // D = 0.5 (direct feedthrough)
        };

        auto step_resp = sys.step(0.0, 5.0);
        CHECK(step_resp.time.size() > 0);
        CHECK(step_resp.output.size() == step_resp.time.size());

        // Initial output should be D*u = 0.5 (due to direct feedthrough)
        CHECK(step_resp.output[0](0, 0) == doctest::Approx(0.5).epsilon(0.1));

        // Steady-state should be 1.5 (DC gain = C*(-A)^(-1)*B + D = 1 + 0.5 = 1.5)
        CHECK(step_resp.output.back()(0, 0) == doctest::Approx(1.5).epsilon(0.01));
    }

    SUBCASE("Step response - discrete") {
        StateSpace sys{
            Matrix::Constant(1, 1, 0.9),  // A
            Matrix::Constant(1, 1, 0.1),  // B
            Matrix::Constant(1, 1, 1.0),  // C
            Matrix::Constant(1, 1, 0.0),  // D
            0.1                           // Ts
        };

        auto step_resp = sys.step(0.0, 1.0);
        CHECK(step_resp.time.size() > 0);
        CHECK(step_resp.output.size() == step_resp.time.size());

        // Check that time points are at discrete intervals
        for (size_t i = 1; i < step_resp.time.size(); ++i) {
            double dt = step_resp.time[i] - step_resp.time[i - 1];
            CHECK(dt == doctest::Approx(0.1).epsilon(0.01));
        }
    }

    SUBCASE("Step response - discrete, custom step amplitude") {
        StateSpace sys{
            Matrix::Constant(1, 1, 0.9),  // A
            Matrix::Constant(1, 1, 0.1),  // B
            Matrix::Constant(1, 1, 1.0),  // C
            Matrix::Constant(1, 1, 0.0),  // D
            0.1                           // Ts
        };

        ColVec uStep(1);
        uStep << 2.0;  // Step of amplitude 2

        auto step_resp = sys.step(0.0, 2.0, uStep);  // Longer time for discrete system
        CHECK(step_resp.time.size() > 0);
        CHECK(step_resp.output.size() == step_resp.time.size());

        // Steady-state should be 2.0 * DC_gain, where DC_gain = 1/(1-0.9) = 10, so 20.0
        CHECK(step_resp.output.back()(0, 0) == doctest::Approx(20.0).epsilon(1.0));
    }

    SUBCASE("Impulse response - continuous") {
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Zero(1, 1),
        };

        auto impulse_resp = sys.impulse(0.0, 2.0);
        CHECK(impulse_resp.time.size() > 0);
        CHECK(impulse_resp.output.size() == impulse_resp.time.size());
    }

    SUBCASE("Impulse response - discrete") {
        StateSpace sys{
            Matrix::Constant(1, 1, 0.9),
            Matrix::Constant(1, 1, 0.1),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Zero(1, 1),
            0.1,
        };

        auto impulse_resp = sys.impulse(0.0, 1.0);
        CHECK(impulse_resp.time.size() > 0);
        CHECK(impulse_resp.output.size() == impulse_resp.time.size());
    }
}

TEST_CASE("StateSpace Frequency Domain Analysis") {
    SUBCASE("Frequency response") {
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),  // A
            Matrix::Constant(1, 1, 1.0),   // B
            Matrix::Constant(1, 1, 1.0),   // C
            Matrix::Zero(1, 1)             // D
        };  // Represents 1/(s+1)

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
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),  // A
            Matrix::Constant(1, 1, 2.0),   // B
            Matrix::Constant(1, 1, 1.0),   // C
            Matrix::Zero(1, 1)             // D
        };  // Represents 2/(s+1)

        std::vector<double> freqs     = {0.0};
        auto                freq_resp = sys.freqresp(freqs);

        CHECK(freq_resp.response[0].real() == doctest::Approx(2.0).epsilon(1e-6));
        CHECK(freq_resp.response[0].imag() == doctest::Approx(0.0).epsilon(1e-6));
    }

    SUBCASE("Frequency response - high frequency") {
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),  // A
            Matrix::Constant(1, 1, 1.0),   // B
            Matrix::Constant(1, 1, 1.0),   // C
            Matrix::Zero(1, 1)             // D
        };  // Represents 1/(s+1)

        std::vector<double> freqs     = {1000.0};
        auto                freq_resp = sys.freqresp(freqs);

        // At high frequency, magnitude should be small
        double mag = std::abs(freq_resp.response[0]);
        CHECK(mag < 0.01);
    }

    SUBCASE("Bode plot - continuous") {
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Zero(1, 1),
        };

        auto bode_resp = sys.bode(0.1, 10.0, 50);
        CHECK(bode_resp.freq.size() > 0);
        CHECK(bode_resp.magnitude.size() == bode_resp.freq.size());
        CHECK(bode_resp.phase.size() == bode_resp.freq.size());
    }

    SUBCASE("Bode plot - discrete") {
        StateSpace sys{
            Matrix::Constant(1, 1, 0.9),
            Matrix::Constant(1, 1, 0.1),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Zero(1, 1),
            0.1,
        };

        auto bode_resp = sys.bode(0.1, 10.0, 50);
        CHECK(bode_resp.freq.size() > 0);
        CHECK(bode_resp.magnitude.size() == bode_resp.freq.size());
        CHECK(bode_resp.phase.size() == bode_resp.freq.size());
    }

    SUBCASE("Nyquist plot") {
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Zero(1, 1),
        };

        auto nyquist_resp = sys.nyquist(0.1, 10.0, 50);
        CHECK(nyquist_resp.freq.size() > 0);
        CHECK(nyquist_resp.response.size() == nyquist_resp.freq.size());
    }

    SUBCASE("Root locus") {
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Zero(1, 1),
        };

        auto rlocus_resp = sys.rlocus(0.0, 10.0, 50);
        CHECK(rlocus_resp.gains.size() > 0);
        CHECK(rlocus_resp.branches.size() == 1);
        CHECK(rlocus_resp.branches[0].size() == rlocus_resp.gains.size());
    }

    SUBCASE("Stability margins - first order system") {
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),  // A
            Matrix::Constant(1, 1, 1.0),   // B
            Matrix::Constant(1, 1, 1.0),   // C
            Matrix::Zero(1, 1)             // D
        };  // Represents 1/(s+1)

        auto margin_info = sys.margin();

        // For 1/(s+1), gain margin should be infinite (no phase crossover in typical range)
        // Phase margin should be positive
        CHECK(margin_info.phaseMargin > 0.0);
        CHECK(margin_info.gainMargin >= 0.0);  // Could be infinite
        CHECK(margin_info.gainCrossover >= 0.0);
        CHECK(margin_info.phaseCrossover >= 0.0);
    }

    SUBCASE("Stability margins - second order system") {
        // Second order system: s^2 + 0.1s + 1 in denominator
        // A = [0, 1; -1, -0.1], B = [0; 1], C = [1, 0], D = 0
        StateSpace sys{
            (Matrix(2, 2) << 0.0, 1.0, -1.0, -0.1).finished(),
            (Matrix(2, 1) << 0.0, 1.0).finished(),
            (Matrix(1, 2) << 1.0, 0.0).finished(),
            Matrix::Zero(1, 1)};

        auto margin_info = sys.margin();

        // Should have finite gain and phase margins
        CHECK(margin_info.phaseMargin > 0.0);
        CHECK(margin_info.gainMargin > 0.0);
        CHECK(margin_info.gainCrossover > 0.0);
        CHECK(margin_info.phaseCrossover > 0.0);
    }

    SUBCASE("Stability margins - discrete system") {
        StateSpace sys{
            Matrix::Constant(1, 1, 0.5),  // A (pole at 0.5)
            Matrix::Constant(1, 1, 1.0),  // B
            Matrix::Constant(1, 1, 1.0),  // C
            Matrix::Zero(1, 1),           // D
            0.1                           // Ts
        };

        auto margin_info = sys.margin();

        // Should compute margins for discrete system
        CHECK(margin_info.gainCrossover >= 0.0);
        CHECK(margin_info.phaseCrossover >= 0.0);
    }
}

TEST_CASE("StateSpace Discretization") {
    StateSpace continuous{
        Matrix::Constant(1, 1, -1.0),
        Matrix::Constant(1, 1, 1.0),
        Matrix::Constant(1, 1, 1.0),
        Matrix::Zero(1, 1),
    };

    SUBCASE("ZOH discretization") {
        auto discrete = continuous.discretize(0.1, DiscretizationMethod::ZOH);
        CHECK(discrete.isDiscrete());
        CHECK(discrete.Ts.value() == doctest::Approx(0.1));
        CHECK(discrete.A.rows() == 1);
        CHECK(discrete.B.rows() == 1);
    }

    SUBCASE("FOH discretization") {
        auto discrete = continuous.discretize(0.1, DiscretizationMethod::FOH);
        CHECK(discrete.isDiscrete());
        CHECK(discrete.Ts.value() == doctest::Approx(0.1));
    }

    SUBCASE("Tustin discretization") {
        auto discrete = continuous.discretize(0.1, DiscretizationMethod::Tustin);
        CHECK(discrete.isDiscrete());
        CHECK(discrete.Ts.value() == doctest::Approx(0.1));
    }

    SUBCASE("Tustin with prewarp") {
        auto discrete = continuous.discretize(0.1, DiscretizationMethod::Tustin, 1.0);
        CHECK(discrete.isDiscrete());
        CHECK(discrete.Ts.value() == doctest::Approx(0.1));
    }

    SUBCASE("Cannot discretize discrete system") {
        StateSpace already_discrete{
            Matrix::Constant(1, 1, 0.9),
            Matrix::Constant(1, 1, 0.1),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Zero(1, 1),
            0.1,
        };

        CHECK_THROWS_AS(already_discrete.discretize(0.2), std::runtime_error);
    }
}

TEST_CASE("StateSpace to TransferFunction Conversion") {
    SUBCASE("SISO system conversion") {
        // Create a simple SISO system: G(s) = 1/(s+1)
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),  // A = -1
            Matrix::Constant(1, 1, 1.0),   // B = 1
            Matrix::Constant(1, 1, 1.0),   // C = 1
            Matrix::Constant(1, 1, 0.0)    // D = 0
        };

        // Convert to transfer function
        auto tf_sys = tf(sys);

        // Should get num=[1], den=[1, 1] representing 1/(s+1)
        CHECK(tf_sys.num.size() == 1);
        CHECK(tf_sys.den.size() == 2);
        CHECK(tf_sys.num[0] == doctest::Approx(1.0));
        CHECK(tf_sys.den[0] == doctest::Approx(1.0));
        CHECK(tf_sys.den[1] == doctest::Approx(1.0));
    }

    SUBCASE("SISO system with D term") {
        // Create a system with feedthrough: G(s) = (s+1)/(s+2)
        StateSpace sys{
            Matrix::Constant(1, 1, -2.0),  // A = -2
            Matrix::Constant(1, 1, 1.0),   // B = 1
            Matrix::Constant(1, 1, -1.0),  // C = -1
            Matrix::Constant(1, 1, 1.0)    // D = 1
        };

        // Convert to transfer function
        auto tf_sys = tf(sys);

        // Check dimensions
        CHECK(tf_sys.num.size() >= 1);
        CHECK(tf_sys.den.size() >= 1);

        // Verify it's normalized
        CHECK(tf_sys.den[0] == doctest::Approx(1.0));
    }

    SUBCASE("Extract SISO from MIMO using indices") {
        // Create a 2x2 MIMO system
        StateSpace mimo_sys{
            (Matrix(2, 2) << -1.0, 0.0, 0.0, -2.0).finished(),  // A (diagonal)
            (Matrix(2, 2) << 1.0, 0.0, 0.0, 1.0).finished(),    // B
            (Matrix(2, 2) << 1.0, 0.0, 0.0, 2.0).finished(),    // C
            Matrix::Zero(2, 2)                                  // D
        };

        // Extract G_00: output 0, input 0
        auto tf_00 = tf(mimo_sys, 0, 0);
        CHECK(tf_00.num.size() >= 1);
        CHECK(tf_00.den.size() >= 1);

        // Extract G_11: output 1, input 1
        auto tf_11 = tf(mimo_sys, 1, 1);
        CHECK(tf_11.num.size() >= 1);
        CHECK(tf_11.den.size() >= 1);
    }

    SUBCASE("Invalid indices throw out_of_range") {
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),  // A
            Matrix::Constant(1, 1, 1.0),   // B
            Matrix::Constant(1, 1, 1.0),   // C
            Matrix::Constant(1, 1, 0.0)    // D
        };

        // Output index out of range
        CHECK_THROWS_AS(tf(sys, 1, 0), std::out_of_range);

        // Input index out of range
        CHECK_THROWS_AS(tf(sys, 0, 1), std::out_of_range);

        // Both out of range
        CHECK_THROWS_AS(tf(sys, 1, 1), std::out_of_range);

        // Negative indices
        CHECK_THROWS_AS(tf(sys, -1, 0), std::out_of_range);
        CHECK_THROWS_AS(tf(sys, 0, -1), std::out_of_range);
    }
}

TEST_CASE("LTI System Arithmetic Operations") {
    // Create test systems
    const StateSpace plant{
        Matrix::Constant(1, 1, -1.0),  // A
        Matrix::Constant(1, 1, 1.0),   // B
        Matrix::Constant(1, 1, 1.0),   // C
        Matrix::Constant(1, 1, 0.0)    // D
    };

    const StateSpace controller{
        Matrix::Zero(0, 0),          // A - No states (pure gain)
        Matrix::Zero(0, 1),          // B
        Matrix::Zero(1, 0),          // C
        Matrix::Constant(1, 1, 2.0)  // D
    };

    const StateSpace sensor{
        Matrix::Zero(0, 0),          // A
        Matrix::Zero(0, 1),          // B
        Matrix::Zero(1, 0),          // C
        Matrix::Constant(1, 1, 1.0)  // D
    };

    SUBCASE("Series Connection: Controller * Plant") {
        // Open-loop: L(s) = C(s) * G(s) = 2/(s+1)
        auto open_loop = controller * plant;

        // Check dimensions
        CHECK(open_loop.A.rows() == 1);
        CHECK(open_loop.A.cols() == 1);
        CHECK(open_loop.B.rows() == 1);
        CHECK(open_loop.B.cols() == 1);
        CHECK(open_loop.C.rows() == 1);
        CHECK(open_loop.C.cols() == 1);
        CHECK(open_loop.D.rows() == 1);
        CHECK(open_loop.D.cols() == 1);

        // Check values (2/(s+1) should have A=-1, B=2, C=1, D=0)
        CHECK(open_loop.A(0, 0) == doctest::Approx(-1.0));
        CHECK(open_loop.B(0, 0) == doctest::Approx(2.0));
        CHECK(open_loop.C(0, 0) == doctest::Approx(1.0));
        CHECK(open_loop.D(0, 0) == doctest::Approx(0.0));
    }

    SUBCASE("Parallel Connection: Plant + Plant") {
        // Sum: G1(s) + G2(s) should give 2/(s+1)
        auto parallel_sum = plant + plant;

        // Check dimensions (2 states from 2 systems)
        CHECK(parallel_sum.A.rows() == 2);
        CHECK(parallel_sum.A.cols() == 2);
        CHECK(parallel_sum.B.rows() == 2);
        CHECK(parallel_sum.C.cols() == 2);

        // Check A is block diagonal
        CHECK(parallel_sum.A(0, 0) == doctest::Approx(-1.0));
        CHECK(parallel_sum.A(1, 1) == doctest::Approx(-1.0));
        CHECK(parallel_sum.A(0, 1) == doctest::Approx(0.0));
        CHECK(parallel_sum.A(1, 0) == doctest::Approx(0.0));

        // Check C combines both outputs
        CHECK(parallel_sum.C(0, 0) == doctest::Approx(1.0));
        CHECK(parallel_sum.C(0, 1) == doctest::Approx(1.0));

        // D should be sum (0 + 0 = 0)
        CHECK(parallel_sum.D(0, 0) == doctest::Approx(0.0));
    }

    SUBCASE("Parallel Subtraction: Plant - Plant") {
        // Difference: G1(s) - G2(s) should give 0
        auto parallel_diff = plant - plant;

        // Check dimensions
        CHECK(parallel_diff.A.rows() == 2);
        CHECK(parallel_diff.A.cols() == 2);

        // Check A is block diagonal
        CHECK(parallel_diff.A(0, 0) == doctest::Approx(-1.0));
        CHECK(parallel_diff.A(1, 1) == doctest::Approx(-1.0));
        CHECK(parallel_diff.A(0, 1) == doctest::Approx(0.0));
        CHECK(parallel_diff.A(1, 0) == doctest::Approx(0.0));

        // Check C has negation on second system
        CHECK(parallel_diff.C(0, 0) == doctest::Approx(1.0));
        CHECK(parallel_diff.C(0, 1) == doctest::Approx(-1.0));

        // D should be difference (0 - 0 = 0)
        CHECK(parallel_diff.D(0, 0) == doctest::Approx(0.0));
    }

    SUBCASE("Negative Feedback: Unity Feedback on Open Loop") {
        // First create open loop
        auto open_loop = controller * plant;

        // Closed-loop with unity feedback: T(s) = L(s) / (1 + L(s))
        // For L(s) = 2/(s+1), T(s) = 2/(s+3)
        auto closed_loop = feedback(open_loop, sensor, -1);

        // Check dimensions (1 state from open_loop, 0 from sensor)
        CHECK(closed_loop.A.rows() == 1);
        CHECK(closed_loop.A.cols() == 1);

        // Check values for 2/(s+3): A=-3, B=2, C=1, D=0
        CHECK(closed_loop.A(0, 0) == doctest::Approx(-3.0));
        CHECK(closed_loop.B(0, 0) == doctest::Approx(2.0));
        CHECK(closed_loop.C(0, 0) == doctest::Approx(1.0));
        CHECK(closed_loop.D(0, 0) == doctest::Approx(0.0));
    }

    SUBCASE("Complete Control System: Feedback(Controller * Plant, Sensor)") {
        // T(s) = C(s)*G(s) / (1 + C(s)*G(s)*H(s))
        auto control_system = feedback(controller * plant, sensor, -1);

        // Check dimensions
        CHECK(control_system.A.rows() == 1);
        CHECK(control_system.A.cols() == 1);

        // Should be same as previous test (2/(s+3))
        CHECK(control_system.A(0, 0) == doctest::Approx(-3.0));
        CHECK(control_system.B(0, 0) == doctest::Approx(2.0));
        CHECK(control_system.C(0, 0) == doctest::Approx(1.0));
        CHECK(control_system.D(0, 0) == doctest::Approx(0.0));

        // Test step response properties
        auto step_resp = control_system.step(0.0, 5.0);
        CHECK(step_resp.time.size() > 0);
        CHECK(step_resp.output.size() == step_resp.time.size());

        // Check steady-state is approximately 2/3
        CHECK(step_resp.output.back()(0, 0) == doctest::Approx(2.0 / 3.0).epsilon(0.01));
    }
}

TEST_CASE("LTI System Type Safety") {
    const StateSpace continuous{
        Matrix::Constant(1, 1, -1.0),
        Matrix::Constant(1, 1, 1.0),
        Matrix::Constant(1, 1, 1.0),
        Matrix::Constant(1, 1, 0.0)};

    const StateSpace discrete{
        Matrix::Constant(1, 1, 0.9),
        Matrix::Constant(1, 1, 0.1),
        Matrix::Constant(1, 1, 1.0),
        Matrix::Constant(1, 1, 0.0),
        0.1  // Ts
    };

    SUBCASE("Same type operations compile") {
        // These should compile fine
        auto result1 = continuous * continuous;
        auto result2 = continuous + continuous;
        auto result3 = continuous - continuous;
        auto result4 = feedback(continuous, continuous);

        CHECK(result1.A.rows() > 0);  // Just verify it compiled
        CHECK(result2.A.rows() > 0);
        CHECK(result3.A.rows() > 0);
        CHECK(result4.A.rows() > 0);
    }

    SUBCASE("Discrete type operations compile") {
        // These should compile fine
        auto result1 = discrete * discrete;
        auto result2 = discrete + discrete;
        auto result3 = discrete - discrete;
        auto result4 = feedback(discrete, discrete);

        CHECK(result1.A.rows() > 0);  // Just verify it compiled
        CHECK(result2.A.rows() > 0);
        CHECK(result3.A.rows() > 0);
        CHECK(result4.A.rows() > 0);
    }
}

TEST_CASE("Integrators with LTI Systems") {
    // Simple decaying system: x' = -x, u = 1 -> solution is e^(-t)
    const Matrix A  = Matrix::Constant(1, 1, -1.0);
    const Matrix B  = Matrix::Constant(1, 1, 1.0);
    const Matrix x0 = Matrix::Zero(1, 1);
    const Matrix u  = Matrix::Constant(1, 1, 1.0);
    const double h  = 0.01;

    SUBCASE("ForwardEuler integrator") {
        ForwardEuler integrator;
        auto         result = integrator.evolve(A, B, x0, u, h);
        CHECK(result.x(0, 0) == doctest::Approx(0.01));  // x + h*(A*x + B*u) = 0 + 0.01*1
    }

    SUBCASE("BackwardEuler integrator") {
        BackwardEuler integrator;
        auto          result = integrator.evolve(A, B, x0, u, h);
        // (I - h*A)^-1 * (x + h*B*u) = (1 + 0.01)^-1 * 0.01 ≈ 0.0099
        CHECK(result.x(0, 0) == doctest::Approx(0.0099).epsilon(0.001));
    }

    SUBCASE("Trapezoidal integrator") {
        Trapezoidal integrator;
        auto        result = integrator.evolve(A, B, x0, u, h);
        // (I - h/2*A)^-1 * ((I + h/2*A)*x + h/2*B*u) ≈ 0.0099
        CHECK(result.x(0, 0) == doctest::Approx(0.0099).epsilon(0.001));
    }

    SUBCASE("RK45 integrator") {
        RK45 integrator;
        auto result = integrator.evolve(A, B, x0, u, h);
        // RK45 with smaller h should give more accurate result
        CHECK(result.x(0, 0) == doctest::Approx(0.00995).epsilon(0.001));
    }

    SUBCASE("Discrete integrator for LTI") {
        Discrete integrator;
        auto     result = integrator.evolve(A, B, x0, u);
        // Simply: A*x + B*u = -1*0 + 1*1 = 1
        CHECK(result.x(0, 0) == doctest::Approx(1.0));
    }
}

TEST_CASE("Integrators with Nonlinear ODEs") {
    // Simple test ODE: x' = -2*x, solution is e^(-2*t)
    auto f = [](double /*t*/, const Matrix& x) -> Matrix {
        return -2.0 * x;
    };

    const Matrix x0 = Matrix::Constant(1, 1, 1.0);
    const double t  = 0.0;
    const double h  = 0.1;

    SUBCASE("ForwardEuler on nonlinear ODE") {
        ForwardEuler integrator;
        auto         result = integrator.evolve(f, x0, t, h);
        // x' = -2*x -> x_new = x + h*f(t,x) = 1 + 0.1*(-2) = 0.8
        CHECK(result.x(0, 0) == doctest::Approx(0.8));
    }

    SUBCASE("RK45 on nonlinear ODE") {
        RK45 integrator;
        auto result = integrator.evolve(f, x0, t, h);
        // RK45 should be more accurate than ForwardEuler
        // Exact: e^(-2*h) ≈ 0.8187
        CHECK(result.x(0, 0) == doctest::Approx(std::exp(-2.0 * h)).epsilon(0.01));
    }

    SUBCASE("BackwardEuler on nonlinear ODE") {
        BackwardEuler integrator;
        auto          result = integrator.evolve(f, x0, t, h);
        // y = x + h*f(t+h, y) = 1 + 0.1*f(0.1, y)
        // Since f(t, x) = -2*x: y = 1 - 0.2*y -> y*(1 + 0.2) = 1 -> y ≈ 0.833
        CHECK(result.x(0, 0) == doctest::Approx(1.0 / 1.2).epsilon(0.001));
    }
}

TEST_CASE("solve() with generic nonlinear ODE") {
    // Van der Pol oscillator: x1' = x2, x2' = (1-x1^2)*x2 - x1
    auto fun = [](double /*t*/, const Matrix& x) -> Matrix {
        const double mu = 1.0;
        Matrix       dx = Matrix::Zero(2, 1);
        dx(0, 0)        = x(1, 0);
        dx(1, 0)        = mu * (1.0 - x(0, 0) * x(0, 0)) * x(1, 0) - x(0, 0);
        return dx;
    };

    const Matrix                    x0          = Matrix::Zero(2, 1);
    const auto                      x0_modified = [&x0]() { auto tmp = x0; tmp(0, 0) = 2.0; return tmp; }();
    const std::pair<double, double> t_span{0.0, 1.0};
    const std::vector<double>       t_eval = {0.0, 0.25, 0.5, 0.75, 1.0};

    SUBCASE("solve with RK45") {
        FixedStepSolver<RK45> solver(0.01);

        auto result = solver.solve(fun, x0_modified, t_span, t_eval);
        CHECK(result.success);
        CHECK(result.t.size() >= 5);
        CHECK(result.x.size() == result.t.size());
        CHECK(result.t.back() <= t_span.second);
    }

    SUBCASE("solve with ForwardEuler") {
        FixedStepSolver<ForwardEuler> solver(0.01);

        auto result = solver.solve(fun, x0_modified, t_span, t_eval);
        CHECK(result.success);
        CHECK(result.t.size() >= 5);
        CHECK(result.x.size() == result.t.size());
    }

    SUBCASE("solve with BackwardEuler") {
        FixedStepSolver<BackwardEuler> solver(0.01);

        auto result = solver.solve(fun, x0_modified, t_span, t_eval);
        CHECK(result.success);
        CHECK(result.t.size() >= 5);
    }
}

TEST_CASE("solve() with LTI constant input") {
    // Simple system: x' = -x, u = 1
    const Matrix                    A       = Matrix::Constant(1, 1, -1.0);
    const Matrix                    B       = Matrix::Constant(1, 1, 1.0);
    const ColVec                    x0      = ColVec::Zero(1);
    const ColVec                    u_const = ColVec::Constant(1, 1.0);
    const std::pair<double, double> t_span{0.0, 1.0};
    const std::vector<double>       t_eval = {0.0, 0.5, 1.0};

    SUBCASE("solve LTI with constant input (using Exact)") {
        ExactSolver solver;

        auto result = solver.solve(A, B, x0, u_const, t_span, t_eval);
        CHECK(result.success);
        CHECK(result.t.size() == 3);
        CHECK(result.x.size() == 3);
        // At t=0: x(0) = 0
        CHECK(result.x[0](0, 0) == doctest::Approx(0.0));
        // At t=1: x(1) = 1 - e^(-1) ≈ 0.632
        CHECK(result.x.back()(0, 0) == doctest::Approx(1.0 - std::exp(-1.0)).epsilon(0.001));
    }

    SUBCASE("solve LTI with Exact integrator") {
        ExactSolver solver;

        auto result = solver.solve(A, B, x0, u_const, t_span, t_eval);
        CHECK(result.success);
        CHECK(result.t.size() == 3);
    }
}

TEST_CASE("solve() with LTI time-varying input") {
    // Simple system: x' = -x + u(t), where u(t) = sin(t)
    const Matrix A  = Matrix::Constant(1, 1, -1.0);
    const Matrix B  = Matrix::Constant(1, 1, 1.0);
    const ColVec x0 = ColVec::Zero(1);

    auto u_func = [](double t) -> ColVec {
        return ColVec::Constant(1, std::sin(t));
    };

    const std::pair<double, double> t_span{0.0, 1.0};
    const std::vector<double>       t_eval = {0.0, 0.5, 1.0};

    SUBCASE("solve LTI with time-varying input and RK45") {
        FixedStepSolver<RK45> solver(0.01);

        auto ode_func = [&A, &B, &u_func](double t, const ColVec& x) -> ColVec {
            return A * x + B * u_func(t);
        };
        auto result = solver.solve(ode_func, x0, t_span, t_eval);
        CHECK(result.success);
        CHECK(result.t.size() >= 3);
        CHECK(result.x.size() == result.t.size());
    }

    SUBCASE("solve LTI with time-varying input and BackwardEuler") {
        FixedStepSolver<BackwardEuler> solver(0.01);

        auto ode_func = [&A, &B, &u_func](double t, const ColVec& x) -> ColVec {
            return A * x + B * u_func(t);
        };
        auto result = solver.solve(ode_func, x0, t_span, t_eval);
        CHECK(result.success);
        CHECK(result.t.size() >= 3);
    }

    SUBCASE("solve LTI with lambda for time-varying input") {
        FixedStepSolver<RK45> solver(0.01);

        auto u_lambda = [](double t) { return Matrix::Constant(1, 1, std::sin(t)); };
        auto ode_func = [&A, &B, &u_lambda](double t, const Matrix& x) -> Matrix {
            return A * x + B * u_lambda(t);
        };
        auto result = solver.solve(ode_func, x0, t_span, t_eval);
        CHECK(result.success);
        CHECK(result.t.size() >= 3);
    }
}

TEST_CASE("Solver API overload disambiguation") {
    // Test that all overloads can be called without ambiguity
    const Matrix A       = Matrix::Constant(1, 1, -1.0);
    const Matrix B       = Matrix::Constant(1, 1, 1.0);
    const Matrix x0      = Matrix::Zero(1, 1);
    const Matrix u_const = Matrix::Constant(1, 1, 1.0);

    const std::pair<double, double> t_span{0.0, 0.1};
    const std::vector<double>       t_eval;

    SUBCASE("LTI with const Matrix input") {
        // Matrix input - use ExactSolver with A, B matrices
        ExactSolver solver;

        auto result = solver.solve(A, B, x0, u_const, t_span, t_eval);
        CHECK(result.success);
    }

    SUBCASE("LTI with function input") {
        // std::function input
        auto u_func = [](double /*t*/) -> ColVec {
            return ColVec::Constant(1, 1.0);
        };
        FixedStepSolver<RK45> solver(0.01);

        auto ode_func = [&A, &B, &u_func](double t, const Matrix& x) -> Matrix {
            return A * x + B * u_func(t);
        };
        auto result = solver.solve(ode_func, x0, t_span, t_eval);
        CHECK(result.success);
    }

    SUBCASE("Generic ODE with explicit integrator and h") {
        // Generic nonlinear ODE solver with required parameters
        FixedStepSolver<RK45> solver(0.01);

        auto result = solver.solve(
            [](double /*t*/, const Matrix& x) -> Matrix { return -x; },
            x0, t_span, t_eval);
        CHECK(result.success);
    }
}