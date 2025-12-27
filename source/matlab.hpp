#pragma once

#include <cmath>
#include <vector>

#include "LTI.hpp"
#include "types.hpp"
#include "utility.hpp"

namespace control {
namespace matlab {
    /*
        Function List:
        System Representations:
            - tf
            - ss
            - zpk

        Transfer Function Operations:
            - c2d
            - ss2tf
            - tf2ss
            - tfdata
            - zpkdata

        Interconnections:
            - series
            - parallel
            - feedback
            - negate
            - connect
            - append

        Gain and Dynamics:
            - dcgain
            - pole
            - zero
            - damp
            - pzmap

        Time Domain Simulations:
            - step
            - impulse
            - lsim
            - initial
            - stepinfo

        Frequency Domain Analysis:
            - bode
            - nyquist
            - margin
            - nichols
            - ngrid
            - freqresp
            - evalfr

        Compensator Design:
            - rlocus
            - sisotool ?
            - place
            - lqr
            - dlqr
            - lqe
            - dlqe

        State-Space:
            - rss - Random state-space system
            - drss - Random discrete state-space system
            - ctrb - Controllability matrix
            - obsv - Observability matrix
            - gram - Controllability/Observability Gramian
            - lyap - Solve continuous Lyapunov equation
            - dlyap - Solve discrete Lyapunov equation
            - care
            - dare

        Utility:
            - linspace
            - logspace
            - mag2db
            - db2mag
            - rad2deg
            - deg2rad
    */

    /* System Representations */
    TransferFunction tf(const LTI& sys);
    TransferFunction tf(std::vector<double>&& num, std::vector<double>&& den, std::optional<double> Ts = std::nullopt);

    StateSpace ss(const LTI& ltiSys);
    StateSpace ss(Matrix&& A, Matrix&& B, Matrix&& C, Matrix&& D, std::optional<double> Ts = std::nullopt);

    ZeroPoleGain zpk(const LTI& ltiSys);
    ZeroPoleGain zpk(std::vector<Zero>&& zeros, std::vector<Pole>&& poles, double gain, std::optional<double> Ts = std::nullopt);

    /* Transfer Function Operations */
    constexpr static StateSpace c2d(const StateSpace&     sys,
                                    double                Ts,
                                    DiscretizationMethod  method  = DiscretizationMethod::ZOH,
                                    std::optional<double> prewarp = std::nullopt) {
        return sys.discretize(Ts, method, prewarp);
    }

    constexpr static TransferFunction ss2tf(const StateSpace& sys, int output_idx, int input_idx) {
        return tf(sys, output_idx, input_idx);
    }

    constexpr static TransferFunction ss2tf(const StateSpace& sys) {
        return tf(sys);
    }

    constexpr static StateSpace tf2ss(const TransferFunction& tf) {
        return ss(tf);
    }

    constexpr static std::pair<std::vector<double>, std::vector<double>> tfdata(const TransferFunction& tf) {
        return {tf.num, tf.den};
    }

    constexpr static std::tuple<std::vector<Zero>, std::vector<Pole>, double> zpkdata(const LTI& sys) {
        ZeroPoleGain zpk_sys = zpk(sys);
        return {sys.zeros(), sys.poles(), zpk_sys.gain()};
    }

    /* Interconnections */
    LTI series(const LTI& sys1, const LTI& sys2, bool connect_direct_feedthrough = true);
    LTI parallel(const LTI& sys1, const LTI& sys2);
    LTI feedback(const LTI& sys1, const LTI& sys2, int sign = -1);
    LTI negate(const LTI& sys);
    LTI append(const std::vector<LTI*>& systems);
    LTI connect(const LTI&              sys,
                const std::vector<int>& inputs,
                const std::vector<int>& outputs,
                int                     num_internal = 0);

    /* Utility Functions*/
    constexpr static double eps = 1e-12;
    constexpr static double inf = std::numeric_limits<double>::infinity();
    constexpr static double nan = std::numeric_limits<double>::quiet_NaN();

    constexpr static std::vector<double> linspace(double start, double end, size_t num) {
        std::vector<double> result;
        result.reserve(num);
        if (num == 1) {
            result.push_back(start);
        } else {
            double step = (end - start) / static_cast<double>(num - 1);
            for (size_t i = 0; i < num; ++i) {
                result.push_back(start + i * step);
            }
        }
        return result;
    }

    constexpr static std::vector<double> linspace(const std::pair<double, double>& span, size_t num) {
        return linspace(span.first, span.second, num);
    }

    constexpr static std::vector<double> logspace(double start_exp, double end_exp, size_t num) {
        std::vector<double> result;
        result.reserve(num);
        if (num == 1) {
            result.push_back(std::pow(10.0, start_exp));
        } else {
            double step = (end_exp - start_exp) / static_cast<double>(num - 1);
            for (size_t i = 0; i < num; ++i) {
                result.push_back(std::pow(10.0, start_exp + i * step));
            }
        }
        return result;
    }

    constexpr static std::vector<double> logspace(const std::pair<double, double>& span, size_t num) {
        return logspace(span.first, span.second, num);
    }

    constexpr static double mag2db(double mag) {
        return 20.0 * std::log10(mag);
    }

    constexpr static double db2mag(double db) {
        return std::pow(10.0, db / 20.0);
    }

    constexpr static double rad2deg(double rad) {
        return rad * (180.0 / std::numbers::pi);
    }

    constexpr static double deg2rad(double deg) {
        return deg * (std::numbers::pi / 180.0);
    }

}  // namespace matlab
}  // namespace control