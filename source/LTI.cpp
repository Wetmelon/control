#include "LTI.hpp"

#include "unsupported/Eigen/MatrixFunctions"
#include <matplot/matplot.h>

namespace control {
StateSpace StateSpace::c2d(const double Ts, const Method method, const std::optional<double> prewarp) const {
    if (this->Ts.has_value()) {
        return *this;
    }

    const auto I    = decltype(A)::Identity(A.rows(), A.cols());
    const auto E    = (A * Ts).exp();
    const auto Ainv = A.inverse();
    const auto I1   = Ainv * (E - I);
    const auto I2   = Ainv * (E * Ts - I1);

    switch (method) {
        case Method::ZOH: {
            return StateSpace{
                E,       // A
                I1 * B,  // B
                C,       // C
                D,       // D
                Ts,      // Ts
            };
        }
        case Method::FOH: {
            const auto Q = I1 - (I2 / Ts);
            const auto P = I1 - Q;
            return StateSpace{
                E,                  // A
                (P + (E * Q)) * B,  // B
                C,                  // C
                C * Q * B + D,      // D
                Ts,                 // Ts
            };
        }
        case Method::Tustin:  // Fallthrough
        case Method::Bilinear: {
            double k = 2.0 / Ts;
            if (prewarp.has_value()) {
                k = prewarp.value() / std::tan(prewarp.value() * Ts / 2.0);
            }

            const auto Q = (k * I - A).inverse();
            return StateSpace{
                Q * (k * I + A),  // A
                (I + A) * Q * B,  // B
                C,                // C
                C * Q * B + D,    // D
                Ts,               // Ts
            };
        }
        default:
            return *this;
    }
}

void plotFrequencyResponse(const FrequencyResponse &response) {
    using namespace matplot;
    
    // Create a tiled layout with 2 rows: magnitude and phase
    tiledlayout(2, 1);
    
    // Plot magnitude (top subplot)
    auto ax1 = nexttile();
    semilogx(ax1, response.freq, response.magnitude);
    ax1->xlabel("Frequency (Hz)");
    ax1->ylabel("Magnitude (dB)");
    ax1->title("Frequency Response - Magnitude");
    ax1->grid(on);
    
    // Plot phase (bottom subplot)
    auto ax2 = nexttile();
    semilogx(ax2, response.freq, response.phase);
    ax2->xlabel("Frequency (Hz)");
    ax2->ylabel("Phase (degrees)");
    ax2->title("Frequency Response - Phase");
    ax2->grid(on);
    
    show();
}
}  // namespace control