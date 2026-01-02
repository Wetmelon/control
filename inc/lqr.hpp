#pragma once

#include <cstddef>

#include "control_design.hpp"
#include "kalman.hpp"
#include "matrix.hpp"

// ============================================================================
// Discrete-time controllers for embedded systems (ISR / RTOS)
// ============================================================================
// Use design:: namespace functions to create these from continuous or discrete models.
// Use online:: namespace for fast runtime linearization and system identification.

// ============================================================================
// LQR: Discrete Linear-Quadratic Regulator
// ============================================================================
template<size_t NX, size_t NU, typename T = double>
struct LQR {
private:
    Matrix<NU, NX, T> K{};

public:
    constexpr LQR() = default;
    constexpr explicit LQR(const Matrix<NU, NX, T>& K_) : K(K_) {}
    constexpr LQR(const design::LQRResult<NX, NU, T>& result) : K(result.K) {}

    template<typename U>
    constexpr LQR(const LQR<NX, NU, U>& other) : K(other.getK()) {}

    // Regulator: u = -K*x (drives x to zero)
    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NX, T>& x) const {
        return ColVec<NU, T>(-K * x);
    }

    // Servo: u = -K*(x - x_ref) (tracks state reference)
    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NX, T>& x, const ColVec<NX, T>& x_ref) const {
        return ColVec<NU, T>(-K * (x - x_ref));
    }

    [[nodiscard]] constexpr const Matrix<NU, NX, T>& getK() const { return K; }
};

// ============================================================================
// LQI: Linear-Quadratic-Integral (output tracking with integral action)
// ============================================================================
template<size_t NX, size_t NU, size_t NY, typename T = double>
struct LQI {
private:
    Matrix<NU, NX, T> Kx{};
    Matrix<NU, NY, T> Ki{};
    ColVec<NY, T>     xi{};

public:
    constexpr LQI() = default;
    constexpr LQI(const Matrix<NU, NX, T>& Kx_, const Matrix<NU, NY, T>& Ki_) : Kx(Kx_), Ki(Ki_) {}
    constexpr LQI(const design::LQIResult<NX, NU, NY, T>& result) : Kx(result.Kx), Ki(result.Ki) {}

    template<typename U>
    constexpr LQI(const LQI<NX, NU, NY, U>& other)
        : Kx(other.getKx()), Ki(other.getKi()), xi(other.getIntegratorState()) {}

    // u = -Kx*x - Ki*xi, where xi integrates (r - y)
    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NX, T>& x, const ColVec<NY, T>& r, const ColVec<NY, T>& y) {
        xi = ColVec<NY, T>(xi + (r - y));
        return ColVec<NU, T>(-Kx * x - Ki * xi);
    }

    constexpr void reset() { xi = ColVec<NY, T>{}; }

    [[nodiscard]] constexpr const Matrix<NU, NX, T>& getKx() const { return Kx; }
    [[nodiscard]] constexpr const Matrix<NU, NY, T>& getKi() const { return Ki; }
    [[nodiscard]] constexpr const ColVec<NY, T>&     getIntegratorState() const { return xi; }
};

// ============================================================================
// LQG: LQR + Kalman Filter
// ============================================================================
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct LQG {
    LQR<NX, NU, T>                      lqr{};
    KalmanFilter<NX, NU, NY, NW, NV, T> kf{};

    constexpr LQG() = default;

    constexpr LQG(const LQR<NX, NU, T>& lqr_, const KalmanFilter<NX, NU, NY, NW, NV, T>& kf_)
        : lqr(lqr_), kf(kf_) {}

    constexpr LQG(const design::LQGResult<NX, NU, NY, NW, NV, T>& result)
        : lqr(result.lqr), kf(result.kalman.sys, result.kalman.Q, result.kalman.R) {}

    template<typename U>
    constexpr LQG(const LQG<NX, NU, NY, NW, NV, U>& other) : lqr(other.lqr), kf(other.kf) {}

    constexpr void predict(const ColVec<NU, T>& u = ColVec<NU, T>{}) { kf.predict(u); }
    constexpr bool update(const ColVec<NY, T>& z, const ColVec<NU, T>& u = ColVec<NU, T>{}) { return kf.update(z, u); }

    [[nodiscard]] constexpr ColVec<NU, T> control() const { return lqr.control(kf.state()); }
    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NX, T>& x_ref) const { return lqr.control(kf.state(), x_ref); }
};

// ============================================================================
// LQGI: LQI + Kalman Filter (output tracking with integral action)
// ============================================================================
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct LQGI {
    LQI<NX, NU, NY, T>                  lqi{};
    KalmanFilter<NX, NU, NY, NW, NV, T> kf{};

    constexpr LQGI() = default;

    constexpr LQGI(const LQI<NX, NU, NY, T>& lqi_, const KalmanFilter<NX, NU, NY, NW, NV, T>& kf_)
        : lqi(lqi_), kf(kf_) {}

    constexpr LQGI(const design::LQGIResult<NX, NU, NY, NW, NV, T>& result)
        : lqi(result.lqi), kf(result.kalman.sys, result.kalman.Q, result.kalman.R) {}

    template<typename U>
    constexpr LQGI(const LQGI<NX, NU, NY, NW, NV, U>& other) : lqi(other.lqi), kf(other.kf) {}

    constexpr void predict(const ColVec<NU, T>& u = ColVec<NU, T>{}) { kf.predict(u); }
    constexpr bool update(const ColVec<NY, T>& z, const ColVec<NU, T>& u = ColVec<NU, T>{}) { return kf.update(z, u); }

    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NY, T>& r) { return lqi.control(kf.state(), r, kf.innovation()); }

    constexpr void reset() { lqi.reset(); }
};
