#pragma once

/**
 * @defgroup synthesis Controller Synthesis APIs
 * @brief High-level glue for design, analysis artifacts, and runtime bundles
 *
 * Provides thin orchestration helpers that connect existing primitives:
 * - design::discrete_lqg(...)
 * - closed-loop analysis models
 * - runtime LQG controller bundles
 * - optional SISO PR internal model composition
 */

#include "wet/controllers/lqg.hpp"
#include "wet/controllers/lqgi.hpp"
#include "wet/controllers/lqi.hpp"
#include "wet/controllers/pr.hpp"
#include "wet/systems/state_space.hpp"

namespace wet {
namespace design {

/**
 * @brief Analysis-oriented models produced from an LQG design
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct LQGAnalysisModels {
    StateSpace<NX, NU, NY, NW, NV, T>   plant{};                      ///< Original plant model
    StateSpace<NX, NU, NY, NW, NV, T>   state_feedback_closed_loop{}; ///< A_cl = A - B*K
    Matrix<NX, NX, T>                   observer_error_dynamics{};    ///< A_e = A - L*C
    StateSpace<2 * NX, NU, NY, 0, 0, T> augmented_closed_loop{};      ///< [x; xhat] closed-loop model
};

/**
 * @brief Runtime bundle for LQG control
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = float>
struct LQGRuntimeBundle {
    design::LQGResult<NX, NU, NY, NW, NV, T> design{};
    LQG<NX, NU, NY, NW, NV, T>               controller{};

    /**
     * @brief One discrete control tick: update estimator, compute control, predict next
     */
    [[nodiscard]] constexpr ColVec<NU, T> step(const ColVec<NY, T>& y) {
        controller.update(y);
        const auto u = controller.control();
        controller.predict(u);
        return u;
    }

    /**
     * @brief One discrete control tick with state-reference tracking
     */
    [[nodiscard]] constexpr ColVec<NU, T> step(const ColVec<NY, T>& y, const ColVec<NX, T>& x_ref) {
        controller.update(y);
        const auto u = controller.control(x_ref);
        controller.predict(u);
        return u;
    }
};

/**
 * @brief Synthesis artifact bundle: design + analysis models + runtime bundle
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double, typename TRuntime = float>
struct LQGArtifacts {
    design::LQGResult<NX, NU, NY, NW, NV, T>       design{};
    LQGAnalysisModels<NX, NU, NY, NW, NV, T>       models{};
    LQGRuntimeBundle<NX, NU, NY, NW, NV, TRuntime> runtime{};
    bool                                           success{false};
};

/**
 * @brief Runtime bundle for SISO LQG + PR internal model compensation
 */
template<size_t NX, size_t NW = NX, size_t NV = 1, typename T = float>
struct LQGPRRuntimeBundle {
    design::LQGResult<NX, 1, 1, NW, NV, T> lqg_design{};
    design::PRResult<T>                    pr_design{};
    LQG<NX, 1, 1, NW, NV, T>               lqg{};
    PRController<T>                        pr{};

    /**
     * @brief One control tick with PR internal model on tracking error
     */
    [[nodiscard]] constexpr ColVec<1, T> step(T reference, T measurement) {
        const ColVec<1, T> y{measurement};
        lqg.update(y);

        ColVec<1, T> u = lqg.control();
        u(0, 0) += pr.control(reference - measurement);

        lqg.predict(u);
        return u;
    }
};

/**
 * @brief Synthesis artifact bundle for SISO LQG + PR
 */
template<size_t NX, size_t NW = NX, size_t NV = 1, typename T = double, typename TRuntime = float>
struct LQGPRArtifacts {
    LQGArtifacts<NX, 1, 1, NW, NV, T, TRuntime> base{};
    LQGPRRuntimeBundle<NX, NW, NV, TRuntime>    runtime_pr{};
    bool                                        success{false};
};

/**
 * @brief Analysis-oriented models produced from an LQI design
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
struct LQIAnalysisModels {
    StateSpace<NX, NU, NY, NW, NV, T>    plant{};                       ///< Original plant model
    StateSpace<NX + NY, NU, NY, 0, 0, T> augmented_servo_closed_loop{}; ///< [x; xi] servo closed-loop model
};

/**
 * @brief Analysis-oriented models produced from an LQGI design
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct LQGIAnalysisModels {
    StateSpace<NX, NU, NY, NW, NV, T>    plant{};                       ///< Original plant model
    StateSpace<NX + NY, NU, NY, 0, 0, T> augmented_servo_closed_loop{}; ///< [x; xi] servo closed-loop model
    Matrix<NX, NX, T>                    observer_error_dynamics{};     ///< A_e = A - L*C
};

/**
 * @brief Runtime bundle for LQI servo control with integrated error state
 */
template<size_t NX, size_t NU, size_t NY, typename T = float>
struct LQIServoRuntimeBundle {
    design::LQIResult<NX, NU, NY, T> design{};
    LQI<NX, NU, NY, T>               controller{};
    ColVec<NY, T>                    xi{};

    /**
     * @brief One discrete servo control tick using full-state feedback
     *
     * @note The integral state uses the unit discrete integrator
     *       xi[k+1] = xi[k] + (r - y) to match the LQI design augmentation
     *       (no Ts scaling); the DARE gain Ki is computed for that convention.
     */
    [[nodiscard]] constexpr ColVec<NU, T> step(
        const ColVec<NX, T>& x,
        const ColVec<NY, T>& y,
        const ColVec<NY, T>& r
    ) {
        const auto e = r - y;
        xi += e;

        ColVec<NX + NY, T> x_aug{};
        x_aug.template block<NX, 1>(0, 0) = x;
        x_aug.template block<NY, 1>(NX, 0) = xi;
        return controller.control(x_aug);
    }

    /**
     * @brief SISO convenience overload
     */
    [[nodiscard]] constexpr ColVec<NU, T> step(
        const ColVec<NX, T>& x,
        T                    y,
        T                    r
    ) {
        static_assert(NY == 1, "Scalar step overload is available only for SISO outputs.");
        return step(x, ColVec<1, T>{y}, ColVec<1, T>{r});
    }

    constexpr void reset() {
        xi = ColVec<NY, T>{};
    }
};

/**
 * @brief Runtime bundle for LQGI servo control (observer + integral tracking)
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = float>
struct LQGIServoRuntimeBundle {
    design::LQGIResult<NX, NU, NY, NW, NV, T> design{};
    KalmanFilter<NX, NU, NY, NW, NV, T>       kf{};

    Matrix<NU, NX, T> Kx{}; ///< State-feedback block of LQI gain
    Matrix<NU, NY, T> Ki{}; ///< Integral block of LQI gain

    ColVec<NY, T> xi{};
    ColVec<NU, T> u_prev{};

    /**
     * @brief One discrete servo control tick using estimated state + integral action
     *
     * @note The integral state uses the unit discrete integrator
     *       xi[k+1] = xi[k] + (r - y) to match the LQI design augmentation
     *       (no Ts scaling); the DARE gain Ki is computed for that convention.
     */
    [[nodiscard]] constexpr ColVec<NU, T> step(
        const ColVec<NY, T>& y,
        const ColVec<NY, T>& r
    ) {
        kf.update(y, u_prev);
        const auto e = r - y;
        xi += e;

        const auto    xhat = kf.state();
        ColVec<NU, T> u = ColVec<NU, T>(-(Kx * xhat + Ki * xi));

        kf.predict(u);
        u_prev = u;
        return u;
    }

    /**
     * @brief SISO convenience overload
     */
    [[nodiscard]] constexpr ColVec<NU, T> step(T y, T r) {
        static_assert(NY == 1, "Scalar step overload is available only for SISO outputs.");
        return step(ColVec<1, T>{y}, ColVec<1, T>{r});
    }

    constexpr void reset() {
        xi = ColVec<NY, T>{};
        u_prev = ColVec<NU, T>{};
    }
};

/**
 * @brief Synthesis artifact bundle: LQI servo design + analysis + runtime bundle
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double, typename TRuntime = float>
struct LQIArtifacts {
    design::LQIResult<NX, NU, NY, T>            design{};
    LQIAnalysisModels<NX, NU, NY, NW, NV, T>    models{};
    LQIServoRuntimeBundle<NX, NU, NY, TRuntime> runtime{};
    bool                                        success{false};
};

/**
 * @brief Synthesis artifact bundle: LQGI servo design + analysis + runtime bundle
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double, typename TRuntime = float>
struct LQGIArtifacts {
    design::LQGIResult<NX, NU, NY, NW, NV, T>            design{};
    LQGIAnalysisModels<NX, NU, NY, NW, NV, T>            models{};
    LQGIServoRuntimeBundle<NX, NU, NY, NW, NV, TRuntime> runtime{};
    bool                                                 success{false};
};

namespace detail {

template<size_t NX, size_t NU, size_t NY, typename T>
[[nodiscard]] constexpr Matrix<NX + NY, NX + NY, T> lqi_augmented_A(const Matrix<NX, NX, T>& A, const Matrix<NY, NX, T>& C) {
    Matrix<NX + NY, NX + NY, T> A_aug{};
    A_aug.template block<NX, NX>(0, 0) = A;
    A_aug.template block<NY, NX>(NX, 0) = -C;
    A_aug.template block<NY, NY>(NX, NX) = Matrix<NY, NY, T>::identity();
    return A_aug;
}

template<size_t NX, size_t NU, size_t NY, typename T>
[[nodiscard]] constexpr Matrix<NX + NY, NU, T> lqi_augmented_B(const Matrix<NX, NU, T>& B) {
    Matrix<NX + NY, NU, T> B_aug{};
    B_aug.template block<NX, NU>(0, 0) = B;
    return B_aug;
}

template<size_t NX, size_t NU, size_t NY, typename T>
[[nodiscard]] constexpr wet::pair<Matrix<NU, NX, T>, Matrix<NU, NY, T>> split_lqi_gain(const Matrix<NU, NX + NY, T>& K) {
    Matrix<NU, NX, T> Kx{};
    Matrix<NU, NY, T> Ki{};

    for (size_t r = 0; r < NU; ++r) {
        for (size_t c = 0; c < NX; ++c) {
            Kx(r, c) = K(r, c);
        }
        for (size_t c = 0; c < NY; ++c) {
            Ki(r, c) = K(r, NX + c);
        }
    }
    return {Kx, Ki};
}

} // namespace detail

/**
 * @brief Build analysis models from an LQG design
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr LQGAnalysisModels<NX, NU, NY, NW, NV, T> build_lqg_analysis_models(
    const StateSpace<NX, NU, NY, NW, NV, T>&        sys,
    const design::LQGResult<NX, NU, NY, NW, NV, T>& lqg
) {
    LQGAnalysisModels<NX, NU, NY, NW, NV, T> models{};
    models.plant = sys;

    const Matrix<NX, NX, T> A_sf = sys.A - sys.B * lqg.lqr.K;
    models.state_feedback_closed_loop = StateSpace<NX, NU, NY, NW, NV, T>{
        A_sf,
        sys.B,
        sys.C,
        sys.D,
        sys.G,
        sys.H,
        sys.Ts
    };

    models.observer_error_dynamics = sys.A - lqg.kalman.L * sys.C;

    Matrix<2 * NX, 2 * NX, T> A_aug{};
    Matrix<2 * NX, NU, T>     B_aug{};
    Matrix<NY, 2 * NX, T>     C_aug{};

    A_aug.template block<NX, NX>(0, 0) = sys.A;
    A_aug.template block<NX, NX>(0, NX) = -sys.B * lqg.lqr.K;
    A_aug.template block<NX, NX>(NX, 0) = lqg.kalman.L * sys.C;
    A_aug.template block<NX, NX>(NX, NX) = sys.A - sys.B * lqg.lqr.K - lqg.kalman.L * sys.C;

    B_aug.template block<NX, NU>(0, 0) = sys.B;
    B_aug.template block<NX, NU>(NX, 0) = sys.B;

    C_aug.template block<NY, NX>(0, 0) = sys.C;

    models.augmented_closed_loop = StateSpace<2 * NX, NU, NY, 0, 0, T>{
        A_aug,
        B_aug,
        C_aug,
        sys.D,
        Matrix<2 * NX, 0, T>{},
        Matrix<NY, 0, T>{},
        sys.Ts
    };

    return models;
}

/**
 * @brief Build analysis models from an LQI servo design
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] constexpr LQIAnalysisModels<NX, NU, NY, NW, NV, T> build_lqi_analysis_models(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const design::LQIResult<NX, NU, NY, T>&  lqi
) {
    LQIAnalysisModels<NX, NU, NY, NW, NV, T> models{};
    models.plant = sys;

    const auto                   A_aug = detail::lqi_augmented_A<NX, NU, NY, T>(sys.A, sys.C);
    const auto                   B_aug = detail::lqi_augmented_B<NX, NU, NY, T>(sys.B);
    const Matrix<NY, NX + NY, T> C_aug = [&]() {
        Matrix<NY, NX + NY, T> C{};
        C.template block<NY, NX>(0, 0) = sys.C;
        return C;
    }();

    const auto A_cl = A_aug - B_aug * lqi.K;
    models.augmented_servo_closed_loop = StateSpace<NX + NY, NU, NY, 0, 0, T>{
        A_cl,
        B_aug,
        C_aug,
        sys.D,
        Matrix<NX + NY, 0, T>{},
        Matrix<NY, 0, T>{},
        sys.Ts
    };

    return models;
}

/**
 * @brief Build analysis models from an LQGI servo design
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr LQGIAnalysisModels<NX, NU, NY, NW, NV, T> build_lqgi_analysis_models(
    const StateSpace<NX, NU, NY, NW, NV, T>&         sys,
    const design::LQGIResult<NX, NU, NY, NW, NV, T>& lqgi
) {
    LQGIAnalysisModels<NX, NU, NY, NW, NV, T> models{};
    models.plant = sys;

    const auto                   A_aug = detail::lqi_augmented_A<NX, NU, NY, T>(sys.A, sys.C);
    const auto                   B_aug = detail::lqi_augmented_B<NX, NU, NY, T>(sys.B);
    const Matrix<NY, NX + NY, T> C_aug = [&]() {
        Matrix<NY, NX + NY, T> C{};
        C.template block<NY, NX>(0, 0) = sys.C;
        return C;
    }();

    const auto A_cl = A_aug - B_aug * lqgi.lqi.K;
    models.augmented_servo_closed_loop = StateSpace<NX + NY, NU, NY, 0, 0, T>{
        A_cl,
        B_aug,
        C_aug,
        sys.D,
        Matrix<NX + NY, 0, T>{},
        Matrix<NY, 0, T>{},
        sys.Ts
    };

    models.observer_error_dynamics = sys.A - lqgi.kalman.L * sys.C;
    return models;
}

/**
 * @brief Synthesize the full LQG artifact bundle in one call
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double, typename TRuntime = float>
[[nodiscard]] constexpr LQGArtifacts<NX, NU, NY, NW, NV, T, TRuntime> synthesize_lqg(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q_lqr,
    const Matrix<NU, NU, T>&                 R_lqr,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    LQGArtifacts<NX, NU, NY, NW, NV, T, TRuntime> out{};

    out.design = design::discrete_lqg(sys, Q_lqr, R_lqr, Q_kf, R_kf, N);
    out.models = build_lqg_analysis_models(sys, out.design);

    const auto runtime_design = out.design.template as<TRuntime>();
    out.runtime.design = runtime_design;
    out.runtime.controller = LQG<NX, NU, NY, NW, NV, TRuntime>{runtime_design};
    out.success = out.design.success;

    return out;
}

/**
 * @brief Synthesize a SISO LQG + PR design with internal-model compensation
 */
template<size_t NX, size_t NW = NX, size_t NV = 1, typename T = double, typename TRuntime = float>
[[nodiscard]] constexpr LQGPRArtifacts<NX, NW, NV, T, TRuntime> synthesize_lqg_pr(
    const StateSpace<NX, 1, 1, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&               Q_lqr,
    const Matrix<1, 1, T>&                 R_lqr,
    const Matrix<NW, NW, T>&               Q_kf,
    const Matrix<NV, NV, T>&               R_kf,
    const design::PRResult<T>&             pr,
    const Matrix<NX, 1, T>&                N = Matrix<NX, 1, T>{}
) {
    LQGPRArtifacts<NX, NW, NV, T, TRuntime> out{};

    out.base = synthesize_lqg<NX, 1, 1, NW, NV, T, TRuntime>(sys, Q_lqr, R_lqr, Q_kf, R_kf, N);

    out.runtime_pr.lqg_design = out.base.design.template as<TRuntime>();
    out.runtime_pr.pr_design = pr.template as<TRuntime>();
    out.runtime_pr.lqg = LQG<NX, 1, 1, NW, NV, TRuntime>{out.runtime_pr.lqg_design};
    out.runtime_pr.pr = PRController<TRuntime>{out.runtime_pr.pr_design};

    out.success = out.base.success;
    return out;
}

/**
 * @brief Synthesize the full LQI servo artifact bundle in one call
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double, typename TRuntime = float>
[[nodiscard]] constexpr LQIArtifacts<NX, NU, NY, NW, NV, T, TRuntime> synthesize_lqi(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q_aug,
    const Matrix<NU, NU, T>&                 R
) {
    LQIArtifacts<NX, NU, NY, NW, NV, T, TRuntime> out{};

    out.design = design::discrete_lqi(sys, Q_aug, R);
    out.models = build_lqi_analysis_models(sys, out.design);

    const auto runtime_design = out.design.template as<TRuntime>();
    out.runtime.design = runtime_design;
    out.runtime.controller = LQI<NX, NU, NY, TRuntime>{runtime_design};
    out.success = out.design.success;

    return out;
}

/**
 * @brief Synthesize the full LQGI servo artifact bundle in one call
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double, typename TRuntime = float>
[[nodiscard]] constexpr LQGIArtifacts<NX, NU, NY, NW, NV, T, TRuntime> synthesize_lqgi(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q_aug,
    const Matrix<NU, NU, T>&                 R,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf
) {
    LQGIArtifacts<NX, NU, NY, NW, NV, T, TRuntime> out{};

    out.design = design::discrete_lqgi(sys, Q_aug, R, Q_kf, R_kf);
    out.models = build_lqgi_analysis_models(sys, out.design);

    const auto runtime_design = out.design.template as<TRuntime>();
    out.runtime.design = runtime_design;
    out.runtime.kf = KalmanFilter<NX, NU, NY, NW, NV, TRuntime>{
        runtime_design.kalman.sys,
        runtime_design.kalman.Q,
        runtime_design.kalman.R,
        ColVec<NX, TRuntime>{},
        runtime_design.kalman.success ? runtime_design.kalman.P : Matrix<NX, NX, TRuntime>::identity()
    };

    const auto split = detail::split_lqi_gain<NX, NU, NY, TRuntime>(runtime_design.lqi.K);
    out.runtime.Kx = split.first;
    out.runtime.Ki = split.second;
    out.success = out.design.success;

    return out;
}

} // namespace design
} // namespace wet
