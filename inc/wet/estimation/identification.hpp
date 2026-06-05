#pragma once

/**
 * @file identification.hpp
 * @brief Identification primitives for parameter estimation and system identification.
 */

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace wet::estimation {

/**
 * @brief One time-aligned sample used for closed-loop or open-loop identification.
 */
template<typename T = double>
struct InputOutputSample {
    using scalar_type = T;

    T t{};
    T u{};
    T y{};
};

/**
 * @brief Compact statistics extracted from a step experiment.
 */
template<typename T = double>
struct StepResponseSummary {
    using scalar_type = T;

    T           sample_time{};
    T           y0{};
    T           yss{};
    T           u0{};
    T           uss{};
    std::size_t sample_count{};
    bool        success{false};
};

/**
 * @brief One frequency-response point used by sweep/chirp-based identification.
 */
template<typename T = double>
struct FrequencyResponsePoint {
    using scalar_type = T;

    T omega{};
    T magnitude{};
    T phase{};
};

/**
 * @brief Summary statistics for a frequency-response data set.
 */
template<typename T = double>
struct FrequencyResponseSummary {
    using scalar_type = T;

    T           coherence_mean{};
    std::size_t point_count{};
    bool        success{false};
};

/**
 * @brief First-order plus dead-time candidate model.
 */
template<typename T = double>
struct FOPDTModel {
    using scalar_type = T;

    T    K{};
    T    tau{};
    T    L{};
    bool success{false};
};

/**
 * @brief Second-order plus dead-time candidate model.
 */
template<typename T = double>
struct SOPDTModel {
    using scalar_type = T;

    T    K{};
    T    tau1{};
    T    tau2{};
    T    L{};
    bool success{false};
};

/**
 * @brief ARX candidate model with fixed numerator/denominator orders.
 */
template<std::size_t NA, std::size_t NB, typename T = double>
struct ARXModel {
    using scalar_type = T;

    T    a[NA]{};
    T    b[NB]{};
    T    delay{};
    bool success{false};
};

/**
 * @brief Frequency-response estimate quality summary.
 */
template<typename T = double>
struct FRFEstimateResult {
    using scalar_type = T;

    T    coherence_mean{};
    bool success{false};
};

/**
 * @brief End-to-end output of a sweep-based identification pass.
 */
template<typename T = double>
struct SweepIdentificationResult {
    using scalar_type = T;

    FRFEstimateResult<T> frf{};
    bool                 success{false};
};

/**
 * @brief Scalar fit metrics used for model comparison and gating.
 */
template<typename T = double>
struct FitMetrics {
    using scalar_type = T;

    T rmse{};
    T nrmse{};
    T coherence{};
};

/**
 * @brief Validation status for a model against held-out or replayed data.
 */
template<typename T = double>
struct ValidationResult {
    using scalar_type = T;

    FitMetrics<T> metrics{};
    bool          residual_whiteness_pass{false};
    bool          success{false};
};

/**
 * @brief Enumerates which reduced model family is selected for downstream design.
 */
enum class IdentificationModelKind : std::uint8_t {
    unknown,
    fopdt,
    sopdt,
    arx,
};

/**
 * @brief Holds all candidate models produced during an identification pass.
 */
template<typename T = double>
struct GreyBoxCandidateModels {
    using scalar_type = T;

    FOPDTModel<T> fopdt{};
    SOPDTModel<T> sopdt{};
    bool          has_fopdt{false};
    bool          has_sopdt{false};
    bool          success{false};
};

/**
 * @brief Selected model used by downstream model-based controller design.
 *
 * Exactly one model family should be active, indicated by @ref kind.
 */
template<typename T = double>
struct GreyBoxIdentificationResult {
    using scalar_type = T;

    IdentificationModelKind kind{IdentificationModelKind::unknown};
    FOPDTModel<T>           fopdt{};
    SOPDTModel<T>           sopdt{};
    FitMetrics<T>           fit{};
    bool                    success{false};
};

/**
 * @brief Aggregated handoff payload from identification to model-based control design.
 */
template<typename T = double>
struct IdentificationHandoff {
    using scalar_type = T;

    GreyBoxCandidateModels<T>      candidates{};
    GreyBoxIdentificationResult<T> model{};
    ValidationResult<T>            validation{};
    bool                           ready_for_control_design{false};
};

/**
 * @brief Result of model-drift monitoring used for adaptation triggers.
 */
template<typename T = double>
struct DriftMonitorResult {
    using scalar_type = T;

    T    drift_score{};
    bool drift_detected{false};
};

/**
 * @brief Gate decision for whether adaptive updates are currently allowed.
 */
template<typename T = double>
struct AdaptationGateDecision {
    using scalar_type = T;

    T    confidence{};
    bool allow_update{false};
};

/**
 * @brief Concept for identified models consumed by downstream design modules.
 */
template<typename Model>
concept IdentifiedModelLike = requires(const Model& model) {
    typename Model::scalar_type;
    requires std::is_floating_point_v<typename Model::scalar_type>;
    { model.success } -> std::convertible_to<bool>;
};

} // namespace wet::estimation
