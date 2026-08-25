#pragma once

#include <fideslib.hpp>

#include "ks_adder_common.h"
#include "subtractor_ks_complex_hoisted_gpu.h"

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ksgreaterthan {

using fideslib::Ciphertext;
using fideslib::CryptoContext;
using fideslib::DCRTPoly;
using fideslib::Plaintext;
using ksadder::Algorithm;
using ksadder::Clock;
using ksadder::Milliseconds;
using ksadder::RuntimeConfig;
using ksadder::Slot;

enum class Relation {
    GreaterThan,
    LessThan,
    GreaterEqual,
    LessEqual,
    Equal,
    NotEqual,
};

inline const char* RelationName(Relation relation) {
    switch (relation) {
        case Relation::GreaterThan:
            return "GT";
        case Relation::LessThan:
            return "LT";
        case Relation::GreaterEqual:
            return "GE";
        case Relation::LessEqual:
            return "LE";
        case Relation::Equal:
            return "EQ";
        case Relation::NotEqual:
            return "NE";
    }
    return "unknown";
}

inline const char* RelationExpression(Relation relation) {
    switch (relation) {
        case Relation::GreaterThan:
            return "unsigned A > B";
        case Relation::LessThan:
            return "unsigned A < B";
        case Relation::GreaterEqual:
            return "unsigned A >= B";
        case Relation::LessEqual:
            return "unsigned A <= B";
        case Relation::Equal:
            return "A == B";
        case Relation::NotEqual:
            return "A != B";
    }
    return "unknown";
}

inline bool IsComplement(Relation relation) {
    return relation == Relation::GreaterEqual ||
           relation == Relation::LessEqual ||
           relation == Relation::Equal;
}

inline bool IsEquality(Relation relation) {
    return relation == Relation::Equal ||
           relation == Relation::NotEqual;
}

inline uint32_t ComparisonCircuitLevelCost(
    const RuntimeConfig& config,
    Relation /* relation */) {
    if (config.algorithm == Algorithm::Baseline) {
        return config.layers + 1;
    }
    // GE/LE use a same-level plaintext subtraction to complement the strict
    // result. Linear addition/subtraction consumes no multiplication level.
    return config.layers + 2;
}

inline void FinalizeComparisonLevelPlacement(
    RuntimeConfig* config,
    Relation relation) {
    config->circuitLevelCost =
        ComparisonCircuitLevelCost(*config, relation);
    constexpr uint32_t outputScaleDegree = 2;
    if (config->multiplicativeDepth <
        config->circuitLevelCost + outputScaleDegree) {
        throw std::invalid_argument(
            "--depth is too small for the selected comparison and the two "
            "towers required by the Bootstrap input");
    }

    if (config->levelPlacement == ksadder::LevelPlacement::Bottom) {
        config->expectedOutputLevel =
            config->multiplicativeDepth - outputScaleDegree;
        config->inputLevel =
            config->expectedOutputLevel - config->circuitLevelCost;
    }
    else {
        config->inputLevel = 0;
        config->expectedOutputLevel = config->circuitLevelCost;
    }
}

struct Masks {
    kssubtractor::Masks borrowPrefix;
    Plaintext msb;
    Plaintext minusHalfIMsb;
    Plaintext result;
};

struct TimingReport {
    kssubtractor::TimingReport prefix;
    double conjugateMs = 0.0;
    double extractMs = 0.0;
    double endpointNormalizeMs = 0.0;
    double resultRotateMs = 0.0;
    double complementMs = 0.0;
    double totalMs = 0.0;
};

namespace detail {

inline void Synchronize(const CryptoContext<DCRTPoly>& cc) {
    cc->Synchronize();
}

template <class Fn>
inline auto TimedCall(const CryptoContext<DCRTPoly>& cc,
                      double* elapsedMs,
                      Fn&& fn) {
    if (elapsedMs == nullptr) {
        return fn();
    }
    Synchronize(cc);
    const auto begin = Clock::now();
    auto result = fn();
    Synchronize(cc);
    *elapsedMs = Milliseconds(begin, Clock::now());
    return result;
}

inline void RequireLevelScale(
    const Ciphertext<DCRTPoly>& ciphertext,
    uint32_t expectedLevel,
    uint32_t expectedScaleDegree,
    const char* stage) {
    if (ciphertext->GetLevel() != expectedLevel ||
        ciphertext->GetNoiseScaleDeg() != expectedScaleDegree) {
        throw std::runtime_error(
            std::string("unexpected equality state after ") + stage +
            ": expected " + std::to_string(expectedLevel) + "/" +
            std::to_string(expectedScaleDegree) + ", got " +
            std::to_string(ciphertext->GetLevel()) + "/" +
            std::to_string(ciphertext->GetNoiseScaleDeg()));
    }
}

inline std::vector<double> MsbMaskValues(
    const RuntimeConfig& config,
    bool negate) {
    std::vector<double> values(config.slots, 0.0);
    const double selected = negate ? -1.0 : 1.0;
    for (uint32_t word = 0; word < config.words; ++word) {
        values[Slot(config, word, config.wordBits - 1U)] = selected;
    }
    return values;
}

inline std::vector<std::complex<double>> ExtractionMaskValues(
    const RuntimeConfig& config,
    bool negate) {
    std::vector<std::complex<double>> values(config.slots);
    const double imaginary = negate ? 0.5 : -0.5;
    for (uint32_t word = 0; word < config.words; ++word) {
        values[Slot(config, word, config.wordBits - 1U)] =
            std::complex<double>(0.0, imaginary);
    }
    return values;
}

inline std::vector<double> EqualityEndpointMaskValues(
    const RuntimeConfig& config) {
    std::vector<double> values(config.slots, 0.0);
    for (uint32_t word = 0; word < config.words; ++word) {
        values[Slot(config, word, config.wordBits - 1U)] =
            1.0;
    }
    return values;
}

}  // namespace detail

inline std::vector<int32_t> OrdinaryRotationIndices(
    const RuntimeConfig& config) {
    std::vector<int32_t> rotations;
    if (config.algorithm != Algorithm::ComplexTwisted) {
        rotations = ksadder::RotationIndices(config);
    }
    // A positive rotation moves the final MSB borrow-generate into bit 0.
    rotations.push_back(
        static_cast<int32_t>(config.wordBits - 1U));
    std::sort(rotations.begin(), rotations.end());
    rotations.erase(
        std::unique(rotations.begin(), rotations.end()),
        rotations.end());
    return rotations;
}

inline Masks BuildAndLoadMasks(const CryptoContext<DCRTPoly>& cc,
                               const RuntimeConfig& config,
                               bool includeComplementMask = false) {
    if (!cc) {
        throw std::invalid_argument(
            "greater-than BuildAndLoadMasks received null context");
    }

    Masks masks;
    masks.borrowPrefix =
        kssubtractor::BuildAndLoadMasks(cc, config, false);

    const uint32_t prefixOutputLevel =
        config.inputLevel + config.layers + 1;
    masks.msb = cc->MakeCKKSPackedPlaintext(
        detail::MsbMaskValues(config, includeComplementMask),
        1, prefixOutputLevel, nullptr, config.slots);
    masks.msb->SetLength(config.slots);
    cc->LoadPlaintext(masks.msb);

    if (config.algorithm != Algorithm::Baseline) {
        const uint32_t outputMaskLevel =
            config.inputLevel + config.layers + 2;
        masks.minusHalfIMsb = cc->MakeCKKSPackedPlaintext(
            detail::ExtractionMaskValues(
                config, includeComplementMask),
            1, outputMaskLevel, nullptr, config.slots);
        masks.minusHalfIMsb->SetLength(config.slots);
        cc->LoadPlaintext(masks.minusHalfIMsb);
    }
    if (includeComplementMask) {
        // Match the strict-result ciphertext exactly. Encoding this plaintext
        // one level lower would force EvalSub to discard an extra modulus.
        const uint32_t resultLevel =
            config.algorithm == Algorithm::Baseline
                ? config.inputLevel + config.layers + 1U
                : config.inputLevel + config.layers + 2U;
        std::vector<double> resultValues(config.slots, 0.0);
        for (uint32_t word = 0; word < config.words; ++word) {
            resultValues[Slot(config, word, 0)] = 1.0;
        }
        masks.result = cc->MakeCKKSPackedPlaintext(
            resultValues, 2, resultLevel, nullptr, config.slots);
        masks.result->SetLength(config.slots);
        cc->LoadPlaintext(masks.result);
    }

    detail::Synchronize(cc);
    return masks;
}

inline Masks BuildAndLoadEqualityMasks(
    const CryptoContext<DCRTPoly>& cc,
    const RuntimeConfig& config,
    bool equal) {
    if (!cc) {
        throw std::invalid_argument(
            "equality BuildAndLoadMasks received null context");
    }

    Masks masks;
    const uint32_t outputLevel =
        config.inputLevel + config.layers + 2U;
    masks.msb = cc->MakeCKKSPackedPlaintext(
        detail::EqualityEndpointMaskValues(config),
        1, outputLevel, nullptr, config.slots);
    masks.msb->SetLength(config.slots);
    cc->LoadPlaintext(masks.msb);

    if (equal) {
        std::vector<double> resultValues(config.slots, 0.0);
        for (uint32_t word = 0; word < config.words; ++word) {
            resultValues[Slot(config, word, 0)] = 1.0;
        }
        masks.result = cc->MakeCKKSPackedPlaintext(
            resultValues, 2, outputLevel, nullptr, config.slots);
        masks.result->SetLength(config.slots);
        cc->LoadPlaintext(masks.result);
    }

    detail::Synchronize(cc);
    return masks;
}

inline Ciphertext<DCRTPoly> EvalConfiguredEquality(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks,
    Relation relation,
    TimingReport* timing = nullptr) {
    if (!cc || !cA || !cB || !masks.msb ||
        !IsEquality(relation)) {
        throw std::invalid_argument(
            "EvalConfiguredEquality received invalid input");
    }
    if (relation == Relation::Equal && !masks.result) {
        throw std::invalid_argument(
            "equality complement mask is not initialized");
    }

    TimingReport local;
    TimingReport& report = timing ? *timing : local;
    const bool profile = timing != nullptr;
    if (profile) {
        report = TimingReport{};
        detail::Synchronize(cc);
    }
    report.prefix.layers.resize(config.layers);
    const auto totalBegin = Clock::now();

    // X=A xor B. NE is the OR of every X bit in one word.
    auto state = detail::TimedCall(
        cc, profile ? &report.prefix.preprocessMs : nullptr,
        [&]() {
            auto ab = cc->EvalMult(cA, cB);
            return cc->EvalSub(
                cc->EvalAdd(cA, cB), cc->EvalAdd(ab, ab));
        });

    for (uint32_t layer = 0; layer < config.layers; ++layer) {
        auto& item = report.prefix.layers[layer];
        if (profile) {
            detail::Synchronize(cc);
        }
        const auto layerBegin = Clock::now();
        const int32_t distance =
            config.directionSign * static_cast<int32_t>(1U << layer);
        auto previous = state;
        auto rotated = detail::TimedCall(
            cc, profile ? &item.transformMs : nullptr,
            [&]() { return cc->EvalRotate(previous, distance); });
        auto product = detail::TimedCall(
            cc, profile ? &item.productMs : nullptr,
            [&]() { return cc->EvalMult(previous, rotated); });
        state = detail::TimedCall(
            cc, profile ? &item.updateMs : nullptr,
            [&]() {
                // Boolean OR(x,y)=x+y-xy.
                return cc->EvalSub(
                    cc->EvalAdd(previous, rotated), product);
            });
        if (profile) {
            detail::Synchronize(cc);
            item.wallMs = Milliseconds(layerBegin, Clock::now());
        }
    }

    // No boundary mask is needed. At distance 2^k, the MSB endpoint of each
    // word depends only on its final 2^(k+1) positions. Since wordBits is a
    // power of two, that interval never crosses the word head. Other slots
    // may be contaminated by cyclic rotation, but they are discarded here.
    auto atMsb = detail::TimedCall(
        cc, profile ? &report.extractMs : nullptr,
        [&]() { return cc->EvalMult(state, masks.msb); });
    auto normalizedEndpoint = detail::TimedCall(
        cc, profile ? &report.endpointNormalizeMs : nullptr,
        [&]() {
            const uint32_t endpointLevel =
                config.inputLevel + config.layers + 1U;
            detail::RequireLevelScale(
                atMsb, endpointLevel, 1, "exact endpoint mask");

            // FIDES FLEXIBLEAUTO deliberately delays rescaling. Use its
            // explicit metadata transitions instead of relying on a later
            // operation to trigger them:
            //   (level,scaleDeg): (L+1,1) -> (L+1,2)
            //                     -> (L+2,1) -> (L+2,2).
            // Both scalar products multiply by one and preserve the message.
            auto scaleRaised = cc->EvalMult(atMsb, 1.0);
            detail::RequireLevelScale(
                scaleRaised, endpointLevel, 2,
                "first scalar scale lift");

            auto rescaled = cc->Rescale(scaleRaised);
            detail::RequireLevelScale(
                rescaled, endpointLevel + 1U, 1,
                "explicit endpoint rescale");

            auto bootstrapReady = cc->EvalMult(rescaled, 1.0);
            detail::RequireLevelScale(
                bootstrapReady, endpointLevel + 1U, 2,
                "second scalar scale lift");
            return bootstrapReady;
        });
    auto result = detail::TimedCall(
        cc, profile ? &report.resultRotateMs : nullptr,
        [&]() {
            return cc->EvalRotate(
                normalizedEndpoint,
                static_cast<int32_t>(config.wordBits - 1U));
        });

    if (relation == Relation::Equal) {
        result = detail::TimedCall(
            cc, profile ? &report.complementMs : nullptr,
            [&]() {
                // Form -NE with same-level ciphertext additions, avoiding
                // FIDES EvalSub(Plaintext,Ciphertext), whose scalar negation
                // path consumes another modulus level.
                auto doubled = cc->EvalAdd(result, result);
                auto negative = cc->EvalSub(result, doubled);
                return cc->EvalAdd(masks.result, negative);
            });
    }

    if (profile) {
        detail::Synchronize(cc);
        report.totalMs = Milliseconds(totalBegin, Clock::now());
    }
    return result;
}

/**
 * Evaluates one unsigned comparison by reusing the subtractor's borrow
 * prefix. GT uses underflow(B-A), LT uses underflow(A-B), and GE/LE negate
 * LT/GT respectively in the public bit-0 result slots.
 */
inline Ciphertext<DCRTPoly> EvalConfiguredComparison(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks,
    Relation relation,
    TimingReport* timing = nullptr) {

    if (IsEquality(relation)) {
        return EvalConfiguredEquality(
            cc, cA, cB, config, masks, relation, timing);
    }

    if (!cc || !cA || !cB || !masks.msb) {
        throw std::invalid_argument(
            "EvalConfiguredComparison received invalid input");
    }
    if (config.algorithm != Algorithm::Baseline &&
        !masks.minusHalfIMsb) {
        throw std::invalid_argument(
            "complex comparison extraction mask is not initialized");
    }
    if (IsComplement(relation) && !masks.result) {
        throw std::invalid_argument(
            "comparison complement mask is not initialized");
    }

    TimingReport local;
    TimingReport& report = timing ? *timing : local;
    const bool profile = timing != nullptr;
    if (profile) {
        report = TimingReport{};
        detail::Synchronize(cc);
    }
    const auto totalBegin = Clock::now();

    // GT/LE start from underflow(B-A); LT/GE start from underflow(A-B).
    const bool reverseOperands =
        relation == Relation::GreaterThan ||
        relation == Relation::LessEqual;
    const auto& minuend = reverseOperands ? cB : cA;
    const auto& subtrahend = reverseOperands ? cA : cB;
    auto prefix = kssubtractor::EvalConfiguredBorrowPrefixState(
        cc, minuend, subtrahend, config, masks.borrowPrefix,
        profile ? &report.prefix : nullptr);

    Ciphertext<DCRTPoly> atMsb;
    if (!prefix.complexPacked) {
        atMsb = detail::TimedCall(
            cc, profile ? &report.extractMs : nullptr,
            [&]() {
                return cc->EvalMult(prefix.value, masks.msb);
            });
    }
    else {
        auto conjugate = detail::TimedCall(
            cc, profile ? &report.conjugateMs : nullptr,
            [&]() { return cc->EvalConjugate(prefix.value); });
        atMsb = detail::TimedCall(
            cc, profile ? &report.extractMs : nullptr,
            [&]() {
                // Strict relations use -i/2 to extract G. Complemented
                // relations use +i/2 to extract -G in the same product,
                // avoiding FIDES GPU multScalar(-1), which consumes a level.
                return cc->EvalMult(
                    cc->EvalSub(prefix.value, conjugate),
                    masks.minusHalfIMsb);
            });
    }

    auto strictResult = detail::TimedCall(
        cc, profile ? &report.resultRotateMs : nullptr,
        [&]() {
            return cc->EvalRotate(
                atMsb,
                static_cast<int32_t>(config.wordBits - 1U));
        });

    auto result = strictResult;
    if (IsComplement(relation)) {
        result = detail::TimedCall(
            cc, profile ? &report.complementMs : nullptr,
            [&]() {
                // strictResult already contains -GT or -LT.
                return cc->EvalAdd(masks.result, strictResult);
            });
    }

    if (profile) {
        detail::Synchronize(cc);
        report.totalMs =
            Milliseconds(totalBegin, Clock::now());
    }
    return result;
}

inline Ciphertext<DCRTPoly> EvalConfiguredGreaterThan(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* timing = nullptr) {
    return EvalConfiguredComparison(
        cc, cA, cB, config, masks, Relation::GreaterThan, timing);
}

inline void InitializeConfiguredComparisonRuntime(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks,
    Relation relation) {
    if (!config.graphInit || config.algorithm == Algorithm::Baseline) {
        return;
    }
    auto disposable = EvalConfiguredComparison(
        cc, cA, cB, config, masks, relation);
    (void)disposable;
    detail::Synchronize(cc);
}

inline void InitializeConfiguredGreaterThanRuntime(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks) {
    InitializeConfiguredComparisonRuntime(
        cc, cA, cB, config, masks, Relation::GreaterThan);
}

}  // namespace ksgreaterthan
