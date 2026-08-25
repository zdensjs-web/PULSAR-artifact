#pragma once

#include <fideslib.hpp>

#include "ks_adder_common.h"

#include <chrono>
#include <complex>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ksadder {

using fideslib::Ciphertext;
using fideslib::CryptoContext;
using fideslib::DCRTPoly;
using fideslib::Plaintext;

struct LayerTiming {
    double transformMs = 0.0;
    double productMs = 0.0;
    double auxiliaryMs = 0.0;
    double updateMs = 0.0;
    double wallMs = 0.0;
};

struct TimingReport {
    double preprocessMs = 0.0;
    double packStateMs = 0.0;
    std::vector<LayerTiming> layers;
    double carryRotateMs = 0.0;
    double outputTermsMs = 0.0;
    double projectionMs = 0.0;
    double reconstructMs = 0.0;
    double totalMs = 0.0;
    bool projectedToReal = false;
};

using AuditCallback = std::function<void(
    const char*, int32_t, const Ciphertext<DCRTPoly>&)>;

struct Masks {
    Plaintext active;
    Plaintext carry;
    Plaintext propagate;
    Plaintext halfPropagate;
    Plaintext iActive;
    Plaintext twoIActive;
    Plaintext minusICarry;
    Plaintext halfActiveProjection;
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

inline std::vector<double> ActiveMaskValues(
    const RuntimeConfig& config) {
    std::vector<double> values(config.slots, 0.0);
    for (uint32_t word = 0; word < config.words; ++word) {
        for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
            values[Slot(config, word, bit)] = 1.0;
        }
    }
    return values;
}

inline std::vector<double> CarryMaskValues(
    const RuntimeConfig& config) {
    std::vector<double> values(config.slots, 0.0);
    for (uint32_t word = 0; word < config.words; ++word) {
        if (config.directionSign == -1) {
            for (uint32_t bit = 1; bit < config.wordBits; ++bit) {
                values[Slot(config, word, bit)] = 1.0;
            }
        }
        else {
            for (uint32_t bit = 0; bit + 1 < config.wordBits; ++bit) {
                values[Slot(config, word, bit)] = 1.0;
            }
        }
    }
    return values;
}

inline std::vector<double> PropagateMaskValues(
    const RuntimeConfig& config,
    double scale) {
    std::vector<double> values(config.slots, 0.0);
    for (uint32_t word = 0; word < config.words; ++word) {
        uint32_t firstBit = 0;
        uint32_t lastBit = config.wordBits;
        if (config.guardFreeLayout) {
            if (config.directionSign == -1) {
                firstBit = 1;
            }
            else {
                lastBit = config.wordBits - 1;
            }
        }
        for (uint32_t bit = firstBit; bit < lastBit; ++bit) {
            values[Slot(config, word, bit)] = scale;
        }
    }
    return values;
}

inline std::vector<std::complex<double>> ImaginaryMaskValues(
    const std::vector<double>& realValues,
    double imaginaryScale) {
    std::vector<std::complex<double>> values(realValues.size());
    for (size_t index = 0; index < realValues.size(); ++index) {
        values[index] =
            std::complex<double>(0.0, imaginaryScale * realValues[index]);
    }
    return values;
}

}  // namespace detail

inline Masks BuildAndLoadMasks(const CryptoContext<DCRTPoly>& cc,
                               const RuntimeConfig& config) {
    if (!cc) {
        throw std::invalid_argument("BuildAndLoadMasks received null context");
    }

    const auto activeValues = detail::ActiveMaskValues(config);
    const auto carryValues = detail::CarryMaskValues(config);
    const auto propagateValues =
        detail::PropagateMaskValues(config, 1.0);
    Masks masks;
    const uint32_t outputMaskLevel =
        config.inputLevel + config.layers + 2;
    masks.active = cc->MakeCKKSPackedPlaintext(
        activeValues, 1, outputMaskLevel, nullptr, config.slots);
    masks.carry = cc->MakeCKKSPackedPlaintext(
        carryValues, 1, outputMaskLevel, nullptr, config.slots);
    masks.active->SetLength(config.slots);
    masks.carry->SetLength(config.slots);
    cc->LoadPlaintext(masks.active);
    cc->LoadPlaintext(masks.carry);

    const uint32_t packMaskLevel = config.inputLevel + 1;
    if (config.algorithm == Algorithm::Baseline) {
        masks.propagate = cc->MakeCKKSPackedPlaintext(
            propagateValues, 1, packMaskLevel, nullptr, config.slots);
        masks.propagate->SetLength(config.slots);
        cc->LoadPlaintext(masks.propagate);
    }
    else {
        const auto halfPropagateValues =
            detail::PropagateMaskValues(config, 0.5);
        masks.halfPropagate = cc->MakeCKKSPackedPlaintext(
            halfPropagateValues, 1, packMaskLevel, nullptr, config.slots);
        masks.iActive = cc->MakeCKKSPackedPlaintext(
            detail::ImaginaryMaskValues(activeValues, 1.0),
            1, packMaskLevel, nullptr, config.slots);
        masks.twoIActive = cc->MakeCKKSPackedPlaintext(
            detail::ImaginaryMaskValues(activeValues, 2.0),
            1, outputMaskLevel, nullptr, config.slots);
        masks.minusICarry = cc->MakeCKKSPackedPlaintext(
            detail::ImaginaryMaskValues(carryValues, -1.0),
            1, outputMaskLevel, nullptr, config.slots);
        masks.halfPropagate->SetLength(config.slots);
        masks.iActive->SetLength(config.slots);
        masks.twoIActive->SetLength(config.slots);
        masks.minusICarry->SetLength(config.slots);
        cc->LoadPlaintext(masks.halfPropagate);
        cc->LoadPlaintext(masks.iActive);
        cc->LoadPlaintext(masks.twoIActive);
        cc->LoadPlaintext(masks.minusICarry);
    }

    detail::Synchronize(cc);
    return masks;
}

inline Ciphertext<DCRTPoly> EvalAddKSBaseline(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* timing = nullptr) {

    if (!cc || !cA || !cB || !masks.active || !masks.carry ||
        !masks.propagate) {
        throw std::invalid_argument("EvalAddKSBaseline received invalid input");
    }

    TimingReport local;
    TimingReport& report = timing ? *timing : local;
    const bool profile = timing != nullptr;
    if (profile) {
        report = TimingReport{};
        report.layers.resize(config.layers);
        detail::Synchronize(cc);
    }
    const auto totalBegin = Clock::now();

    const auto preprocessBegin = Clock::now();
    auto g = cc->EvalMult(cA, cB);
    auto p = cc->EvalSub(cc->EvalAdd(cA, cB), cc->EvalAdd(g, g));
    if (profile) {
        detail::Synchronize(cc);
        report.preprocessMs =
            Milliseconds(preprocessBegin, Clock::now());
    }

    auto G = g;
    // In the dense layout, P=0 at every word head is the absorbing segment
    // delimiter. This matched-level plaintext product replaces the old
    // guard slots and is not an additional prefix-layer operation.
    auto P = cc->EvalMult(p, masks.propagate);
    for (uint32_t layer = 0; layer < config.layers; ++layer) {
        const auto layerBegin = Clock::now();
        const int32_t rotation =
            config.directionSign * static_cast<int32_t>(1U << layer);
        const auto Gold = G;
        const auto Pold = P;

        auto Grot = detail::TimedCall(
            cc, profile ? &report.layers[layer].transformMs : nullptr,
            [&]() { return cc->EvalRotate(Gold, rotation); });
        auto term = detail::TimedCall(
            cc, profile ? &report.layers[layer].productMs : nullptr,
            [&]() { return cc->EvalMult(Pold, Grot); });
        G = detail::TimedCall(
            cc, profile ? &report.layers[layer].updateMs : nullptr,
            [&]() { return cc->EvalAdd(Gold, term); });

        if (layer + 1 < config.layers) {
            auto Prot = cc->EvalRotate(Pold, rotation);
            P = detail::TimedCall(
                cc, profile ? &report.layers[layer].auxiliaryMs : nullptr,
                [&]() { return cc->EvalMult(Pold, Prot); });
        }
        if (profile) {
            detail::Synchronize(cc);
            report.layers[layer].wallMs =
                Milliseconds(layerBegin, Clock::now());
        }
    }

    auto carry = detail::TimedCall(
        cc, profile ? &report.carryRotateMs : nullptr,
        [&]() { return cc->EvalRotate(G, config.directionSign); });

    const auto outputBegin = Clock::now();
    carry = cc->EvalMult(carry, masks.carry);
    G = cc->EvalMult(G, masks.active);
    if (profile) {
        detail::Synchronize(cc);
        report.outputTermsMs = Milliseconds(outputBegin, Clock::now());
    }

    const auto reconstructBegin = Clock::now();
    auto result = cc->EvalAdd(cA, cB);
    result = cc->EvalAdd(result, carry);
    result = cc->EvalSub(result, cc->EvalAdd(G, G));
    if (profile) {
        detail::Synchronize(cc);
        report.reconstructMs =
            Milliseconds(reconstructBegin, Clock::now());
        report.totalMs = Milliseconds(totalBegin, Clock::now());
    }
    return result;
}

inline void InitializeHoistedKoggeStoneGraphs(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config) {

    if (config.algorithm != Algorithm::ComplexHoisted) {
        return;
    }
    auto probe = cc->EvalMult(cA, cB);
    probe = cc->EvalMult(probe, 0.5);
    for (uint32_t layer = 0; layer < config.layers; ++layer) {
        const int32_t rotation =
            config.directionSign * static_cast<int32_t>(1U << layer);
        auto pair = cc->EvalRotateConjugateHoisted(probe, rotation);
        probe = cc->EvalMult(pair.first, 0.5);
    }
    auto finalRotation = cc->EvalRotate(probe, config.directionSign);
    (void)finalRotation;
    detail::Synchronize(cc);
}

inline Ciphertext<DCRTPoly> PackComplexState(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    Masks& masks,
    TimingReport* report,
    const AuditCallback& audit) {

    const bool profile = report != nullptr;
    const auto preprocessBegin = Clock::now();
    auto g = cc->EvalMult(cA, cB);
    auto p = cc->EvalSub(cc->EvalAdd(cA, cB), cc->EvalAdd(g, g));
    if (profile) {
        detail::Synchronize(cc);
        report->preprocessMs =
            Milliseconds(preprocessBegin, Clock::now());
    }
    if (audit) {
        audit("g", -1, g);
        audit("p", -1, p);
    }

    const auto packBegin = Clock::now();
    // The same P/2 multiplication also inserts P=0 at each packed-word
    // head. Rotated states from the preceding word are therefore absorbed
    // without guard slots or a boundary mask in each prefix layer.
    auto halfP = cc->EvalMult(p, masks.halfPropagate);
    auto iG = cc->EvalMult(g, masks.iActive);
    auto state = cc->EvalAdd(halfP, iG);
    if (profile) {
        detail::Synchronize(cc);
        report->packStateMs = Milliseconds(packBegin, Clock::now());
    }
    if (audit) {
        audit("Z", -1, state);
    }
    return state;
}

inline Ciphertext<DCRTPoly> ReconstructComplexSum(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    Ciphertext<DCRTPoly> state,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* report,
    const AuditCallback& audit) {

    const bool profile = report != nullptr;
    auto carryState = detail::TimedCall(
        cc, profile ? &report->carryRotateMs : nullptr,
        [&]() { return cc->EvalRotate(state, config.directionSign); });
    if (audit) {
        audit("carry-state", static_cast<int32_t>(config.layers), carryState);
    }

    const auto outputBegin = Clock::now();
    auto minusTwoGenerate = cc->EvalMult(state, masks.twoIActive);
    auto carry = cc->EvalMult(carryState, masks.minusICarry);
    if (profile) {
        detail::Synchronize(cc);
        report->outputTermsMs = Milliseconds(outputBegin, Clock::now());
    }
    if (audit) {
        audit("minus-two-G", static_cast<int32_t>(config.layers),
              minusTwoGenerate);
        audit("carry-term", static_cast<int32_t>(config.layers), carry);
    }

    const auto reconstructBegin = Clock::now();
    auto result = cc->EvalAdd(cA, cB);
    result = cc->EvalAdd(result, minusTwoGenerate);
    result = cc->EvalAdd(result, carry);
    if (profile) {
        detail::Synchronize(cc);
        report->reconstructMs =
            Milliseconds(reconstructBegin, Clock::now());
    }
    if (audit) {
        audit("complex-result", static_cast<int32_t>(config.layers), result);
    }

    if (config.projectOutputToReal) {
        const auto projectionBegin = Clock::now();
        auto conjugate = cc->EvalConjugate(result);
        auto realSum = cc->EvalAdd(result, conjugate);
        result = masks.halfActiveProjection
            ? cc->EvalMult(realSum, masks.halfActiveProjection)
            : cc->EvalMult(realSum, 0.5);
        if (profile) {
            detail::Synchronize(cc);
            report->projectionMs =
                Milliseconds(projectionBegin, Clock::now());
            report->projectedToReal = true;
        }
        if (audit) {
            audit("real-result", static_cast<int32_t>(config.layers), result);
        }
    }
    return result;
}

inline Ciphertext<DCRTPoly> EvalAddKSComplexStandard(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* timing = nullptr,
    const AuditCallback& audit = {}) {

    TimingReport local;
    TimingReport& report = timing ? *timing : local;
    const bool profile = timing != nullptr;
    if (profile) {
        report = TimingReport{};
        report.layers.resize(config.layers);
        detail::Synchronize(cc);
    }
    const auto totalBegin = Clock::now();
    auto state = PackComplexState(
        cc, cA, cB, masks, profile ? &report : nullptr, audit);

    for (uint32_t layer = 0; layer < config.layers; ++layer) {
        const auto layerBegin = Clock::now();
        const int32_t rotation =
            config.directionSign * static_cast<int32_t>(1U << layer);
        const auto oldState = state;
        auto transforms = detail::TimedCall(
            cc, profile ? &report.layers[layer].transformMs : nullptr,
            [&]() {
                return std::make_pair(
                    cc->EvalRotate(oldState, rotation),
                    cc->EvalConjugate(oldState));
            });
        auto product = detail::TimedCall(
            cc, profile ? &report.layers[layer].productMs : nullptr,
            [&]() {
                auto propagate =
                    cc->EvalAdd(oldState, transforms.second);
                return cc->EvalMult(propagate, transforms.first);
            });
        auto keepGenerate = detail::TimedCall(
            cc, profile ? &report.layers[layer].auxiliaryMs : nullptr,
            [&]() {
                return cc->EvalMult(
                    cc->EvalSub(oldState, transforms.second), 0.5);
            });
        state = detail::TimedCall(
            cc, profile ? &report.layers[layer].updateMs : nullptr,
            [&]() { return cc->EvalAdd(product, keepGenerate); });

        if (audit) {
            audit("standard-Z", static_cast<int32_t>(layer), state);
        }
        if (profile) {
            detail::Synchronize(cc);
            report.layers[layer].wallMs =
                Milliseconds(layerBegin, Clock::now());
        }
    }

    auto result = ReconstructComplexSum(
        cc, cA, cB, state, config, masks,
        profile ? &report : nullptr, audit);
    if (profile) {
        report.totalMs = Milliseconds(totalBegin, Clock::now());
    }
    return result;
}

inline Ciphertext<DCRTPoly> EvalAddKSComplexHoisted(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* timing = nullptr,
    const AuditCallback& audit = {}) {

    TimingReport local;
    TimingReport& report = timing ? *timing : local;
    const bool profile = timing != nullptr;
    if (profile) {
        report = TimingReport{};
        report.layers.resize(config.layers);
        detail::Synchronize(cc);
    }
    const auto totalBegin = Clock::now();
    auto state = PackComplexState(
        cc, cA, cB, masks, profile ? &report : nullptr, audit);

    for (uint32_t layer = 0; layer < config.layers; ++layer) {
        const auto layerBegin = Clock::now();
        const int32_t rotation =
            config.directionSign * static_cast<int32_t>(1U << layer);
        const auto oldState = state;
        auto pair = detail::TimedCall(
            cc, profile ? &report.layers[layer].transformMs : nullptr,
            [&]() {
                return cc->EvalRotateConjugateHoisted(oldState, rotation);
            });

        auto product = detail::TimedCall(
            cc, profile ? &report.layers[layer].productMs : nullptr,
            [&]() {
                auto propagate = cc->EvalAdd(oldState, pair.second);
                return cc->EvalMult(propagate, pair.first);
            });
        auto keepGenerate = detail::TimedCall(
            cc, profile ? &report.layers[layer].auxiliaryMs : nullptr,
            [&]() {
                return cc->EvalMult(
                    cc->EvalSub(oldState, pair.second), 0.5);
            });
        state = detail::TimedCall(
            cc, profile ? &report.layers[layer].updateMs : nullptr,
            [&]() { return cc->EvalAdd(product, keepGenerate); });

        if (profile) {
            detail::Synchronize(cc);
            report.layers[layer].wallMs =
                Milliseconds(layerBegin, Clock::now());
        }
    }

    auto result = ReconstructComplexSum(
        cc, cA, cB, state, config, masks,
        profile ? &report : nullptr, audit);
    if (profile) {
        report.totalMs = Milliseconds(totalBegin, Clock::now());
    }
    return result;
}

inline Ciphertext<DCRTPoly> EvalAddKSComplexTwisted(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* timing = nullptr,
    const AuditCallback& audit = {}) {

    TimingReport local;
    TimingReport& report = timing ? *timing : local;
    const bool profile = timing != nullptr;
    if (profile) {
        report = TimingReport{};
        report.layers.resize(config.layers);
        detail::Synchronize(cc);
    }
    const auto totalBegin = Clock::now();
    auto state = PackComplexState(
        cc, cA, cB, masks, profile ? &report : nullptr, audit);

    uint32_t publicTag = 1;
    for (uint32_t layer = 0; layer < config.layers; ++layer) {
        const auto layerBegin = Clock::now();
        const int32_t rotation =
            config.directionSign * static_cast<int32_t>(publicTag);
        const auto oldState = state;

        auto conjugate = detail::TimedCall(
            cc, profile ? &report.layers[layer].transformMs : nullptr,
            [&]() { return cc->EvalConjugate(oldState); });
        auto product = detail::TimedCall(
            cc, profile ? &report.layers[layer].productMs : nullptr,
            [&]() {
                auto propagate = cc->EvalAdd(oldState, conjugate);
                return cc->EvalTwistedProduct(
                    propagate, oldState, rotation);
            });
        auto keepGenerate = detail::TimedCall(
            cc, profile ? &report.layers[layer].auxiliaryMs : nullptr,
            [&]() {
                return cc->EvalMult(
                    cc->EvalSub(oldState, conjugate), 0.5);
            });
        state = detail::TimedCall(
            cc, profile ? &report.layers[layer].updateMs : nullptr,
            [&]() { return cc->EvalAdd(product, keepGenerate); });
        publicTag <<= 1U;

        if (audit) {
            audit("twisted-Z", static_cast<int32_t>(layer), state);
        }
        if (profile) {
            detail::Synchronize(cc);
            report.layers[layer].wallMs =
                Milliseconds(layerBegin, Clock::now());
        }
    }

    auto result = ReconstructComplexSum(
        cc, cA, cB, state, config, masks,
        profile ? &report : nullptr, audit);
    if (profile) {
        report.totalMs = Milliseconds(totalBegin, Clock::now());
    }
    return result;
}

inline Ciphertext<DCRTPoly> EvalConfiguredAdder(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* timing = nullptr,
    const AuditCallback& audit = {}) {
    switch (config.algorithm) {
        case Algorithm::Baseline:
            return EvalAddKSBaseline(
                cc, cA, cB, config, masks, timing);
        case Algorithm::ComplexStandard:
            return EvalAddKSComplexStandard(
                cc, cA, cB, config, masks, timing, audit);
        case Algorithm::ComplexHoisted:
            return EvalAddKSComplexHoisted(
                cc, cA, cB, config, masks, timing, audit);
        case Algorithm::ComplexTwisted:
            return EvalAddKSComplexTwisted(
                cc, cA, cB, config, masks, timing, audit);
    }
    throw std::logic_error("unknown adder algorithm");
}

/**
 * Initializes level-specific ModUp graph/workspace state with one disposable
 * chain. No user-visible output is retained. This is setup, not a benchmark
 * repetition; the caller times the following real addition exactly once.
 */
inline void InitializeConfiguredAdderRuntime(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks) {
    if (!config.graphInit || config.algorithm == Algorithm::Baseline) {
        return;
    }
    if (config.algorithm == Algorithm::ComplexHoisted) {
        InitializeHoistedKoggeStoneGraphs(cc, cA, cB, config);
        return;
    }
    if (config.algorithm == Algorithm::ComplexStandard) {
        auto disposable = EvalAddKSComplexStandard(
            cc, cA, cB, config, masks);
        (void)disposable;
        detail::Synchronize(cc);
        return;
    }

    auto disposable = EvalAddKSComplexTwisted(
        cc, cA, cB, config, masks);
    (void)disposable;
    detail::Synchronize(cc);
}

}  // namespace ksadder
