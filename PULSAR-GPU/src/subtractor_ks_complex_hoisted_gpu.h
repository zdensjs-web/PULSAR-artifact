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

namespace kssubtractor {

using fideslib::Ciphertext;
using fideslib::CryptoContext;
using fideslib::DCRTPoly;
using fideslib::Plaintext;
using ksadder::Algorithm;
using ksadder::Clock;
using ksadder::Milliseconds;
using ksadder::RuntimeConfig;
using ksadder::Slot;

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
    double borrowRotateMs = 0.0;
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
    Plaintext borrow;
    Plaintext propagate;
    Plaintext halfPropagate;
    Plaintext iActive;
    Plaintext minusTwoIActive;
    Plaintext plusIBorrow;
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

inline std::vector<double> BorrowMaskValues(
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
                               const RuntimeConfig& config,
                               bool includeDifferenceOutput = true) {
    if (!cc) {
        throw std::invalid_argument("BuildAndLoadMasks received null context");
    }

    const auto activeValues = detail::ActiveMaskValues(config);
    const auto borrowValues = detail::BorrowMaskValues(config);
    Masks masks;
    const uint32_t outputMaskLevel =
        config.inputLevel + config.layers + 2;
    masks.active = cc->MakeCKKSPackedPlaintext(
        activeValues, 1, outputMaskLevel, nullptr, config.slots);
    masks.borrow = cc->MakeCKKSPackedPlaintext(
        borrowValues, 1, outputMaskLevel, nullptr, config.slots);
    masks.active->SetLength(config.slots);
    masks.borrow->SetLength(config.slots);
    cc->LoadPlaintext(masks.active);
    cc->LoadPlaintext(masks.borrow);

    const uint32_t packMaskLevel = config.inputLevel + 1;
    if (config.algorithm == Algorithm::Baseline &&
        config.guardFreeLayout) {
        masks.propagate = cc->MakeCKKSPackedPlaintext(
            detail::PropagateMaskValues(config, 1.0),
            1, packMaskLevel, nullptr, config.slots);
        masks.propagate->SetLength(config.slots);
        cc->LoadPlaintext(masks.propagate);
    }
    else if (config.algorithm != Algorithm::Baseline) {
        if (config.guardFreeLayout) {
            masks.halfPropagate = cc->MakeCKKSPackedPlaintext(
                detail::PropagateMaskValues(config, 0.5),
                1, packMaskLevel, nullptr, config.slots);
            masks.halfPropagate->SetLength(config.slots);
            cc->LoadPlaintext(masks.halfPropagate);
        }
        masks.iActive = cc->MakeCKKSPackedPlaintext(
            detail::ImaginaryMaskValues(activeValues, 1.0),
            1, packMaskLevel, nullptr, config.slots);
        masks.iActive->SetLength(config.slots);
        cc->LoadPlaintext(masks.iActive);
        if (includeDifferenceOutput) {
            masks.minusTwoIActive = cc->MakeCKKSPackedPlaintext(
                detail::ImaginaryMaskValues(activeValues, -2.0),
                1, outputMaskLevel, nullptr, config.slots);
            masks.plusIBorrow = cc->MakeCKKSPackedPlaintext(
                detail::ImaginaryMaskValues(borrowValues, 1.0),
                1, outputMaskLevel, nullptr, config.slots);
            masks.minusTwoIActive->SetLength(config.slots);
            masks.plusIBorrow->SetLength(config.slots);
            cc->LoadPlaintext(masks.minusTwoIActive);
            cc->LoadPlaintext(masks.plusIBorrow);
        }
    }

    detail::Synchronize(cc);
    return masks;
}

inline Ciphertext<DCRTPoly> EvalBorrowPrefixBaseline(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* report = nullptr) {

    const bool profile = report != nullptr;
    const auto preprocessBegin = Clock::now();
    auto ab = cc->EvalMult(cA, cB);
    auto xorAB = cc->EvalSub(
        cc->EvalAdd(cA, cB), cc->EvalAdd(ab, ab));
    auto G = cc->EvalSub(cB, ab);
    // Q=-P=XOR-1 avoids plaintext-minus-ciphertext level mismatches.
    auto q = cc->EvalSub(xorAB, 1.0);
    auto Q = config.guardFreeLayout
        ? cc->EvalMult(q, masks.propagate)
        : q;
    if (profile) {
        detail::Synchronize(cc);
        report->preprocessMs =
            Milliseconds(preprocessBegin, Clock::now());
        report->layers.resize(config.layers);
    }

    for (uint32_t layer = 0; layer < config.layers; ++layer) {
        const auto layerBegin = Clock::now();
        const int32_t rotation =
            config.directionSign * static_cast<int32_t>(1U << layer);
        const auto Gold = G;
        const auto Qold = Q;

        auto Grot = detail::TimedCall(
            cc, profile ? &report->layers[layer].transformMs : nullptr,
            [&]() { return cc->EvalRotate(Gold, rotation); });
        auto term = detail::TimedCall(
            cc, profile ? &report->layers[layer].productMs : nullptr,
            [&]() { return cc->EvalMult(Qold, Grot); });
        G = detail::TimedCall(
            cc, profile ? &report->layers[layer].updateMs : nullptr,
            [&]() { return cc->EvalSub(Gold, term); });

        if (layer + 1 < config.layers) {
            auto Qrot = cc->EvalRotate(Qold, rotation);
            Q = detail::TimedCall(
                cc, profile ? &report->layers[layer].auxiliaryMs : nullptr,
                [&]() {
                    return cc->EvalNegate(cc->EvalMult(Qold, Qrot));
                });
        }
        if (profile) {
            detail::Synchronize(cc);
            report->layers[layer].wallMs =
                Milliseconds(layerBegin, Clock::now());
        }
    }
    return G;
}

inline Ciphertext<DCRTPoly> EvalSubKSBaseline(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* timing = nullptr) {

    if (!cc || !cA || !cB || !masks.active || !masks.borrow ||
        (config.guardFreeLayout && !masks.propagate)) {
        throw std::invalid_argument("EvalSubKSBaseline received invalid input");
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

    auto G = EvalBorrowPrefixBaseline(
        cc, cA, cB, config, masks, profile ? &report : nullptr);

    auto borrow = detail::TimedCall(
        cc, profile ? &report.borrowRotateMs : nullptr,
        [&]() { return cc->EvalRotate(G, config.directionSign); });

    const auto outputBegin = Clock::now();
    borrow = cc->EvalMult(borrow, masks.borrow);
    G = cc->EvalMult(G, masks.active);
    if (profile) {
        detail::Synchronize(cc);
        report.outputTermsMs = Milliseconds(outputBegin, Clock::now());
    }

    const auto reconstructBegin = Clock::now();
    auto result = cc->EvalSub(cA, cB);
    result = cc->EvalSub(result, borrow);
    result = cc->EvalAdd(result, cc->EvalAdd(G, G));
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

/**
 * Packs the borrow-affine state Z=Q/2+iG, where
 *
 *   G=(not A) and B, P=not(A xor B), Q=-P.
 *
 * Q is used because XOR-1 is a ciphertext-scalar subtraction with stable
 * level/scale behavior in FIDES. Guarded callers retain Q=-1 in guard slots.
 * Guard-free callers set Q=0 at each word head in the existing Q/2 product,
 * which creates an absorbing segment delimiter without another operation.
 */
inline Ciphertext<DCRTPoly> PackComplexBorrowState(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* report,
    const AuditCallback& audit) {

    if (!cc || !cA || !cB || !masks.iActive ||
        (config.guardFreeLayout && !masks.halfPropagate)) {
        throw std::invalid_argument(
            "PackComplexBorrowState received invalid input or masks");
    }

    const bool profile = report != nullptr;
    const auto preprocessBegin = Clock::now();
    auto ab = cc->EvalMult(cA, cB);
    auto xorAB = cc->EvalSub(
        cc->EvalAdd(cA, cB), cc->EvalAdd(ab, ab));
    auto g = cc->EvalSub(cB, ab);
    auto q = cc->EvalSub(xorAB, 1.0);
    if (profile) {
        detail::Synchronize(cc);
        report->preprocessMs =
            Milliseconds(preprocessBegin, Clock::now());
    }
    if (audit) {
        audit("g", -1, g);
        audit("q=-p", -1, q);
    }

    const auto packBegin = Clock::now();
    auto halfQ = config.guardFreeLayout
        ? cc->EvalMult(q, masks.halfPropagate)
        : cc->EvalMult(q, 0.5);
    auto iG = cc->EvalMult(g, masks.iActive);
    auto state = cc->EvalAdd(halfQ, iG);
    if (profile) {
        detail::Synchronize(cc);
        report->packStateMs = Milliseconds(packBegin, Clock::now());
    }
    if (audit) {
        audit("Z", -1, state);
    }
    return state;
}

inline Ciphertext<DCRTPoly> EvalBorrowPrefixComplexStandard(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* report = nullptr,
    const AuditCallback& audit = {}) {

    const bool profile = report != nullptr;
    if (profile) {
        report->layers.resize(config.layers);
    }
    auto state = PackComplexBorrowState(
        cc, cA, cB, config, masks, report, audit);

    for (uint32_t layer = 0; layer < config.layers; ++layer) {
        const auto layerBegin = Clock::now();
        const int32_t rotation =
            config.directionSign * static_cast<int32_t>(1U << layer);
        const auto oldState = state;
        auto transforms = detail::TimedCall(
            cc, profile ? &report->layers[layer].transformMs : nullptr,
            [&]() {
                return std::make_pair(
                    cc->EvalRotate(oldState, rotation),
                    cc->EvalConjugate(oldState));
            });
        auto product = detail::TimedCall(
            cc, profile ? &report->layers[layer].productMs : nullptr,
            [&]() {
                auto propagate =
                    cc->EvalAdd(oldState, transforms.second);
                return cc->EvalMult(propagate, transforms.first);
            });
        auto keepGenerate = detail::TimedCall(
            cc, profile ? &report->layers[layer].auxiliaryMs : nullptr,
            [&]() {
                return cc->EvalMult(
                    cc->EvalSub(oldState, transforms.second), 0.5);
            });
        state = detail::TimedCall(
            cc, profile ? &report->layers[layer].updateMs : nullptr,
            [&]() { return cc->EvalSub(keepGenerate, product); });

        if (audit) {
            audit("standard-Z", static_cast<int32_t>(layer), state);
        }
        if (profile) {
            detail::Synchronize(cc);
            report->layers[layer].wallMs =
                Milliseconds(layerBegin, Clock::now());
        }
    }
    return state;
}

inline Ciphertext<DCRTPoly> EvalBorrowPrefixComplexHoisted(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* report = nullptr,
    const AuditCallback& audit = {}) {

    const bool profile = report != nullptr;
    if (profile) {
        report->layers.resize(config.layers);
    }
    auto state = PackComplexBorrowState(
        cc, cA, cB, config, masks, report, audit);

    for (uint32_t layer = 0; layer < config.layers; ++layer) {
        const auto layerBegin = Clock::now();
        const int32_t rotation =
            config.directionSign * static_cast<int32_t>(1U << layer);
        const auto oldState = state;
        auto pair = detail::TimedCall(
            cc, profile ? &report->layers[layer].transformMs : nullptr,
            [&]() {
                return cc->EvalRotateConjugateHoisted(oldState, rotation);
            });

        auto product = detail::TimedCall(
            cc, profile ? &report->layers[layer].productMs : nullptr,
            [&]() {
                auto propagate = cc->EvalAdd(oldState, pair.second);
                return cc->EvalMult(propagate, pair.first);
            });
        auto keepGenerate = detail::TimedCall(
            cc, profile ? &report->layers[layer].auxiliaryMs : nullptr,
            [&]() {
                return cc->EvalMult(
                    cc->EvalSub(oldState, pair.second), 0.5);
            });
        state = detail::TimedCall(
            cc, profile ? &report->layers[layer].updateMs : nullptr,
            [&]() { return cc->EvalSub(keepGenerate, product); });

        if (profile) {
            detail::Synchronize(cc);
            report->layers[layer].wallMs =
                Milliseconds(layerBegin, Clock::now());
        }
    }
    return state;
}

inline Ciphertext<DCRTPoly> EvalBorrowPrefixComplexTwisted(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* report = nullptr,
    const AuditCallback& audit = {}) {

    const bool profile = report != nullptr;
    if (profile) {
        report->layers.resize(config.layers);
    }
    auto state = PackComplexBorrowState(
        cc, cA, cB, config, masks, report, audit);

    uint32_t publicTag = 1;
    for (uint32_t layer = 0; layer < config.layers; ++layer) {
        const auto layerBegin = Clock::now();
        const int32_t rotation =
            config.directionSign * static_cast<int32_t>(publicTag);
        const auto oldState = state;

        auto conjugate = detail::TimedCall(
            cc, profile ? &report->layers[layer].transformMs : nullptr,
            [&]() { return cc->EvalConjugate(oldState); });
        auto product = detail::TimedCall(
            cc, profile ? &report->layers[layer].productMs : nullptr,
            [&]() {
                auto propagate = cc->EvalAdd(oldState, conjugate);
                return cc->EvalTwistedProduct(
                    propagate, oldState, rotation);
            });
        auto keepGenerate = detail::TimedCall(
            cc, profile ? &report->layers[layer].auxiliaryMs : nullptr,
            [&]() {
                return cc->EvalMult(
                    cc->EvalSub(oldState, conjugate), 0.5);
            });
        state = detail::TimedCall(
            cc, profile ? &report->layers[layer].updateMs : nullptr,
            [&]() { return cc->EvalSub(keepGenerate, product); });
        publicTag <<= 1U;

        if (audit) {
            audit("twisted-Z", static_cast<int32_t>(layer), state);
        }
        if (profile) {
            detail::Synchronize(cc);
            report->layers[layer].wallMs =
                Milliseconds(layerBegin, Clock::now());
        }
    }
    return state;
}

inline Ciphertext<DCRTPoly> ReconstructComplexDifference(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    Ciphertext<DCRTPoly> state,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* report,
    const AuditCallback& audit) {

    const bool profile = report != nullptr;
    auto borrowState = detail::TimedCall(
        cc, profile ? &report->borrowRotateMs : nullptr,
        [&]() { return cc->EvalRotate(state, config.directionSign); });

    const auto outputBegin = Clock::now();
    auto plusTwoGenerate =
        cc->EvalMult(state, masks.minusTwoIActive);
    auto minusBorrow =
        cc->EvalMult(borrowState, masks.plusIBorrow);
    if (profile) {
        detail::Synchronize(cc);
        report->outputTermsMs = Milliseconds(outputBegin, Clock::now());
    }

    const auto reconstructBegin = Clock::now();
    // D=A-B-r_i+2r_(i+1). The real parts of the two complex
    // plaintext products above are +2G and -borrow respectively.
    auto result = cc->EvalSub(cA, cB);
    result = cc->EvalAdd(result, plusTwoGenerate);
    result = cc->EvalAdd(result, minusBorrow);
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
        result = cc->EvalMult(cc->EvalAdd(result, conjugate), 0.5);
        if (profile) {
            detail::Synchronize(cc);
            report->projectionMs =
                Milliseconds(projectionBegin, Clock::now());
            report->projectedToReal = true;
        }
    }
    return result;
}

inline Ciphertext<DCRTPoly> EvalSubKSComplexStandard(
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
    auto state = EvalBorrowPrefixComplexStandard(
        cc, cA, cB, config, masks,
        profile ? &report : nullptr, audit);

    auto result = ReconstructComplexDifference(
        cc, cA, cB, state, config, masks,
        profile ? &report : nullptr, audit);
    if (profile) {
        report.totalMs = Milliseconds(totalBegin, Clock::now());
    }
    return result;
}

inline Ciphertext<DCRTPoly> EvalSubKSComplexHoisted(
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
    auto state = EvalBorrowPrefixComplexHoisted(
        cc, cA, cB, config, masks,
        profile ? &report : nullptr, audit);

    auto result = ReconstructComplexDifference(
        cc, cA, cB, state, config, masks,
        profile ? &report : nullptr, audit);
    if (profile) {
        report.totalMs = Milliseconds(totalBegin, Clock::now());
    }
    return result;
}

inline Ciphertext<DCRTPoly> EvalSubKSComplexTwisted(
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
    auto state = EvalBorrowPrefixComplexTwisted(
        cc, cA, cB, config, masks,
        profile ? &report : nullptr, audit);

    auto result = ReconstructComplexDifference(
        cc, cA, cB, state, config, masks,
        profile ? &report : nullptr, audit);
    if (profile) {
        report.totalMs = Milliseconds(totalBegin, Clock::now());
    }
    return result;
}

struct BorrowPrefixState {
    Ciphertext<DCRTPoly> value;
    bool complexPacked = false;
};

/**
 * Evaluates only the associative borrow-prefix network. The returned state is
 * G for baseline mode and Z=Q/2+iG for either complex mode. This interface is
 * shared by subtraction and unsigned comparison circuits.
 */
inline BorrowPrefixState EvalConfiguredBorrowPrefixState(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* timing = nullptr,
    const AuditCallback& audit = {}) {

    if (!cc || !cA || !cB || !masks.active || !masks.borrow) {
        throw std::invalid_argument(
            "EvalConfiguredBorrowPrefixState received invalid input");
    }
    if (config.algorithm != Algorithm::Baseline && !masks.iActive) {
        throw std::invalid_argument(
            "complex borrow-prefix masks are not initialized");
    }
    if (config.guardFreeLayout &&
        config.algorithm == Algorithm::Baseline &&
        !masks.propagate) {
        throw std::invalid_argument(
            "head-zero baseline propagate mask is not initialized");
    }
    if (config.guardFreeLayout &&
        config.algorithm != Algorithm::Baseline &&
        !masks.halfPropagate) {
        throw std::invalid_argument(
            "head-zero complex propagate mask is not initialized");
    }

    TimingReport local;
    TimingReport& report = timing ? *timing : local;
    const bool profile = timing != nullptr;
    if (profile) {
        report = TimingReport{};
        report.layers.resize(config.layers);
        detail::Synchronize(cc);
    }
    const auto begin = Clock::now();

    BorrowPrefixState result;
    switch (config.algorithm) {
        case Algorithm::Baseline:
            result.value = EvalBorrowPrefixBaseline(
                cc, cA, cB, config, masks,
                profile ? &report : nullptr);
            break;
        case Algorithm::ComplexStandard:
            result.value = EvalBorrowPrefixComplexStandard(
                cc, cA, cB, config, masks,
                profile ? &report : nullptr, audit);
            result.complexPacked = true;
            break;
        case Algorithm::ComplexHoisted:
            result.value = EvalBorrowPrefixComplexHoisted(
                cc, cA, cB, config, masks,
                profile ? &report : nullptr, audit);
            result.complexPacked = true;
            break;
        case Algorithm::ComplexTwisted:
            result.value = EvalBorrowPrefixComplexTwisted(
                cc, cA, cB, config, masks,
                profile ? &report : nullptr, audit);
            result.complexPacked = true;
            break;
    }
    if (profile) {
        detail::Synchronize(cc);
        report.totalMs = Milliseconds(begin, Clock::now());
    }
    return result;
}

inline Ciphertext<DCRTPoly> EvalConfiguredSubtractor(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* timing = nullptr,
    const AuditCallback& audit = {}) {
    switch (config.algorithm) {
        case Algorithm::Baseline:
            return EvalSubKSBaseline(
                cc, cA, cB, config, masks, timing);
        case Algorithm::ComplexStandard:
            return EvalSubKSComplexStandard(
                cc, cA, cB, config, masks, timing, audit);
        case Algorithm::ComplexHoisted:
            return EvalSubKSComplexHoisted(
                cc, cA, cB, config, masks, timing, audit);
        case Algorithm::ComplexTwisted:
            return EvalSubKSComplexTwisted(
                cc, cA, cB, config, masks, timing, audit);
    }
    throw std::logic_error("unknown subtractor algorithm");
}

/**
 * Initializes level-specific ModUp graph/workspace state with one disposable
 * chain. No user-visible output is retained. This is setup, not a benchmark
 * repetition; the caller times the following real subtraction exactly once.
 */
inline void InitializeConfiguredSubtractorRuntime(
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
        auto disposable = EvalSubKSComplexStandard(
            cc, cA, cB, config, masks);
        (void)disposable;
        detail::Synchronize(cc);
        return;
    }

    auto disposable = EvalSubKSComplexTwisted(
        cc, cA, cB, config, masks);
    (void)disposable;
    detail::Synchronize(cc);
}

}  // namespace kssubtractor
