#pragma once

#include <fideslib.hpp>

#include "cipher_divider_runtime_common.h"
#include "subtractor_ks_complex_hoisted_gpu.h"

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ksdivcipher {

using fideslib::Ciphertext;
using fideslib::CryptoContext;
using fideslib::DCRTPoly;
using fideslib::Plaintext;

struct TimingReport {
    double zeroDetectionMs = 0.0;
    double inputRefreshMs = 0.0;
    double multiplePreparationMs = 0.0;
    double nibbleFormationMs = 0.0;
    double candidateSubtractionMs = 0.0;
    double prefixRefreshMs = 0.0;
    double borrowProjectionMs = 0.0;
    double candidateMuxMs = 0.0;
    double stateRefreshMs = 0.0;
    double quotientUpdateMs = 0.0;
    double zeroResultSelectionMs = 0.0;
    double finalRefreshMs = 0.0;
    double levelAlignmentMs = 0.0;
    double setupExcludedMs = 0.0;
    double totalWallMs = 0.0;
    uint32_t radixRounds = 0;
    uint32_t candidateSubtractions = 0;
    uint32_t stateBootstraps = 0;
    uint32_t prefixBootstraps = 0;
    uint32_t totalBootstraps = 0;
    uint32_t booleanPurifications = 0;
    uint32_t asyncPreparationBatches = 0;
};

struct DivisionResult {
    Ciphertext<DCRTPoly> quotient;
    Ciphertext<DCRTPoly> remainder;
};

namespace detail {

using Clock = std::chrono::steady_clock;

inline double Milliseconds(const Clock::time_point& begin,
                           const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(
        end - begin).count();
}

inline void Synchronize(const CryptoContext<DCRTPoly>& cc) {
    cc->Synchronize();
}

inline uint32_t RemainingLevels(
    const RuntimeConfig& config,
    const Ciphertext<DCRTPoly>& ciphertext) {
    const uint32_t used =
        ciphertext->GetLevel() + ciphertext->GetNoiseScaleDeg();
    return config.multiplicativeDepth > used
        ? config.multiplicativeDepth - used
        : 0U;
}

inline void RequireState(
    const char* label,
    const Ciphertext<DCRTPoly>& value,
    uint32_t expectedLevel,
    uint32_t expectedScaleDegree) {
    if (!value ||
        value->GetLevel() != expectedLevel ||
        value->GetNoiseScaleDeg() != expectedScaleDegree) {
        std::ostringstream message;
        message << label << " has state ";
        if (value) {
            message << value->GetLevel() << "/"
                    << value->GetNoiseScaleDeg();
        }
        else {
            message << "<null>";
        }
        message << ", expected " << expectedLevel << "/"
                << expectedScaleDegree;
        throw std::runtime_error(message.str());
    }
}

inline std::vector<double> ActiveValues(
    const RuntimeConfig& config,
    uint32_t width) {
    std::vector<double> values(config.slots, 0.0);
    for (uint32_t word = 0; word < config.words; ++word) {
        for (uint32_t bit = 0; bit < width; ++bit) {
            values[ksword::Slot(config, word, bit)] = 1.0;
        }
    }
    return values;
}

inline std::vector<double> CarryValues(
    const RuntimeConfig& config,
    uint32_t width) {
    auto values = ActiveValues(config, width);
    for (uint32_t word = 0; word < config.words; ++word) {
        values[ksword::Slot(config, word, 0)] = 0.0;
    }
    return values;
}

inline std::vector<double> BitZeroValues(
    const RuntimeConfig& config) {
    std::vector<double> values(config.slots, 0.0);
    for (uint32_t word = 0; word < config.words; ++word) {
        values[ksword::Slot(config, word, 0)] = 1.0;
    }
    return values;
}

inline std::vector<double> LowNibbleValues(
    const RuntimeConfig& config) {
    std::vector<double> values(config.slots, 0.0);
    for (uint32_t word = 0; word < config.words; ++word) {
        for (uint32_t bit = 0; bit < kRadixBits; ++bit) {
            values[ksword::Slot(config, word, bit)] = 1.0;
        }
    }
    return values;
}

inline std::vector<double> MsbValues(
    const RuntimeConfig& config,
    uint32_t width) {
    std::vector<double> values(config.slots, 0.0);
    for (uint32_t word = 0; word < config.words; ++word) {
        values[ksword::Slot(config, word, width - 1U)] = 1.0;
    }
    return values;
}

inline std::vector<std::complex<double>> ImaginaryValues(
    const std::vector<double>& realValues,
    double scale) {
    std::vector<std::complex<double>> values(realValues.size());
    for (size_t index = 0; index < realValues.size(); ++index) {
        values[index] =
            std::complex<double>(0.0, scale * realValues[index]);
    }
    return values;
}

struct CachedPlaintext {
    const void* context = nullptr;
    uint64_t contentHash = 0;
    size_t valueCount = 0;
    uint32_t level = 0;
    uint32_t scaleDegree = 0;
    bool complexValues = false;
    Plaintext plaintext;
};

inline uint64_t HashMix(uint64_t hash, uint64_t value) {
    constexpr uint64_t prime = 1099511628211ULL;
    hash ^= value;
    hash *= prime;
    return hash;
}

inline uint64_t HashValues(const std::vector<double>& values) {
    uint64_t hash = 1469598103934665603ULL;
    for (double value : values) {
        uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        hash = HashMix(hash, bits);
    }
    return hash;
}

inline uint64_t HashValues(
    const std::vector<std::complex<double>>& values) {
    uint64_t hash = 1469598103934665603ULL;
    for (const auto& value : values) {
        uint64_t realBits = 0;
        uint64_t imaginaryBits = 0;
        const double real = value.real();
        const double imaginary = value.imag();
        std::memcpy(&realBits, &real, sizeof(realBits));
        std::memcpy(&imaginaryBits, &imaginary, sizeof(imaginaryBits));
        hash = HashMix(HashMix(hash, realBits), imaginaryBits);
    }
    return hash;
}

inline std::vector<CachedPlaintext>& LoadedPlaintextCache() {
    static std::vector<CachedPlaintext> cache;
    return cache;
}

inline Plaintext MakeAndLoad(
    const CryptoContext<DCRTPoly>& cc,
    const RuntimeConfig& config,
    const std::vector<double>& values,
    uint32_t level,
    uint32_t scaleDegree = 1) {
    const uint64_t contentHash = HashValues(values);
    auto& cache = LoadedPlaintextCache();
    for (const auto& entry : cache) {
        if (entry.context == cc.get() &&
            entry.contentHash == contentHash &&
            entry.valueCount == values.size() &&
            entry.level == level &&
            entry.scaleDegree == scaleDegree &&
            !entry.complexValues) {
            return entry.plaintext;
        }
    }
    auto plaintext = cc->MakeCKKSPackedPlaintext(
        values, scaleDegree, level, nullptr, config.slots);
    plaintext->SetLength(config.slots);
    cc->LoadPlaintext(plaintext);
    cache.push_back({
        cc.get(), contentHash, values.size(), level, scaleDegree,
        false, plaintext});
    return plaintext;
}

inline Plaintext MakeAndLoadComplex(
    const CryptoContext<DCRTPoly>& cc,
    const RuntimeConfig& config,
    const std::vector<std::complex<double>>& values,
    uint32_t level) {
    const uint64_t contentHash = HashValues(values);
    auto& cache = LoadedPlaintextCache();
    for (const auto& entry : cache) {
        if (entry.context == cc.get() &&
            entry.contentHash == contentHash &&
            entry.valueCount == values.size() &&
            entry.level == level &&
            entry.scaleDegree == 1U &&
            entry.complexValues) {
            return entry.plaintext;
        }
    }
    auto plaintext = cc->MakeCKKSPackedPlaintext(
        values, 1, level, nullptr, config.slots);
    plaintext->SetLength(config.slots);
    cc->LoadPlaintext(plaintext);
    cache.push_back({
        cc.get(), contentHash, values.size(), level, 1U,
        true, plaintext});
    return plaintext;
}

inline Ciphertext<DCRTPoly> ApplyMask(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> value,
    const RuntimeConfig& config,
    const std::vector<double>& maskValues,
    TimingReport* timing) {
    const auto setupBegin = Clock::now();
    auto mask = MakeAndLoad(
        cc, config, maskValues, value->GetLevel());
    Synchronize(cc);
    if (timing) {
        timing->setupExcludedMs +=
            Milliseconds(setupBegin, Clock::now());
    }
    return cc->EvalMult(value, mask);
}

inline Ciphertext<DCRTPoly> OneMinusWithinMask(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& value,
    const RuntimeConfig& config,
    const std::vector<double>& maskValues,
    TimingReport* timing) {
    const auto setupBegin = Clock::now();
    auto mask = MakeAndLoad(
        cc,
        config,
        maskValues,
        value->GetLevel(),
        value->GetNoiseScaleDeg());
    Synchronize(cc);
    if (timing) {
        timing->setupExcludedMs +=
            Milliseconds(setupBegin, Clock::now());
    }

    // FIDES implements Plaintext-Ciphertext by multScalar(-1), which
    // rescales a scaleDeg=2 ciphertext. Build Enc(mask) at the same state
    // first, so both following subtractions are genuinely level preserving.
    auto encryptedMask = cc->EvalSub(value, value);
    encryptedMask = cc->EvalAdd(encryptedMask, mask);
    return cc->EvalSub(encryptedMask, value);
}

inline Ciphertext<DCRTPoly> MaskWidth(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> value,
    const RuntimeConfig& config,
    uint32_t width,
    TimingReport* timing) {
    return ApplyMask(
        cc,
        std::move(value),
        config,
        ActiveValues(config, width),
        timing);
}

inline Ciphertext<DCRTPoly> PadToLevel(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> value,
    uint32_t targetLevel,
    const RuntimeConfig& config,
    uint32_t width,
    TimingReport* timing);

inline Ciphertext<DCRTPoly> ProjectBoolean(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> value,
    const RuntimeConfig& config,
    uint32_t width,
    TimingReport* timing) {
    if (value->GetNoiseScaleDeg() != 2U) {
        throw std::runtime_error(
            "Boolean projection expects scaleDeg=2");
    }
    const uint32_t inputLevel = value->GetLevel();
    auto squared = cc->EvalMult(value, value);
    auto cubed = cc->EvalMult(squared, value);
    auto threeSquared =
        cc->EvalAdd(squared, cc->EvalAdd(squared, squared));
    auto twoCubed = cc->EvalAdd(cubed, cubed);
    auto purified = cc->EvalSub(threeSquared, twoCubed);
    purified = MaskWidth(
        cc, std::move(purified), config, width, timing);
    Synchronize(cc);
    if (purified->GetLevel() !=
            inputLevel + kBooleanPurificationLevels ||
        purified->GetNoiseScaleDeg() != 2U) {
        throw std::runtime_error(
            "Boolean projection consumed an unexpected number of modulus "
            "levels");
    }
    if (timing) {
        ++timing->booleanPurifications;
    }
    return purified;
}

inline Ciphertext<DCRTPoly> BootstrapRaw(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> value,
    const RuntimeConfig& config,
    TimingReport* timing,
    bool stateRefresh = false) {
    auto refreshed = cc->EvalBootstrap(value);
    Synchronize(cc);
    if (refreshed->GetLevel() != kBootstrapOutputLevel ||
        refreshed->GetNoiseScaleDeg() != 2U) {
        std::ostringstream message;
        message
            << "unexpected Bootstrap output state: expected "
            << kBootstrapOutputLevel << "/2, got "
            << refreshed->GetLevel() << "/"
            << refreshed->GetNoiseScaleDeg()
            << " for levelBudget={"
            << kBootstrapEncodingLevelBudget << ","
            << kBootstrapDecodingLevelBudget << "}";
        throw std::runtime_error(message.str());
    }

    if (timing) {
        ++timing->totalBootstraps;
        if (stateRefresh) {
            ++timing->stateBootstraps;
        }
    }
    return refreshed;
}

inline Ciphertext<DCRTPoly> BootstrapThenProject(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> value,
    const RuntimeConfig& config,
    uint32_t width,
    TimingReport* timing,
    double* elapsedMs,
    bool stateRefresh = false) {
    Synchronize(cc);
    const auto begin = Clock::now();
    auto refreshed = BootstrapRaw(
        cc,
        std::move(value),
        config,
        timing,
        stateRefresh);
    refreshed = ProjectBoolean(
        cc,
        std::move(refreshed),
        config,
        width,
        timing);
    RequireState(
        "post-Bootstrap Boolean state",
        refreshed,
        kCanonicalStateLevel,
        2U);
    if (elapsedMs) {
        *elapsedMs += Milliseconds(begin, Clock::now());
    }
    return refreshed;
}

inline Ciphertext<DCRTPoly> PadToLevel(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> value,
    uint32_t targetLevel,
    const RuntimeConfig& config,
    uint32_t width,
    TimingReport* timing) {
    if (value->GetLevel() > targetLevel) {
        std::ostringstream message;
        message
            << "cannot align ciphertext "
            << value->GetLevel() << "/"
            << value->GetNoiseScaleDeg()
            << " to earlier target " << targetLevel << "/2";
        throw std::runtime_error(message.str());
    }
    // A fresh scaleDeg=1 ciphertext is first normalized to scaleDeg=2 by an
    // identity plaintext multiplication without dropping a modulus tower.
    // Once normalized, each identical multiplication advances one level.
    while (value->GetLevel() < targetLevel ||
           value->GetNoiseScaleDeg() != 2U) {
        const uint32_t previousLevel = value->GetLevel();
        const uint32_t previousScaleDegree =
            value->GetNoiseScaleDeg();
        if (previousScaleDegree == 0U ||
            previousScaleDegree > 2U) {
            throw std::runtime_error(
                "plaintext-mask alignment expects scaleDeg 1 or 2");
        }
        value = MaskWidth(
            cc, std::move(value), config, width, timing);
        Synchronize(cc);

        const uint32_t nextLevel = value->GetLevel();
        const uint32_t nextScaleDegree =
            value->GetNoiseScaleDeg();
        const bool normalizedScale =
            nextLevel == previousLevel &&
            previousScaleDegree == 1U &&
            nextScaleDegree == 2U;
        const bool droppedOneLevel =
            nextLevel == previousLevel + 1U &&
            nextScaleDegree == 2U;
        if ((!normalizedScale && !droppedOneLevel) ||
            nextLevel > targetLevel) {
            std::ostringstream message;
            message
                << "unexpected plaintext-mask alignment transition: "
                << previousLevel << "/" << previousScaleDegree
                << " -> " << nextLevel << "/" << nextScaleDegree
                << ", target=" << targetLevel << "/2";
            throw std::runtime_error(
                message.str());
        }
    }
    return value;
}

inline void AlignLevels(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly>* left,
    Ciphertext<DCRTPoly>* right,
    const RuntimeConfig& config,
    uint32_t width,
    TimingReport* timing) {
    const auto begin = Clock::now();
    const uint32_t target =
        std::max((*left)->GetLevel(), (*right)->GetLevel());
    *left = PadToLevel(
        cc, std::move(*left), target,
        config, width, timing);
    *right = PadToLevel(
        cc, std::move(*right), target,
        config, width, timing);
    if ((*left)->GetNoiseScaleDeg() !=
        (*right)->GetNoiseScaleDeg()) {
        throw std::runtime_error(
            "level-aligned ciphertexts have different scale degrees");
    }
    if (timing) {
        timing->levelAlignmentMs +=
            Milliseconds(begin, Clock::now());
    }
}

inline ksadder::RuntimeConfig MakeSubtractorConfig(
    const RuntimeConfig& config) {
    ksadder::RuntimeConfig result;
    result.wordBits = config.wideBits();
    result.guardBits = config.stride - config.wideBits();
    result.stride = config.stride;
    result.words = config.words;
    result.slots = config.slots;
    result.layers = config.wideLayers();
    result.multiplicativeDepth = config.multiplicativeDepth;
    result.directionSign = -1;
    result.backend =
        config.backend == ksword::Backend::Gpu
            ? ksadder::Backend::Gpu
            : ksadder::Backend::Cpu;
    result.algorithm = ksadder::Algorithm::ComplexStandard;
    result.devices = config.devices;
    result.bootstrap = true;
    result.detailedProfile = config.detailedProfile;
    result.graphInit = config.graphInit;
    result.projectOutputToReal = true;
    result.inputLevel = config.expectedTrialInputLevel();
    result.circuitLevelCost = config.wideLayers() + 4U;
    result.expectedOutputLevel =
        result.inputLevel + result.circuitLevelCost;
    result.wordsExplicit = true;
    result.layersExplicit = true;
    result.algorithmExplicit = true;
    return result;
}

struct BorrowRuntime {
    uint32_t inputLevel = 0;
    uint32_t inputScaleDegree = 0;
    kssubtractor::Masks masks;
    Plaintext minusHalfIMsb;
};

inline BorrowRuntime BuildBorrowRuntime(
    const CryptoContext<DCRTPoly>& cc,
    const RuntimeConfig& config,
    uint32_t inputLevel,
    uint32_t inputScaleDegree) {
    const uint32_t width = config.wideBits();
    const auto activeValues = ActiveValues(config, width);
    const auto borrowValues = CarryValues(config, width);
    const uint32_t packLevel = inputLevel + inputScaleDegree;
    const uint32_t outputLevel =
        packLevel + config.wideLayers() + 1U;
    if (outputLevel >= config.multiplicativeDepth) {
        throw std::runtime_error(
            "wide subtractor masks exceed the configured modulus chain");
    }

    BorrowRuntime runtime;
    runtime.inputLevel = inputLevel;
    runtime.inputScaleDegree = inputScaleDegree;
    runtime.masks.active = MakeAndLoad(
        cc, config, activeValues, inputLevel);
    runtime.masks.borrow = MakeAndLoad(
        cc, config, borrowValues, inputLevel);
    runtime.masks.iActive = MakeAndLoadComplex(
        cc,
        config,
        ImaginaryValues(activeValues, 1.0),
        packLevel);
    runtime.masks.minusTwoIActive = MakeAndLoadComplex(
        cc,
        config,
        ImaginaryValues(activeValues, -2.0),
        outputLevel);
    runtime.masks.plusIBorrow = MakeAndLoadComplex(
        cc,
        config,
        ImaginaryValues(borrowValues, 1.0),
        outputLevel);
    runtime.minusHalfIMsb = MakeAndLoadComplex(
        cc,
        config,
        ImaginaryValues(MsbValues(config, width), -0.5),
        outputLevel);
    Synchronize(cc);
    return runtime;
}

class BorrowRuntimeCache {
public:
    BorrowRuntime& Get(
        const CryptoContext<DCRTPoly>& cc,
        const RuntimeConfig& config,
        const Ciphertext<DCRTPoly>& left,
        const Ciphertext<DCRTPoly>& right,
        TimingReport* timing) {
        if (left->GetLevel() != right->GetLevel() ||
            left->GetNoiseScaleDeg() != right->GetNoiseScaleDeg()) {
            throw std::invalid_argument(
                "borrow runtime requires level/scale-aligned inputs");
        }
        const uint32_t level = left->GetLevel();
        const uint32_t scaleDegree = left->GetNoiseScaleDeg();
        for (auto& item : entries_) {
            if (item.inputLevel == level &&
                item.inputScaleDegree == scaleDegree) {
                return item;
            }
        }

        const auto begin = Clock::now();
        entries_.push_back(
            BuildBorrowRuntime(cc, config, level, scaleDegree));
        auto& item = entries_.back();
        if (config.graphInit) {
            auto subConfig =
                MakeSubtractorConfig(config);
            auto prefix =
                kssubtractor::EvalConfiguredBorrowPrefixState(
                    cc,
                    left,
                    right,
                    subConfig,
                    item.masks);
            auto disposableDifference =
                kssubtractor::ReconstructComplexDifference(
                    cc,
                    left,
                    right,
                    prefix.value,
                    subConfig,
                    item.masks,
                    nullptr,
                    {});
            auto conjugate =
                cc->EvalConjugate(prefix.value);
            auto atMsb = cc->EvalMult(
                cc->EvalSub(prefix.value, conjugate),
                item.minusHalfIMsb);
            auto disposableBorrow = cc->EvalRotate(
                atMsb,
                static_cast<int32_t>(config.wideBits() - 1U));
            (void)disposableDifference;
            (void)disposableBorrow;
            Synchronize(cc);
        }
        if (timing) {
            timing->setupExcludedMs +=
                Milliseconds(begin, Clock::now());
        }
        return item;
    }

private:
    std::vector<BorrowRuntime> entries_;
};

struct TrialResult {
    Ciphertext<DCRTPoly> state;
    Ciphertext<DCRTPoly> quotientBit;
};

inline Ciphertext<DCRTPoly> RotateByPowers(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& value,
    int32_t rotation) {
    if (rotation == 0) {
        return value;
    }
    const bool negative = rotation < 0;
    uint32_t amount = negative
        ? static_cast<uint32_t>(-static_cast<int64_t>(rotation))
        : static_cast<uint32_t>(rotation);
    int32_t step = 1;
    auto result = value;
    std::vector<Ciphertext<DCRTPoly>> keepAlive;
    keepAlive.reserve(10);
    while (amount != 0U) {
        if ((amount & 1U) != 0U) {
            auto next = cc->EvalRotate(
                result, negative ? -step : step);
            keepAlive.push_back(result);
            result = std::move(next);
        }
        amount >>= 1U;
        step <<= 1;
    }
    Synchronize(cc);
    return result;
}

inline Ciphertext<DCRTPoly> BroadcastBitZero(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> bit,
    const RuntimeConfig& config,
    uint32_t width) {
    if (!ksword::IsPowerOfTwo(width) ||
        width > config.wideBits()) {
        throw std::invalid_argument(
            "broadcast width must be a power of two within the active lane");
    }
    auto result = std::move(bit);
    for (uint32_t layer = 0;
         layer < ksword::IntegerLog2(width);
         ++layer) {
        const int32_t rotation =
            -static_cast<int32_t>(1U << layer);
        result = cc->EvalAdd(
            result, cc->EvalRotate(result, rotation));
    }
    return result;
}

struct StableBorrowResult {
    Ciphertext<DCRTPoly> difference;
    Ciphertext<DCRTPoly> borrowAtZero;
};

/**
 * Evaluates one exact wide subtraction with the precision schedule used by
 * the original divider.  A continuous 2w-bit prefix lets approximation and
 * key-switch noise grow through too many dependent products.  We therefore
 * refresh the Boolean G and P channels after four prefix layers, then finish
 * the remaining layers from canonical ciphertexts.
 */
inline StableBorrowResult EvalMidpointRefreshedBorrow(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& left,
    const Ciphertext<DCRTPoly>& right,
    const RuntimeConfig& config,
    TimingReport* timing) {
    const uint32_t width = config.wideBits();
    const uint32_t layers = config.wideLayers();
    const uint32_t refreshAfter =
        config.prefixLayersBeforeRefresh();
    const uint32_t bottomLevel =
        config.expectedTrialOutputLevel();

    auto ab = cc->EvalMult(left, right);
    auto xorValue = cc->EvalSub(
        cc->EvalAdd(left, right),
        cc->EvalAdd(ab, ab));
    auto generate = cc->EvalSub(right, ab);
    auto propagate = OneMinusWithinMask(
        cc,
        xorValue,
        config,
        ActiveValues(config, width),
        timing);

    std::vector<Ciphertext<DCRTPoly>> keepAlive;
    keepAlive.reserve(5U * layers + 8U);
    for (uint32_t layer = 0; layer < layers; ++layer) {
        const int32_t rotation =
            -static_cast<int32_t>(1U << layer);
        const auto oldGenerate = generate;
        const auto oldPropagate = propagate;
        auto rotatedGenerate =
            cc->EvalRotate(oldGenerate, rotation);
        auto rotatedPropagate =
            cc->EvalRotate(oldPropagate, rotation);
        auto term = cc->EvalMult(
            rotatedGenerate, oldPropagate);
        auto nextGenerate =
            cc->EvalAdd(oldGenerate, term);
        const bool needNextPropagate =
            layer + 1U < layers ||
            layer + 1U == refreshAfter;
        Ciphertext<DCRTPoly> nextPropagate;
        if (needNextPropagate) {
            nextPropagate = cc->EvalMult(
                rotatedPropagate, oldPropagate);
        }

        keepAlive.push_back(oldGenerate);
        keepAlive.push_back(oldPropagate);
        keepAlive.push_back(std::move(rotatedGenerate));
        keepAlive.push_back(std::move(rotatedPropagate));
        keepAlive.push_back(std::move(term));
        generate = std::move(nextGenerate);
        if (needNextPropagate) {
            propagate = std::move(nextPropagate);
        }

        if (layer + 1U == refreshAfter) {
            Synchronize(cc);
            const uint32_t naturalRefreshLevel =
                kCanonicalStateLevel + 1U + refreshAfter;
            RequireState(
                "midpoint G before bottom placement",
                generate,
                naturalRefreshLevel,
                2U);
            RequireState(
                "midpoint P before bottom placement",
                propagate,
                naturalRefreshLevel,
                2U);
            generate = PadToLevel(
                cc,
                std::move(generate),
                bottomLevel,
                config,
                width,
                timing);
            propagate = PadToLevel(
                cc,
                std::move(propagate),
                bottomLevel,
                config,
                width,
                timing);
            generate = BootstrapThenProject(
                cc,
                std::move(generate),
                config,
                width,
                timing,
                timing ? &timing->prefixRefreshMs : nullptr);
            propagate = BootstrapThenProject(
                cc,
                std::move(propagate),
                config,
                width,
                timing,
                timing ? &timing->prefixRefreshMs : nullptr);
            if (timing) {
                timing->prefixBootstraps += 2U;
            }
            RequireState(
                "midpoint-refreshed G",
                generate,
                kCanonicalStateLevel,
                2U);
            RequireState(
                "midpoint-refreshed P",
                propagate,
                kCanonicalStateLevel,
                2U);
            keepAlive.clear();
        }
    }
    Synchronize(cc);
    const uint32_t finalPrefixLevel =
        kCanonicalStateLevel + config.prefixLayersAfterRefresh();
    RequireState(
        "completed midpoint-refreshed G prefix",
        generate,
        finalPrefixLevel,
        2U);

    auto borrowIn = cc->EvalRotate(generate, -1);
    borrowIn = ApplyMask(
        cc,
        std::move(borrowIn),
        config,
        CarryValues(config, width),
        timing);
    auto maskedGenerate = ApplyMask(
        cc,
        generate,
        config,
        ActiveValues(config, width),
        timing);

    auto difference = cc->EvalSub(left, right);
    difference = cc->EvalSub(difference, borrowIn);
    difference = cc->EvalAdd(
        difference,
        cc->EvalAdd(maskedGenerate, maskedGenerate));

    auto borrowAtZero = RotateByPowers(
        cc,
        generate,
        static_cast<int32_t>(width - 1U));
    borrowAtZero = ApplyMask(
        cc,
        std::move(borrowAtZero),
        config,
        BitZeroValues(config),
        timing);
    Synchronize(cc);
    const uint32_t expectedOutputLevel = finalPrefixLevel + 1U;
    RequireState(
        "midpoint-refreshed difference",
        difference,
        expectedOutputLevel,
        2U);
    RequireState(
        "midpoint-refreshed raw borrow",
        borrowAtZero,
        expectedOutputLevel,
        2U);
    return {
        std::move(difference),
        std::move(borrowAtZero),
    };
}

inline TrialResult EvalCandidateTrial(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> state,
    Ciphertext<DCRTPoly> multiple,
    const RuntimeConfig& config,
    BorrowRuntimeCache* cache,
    TimingReport* timing,
    uint32_t refreshedStateWidth) {
    if (refreshedStateWidth != config.wordBits &&
        refreshedStateWidth != config.wideBits()) {
        throw std::invalid_argument(
            "candidate refresh width must be w or 2w bits");
    }
    const uint32_t targetInputLevel =
        config.expectedTrialInputLevel();
    const auto prepareOperand = [&](Ciphertext<DCRTPoly> value) {
        if (value->GetLevel() > targetInputLevel ||
            value->GetNoiseScaleDeg() != 2U) {
            value = BootstrapThenProject(
                cc,
                std::move(value),
                config,
                config.wideBits(),
                timing,
                nullptr);
        }
        if (value->GetLevel() > targetInputLevel) {
            throw std::runtime_error(
                "Bootstrap output is later than the scheduled candidate-trial "
                "input; increase --depth");
        }
        value = PadToLevel(
            cc,
            std::move(value),
            targetInputLevel,
            config,
            config.wideBits(),
            timing);
        if (value->GetNoiseScaleDeg() != 2U) {
            throw std::runtime_error(
                "candidate trial requires canonical scaleDeg=2 inputs");
        }
        return value;
    };

    state = prepareOperand(std::move(state));
    multiple = prepareOperand(std::move(multiple));
    AlignLevels(
        cc, &state, &multiple,
        config, config.wideBits(), timing);
    const uint32_t required = config.trialLevelCost();
    if (state->GetLevel() != targetInputLevel ||
        multiple->GetLevel() != targetInputLevel ||
        RemainingLevels(config, state) != required) {
        throw std::runtime_error(
            "candidate operands were not placed at the deterministic "
            "bottom-level schedule");
    }
    (void)cache;

    Synchronize(cc);
    auto begin = Clock::now();
    auto subtraction = EvalMidpointRefreshedBorrow(
        cc, state, multiple, config, timing);
    auto difference = std::move(subtraction.difference);
    auto borrow = std::move(subtraction.borrowAtZero);
    Synchronize(cc);
    if (difference->GetLevel() > config.expectedTrialOutputLevel() ||
        borrow->GetLevel() > config.expectedTrialOutputLevel() ||
        difference->GetNoiseScaleDeg() != 2U ||
        borrow->GetNoiseScaleDeg() != 2U) {
        throw std::runtime_error(
            "midpoint-refreshed borrow path exceeded the scheduled "
            "Bootstrap boundary");
    }
    if (timing) {
        timing->candidateSubtractionMs +=
            Milliseconds(begin, Clock::now());
        ++timing->candidateSubtractions;
    }

    begin = Clock::now();
    // The borrow controls both the quotient bit and the encrypted mux. It is
    // therefore projected before either use. Deferring this projection until
    // the final quotient refresh lets Bootstrap approximation error enter the
    // next restoring trial and accumulate across all n trials.
    borrow = ProjectBoolean(
        cc,
        std::move(borrow),
        config,
        1U,
        timing);
    auto quotientBit = OneMinusWithinMask(
        cc,
        borrow,
        config,
        BitZeroValues(config),
        timing);
    Synchronize(cc);
    RequireState(
        "candidate quotient bit",
        quotientBit,
        borrow->GetLevel(),
        2U);
    if (timing) {
        timing->borrowProjectionMs +=
            Milliseconds(begin, Clock::now());
    }

    begin = Clock::now();
    auto borrowWide = BroadcastBitZero(
        cc, borrow, config, config.wideBits());
    // The source occupies bit zero of each stride. Doubling rotations fill
    // exactly [0, 2w) and cannot reach the following 2w guard, so a boundary
    // plaintext product would be redundant here.
    auto delta = cc->EvalSub(state, difference);
    AlignLevels(
        cc,
        &delta,
        &borrowWide,
        config,
        config.wideBits(),
        timing);
    auto selectedDelta = cc->EvalMult(delta, borrowWide);
    auto selectedState =
        cc->EvalAdd(difference, selectedDelta);
    selectedState = PadToLevel(
        cc,
        std::move(selectedState),
        config.expectedTrialOutputLevel(),
        config,
        refreshedStateWidth,
        timing);
    Synchronize(cc);
    if (selectedState->GetLevel() !=
            config.expectedTrialOutputLevel() ||
        RemainingLevels(config, selectedState) != 0U) {
        throw std::runtime_error(
            "candidate mux did not end at the scheduled Bootstrap boundary: "
            "expected level " +
            std::to_string(config.expectedTrialOutputLevel()) +
            ", got " + std::to_string(selectedState->GetLevel()));
    }
    if (timing) {
        timing->candidateMuxMs +=
            Milliseconds(begin, Clock::now());
    }

    // Bootstrap resets cryptographic noise; the Boolean projection follows
    // it so that the approximation introduced by Bootstrap is removed before
    // this state is reused by the next dependent trial.
    selectedState = BootstrapThenProject(
        cc,
        std::move(selectedState),
        config,
        refreshedStateWidth,
        timing,
        timing ? &timing->stateRefreshMs : nullptr,
        true);
    return {
        std::move(selectedState),
        std::move(quotientBit),
    };
}

inline Ciphertext<DCRTPoly> DetectDivisorZero(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& divisor,
    const RuntimeConfig& config,
    TimingReport* timing) {
    Synchronize(cc);
    const auto begin = Clock::now();
    auto suffixOr = divisor;
    for (uint32_t layer = 0;
         layer < ksword::IntegerLog2(config.wordBits);
         ++layer) {
        const int32_t rotation =
            static_cast<int32_t>(1U << layer);
        auto shifted = cc->EvalRotate(suffixOr, rotation);
        auto both = cc->EvalMult(suffixOr, shifted);
        suffixOr = cc->EvalSub(
            cc->EvalAdd(suffixOr, shifted), both);
    }
    auto anyBit = ApplyMask(
        cc,
        std::move(suffixOr),
        config,
        BitZeroValues(config),
        timing);

    auto zero = OneMinusWithinMask(
        cc,
        anyBit,
        config,
        BitZeroValues(config),
        timing);
    zero = BootstrapThenProject(
        cc, std::move(zero), config, 1U, timing, nullptr);
    Synchronize(cc);
    if (timing) {
        timing->zeroDetectionMs +=
            Milliseconds(begin, Clock::now());
    }
    return zero;
}

inline std::vector<Ciphertext<DCRTPoly>> BuildMultiples(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& divisor,
    const RuntimeConfig& config,
    TimingReport* timing) {
    Synchronize(cc);
    const auto begin = Clock::now();
    std::vector<Ciphertext<DCRTPoly>> multiples(kRadix);
    multiples[1] = divisor;

    uint32_t queued = 0;
    for (uint32_t bit = 1; bit < kRadixBits; ++bit) {
        const uint32_t weight = 1U << bit;
        multiples[weight] = cc->EvalRotate(
            divisor, -static_cast<int32_t>(bit));
        ++queued;
        if (queued >= config.effectiveParallelBlocks()) {
            Synchronize(cc);
            queued = 0;
            if (timing) {
                ++timing->asyncPreparationBatches;
            }
        }
    }
    Synchronize(cc);
    if (queued != 0 && timing) {
        ++timing->asyncPreparationBatches;
    }
    const uint32_t targetLevel = config.expectedTrialInputLevel();
    for (uint32_t bit = 0; bit < kRadixBits; ++bit) {
        const uint32_t weight = 1U << bit;
        multiples[weight] = PadToLevel(
            cc,
            std::move(multiples[weight]),
            targetLevel,
            config,
            config.wideBits(),
            timing);
        RequireState(
            "prepared divisor multiple",
            multiples[weight],
            targetLevel,
            2U);
    }
    Synchronize(cc);
    if (timing) {
        timing->multiplePreparationMs +=
            Milliseconds(begin, Clock::now());
    }
    return multiples;
}

inline Ciphertext<DCRTPoly> FormNextNibble(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& dividend,
    uint32_t round,
    const RuntimeConfig& config,
    TimingReport* timing) {
    const auto begin = Clock::now();
    const uint32_t sourceLowBit =
        config.wordBits - kRadixBits * (round + 1U);
    // Every radix digit is extracted from the same refreshed dividend. A
    // power-of-two rotation decomposition bounds each output path by log2(w)
    // KSKs instead of accumulating one KSK across all radix rounds.
    auto aligned = RotateByPowers(
        cc,
        dividend,
        static_cast<int32_t>(sourceLowBit));
    auto nibble = ApplyMask(
        cc,
        std::move(aligned),
        config,
        LowNibbleValues(config),
        timing);
    Synchronize(cc);
    RequireState(
        "extracted dividend nibble",
        nibble,
        kBootstrapOutputLevel + 1U,
        2U);
    if (timing) {
        timing->nibbleFormationMs +=
            Milliseconds(begin, Clock::now());
    }
    return nibble;
}

inline Ciphertext<DCRTPoly> BuildDigit(
    const CryptoContext<DCRTPoly>& cc,
    const std::vector<Ciphertext<DCRTPoly>>& bits,
    const RuntimeConfig& config,
    TimingReport* timing) {
    if (bits.size() != kRadixBits) {
        throw std::invalid_argument(
            "radix-16 digit requires four quotient bits");
    }
    auto digit = bits[0];
    uint32_t queued = 0;
    for (uint32_t bit = 1; bit < kRadixBits; ++bit) {
        auto placed = cc->EvalRotate(
            bits[bit], -static_cast<int32_t>(bit));
        digit = cc->EvalAdd(digit, placed);
        ++queued;
        if (queued >= config.effectiveParallelBlocks()) {
            Synchronize(cc);
            queued = 0;
            if (timing) {
                ++timing->asyncPreparationBatches;
            }
        }
    }
    Synchronize(cc);
    if (queued != 0 && timing) {
        ++timing->asyncPreparationBatches;
    }
    return digit;
}

}  // namespace detail

inline std::vector<int32_t> OrdinaryRotationIndices(
    const RuntimeConfig& config) {
    std::vector<int32_t> rotations{
        -1,
        -2,
        -3,
        -static_cast<int32_t>(kRadixBits),
        static_cast<int32_t>(
            config.wordBits - kRadixBits),
        -static_cast<int32_t>(config.wordBits),
        static_cast<int32_t>(
            config.wideBits() - 1U),
    };
    for (uint32_t layer = 0;
         layer < config.wideLayers();
         ++layer) {
        const int32_t step =
            static_cast<int32_t>(1U << layer);
        rotations.push_back(-step);
        if (layer < ksword::IntegerLog2(config.wordBits)) {
            rotations.push_back(step);
        }
    }
    std::sort(rotations.begin(), rotations.end());
    rotations.erase(
        std::unique(rotations.begin(), rotations.end()),
        rotations.end());
    return rotations;
}

inline DivisionResult EvalDivideEncrypted(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& dividend,
    const Ciphertext<DCRTPoly>& divisor,
    const RuntimeConfig& config,
    TimingReport* timing = nullptr) {
    if (!cc || !dividend || !divisor) {
        throw std::invalid_argument(
            "encrypted divider received a null input");
    }
    TimingReport local;
    TimingReport& report = timing ? *timing : local;
    report = TimingReport{};

    detail::Synchronize(cc);
    const auto totalBegin = detail::Clock::now();

    std::cout << "[divider] encrypted zero-divisor detection\n";
    auto zeroFlag = detail::DetectDivisorZero(
        cc, divisor, config, &report);
    std::cout << "[divider] canonical input refresh\n";
    auto divisorWork = detail::BootstrapThenProject(
        cc,
        divisor,
        config,
        config.wordBits,
        &report,
        &report.inputRefreshMs);
    detail::Synchronize(cc);
    const auto dividendRefreshBegin = detail::Clock::now();
    auto dividendWork = detail::BootstrapRaw(
        cc, dividend, config, &report);
    report.inputRefreshMs += detail::Milliseconds(
        dividendRefreshBegin, detail::Clock::now());

    detail::RequireState(
        "refreshed divisor",
        divisorWork,
        kCanonicalStateLevel,
        2U);
    detail::RequireState(
        "refreshed zero flag",
        zeroFlag,
        kCanonicalStateLevel,
        2U);
    detail::RequireState(
        "raw dividend cursor",
        dividendWork,
        kBootstrapOutputLevel,
        2U);

    std::cout << "[divider] refreshed divisor state "
              << divisorWork->GetLevel() << "/"
              << divisorWork->GetNoiseScaleDeg()
              << ", zero flag "
              << zeroFlag->GetLevel() << "/"
              << zeroFlag->GetNoiseScaleDeg()
              << ", immutable dividend "
              << dividendWork->GetLevel() << "/"
              << dividendWork->GetNoiseScaleDeg() << "\n";

    std::cout << "[divider] prepare D/2D/4D/8D\n";
    auto multiples = detail::BuildMultiples(
        cc, divisorWork, config, &report);
    auto remainder =
        cc->EvalSub(divisorWork, divisorWork);
    Ciphertext<DCRTPoly> quotient;
    detail::BorrowRuntimeCache cache;

    std::cout << "[divider] " << config.radixRounds()
              << " radix-16 rounds, "
              << 4U * config.radixRounds()
              << " sequential latest-borrow trials\n";
    for (uint32_t round = 0;
         round < config.radixRounds();
         ++round) {
        if (round == 0U ||
            round + 1U == config.radixRounds() ||
            (round + 1U) % 4U == 0U) {
            std::cout << "[divider] round "
                      << (round + 1U) << "/"
                      << config.radixRounds() << "\n";
        }
        auto nibble = detail::FormNextNibble(
            cc, dividendWork, round, config, &report);

        auto shiftedR = cc->EvalRotate(
            remainder,
            -static_cast<int32_t>(kRadixBits));
        detail::AlignLevels(
            cc,
            &shiftedR,
            &nibble,
            config,
            config.wideBits(),
            &report);
        auto state = cc->EvalAdd(shiftedR, nibble);
        detail::Synchronize(cc);

        std::vector<Ciphertext<DCRTPoly>> digitBits(
            kRadixBits);
        for (int32_t bit =
                 static_cast<int32_t>(kRadixBits) - 1;
             bit >= 0;
             --bit) {
            const uint32_t index =
                static_cast<uint32_t>(bit);
            const uint32_t weight = 1U << index;
            const bool finalOutputTrial =
                round + 1U == config.radixRounds() && index == 0U;
            auto trial = detail::EvalCandidateTrial(
                cc,
                state,
                multiples[weight],
                config,
                &cache,
                &report,
                finalOutputTrial
                    ? config.wordBits
                    : config.wideBits());
            state = std::move(trial.state);
            digitBits[index] =
                std::move(trial.quotientBit);
        }
        remainder = std::move(state);

        const auto quotientBegin =
            detail::Clock::now();
        auto digit = detail::BuildDigit(
            cc, digitBits, config, &report);
        const uint32_t destinationLowBit =
            config.wordBits - kRadixBits * (round + 1U);
        auto placedDigit = detail::RotateByPowers(
            cc,
            digit,
            -static_cast<int32_t>(destinationLowBit));
        if (!quotient) {
            quotient = std::move(placedDigit);
        }
        else {
            detail::AlignLevels(
                cc,
                &quotient,
                &placedDigit,
                config,
                config.wordBits,
                &report);
            quotient = cc->EvalAdd(quotient, placedDigit);
        }
        detail::Synchronize(cc);
        report.quotientUpdateMs +=
            detail::Milliseconds(
                quotientBegin, detail::Clock::now());
        ++report.radixRounds;
    }

    // For D=0, every subtraction of zero is accepted. The recurrence leaves
    // the remainder equal to the dividend and constructs an all-one quotient.
    // Gate only that quotient with the encrypted zero flag to obtain (0, A)
    // without mixing independently refreshed paths before the trials.
    const auto zeroSelectionBegin = detail::Clock::now();
    auto zeroWide = detail::BroadcastBitZero(
        cc, zeroFlag, config, config.wordBits);
    auto nonZeroWide = detail::OneMinusWithinMask(
        cc,
        zeroWide,
        config,
        detail::ActiveValues(config, config.wordBits),
        &report);
    detail::AlignLevels(
        cc,
        &quotient,
        &nonZeroWide,
        config,
        config.wordBits,
        &report);
    quotient = cc->EvalMult(quotient, nonZeroWide);
    detail::Synchronize(cc);
    report.zeroResultSelectionMs +=
        detail::Milliseconds(
            zeroSelectionBegin, detail::Clock::now());

    // nonZeroWide is already zero outside the w-bit result lane, so this
    // ciphertext product also performs the final quotient-width masking. The
    // last candidate trial already Bootstrapped the remainder and used the
    // w-bit post-Bootstrap mask; bootstrapping it again is both redundant and
    // numerically unsafe in the current FIDES path.
    detail::RequireState(
        "final quotient Bootstrap input",
        quotient,
        config.expectedTrialOutputLevel(),
        2U);
    detail::RequireState(
        "final remainder from last trial Bootstrap",
        remainder,
        kCanonicalStateLevel,
        2U);
    std::cout << "[divider] final quotient Bootstrap input "
              << quotient->GetLevel() << "/"
              << quotient->GetNoiseScaleDeg()
              << ", remainder already refreshed "
              << remainder->GetLevel() << "/"
              << remainder->GetNoiseScaleDeg() << "\n";
    quotient = detail::BootstrapThenProject(
        cc,
        std::move(quotient),
        config,
        config.wordBits,
        &report,
        &report.finalRefreshMs);
    detail::Synchronize(cc);
    report.totalWallMs =
        detail::Milliseconds(
            totalBegin, detail::Clock::now());
    return {
        std::move(quotient),
        std::move(remainder),
    };
}

}  // namespace ksdivcipher
