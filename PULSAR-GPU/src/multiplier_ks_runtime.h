#pragma once

#include <fideslib.hpp>

#include "adder_ks_complex_hoisted_gpu.h"
#include "multiplier_runtime_common.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ksmul {

using fideslib::Ciphertext;
using fideslib::CryptoContext;
using fideslib::DCRTPoly;
using fideslib::Plaintext;

struct DaddaStageTiming {
    uint32_t inputRows = 0;
    uint32_t targetRows = 0;
    uint32_t fullAdders = 0;
    uint32_t passthroughRows = 0;
    double wallMs = 0.0;
    double purificationMs = 0.0;
    uint32_t purifiedRows = 0;
};

struct TimingReport {
    double partialProductsMs = 0.0;
    std::vector<DaddaStageTiming> daddaStages;
    ksadder::TimingReport finalAdder;
    double finalAdderMaskReloadMs = 0.0;
    double finalAdderInputRescaleMs = 0.0;
    double finalAdderGraphInitMs = 0.0;
    double daddaRowRefreshMs = 0.0;
    double totalMs = 0.0;
    uint32_t daddaPreRefreshLevel = 0;
    uint32_t daddaOutputLevel = 0;
    uint32_t finalAdderInputLevel = 0;

    uint64_t inputAlignmentRotations = 0;
    uint64_t broadcastRotations = 0;
    uint64_t selectorPlainMults = 0;
    uint64_t partialCipherMults = 0;
    uint64_t compressorCipherMults = 0;
    uint64_t compressorCarryRotations = 0;
    uint64_t booleanPurifications = 0;
    uint64_t daddaRowBootstraps = 0;
    uint64_t alignmentPlainMults = 0;
    uint64_t streamingSynchronizations = 0;
    uint32_t peakBufferedRows = 0;
    std::array<Ciphertext<DCRTPoly>, 2> auditDaddaRows{};
    Ciphertext<DCRTPoly> auditBaselineProduct;
};

struct LevelMask {
    uint32_t level = 0;
    Plaintext plaintext;
};

struct Masks {
    Plaintext bitZero;
    ksadder::Masks finalAdder;
    uint32_t finalAdderInputLevel = 0;
    std::vector<LevelMask> activeMasks;
};

namespace detail {

using Clock = std::chrono::steady_clock;

inline double Milliseconds(const Clock::time_point& begin,
                           const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

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
            values[ksword::Slot(config, word, bit)] = 1.0;
        }
    }
    return values;
}

inline std::vector<double> CarryMaskValues(
    const RuntimeConfig& config) {
    std::vector<double> values(config.slots, 0.0);
    for (uint32_t word = 0; word < config.words; ++word) {
        for (uint32_t bit = 1; bit < config.wordBits; ++bit) {
            values[ksword::Slot(config, word, bit)] = 1.0;
        }
    }
    return values;
}

inline std::vector<double> BitZeroMaskValues(
    const RuntimeConfig& config) {
    std::vector<double> values(config.slots, 0.0);
    for (uint32_t word = 0; word < config.words; ++word) {
        values[ksword::Slot(config, word, 0)] = 1.0;
    }
    return values;
}

inline Plaintext MakeAndLoad(
    const CryptoContext<DCRTPoly>& cc,
    const RuntimeConfig& config,
    const std::vector<double>& values,
    uint32_t level = 0) {
    auto plaintext = cc->MakeCKKSPackedPlaintext(
        values, 1, level, nullptr, config.slots);
    plaintext->SetLength(config.slots);
    cc->LoadPlaintext(plaintext);
    return plaintext;
}

inline Plaintext ActiveMaskAtLevel(
    const CryptoContext<DCRTPoly>& cc,
    const RuntimeConfig& config,
    Masks& masks,
    uint32_t level) {
    for (const auto& item : masks.activeMasks) {
        if (item.level == level) {
            return item.plaintext;
        }
    }
    auto plaintext = MakeAndLoad(
        cc, config, ActiveMaskValues(config), level);
    masks.activeMasks.push_back({level, plaintext});
    Synchronize(cc);
    return plaintext;
}

inline Ciphertext<DCRTPoly> ApplyActiveMask(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> value,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* timing) {
    auto mask = ActiveMaskAtLevel(
        cc, config, masks, value->GetLevel());
    auto result = cc->EvalMult(value, mask);
    if (timing) {
        ++timing->alignmentPlainMults;
    }
    return result;
}

inline Ciphertext<DCRTPoly> PadToLevel(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> value,
    uint32_t targetLevel,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* timing) {
    if (value->GetLevel() > targetLevel) {
        std::ostringstream message;
        message << "cannot align multiplier ciphertext "
                << value->GetLevel() << "/"
                << value->GetNoiseScaleDeg()
                << " to earlier target " << targetLevel << "/2";
        throw std::runtime_error(message.str());
    }
    while (value->GetLevel() < targetLevel ||
           value->GetNoiseScaleDeg() != 2U) {
        const uint32_t previousLevel = value->GetLevel();
        const uint32_t previousScale = value->GetNoiseScaleDeg();
        if (previousScale == 0U || previousScale > 2U) {
            throw std::runtime_error(
                "multiplier level alignment expects scale degree 1 or 2");
        }
        value = ApplyActiveMask(
            cc, std::move(value), config, masks, timing);
        Synchronize(cc);
        const bool normalized =
            previousScale == 1U &&
            value->GetLevel() == previousLevel &&
            value->GetNoiseScaleDeg() == 2U;
        const bool advanced =
            value->GetLevel() == previousLevel + 1U &&
            value->GetNoiseScaleDeg() == 2U;
        if ((!normalized && !advanced) ||
            value->GetLevel() > targetLevel) {
            std::ostringstream message;
            message << "unexpected multiplier alignment transition: "
                    << previousLevel << "/" << previousScale
                    << " -> " << value->GetLevel() << "/"
                    << value->GetNoiseScaleDeg()
                    << ", target=" << targetLevel << "/2";
            throw std::runtime_error(message.str());
        }
    }
    return value;
}

inline Ciphertext<DCRTPoly> ProjectBoolean(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> value,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* timing) {
    if (value->GetNoiseScaleDeg() != 2U) {
        throw std::runtime_error(
            "multiplier Boolean projection expects scale degree 2");
    }
    const uint32_t inputLevel = value->GetLevel();
    auto squared = cc->EvalMult(value, value);
    auto cubed = cc->EvalMult(squared, value);
    auto purified = cc->EvalSub(
        cc->EvalAdd(squared, cc->EvalAdd(squared, squared)),
        cc->EvalAdd(cubed, cubed));
    purified = ApplyActiveMask(
        cc, std::move(purified), config, masks, timing);
    Synchronize(cc);
    if (purified->GetLevel() !=
            inputLevel + kBooleanProjectionLevels ||
        purified->GetNoiseScaleDeg() != 2U) {
        throw std::runtime_error(
            "multiplier Boolean projection consumed an unexpected number "
            "of modulus levels");
    }
    if (timing) {
        ++timing->booleanPurifications;
    }
    return purified;
}

inline Ciphertext<DCRTPoly> BootstrapThenProject(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> value,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* timing,
    double* elapsedMs,
    bool daddaRowRefresh) {
    Synchronize(cc);
    const auto begin = Clock::now();
    auto refreshed = cc->EvalBootstrap(value);
    Synchronize(cc);
    if (refreshed->GetLevel() != kBootstrapOutputLevel ||
        refreshed->GetNoiseScaleDeg() != 2U) {
        std::ostringstream message;
        message << "unexpected multiplier Bootstrap output: expected "
                << kBootstrapOutputLevel << "/2, got "
                << refreshed->GetLevel() << "/"
                << refreshed->GetNoiseScaleDeg();
        throw std::runtime_error(message.str());
    }
    refreshed = ProjectBoolean(
        cc, std::move(refreshed), config, masks, timing);
    if (refreshed->GetLevel() != kCanonicalBooleanLevel ||
        refreshed->GetNoiseScaleDeg() != 2U) {
        throw std::runtime_error(
            "multiplier post-Bootstrap Boolean projection is not canonical");
    }
    if (timing && daddaRowRefresh) {
        ++timing->daddaRowBootstraps;
    }
    if (elapsedMs) {
        *elapsedMs += Milliseconds(begin, Clock::now());
    }
    return refreshed;
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

inline Plaintext MakeAndLoadComplex(
    const CryptoContext<DCRTPoly>& cc,
    const RuntimeConfig& config,
    const std::vector<std::complex<double>>& values,
    uint32_t level) {
    auto plaintext = cc->MakeCKKSPackedPlaintext(
        values, 1, level, nullptr, config.slots);
    plaintext->SetLength(config.slots);
    cc->LoadPlaintext(plaintext);
    return plaintext;
}

inline uint32_t ExpectedDaddaOutputLevel(
    const RuntimeConfig& config) {
    return config.expectedDaddaOutputLevel;
}

inline uint32_t ExpectedDaddaPreRefreshLevel(
    const RuntimeConfig& config) {
    return config.expectedDaddaPreRefreshLevel;
}

inline uint32_t FinalAdderRequiredLevels(
    const RuntimeConfig& config) {
    // Dadda emits scale-degree-2 rows. One explicit rescale first converts
    // them to the scale-degree-1 input contract used by the validated
    // complex-standard adder; that adder then consumes layers+3 levels.
    return config.controlBits() + 4U;
}

inline uint32_t RemainingLevels(
    const RuntimeConfig& config,
    const Ciphertext<DCRTPoly>& ciphertext) {
    const uint32_t scalePenalty =
        ciphertext->GetNoiseScaleDeg();
    const uint32_t used =
        ciphertext->GetLevel() + scalePenalty;
    return config.multiplicativeDepth > used
        ? config.multiplicativeDepth - used
        : 0U;
}

inline ksadder::RuntimeConfig MakeFinalAdderConfig(
    const RuntimeConfig& config) {
    ksadder::RuntimeConfig adder;
    adder.wordBits = config.wordBits;
    adder.guardBits = config.guardBits;
    adder.stride = config.stride;
    adder.words = config.words;
    adder.slots = config.slots;
    adder.layers = config.controlBits();
    adder.multiplicativeDepth =
        config.multiplicativeDepth;
    adder.directionSign = -1;
    adder.backend =
        config.backend == ksword::Backend::Gpu
            ? ksadder::Backend::Gpu
            : ksadder::Backend::Cpu;
    adder.algorithm = ksadder::Algorithm::ComplexStandard;
    adder.levelPlacement = ksadder::LevelPlacement::Bottom;
    adder.devices = config.devices;
    adder.bootstrap = true;
    adder.detailedProfile = config.detailedProfile;
    adder.graphInit = config.graphInit;
    adder.projectOutputToReal = true;
    adder.wordsExplicit = true;
    adder.layersExplicit = true;
    adder.algorithmExplicit = true;
    adder.guardFreeLayout = false;
    adder.inputLevel = config.expectedDaddaOutputLevel + 1U;
    adder.circuitLevelCost =
        ksadder::PrefixCircuitLevelCost(adder);
    adder.expectedOutputLevel =
        adder.inputLevel + adder.circuitLevelCost;
    return adder;
}

inline ksadder::Masks BuildAndLoadFinalAdderMasks(
    const CryptoContext<DCRTPoly>& cc,
    const RuntimeConfig& config,
    uint32_t inputLevel) {
    const auto activeValues = ActiveMaskValues(config);
    const auto carryValues = CarryMaskValues(config);
    auto halfPropagateValues = activeValues;
    for (double& value : halfPropagateValues) {
        value *= 0.5;
    }
    const uint32_t packMaskLevel = inputLevel + 1U;
    const uint32_t outputMaskLevel =
        inputLevel + config.controlBits() + 2U;
    const uint32_t projectedOutputLevel = outputMaskLevel + 1U;

    if (projectedOutputLevel >= config.multiplicativeDepth) {
        throw std::runtime_error(
            "final complex-standard adder masks exceed the "
            "configured modulus chain");
    }

    ksadder::Masks masks;
    masks.active = MakeAndLoad(
        cc, config, activeValues, projectedOutputLevel);
    masks.carry = MakeAndLoad(
        cc, config, carryValues, projectedOutputLevel);
    masks.halfPropagate = MakeAndLoad(
        cc, config, halfPropagateValues, packMaskLevel);
    masks.iActive = MakeAndLoadComplex(
        cc,
        config,
        ImaginaryValues(activeValues, 1.0),
        packMaskLevel);
    masks.twoIActive = MakeAndLoadComplex(
        cc,
        config,
        ImaginaryValues(activeValues, 2.0),
        outputMaskLevel);
    masks.minusICarry = MakeAndLoadComplex(
        cc,
        config,
        ImaginaryValues(carryValues, -1.0),
        outputMaskLevel);
    // The output terms advance to the RNS level after outputMaskLevel.  A
    // plaintext encoded at outputMaskLevel therefore cannot also perform the
    // subsequent real projection.  ReconstructComplexSum falls back to its
    // scalar 0.5 projection; the masks above already keep guard slots zero.
    masks.halfActiveProjection = {};
    Synchronize(cc);
    return masks;
}

inline Ciphertext<DCRTPoly> EvalOriginalFinalAdder(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config) {
    auto g = cc->EvalMult(cA, cB);
    auto p = cc->EvalSub(
        cc->EvalAdd(cA, cB), cc->EvalAdd(g, g));
    auto G = g;
    auto P = p;

    for (uint32_t layer = 0;
         layer < config.controlBits();
         ++layer) {
        const int32_t rotation =
            -static_cast<int32_t>(1U << layer);
        const auto oldG = G;
        const auto oldP = P;
        auto rotatedG = cc->EvalRotate(oldG, rotation);
        auto term = cc->EvalMult(rotatedG, oldP);
        G = cc->EvalAdd(oldG, term);
        if (layer + 1U < config.controlBits()) {
            auto rotatedP = cc->EvalRotate(oldP, rotation);
            P = cc->EvalMult(rotatedP, oldP);
        }
    }

    const uint32_t outputMaskLevel =
        cA->GetLevel() + config.controlBits() + 1U;
    auto active = MakeAndLoad(
        cc, config, ActiveMaskValues(config), outputMaskLevel);
    auto carryMask = MakeAndLoad(
        cc, config, CarryMaskValues(config), outputMaskLevel);
    auto finalActive = MakeAndLoad(
        cc, config, ActiveMaskValues(config), outputMaskLevel + 1U);
    auto carry = cc->EvalMult(cc->EvalRotate(G, -1), carryMask);
    G = cc->EvalMult(G, active);

    auto result = cc->EvalAdd(cA, cB);
    result = cc->EvalAdd(result, carry);
    result = cc->EvalSub(result, cc->EvalAdd(G, G));
    return cc->EvalMult(result, finalActive);
}

struct CompressorRows {
    Ciphertext<DCRTPoly> sum;
    Ciphertext<DCRTPoly> alignedCarry;
};

inline CompressorRows EvalCompressor3To2(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& x,
    const Ciphertext<DCRTPoly>& y,
    const Ciphertext<DCRTPoly>& z,
    TimingReport* timing) {
    auto xy = cc->EvalMult(x, y);
    auto parity =
        cc->EvalSub(cc->EvalAdd(x, y), cc->EvalAdd(xy, xy));

    auto parityZ = cc->EvalMult(parity, z);
    auto sum = cc->EvalSub(
        cc->EvalAdd(parity, z),
        cc->EvalAdd(parityZ, parityZ));
    auto carry = cc->EvalAdd(xy, parityZ);
    auto alignedCarry = cc->EvalRotate(carry, -1);

    if (timing) {
        timing->compressorCipherMults += 2;
        timing->compressorCarryRotations += 1;
    }
    return {std::move(sum), std::move(alignedCarry)};
}

class PartialProductGenerator {
public:
    PartialProductGenerator(
        const CryptoContext<DCRTPoly>& cc,
        const Ciphertext<DCRTPoly>& cA,
        const Ciphertext<DCRTPoly>& cB,
        const RuntimeConfig& config,
        Masks& masks,
        TimingReport* timing)
        : cc_(cc),
          shiftedA_(cA),
          alignedB_(cB),
          config_(config),
          masks_(masks),
          timing_(timing) {}

    Ciphertext<DCRTPoly> Next() {
        if (nextRow_ >= config_.wordBits) {
            throw std::out_of_range(
                "partial-product generator exhausted");
        }

        // Complete the previous streaming cascade before admitting another
        // source row. This keeps row generation itself at constant memory.
        Synchronize(cc_);
        const auto begin = Clock::now();

        auto selected = cc_->EvalMult(alignedB_, masks_.bitZero);
        auto broadcast = selected;
        for (uint32_t layer = 0;
             layer < config_.controlBits();
             ++layer) {
            const int32_t rotation =
                -static_cast<int32_t>(1U << layer);
            broadcast = cc_->EvalAdd(
                broadcast, cc_->EvalRotate(broadcast, rotation));
        }

        auto partial = cc_->EvalMult(shiftedA_, broadcast);

        ++nextRow_;
        if (nextRow_ < config_.wordBits) {
            shiftedA_ = cc_->EvalRotate(shiftedA_, -1);
            alignedB_ = cc_->EvalRotate(alignedB_, 1);
        }

        Synchronize(cc_);
        if (timing_) {
            timing_->partialProductsMs +=
                Milliseconds(begin, Clock::now());
            timing_->selectorPlainMults += 1;
            timing_->partialCipherMults += 1;
            timing_->broadcastRotations += config_.controlBits();
            if (nextRow_ < config_.wordBits) {
                timing_->inputAlignmentRotations += 2;
            }
        }
        return partial;
    }

    uint32_t RowsGenerated() const noexcept {
        return nextRow_;
    }

private:
    CryptoContext<DCRTPoly> cc_;
    Ciphertext<DCRTPoly> shiftedA_;
    Ciphertext<DCRTPoly> alignedB_;
    const RuntimeConfig& config_;
    Masks& masks_;
    TimingReport* timing_;
    uint32_t nextRow_ = 0;
};

struct StreamingStageState {
    uint32_t inputRows = 0;
    uint32_t targetRows = 0;
    uint32_t fullAdders = 0;
    uint32_t passthroughRows = 0;
    uint32_t receivedRows = 0;
    uint32_t emittedRows = 0;
    uint32_t completedFullAdders = 0;
    std::array<Ciphertext<DCRTPoly>, 3> pending{};
    uint32_t pendingCount = 0;
};

class StreamingDaddaPipeline {
public:
    StreamingDaddaPipeline(
        const CryptoContext<DCRTPoly>& cc,
        const RuntimeConfig& config,
        Masks& masks,
        TimingReport* timing)
        : cc_(cc),
          config_(config),
          masks_(masks),
          mergeBatch_(config.mergeBatch),
          timing_(timing) {
        const auto targets = DaddaTargets(config.wordBits);
        stages_.resize(targets.size());
        if (timing_) {
            timing_->daddaStages.resize(targets.size());
        }

        uint32_t inputRows = config.wordBits;
        for (uint32_t stage = 0;
             stage < targets.size();
             ++stage) {
            const uint32_t targetRows = targets[stage];
            if (targetRows >= inputRows ||
                targetRows < ((2U * inputRows + 2U) / 3U)) {
                throw std::invalid_argument(
                    "invalid Dadda target sequence");
            }

            auto& state = stages_[stage];
            state.inputRows = inputRows;
            state.targetRows = targetRows;
            state.fullAdders = inputRows - targetRows;
            state.passthroughRows =
                3U * targetRows - 2U * inputRows;

            if (3U * state.fullAdders +
                        state.passthroughRows !=
                    inputRows ||
                2U * state.fullAdders +
                        state.passthroughRows !=
                    targetRows) {
                throw std::runtime_error(
                    "Dadda row-count identity failed");
            }

            if (timing_) {
                auto& item = timing_->daddaStages[stage];
                item.inputRows = inputRows;
                item.targetRows = targetRows;
                item.fullAdders = state.fullAdders;
                item.passthroughRows =
                    state.passthroughRows;
            }
            inputRows = targetRows;
        }
    }

    void Push(Ciphertext<DCRTPoly> row) {
        if (!row) {
            throw std::invalid_argument(
                "Dadda pipeline received a null row");
        }
        PushToStage(0, std::move(row));
        UpdatePeakBufferedRows();
    }

    std::array<Ciphertext<DCRTPoly>, 2> Finish() {
        Synchronize(cc_);
        if (timing_) {
            ++timing_->streamingSynchronizations;
        }

        for (const auto& state : stages_) {
            if (state.receivedRows != state.inputRows ||
                state.emittedRows != state.targetRows ||
                state.completedFullAdders != state.fullAdders ||
                state.pendingCount != 0) {
                throw std::runtime_error(
                    "Dadda stage finished in an invalid state");
            }
        }
        if (finalCount_ != 2 ||
            !finalRows_[0] ||
            !finalRows_[1]) {
            throw std::runtime_error(
                "Dadda tree did not terminate with two rows");
        }
        return {
            std::move(finalRows_[0]),
            std::move(finalRows_[1]),
        };
    }

private:
    bool IsPurificationBoundary(uint32_t stageIndex) const {
        return std::find(
                   config_.daddaPurificationStages.begin(),
                   config_.daddaPurificationStages.end(),
                   stageIndex) !=
               config_.daddaPurificationStages.end();
    }

    void ForwardRow(uint32_t completedStage,
                    Ciphertext<DCRTPoly> row) {
        if (IsPurificationBoundary(completedStage)) {
            Synchronize(cc_);
            const auto begin = Clock::now();
            row = ProjectBoolean(
                cc_, std::move(row), config_, masks_, timing_);
            Synchronize(cc_);
            if (timing_) {
                auto& stage = timing_->daddaStages[completedStage];
                stage.purificationMs +=
                    Milliseconds(begin, Clock::now());
                ++stage.purifiedRows;
            }
        }

        if (completedStage + 1U == stages_.size()) {
            if (timing_) {
                timing_->daddaPreRefreshLevel = std::max(
                    timing_->daddaPreRefreshLevel,
                    static_cast<uint32_t>(row->GetLevel()));
            }
            if (row->GetLevel() >
                config_.expectedDaddaPreRefreshLevel) {
                throw std::runtime_error(
                    "Dadda row exceeded the scheduled pre-refresh level");
            }
            row = PadToLevel(
                cc_,
                std::move(row),
                config_.expectedOutputLevel,
                config_,
                masks_,
                timing_);
            row = BootstrapThenProject(
                cc_,
                std::move(row),
                config_,
                masks_,
                timing_,
                timing_ ? &timing_->daddaRowRefreshMs : nullptr,
                true);
            row = PadToLevel(
                cc_,
                std::move(row),
                config_.finalAdderInputLevel,
                config_,
                masks_,
                timing_);
        }
        PushToStage(completedStage + 1U, std::move(row));
    }

    void PushToStage(uint32_t stageIndex,
                     Ciphertext<DCRTPoly> row) {
        if (stageIndex == stages_.size()) {
            if (finalCount_ >= finalRows_.size()) {
                throw std::runtime_error(
                    "Dadda pipeline emitted too many final rows");
            }
            finalRows_[finalCount_++] = std::move(row);
            return;
        }

        auto& state = stages_[stageIndex];
        if (state.receivedRows >= state.inputRows) {
            throw std::runtime_error(
                "Dadda stage received too many rows");
        }

        const uint32_t compressedInputRows =
            3U * state.fullAdders;
        ++state.receivedRows;

        if (state.receivedRows <= compressedInputRows) {
            state.pending[state.pendingCount++] = std::move(row);
            if (state.pendingCount != 3) {
                return;
            }

            const auto begin = Clock::now();
            auto compressed = EvalCompressor3To2(
                cc_,
                state.pending[0],
                state.pending[1],
                state.pending[2],
                timing_);

            state.pending = {};
            state.pendingCount = 0;
            ++state.completedFullAdders;
            state.emittedRows += 2;
            ++compressorsSinceSync_;
            MaybeSynchronize();

            if (timing_) {
                timing_->daddaStages[stageIndex].wallMs +=
                    Milliseconds(begin, Clock::now());
            }

            ForwardRow(stageIndex, std::move(compressed.sum));
            ForwardRow(
                stageIndex, std::move(compressed.alignedCarry));
        }
        else {
            ++state.emittedRows;
            ForwardRow(stageIndex, std::move(row));
        }
    }

    void MaybeSynchronize() {
        if (compressorsSinceSync_ < mergeBatch_) {
            return;
        }
        Synchronize(cc_);
        compressorsSinceSync_ = 0;
        if (timing_) {
            ++timing_->streamingSynchronizations;
        }
    }

    void UpdatePeakBufferedRows() {
        if (!timing_) {
            return;
        }
        uint32_t buffered = finalCount_;
        for (const auto& stage : stages_) {
            buffered += stage.pendingCount;
        }
        timing_->peakBufferedRows =
            std::max(timing_->peakBufferedRows, buffered);
    }

    CryptoContext<DCRTPoly> cc_;
    const RuntimeConfig& config_;
    Masks& masks_;
    uint32_t mergeBatch_;
    TimingReport* timing_;
    std::vector<StreamingStageState> stages_;
    std::array<Ciphertext<DCRTPoly>, 2> finalRows_{};
    uint32_t finalCount_ = 0;
    uint32_t compressorsSinceSync_ = 0;
};

}  // namespace detail

inline Masks BuildAndLoadMasks(
    const CryptoContext<DCRTPoly>& cc,
    const RuntimeConfig& config) {
    if (!cc) {
        throw std::invalid_argument(
            "multiplier mask builder received null context");
    }

    Masks masks;
    masks.bitZero = detail::MakeAndLoad(
        cc,
        config,
        detail::BitZeroMaskValues(config),
        config.inputLevel);
    for (uint32_t level : RequiredActiveMaskLevels(config)) {
        (void)detail::ActiveMaskAtLevel(
            cc, config, masks, level);
    }
    masks.finalAdderInputLevel =
        detail::ExpectedDaddaOutputLevel(config) + 1U;
    masks.finalAdder = detail::BuildAndLoadFinalAdderMasks(
        cc, config, masks.finalAdderInputLevel);
    detail::Synchronize(cc);
    return masks;
}

inline std::vector<int32_t> RotationIndices(
    const RuntimeConfig& config) {
    std::vector<int32_t> rotations{1};
    for (uint32_t layer = 0;
         layer < config.controlBits();
         ++layer) {
        rotations.push_back(
            -static_cast<int32_t>(1U << layer));
    }
    return rotations;
}

inline Ciphertext<DCRTPoly> EvalMultiplyDaddaKS(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    Masks& masks,
    TimingReport* timing = nullptr,
    const ksadder::AuditCallback& finalAdderAudit = {}) {
    if (!cc || !cA || !cB ||
        !masks.bitZero ||
        !masks.finalAdder.halfPropagate ||
        !masks.finalAdder.iActive ||
        !masks.finalAdder.twoIActive ||
        !masks.finalAdder.minusICarry) {
        throw std::invalid_argument(
            "EvalMultiplyDaddaKS received invalid input");
    }

    TimingReport local;
    TimingReport& report = timing ? *timing : local;
    report = TimingReport{};
    const bool detailed =
        timing != nullptr && config.detailedProfile;

    detail::Synchronize(cc);
    const auto totalBegin = detail::Clock::now();

    detail::PartialProductGenerator generator(
        cc, cA, cB, config, masks,
        detailed ? &report : nullptr);
    detail::StreamingDaddaPipeline pipeline(
        cc, config, masks, detailed ? &report : nullptr);

    for (uint32_t row = 0; row < config.wordBits; ++row) {
        pipeline.Push(generator.Next());
    }
    if (generator.RowsGenerated() != config.wordBits) {
        throw std::runtime_error(
            "partial-product row-count mismatch");
    }

    auto rows = pipeline.Finish();
    if (rows[0]->GetLevel() != rows[1]->GetLevel() ||
        rows[0]->GetNoiseScaleDeg() !=
            rows[1]->GetNoiseScaleDeg()) {
        throw std::runtime_error(
            "Dadda final rows have incompatible levels/scales");
    }

    report.daddaOutputLevel = rows[0]->GetLevel();
    if (report.daddaOutputLevel !=
        config.expectedDaddaOutputLevel) {
        throw std::runtime_error(
            "Dadda reduction consumed an unexpected number of modulus "
            "levels: expected " +
            std::to_string(config.expectedDaddaOutputLevel) +
            ", got " + std::to_string(report.daddaOutputLevel));
    }
    if (rows[0]->GetNoiseScaleDeg() != 2U) {
        throw std::runtime_error(
            "Dadda output has an unexpected scale degree: expected 2, got " +
            std::to_string(rows[0]->GetNoiseScaleDeg()));
    }
    if (config.audit) {
        report.auditDaddaRows = rows;
    }
    const uint32_t requiredLevels =
        detail::FinalAdderRequiredLevels(config);
    if (detail::RemainingLevels(config, rows[0]) <
        requiredLevels) {
        throw std::runtime_error(
            "Dadda output does not leave the scheduled levels for the "
            "final complex-standard adder");
    }

    // Bootstrap and h(x) return scale-degree-2 Boolean rows.  The standalone
    // complex adder was validated for scale-degree-1 inputs.  Normalize both
    // rows explicitly instead of trying to compensate by shifting plaintext
    // mask levels; the latter leaves the complex G/P path semantically wrong.
    detail::Synchronize(cc);
    const auto normalizeBegin = detail::Clock::now();
    rows[0] = cc->Rescale(rows[0]);
    rows[1] = cc->Rescale(rows[1]);
    detail::Synchronize(cc);
    report.finalAdderInputRescaleMs = detail::Milliseconds(
        normalizeBegin, detail::Clock::now());

    const uint32_t normalizedInputLevel =
        config.expectedDaddaOutputLevel + 1U;
    for (const auto& row : rows) {
        if (row->GetLevel() != normalizedInputLevel ||
            row->GetNoiseScaleDeg() != 1U) {
            throw std::runtime_error(
                "final-adder input normalization failed: expected " +
                std::to_string(normalizedInputLevel) + "/1, got " +
                std::to_string(row->GetLevel()) + "/" +
                std::to_string(row->GetNoiseScaleDeg()));
        }
    }

    report.finalAdderInputLevel = rows[0]->GetLevel();
    if (masks.finalAdderInputLevel !=
        report.finalAdderInputLevel) {
        detail::Synchronize(cc);
        const auto maskBegin = detail::Clock::now();
        masks.finalAdder = detail::BuildAndLoadFinalAdderMasks(
            cc, config, report.finalAdderInputLevel);
        masks.finalAdderInputLevel =
            report.finalAdderInputLevel;
        report.finalAdderMaskReloadMs =
            detail::Milliseconds(maskBegin, detail::Clock::now());
    }

    const auto adderConfig =
        detail::MakeFinalAdderConfig(config);
    if (adderConfig.inputLevel != report.finalAdderInputLevel ||
        adderConfig.expectedOutputLevel != config.expectedOutputLevel) {
        throw std::runtime_error(
            "final complex-adder level schedule is inconsistent");
    }
    if (config.graphInit) {
        detail::Synchronize(cc);
        const auto initBegin = detail::Clock::now();
        ksadder::InitializeConfiguredAdderRuntime(
            cc,
            rows[0],
            rows[1],
            adderConfig,
            masks.finalAdder);
        detail::Synchronize(cc);
        report.finalAdderGraphInitMs =
            detail::Milliseconds(initBegin, detail::Clock::now());
    }

    auto product = ksadder::EvalAddKSComplexStandard(
        cc,
        rows[0],
        rows[1],
        adderConfig,
        masks.finalAdder,
        detailed ? &report.finalAdder : nullptr,
        finalAdderAudit);

    if (config.audit) {
        report.auditBaselineProduct = detail::EvalOriginalFinalAdder(
            cc, rows[0], rows[1], config);
        detail::Synchronize(cc);
    }

    if (product->GetLevel() != config.expectedOutputLevel) {
        throw std::runtime_error(
            "multiplier consumed an unexpected number of modulus levels: "
            "expected raw level " +
            std::to_string(config.expectedOutputLevel) +
            ", got " + std::to_string(product->GetLevel()));
    }
    if (ksadder::UsableMultiplicationLevels(
            config.multiplicativeDepth,
            product->GetLevel(),
            product->GetNoiseScaleDeg()) != 0) {
        throw std::runtime_error(
            "bottom placement did not exhaust all circuit-usable levels");
    }

    if (timing) {
        detail::Synchronize(cc);
        report.totalMs = detail::Milliseconds(
            totalBegin, detail::Clock::now());
    }
    return product;
}

inline Ciphertext<DCRTPoly> EvalFinalizeProduct(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> product,
    const RuntimeConfig& config,
    Masks& masks,
    double* elapsedMs = nullptr) {
    if (!cc || !product) {
        throw std::invalid_argument(
            "EvalFinalizeProduct received invalid input");
    }
    if (product->GetLevel() != config.expectedOutputLevel ||
        product->GetNoiseScaleDeg() != 2U) {
        throw std::runtime_error(
            "final multiplier Bootstrap input is not at the scheduled "
            "bottom-level state");
    }
    return detail::BootstrapThenProject(
        cc,
        std::move(product),
        config,
        masks,
        nullptr,
        elapsedMs,
        false);
}

}  // namespace ksmul
