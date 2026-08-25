#pragma once

#include <fideslib.hpp>

#include "scalar_divider_runtime_common.h"
#include "subtractor_ks_complex_hoisted_gpu.h"
#include "adder_ks_complex_hoisted_gpu.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ksdivscalar {

using fideslib::Ciphertext;
using fideslib::CryptoContext;
using fideslib::DCRTPoly;
using fideslib::Plaintext;

struct PublicMultiplyTiming {
    double rowGenerationMs = 0.0;
    double daddaMs = 0.0;
    double finalAdderMs = 0.0;
    double setupExcludedMs = 0.0;
    uint32_t inputRows = 0;
    uint32_t daddaStages = 0;
    uint32_t shiftStreams = 0;
    uint32_t rowRotations = 0;
    uint32_t rowPlainMults = 0;
    uint32_t compressorCipherMults = 0;
    uint32_t compressorCarryRotations = 0;
    uint32_t synchronizations = 0;
};

struct TimingReport {
    PublicMultiplyTiming reciprocalMultiply;
    PublicMultiplyTiming divisorMultiply;
    double quotientExtractMs = 0.0;
    double quotientRefreshMs = 0.0;
    double productRefreshMs = 0.0;
    double numeratorRefreshMs = 0.0;
    double remainderSubtractMs = 0.0;
    double remainderRefreshMs = 0.0;
    double correctionBorrowMs = 0.0;
    double correctionAddMs = 0.0;
    double finalRefreshMs = 0.0;
    double levelAlignmentMs = 0.0;
    double setupExcludedMs = 0.0;
    double totalWallMs = 0.0;
    uint32_t bootstrapCount = 0;
    uint32_t purificationCount = 0;
};

struct DivisionResult {
    Ciphertext<DCRTPoly> quotient;
    Ciphertext<DCRTPoly> borrowBit;
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
    const uint32_t scalePenalty =
        ciphertext->GetNoiseScaleDeg() > 0
            ? ciphertext->GetNoiseScaleDeg() - 1U
            : 0U;
    const uint32_t used =
        ciphertext->GetLevel() + scalePenalty;
    return config.multiplicativeDepth > used
        ? config.multiplicativeDepth - used
        : 0U;
}

inline std::vector<double> ActiveValues(
    const RuntimeConfig& config,
    uint32_t activeBits) {
    std::vector<double> values(config.slots, 0.0);
    for (uint32_t word = 0; word < config.words; ++word) {
        for (uint32_t bit = 0; bit < activeBits; ++bit) {
            values[ksword::Slot(config, word, bit)] = 1.0;
        }
    }
    return values;
}

inline std::vector<double> CarryValues(
    const RuntimeConfig& config,
    uint32_t activeBits) {
    auto values = ActiveValues(config, activeBits);
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

inline std::vector<double> MsbValues(
    const RuntimeConfig& config) {
    std::vector<double> values(config.slots, 0.0);
    for (uint32_t word = 0; word < config.words; ++word) {
        values[ksword::Slot(
            config, word, config.wordBits - 1U)] = 1.0;
    }
    return values;
}

inline std::vector<double> PublicBitsValues(
    const RuntimeConfig& config,
    const std::vector<uint8_t>& bits) {
    std::vector<double> values(config.slots, 0.0);
    for (uint32_t word = 0; word < config.words; ++word) {
        for (uint32_t bit = 0;
             bit < config.wordBits && bit < bits.size();
             ++bit) {
            values[ksword::Slot(config, word, bit)] =
                static_cast<double>(bits[bit]);
        }
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

inline Plaintext MakeAndLoad(
    const CryptoContext<DCRTPoly>& cc,
    const RuntimeConfig& config,
    const std::vector<double>& values,
    uint32_t level,
    uint32_t scaleDegree = 1) {
    auto plaintext = cc->MakeCKKSPackedPlaintext(
        values, scaleDegree, level, nullptr, config.slots);
    plaintext->SetLength(config.slots);
    cc->LoadPlaintext(plaintext);
    return plaintext;
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

inline Ciphertext<DCRTPoly> MaskAtCurrentLevel(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> value,
    const RuntimeConfig& config,
    uint32_t activeBits,
    double* setupMs = nullptr) {
    const auto setupBegin = Clock::now();
    auto active = MakeAndLoad(
        cc,
        config,
        ActiveValues(config, activeBits),
        value->GetLevel());
    Synchronize(cc);
    if (setupMs) {
        *setupMs += Milliseconds(setupBegin, Clock::now());
    }
    return cc->EvalMult(value, active);
}

inline Ciphertext<DCRTPoly> PurifyThenBootstrap(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> value,
    const RuntimeConfig& config,
    uint32_t activeBits,
    TimingReport* timing,
    double* elapsedMs) {
    Synchronize(cc);
    const auto begin = Clock::now();

    if (RemainingLevels(config, value) < 3U) {
        throw std::runtime_error(
            "fewer than three levels remain for Boolean purification before "
            "Bootstrap; increase --depth or reduce the divisor popcount");
    }

    auto squared = cc->EvalMult(value, value);
    auto cubed = cc->EvalMult(squared, value);
    auto threeSquared =
        cc->EvalAdd(squared, cc->EvalAdd(squared, squared));
    auto twoCubed = cc->EvalAdd(cubed, cubed);
    auto purified = cc->EvalSub(threeSquared, twoCubed);
    purified = MaskAtCurrentLevel(
        cc,
        std::move(purified),
        config,
        activeBits,
        timing ? &timing->setupExcludedMs : nullptr);
    Synchronize(cc);

    auto refreshed = cc->EvalBootstrap(purified);
    Synchronize(cc);
    refreshed = MaskAtCurrentLevel(
        cc,
        std::move(refreshed),
        config,
        activeBits,
        timing ? &timing->setupExcludedMs : nullptr);
    Synchronize(cc);

    if (timing) {
        ++timing->bootstrapCount;
        ++timing->purificationCount;
    }
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
    uint32_t activeBits,
    TimingReport* timing) {
    if (value->GetLevel() > targetLevel) {
        throw std::runtime_error(
            "cannot align a ciphertext to an earlier CKKS level");
    }
    while (value->GetLevel() < targetLevel) {
        value = MaskAtCurrentLevel(
            cc,
            std::move(value),
            config,
            activeBits,
            timing ? &timing->setupExcludedMs : nullptr);
        Synchronize(cc);
        if (value->GetLevel() > targetLevel) {
            throw std::runtime_error(
                "plaintext-mask level alignment overshot its target");
        }
    }
    return value;
}

inline void AlignLevels(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly>* left,
    Ciphertext<DCRTPoly>* right,
    const RuntimeConfig& config,
    uint32_t activeBits,
    TimingReport* timing) {
    const auto begin = Clock::now();
    const uint32_t target =
        std::max((*left)->GetLevel(), (*right)->GetLevel());
    *left = PadToLevel(
        cc, std::move(*left), target,
        config, activeBits, timing);
    *right = PadToLevel(
        cc, std::move(*right), target,
        config, activeBits, timing);
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

inline ksadder::RuntimeConfig MakePrefixConfig(
    const RuntimeConfig& config,
    uint32_t wordBits) {
    ksadder::RuntimeConfig prefix;
    prefix.wordBits = wordBits;
    prefix.guardBits = config.stride - wordBits;
    prefix.stride = config.stride;
    prefix.words = config.words;
    prefix.slots = config.slots;
    prefix.layers = ksword::IntegerLog2(wordBits);
    prefix.multiplicativeDepth = config.multiplicativeDepth;
    prefix.directionSign = -1;
    prefix.backend =
        config.backend == ksword::Backend::Gpu
            ? ksadder::Backend::Gpu
            : ksadder::Backend::Cpu;
    prefix.algorithm = ksadder::Algorithm::ComplexTwisted;
    prefix.devices = config.devices;
    prefix.bootstrap = true;
    prefix.detailedProfile = config.detailedProfile;
    prefix.graphInit = config.graphInit;
    prefix.projectOutputToReal = true;
    prefix.wordsExplicit = true;
    prefix.layersExplicit = true;
    prefix.algorithmExplicit = true;
    return prefix;
}

inline ksadder::Masks BuildAdderMasks(
    const CryptoContext<DCRTPoly>& cc,
    const RuntimeConfig& config,
    uint32_t wordBits,
    uint32_t inputLevel) {
    const auto activeValues = ActiveValues(config, wordBits);
    const auto carryValues = CarryValues(config, wordBits);
    const uint32_t packLevel = inputLevel + 1U;
    const uint32_t outputLevel =
        inputLevel + ksword::IntegerLog2(wordBits) + 2U;
    if (outputLevel >= config.multiplicativeDepth) {
        throw std::runtime_error(
            "latest adder masks exceed the configured modulus chain");
    }

    ksadder::Masks masks;
    masks.active = MakeAndLoad(
        cc, config, activeValues, inputLevel);
    masks.carry = MakeAndLoad(
        cc, config, carryValues, inputLevel);
    masks.iActive = MakeAndLoadComplex(
        cc, config, ImaginaryValues(activeValues, 1.0), packLevel);
    masks.twoIActive = MakeAndLoadComplex(
        cc, config, ImaginaryValues(activeValues, 2.0), outputLevel);
    masks.minusICarry = MakeAndLoadComplex(
        cc, config, ImaginaryValues(carryValues, -1.0), outputLevel);
    Synchronize(cc);
    return masks;
}

inline kssubtractor::Masks BuildSubtractorMasks(
    const CryptoContext<DCRTPoly>& cc,
    const RuntimeConfig& config,
    uint32_t inputLevel,
    bool includeDifference) {
    const auto activeValues =
        ActiveValues(config, config.wordBits);
    const auto borrowValues =
        CarryValues(config, config.wordBits);
    const uint32_t packLevel = inputLevel + 1U;
    const uint32_t outputLevel =
        inputLevel + ksword::IntegerLog2(config.wordBits) + 2U;
    if (outputLevel >= config.multiplicativeDepth) {
        throw std::runtime_error(
            "latest subtractor masks exceed the configured modulus chain");
    }

    kssubtractor::Masks masks;
    masks.active = MakeAndLoad(
        cc, config, activeValues, inputLevel);
    masks.borrow = MakeAndLoad(
        cc, config, borrowValues, inputLevel);
    masks.iActive = MakeAndLoadComplex(
        cc, config, ImaginaryValues(activeValues, 1.0), packLevel);
    if (includeDifference) {
        masks.minusTwoIActive = MakeAndLoadComplex(
            cc,
            config,
            ImaginaryValues(activeValues, -2.0),
            outputLevel);
        masks.plusIBorrow = MakeAndLoadComplex(
            cc,
            config,
            ImaginaryValues(borrowValues, 1.0),
            outputLevel);
    }
    Synchronize(cc);
    return masks;
}

inline Plaintext BuildBorrowExtractionMask(
    const CryptoContext<DCRTPoly>& cc,
    const RuntimeConfig& config,
    uint32_t level) {
    return MakeAndLoadComplex(
        cc,
        config,
        ImaginaryValues(MsbValues(config), -0.5),
        level);
}

struct CompressorRows {
    Ciphertext<DCRTPoly> sum;
    Ciphertext<DCRTPoly> carry;
};

inline CompressorRows EvalCompressor(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& x,
    const Ciphertext<DCRTPoly>& y,
    const Ciphertext<DCRTPoly>& z,
    PublicMultiplyTiming* timing) {
    auto xy = cc->EvalMult(x, y);
    auto parity =
        cc->EvalSub(cc->EvalAdd(x, y), cc->EvalAdd(xy, xy));
    auto parityZ = cc->EvalMult(parity, z);
    auto sum = cc->EvalSub(
        cc->EvalAdd(parity, z),
        cc->EvalAdd(parityZ, parityZ));
    auto carry = cc->EvalAdd(xy, parityZ);
    carry = cc->EvalRotate(carry, -1);
    if (timing) {
        timing->compressorCipherMults += 2;
        ++timing->compressorCarryRotations;
    }
    return {std::move(sum), std::move(carry)};
}

struct DaddaStage {
    uint32_t inputRows = 0;
    uint32_t targetRows = 0;
    uint32_t fullAdders = 0;
    uint32_t passRows = 0;
    uint32_t received = 0;
    uint32_t emitted = 0;
    std::array<Ciphertext<DCRTPoly>, 3> pending{};
    uint32_t pendingCount = 0;
};

class DynamicDadda {
public:
    DynamicDadda(const CryptoContext<DCRTPoly>& cc,
                 uint32_t inputRows,
                 uint32_t mergeBatch,
                 PublicMultiplyTiming* timing)
        : cc_(cc),
          mergeBatch_(mergeBatch),
          timing_(timing) {
        const auto targets = DaddaTargets(inputRows);
        if (timing_) {
            timing_->daddaStages =
                static_cast<uint32_t>(targets.size());
        }
        stages_.resize(targets.size());
        uint32_t rows = inputRows;
        for (uint32_t index = 0;
             index < targets.size();
             ++index) {
            auto& stage = stages_[index];
            stage.inputRows = rows;
            stage.targetRows = targets[index];
            stage.fullAdders = rows - stage.targetRows;
            stage.passRows =
                3U * stage.targetRows - 2U * rows;
            if (3U * stage.fullAdders + stage.passRows != rows ||
                2U * stage.fullAdders + stage.passRows !=
                    stage.targetRows) {
                throw std::runtime_error(
                    "invalid dynamic Dadda row identity");
            }
            rows = stage.targetRows;
        }
    }

    void Push(Ciphertext<DCRTPoly> row) {
        PushAt(0, std::move(row));
    }

    std::vector<Ciphertext<DCRTPoly>> Finish() {
        Synchronize(cc_);
        if (timing_) {
            ++timing_->synchronizations;
        }
        for (const auto& stage : stages_) {
            if (stage.received != stage.inputRows ||
                stage.emitted != stage.targetRows ||
                stage.pendingCount != 0) {
                throw std::runtime_error(
                    "dynamic Dadda pipeline ended in an invalid state");
            }
        }
        if (finalRows_.empty() || finalRows_.size() > 2) {
            throw std::runtime_error(
                "dynamic Dadda did not produce one or two rows");
        }
        return std::move(finalRows_);
    }

private:
    void PushAt(uint32_t index,
                Ciphertext<DCRTPoly> row) {
        if (index == stages_.size()) {
            finalRows_.push_back(std::move(row));
            return;
        }

        auto& stage = stages_[index];
        ++stage.received;
        if (stage.received <= 3U * stage.fullAdders) {
            stage.pending[stage.pendingCount++] = std::move(row);
            if (stage.pendingCount < 3) {
                return;
            }
            auto rows = EvalCompressor(
                cc_,
                stage.pending[0],
                stage.pending[1],
                stage.pending[2],
                timing_);
            stage.pending = {};
            stage.pendingCount = 0;
            stage.emitted += 2;
            ++sinceSync_;
            if (sinceSync_ >= mergeBatch_) {
                Synchronize(cc_);
                sinceSync_ = 0;
                if (timing_) {
                    ++timing_->synchronizations;
                }
            }
            PushAt(index + 1, std::move(rows.sum));
            PushAt(index + 1, std::move(rows.carry));
        }
        else {
            ++stage.emitted;
            PushAt(index + 1, std::move(row));
        }
    }

    CryptoContext<DCRTPoly> cc_;
    uint32_t mergeBatch_;
    PublicMultiplyTiming* timing_;
    std::vector<DaddaStage> stages_;
    std::vector<Ciphertext<DCRTPoly>> finalRows_;
    uint32_t sinceSync_ = 0;
};

struct ShiftStream {
    uint32_t begin = 0;
    uint32_t end = 0;
    uint32_t position = 0;
    Ciphertext<DCRTPoly> shifted;
};

inline std::vector<ShiftStream> BuildShiftStreams(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& input,
    uint32_t positions,
    uint32_t requestedBlocks) {
    const uint32_t blocks =
        std::min(requestedBlocks, positions);
    std::vector<ShiftStream> streams;
    streams.reserve(blocks);
    for (uint32_t block = 0; block < blocks; ++block) {
        const uint32_t begin =
            (positions * block) / blocks;
        const uint32_t end =
            (positions * (block + 1U)) / blocks;
        if (begin == end) {
            continue;
        }
        auto shifted = begin == 0
            ? input
            : cc->EvalRotate(
                input, -static_cast<int32_t>(begin));
        streams.push_back(
            {begin, end, begin, std::move(shifted)});
    }
    return streams;
}

inline Ciphertext<DCRTPoly> EvalPublicMultiply(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> input,
    const std::vector<uint8_t>& scalarBits,
    uint32_t outputBits,
    const RuntimeConfig& config,
    TimingReport* divisionTiming,
    PublicMultiplyTiming* timing) {
    if (!cc || !input) {
        throw std::invalid_argument(
            "public multiplier received a null input");
    }
    const uint32_t rows = static_cast<uint32_t>(
        std::count(scalarBits.begin(), scalarBits.end(), uint8_t{1}));
    if (rows == 0) {
        throw std::invalid_argument(
            "public multiplier received the zero scalar");
    }
    if (timing) {
        *timing = PublicMultiplyTiming{};
        timing->inputRows = rows;
    }

    const uint32_t required =
        PublicMultiplyDepth(rows, outputBits) + 3U;
    if (RemainingLevels(config, input) < required) {
        input = PurifyThenBootstrap(
            cc, std::move(input), config, outputBits,
            divisionTiming, nullptr);
    }
    if (RemainingLevels(config, input) < required) {
        throw std::runtime_error(
            "Bootstrap did not leave enough levels for this public "
            "constant multiplication; increase --depth");
    }

    const auto setupBegin = Clock::now();
    auto active = MakeAndLoad(
        cc,
        config,
        ActiveValues(config, outputBits),
        input->GetLevel());
    Synchronize(cc);
    if (timing) {
        timing->setupExcludedMs +=
            Milliseconds(setupBegin, Clock::now());
    }

    const auto rowBegin = Clock::now();
    DynamicDadda pipeline(
        cc, rows, config.mergeBatch, timing);
    auto streams = BuildShiftStreams(
        cc,
        input,
        static_cast<uint32_t>(scalarBits.size()),
        config.parallelBlocks);
    if (timing) {
        timing->shiftStreams =
            static_cast<uint32_t>(streams.size());
    }

    uint32_t emitted = 0;
    bool work = true;
    while (work) {
        work = false;
        for (auto& stream : streams) {
            if (stream.position >= stream.end) {
                continue;
            }
            work = true;
            const uint32_t bit = stream.position;
            if (scalarBits[bit]) {
                pipeline.Push(
                    cc->EvalMult(stream.shifted, active));
                ++emitted;
                if (timing) {
                    ++timing->rowPlainMults;
                }
            }
            ++stream.position;
            if (stream.position < stream.end) {
                stream.shifted =
                    cc->EvalRotate(stream.shifted, -1);
                if (timing) {
                    ++timing->rowRotations;
                }
            }
        }
    }
    if (emitted != rows) {
        throw std::runtime_error(
            "public partial-product row count mismatch");
    }
    if (timing) {
        Synchronize(cc);
        timing->rowGenerationMs =
            Milliseconds(rowBegin, Clock::now());
    }

    const auto daddaBegin = Clock::now();
    auto finalRows = pipeline.Finish();
    if (timing) {
        timing->daddaMs =
            Milliseconds(daddaBegin, Clock::now());
    }
    if (finalRows.size() == 1) {
        return finalRows.front();
    }
    if (finalRows[0]->GetLevel() !=
            finalRows[1]->GetLevel() ||
        finalRows[0]->GetNoiseScaleDeg() !=
            finalRows[1]->GetNoiseScaleDeg()) {
        throw std::runtime_error(
            "Dadda final rows have incompatible level/scale");
    }

    const uint32_t adderRequired =
        ksword::IntegerLog2(outputBits) + 3U;
    if (RemainingLevels(config, finalRows[0]) <
        adderRequired + 3U) {
        finalRows[0] = PurifyThenBootstrap(
            cc, std::move(finalRows[0]), config,
            outputBits, divisionTiming, nullptr);
        finalRows[1] = PurifyThenBootstrap(
            cc, std::move(finalRows[1]), config,
            outputBits, divisionTiming, nullptr);
        AlignLevels(
            cc, &finalRows[0], &finalRows[1],
            config, outputBits, divisionTiming);
    }
    if (RemainingLevels(config, finalRows[0]) <
        adderRequired + 3U) {
        throw std::runtime_error(
            "not enough post-Bootstrap levels for the latest final adder");
    }

    const auto adderSetupBegin = Clock::now();
    auto adderConfig =
        MakePrefixConfig(config, outputBits);
    auto adderMasks = BuildAdderMasks(
        cc, config, outputBits, finalRows[0]->GetLevel());
    if (config.graphInit) {
        ksadder::InitializeConfiguredAdderRuntime(
            cc,
            finalRows[0],
            finalRows[1],
            adderConfig,
            adderMasks);
    }
    Synchronize(cc);
    if (timing) {
        timing->setupExcludedMs +=
            Milliseconds(adderSetupBegin, Clock::now());
    }

    const auto adderBegin = Clock::now();
    auto product = ksadder::EvalAddKSComplexTwisted(
        cc,
        finalRows[0],
        finalRows[1],
        adderConfig,
        adderMasks);
    Synchronize(cc);
    if (timing) {
        timing->finalAdderMs =
            Milliseconds(adderBegin, Clock::now());
    }
    return product;
}

inline Ciphertext<DCRTPoly> MakePublicCiphertext(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& model,
    const RuntimeConfig& config,
    const std::vector<uint8_t>& bits,
    TimingReport* timing) {
    const auto begin = Clock::now();
    auto values = PublicBitsValues(config, bits);
    auto plaintext = MakeAndLoad(
        cc,
        config,
        values,
        model->GetLevel(),
        model->GetNoiseScaleDeg());
    auto zero = cc->EvalSub(model, model);
    auto result = cc->EvalAdd(zero, plaintext);
    Synchronize(cc);
    if (timing) {
        timing->setupExcludedMs +=
            Milliseconds(begin, Clock::now());
    }
    return result;
}

inline Ciphertext<DCRTPoly> EvalLatestSubtract(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> left,
    Ciphertext<DCRTPoly> right,
    const RuntimeConfig& config,
    TimingReport* timing,
    double* elapsedMs) {
    AlignLevels(
        cc, &left, &right,
        config, config.wordBits, timing);
    const uint32_t required =
        ksword::IntegerLog2(config.wordBits) + 3U;
    if (RemainingLevels(config, left) < required + 3U) {
        left = PurifyThenBootstrap(
            cc, std::move(left), config,
            config.wordBits, timing, nullptr);
        right = PurifyThenBootstrap(
            cc, std::move(right), config,
            config.wordBits, timing, nullptr);
        AlignLevels(
            cc, &left, &right,
            config, config.wordBits, timing);
    }
    if (RemainingLevels(config, left) < required + 3U) {
        throw std::runtime_error(
            "not enough post-Bootstrap levels for the latest remainder "
            "subtractor");
    }

    const auto setupBegin = Clock::now();
    auto subConfig =
        MakePrefixConfig(config, config.wordBits);
    auto masks = BuildSubtractorMasks(
        cc, config, left->GetLevel(), true);
    if (config.graphInit) {
        kssubtractor::InitializeConfiguredSubtractorRuntime(
            cc, left, right, subConfig, masks);
    }
    Synchronize(cc);
    if (timing) {
        timing->setupExcludedMs +=
            Milliseconds(setupBegin, Clock::now());
    }

    const auto begin = Clock::now();
    auto result = kssubtractor::EvalSubKSComplexTwisted(
        cc, left, right, subConfig, masks);
    Synchronize(cc);
    if (elapsedMs) {
        *elapsedMs += Milliseconds(begin, Clock::now());
    }
    return result;
}

inline Ciphertext<DCRTPoly> EvalCorrectionBorrow(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> remainder,
    Ciphertext<DCRTPoly> divisor,
    const RuntimeConfig& config,
    TimingReport* timing) {
    AlignLevels(
        cc, &remainder, &divisor,
        config, config.wordBits, timing);
    const uint32_t required =
        ksword::IntegerLog2(config.wordBits) + 3U;
    if (RemainingLevels(config, remainder) <
        required + 3U) {
        remainder = PurifyThenBootstrap(
            cc, std::move(remainder),
            config, config.wordBits, timing, nullptr);
        divisor = PurifyThenBootstrap(
            cc, std::move(divisor),
            config, config.wordBits, timing, nullptr);
        AlignLevels(
            cc, &remainder, &divisor,
            config, config.wordBits, timing);
    }
    if (RemainingLevels(config, remainder) <
        required + 3U) {
        throw std::runtime_error(
            "not enough post-Bootstrap levels for the correction "
            "borrow prefix");
    }
    const auto setupBegin = Clock::now();
    auto subConfig =
        MakePrefixConfig(config, config.wordBits);
    auto masks = BuildSubtractorMasks(
        cc, config, remainder->GetLevel(), false);
    const uint32_t extractionLevel =
        remainder->GetLevel() +
        ksword::IntegerLog2(config.wordBits) + 2U;
    auto minusHalfIMsb =
        BuildBorrowExtractionMask(
            cc, config, extractionLevel);
    if (config.graphInit) {
        auto disposable =
            kssubtractor::EvalConfiguredBorrowPrefixState(
                cc,
                remainder,
                divisor,
                subConfig,
                masks);
        auto disposableConjugate =
            cc->EvalConjugate(disposable.value);
        auto disposableAtMsb = cc->EvalMult(
            cc->EvalSub(
                disposable.value, disposableConjugate),
            minusHalfIMsb);
        auto disposableBorrow = cc->EvalRotate(
            disposableAtMsb,
            static_cast<int32_t>(config.wordBits - 1U));
        (void)disposableBorrow;
        Synchronize(cc);
    }
    if (timing) {
        timing->setupExcludedMs +=
            Milliseconds(setupBegin, Clock::now());
    }

    const auto begin = Clock::now();
    auto prefix =
        kssubtractor::EvalConfiguredBorrowPrefixState(
            cc,
            remainder,
            divisor,
            subConfig,
            masks);
    auto conjugate = cc->EvalConjugate(prefix.value);
    auto atMsb = cc->EvalMult(
        cc->EvalSub(prefix.value, conjugate),
        minusHalfIMsb);
    auto borrow = cc->EvalRotate(
        atMsb,
        static_cast<int32_t>(config.wordBits - 1U));
    Synchronize(cc);
    if (timing) {
        timing->correctionBorrowMs +=
            Milliseconds(begin, Clock::now());
    }
    return borrow;
}

inline Ciphertext<DCRTPoly> EvalCorrectionAdd(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> quotientBase,
    Ciphertext<DCRTPoly> borrow,
    const RuntimeConfig& config,
    TimingReport* timing) {
    const auto setupBegin = Clock::now();
    auto one = MakeAndLoad(
        cc,
        config,
        BitZeroValues(config),
        borrow->GetLevel(),
        borrow->GetNoiseScaleDeg());
    auto increment = cc->EvalSub(one, borrow);
    Synchronize(cc);
    if (timing) {
        timing->setupExcludedMs +=
            Milliseconds(setupBegin, Clock::now());
    }

    AlignLevels(
        cc, &quotientBase, &increment,
        config, config.wordBits, timing);
    const uint32_t required =
        ksword::IntegerLog2(config.wordBits) + 3U;
    if (RemainingLevels(config, quotientBase) <
        required + 3U) {
        quotientBase = PurifyThenBootstrap(
            cc, std::move(quotientBase),
            config, config.wordBits, timing, nullptr);
        increment = PurifyThenBootstrap(
            cc, std::move(increment),
            config, config.wordBits, timing, nullptr);
        AlignLevels(
            cc, &quotientBase, &increment,
            config, config.wordBits, timing);
    }
    if (RemainingLevels(config, quotientBase) <
        required + 3U) {
        throw std::runtime_error(
            "not enough post-Bootstrap levels for quotient correction "
            "addition");
    }

    const auto adderSetupBegin = Clock::now();
    auto adderConfig =
        MakePrefixConfig(config, config.wordBits);
    auto masks = BuildAdderMasks(
        cc,
        config,
        config.wordBits,
        quotientBase->GetLevel());
    if (config.graphInit) {
        ksadder::InitializeConfiguredAdderRuntime(
            cc,
            quotientBase,
            increment,
            adderConfig,
            masks);
    }
    Synchronize(cc);
    if (timing) {
        timing->setupExcludedMs +=
            Milliseconds(adderSetupBegin, Clock::now());
    }

    const auto begin = Clock::now();
    auto quotient = ksadder::EvalAddKSComplexTwisted(
        cc,
        quotientBase,
        increment,
        adderConfig,
        masks);
    Synchronize(cc);
    if (timing) {
        timing->correctionAddMs +=
            Milliseconds(begin, Clock::now());
    }
    return quotient;
}

}  // namespace detail

inline std::vector<int32_t> RotationIndices(
    const RuntimeConfig& config) {
    std::vector<int32_t> rotations{
        -1,
        static_cast<int32_t>(config.wordBits),
        static_cast<int32_t>(config.wordBits - 1U),
    };
    const auto addBlockStarts =
        [&](uint32_t positions) {
            const uint32_t blocks =
                std::min(config.parallelBlocks, positions);
            for (uint32_t block = 1;
                 block < blocks;
                 ++block) {
                const uint32_t begin =
                    (positions * block) / blocks;
                rotations.push_back(
                    -static_cast<int32_t>(begin));
            }
        };
    addBlockStarts(config.wordBits + 1U);
    addBlockStarts(config.wordBits);

    std::sort(rotations.begin(), rotations.end());
    rotations.erase(
        std::unique(rotations.begin(), rotations.end()),
        rotations.end());
    return rotations;
}

inline std::vector<int32_t> TwistedIndices(
    const RuntimeConfig& config) {
    std::vector<int32_t> rotations;
    for (uint32_t layer = 0;
         layer < config.internalLayers();
         ++layer) {
        rotations.push_back(
            -static_cast<int32_t>(1U << layer));
    }
    return rotations;
}

inline DivisionResult EvalDividePublicScalar(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& numerator,
    const RuntimeConfig& config,
    TimingReport* timing = nullptr) {
    if (!cc || !numerator) {
        throw std::invalid_argument(
            "scalar divider received a null input");
    }
    TimingReport local;
    TimingReport& report = timing ? *timing : local;
    report = TimingReport{};

    detail::Synchronize(cc);
    const auto totalBegin = detail::Clock::now();

    const auto reciprocalBits =
        ToBits(Reciprocal(config), config.wordBits + 1U);
    const auto divisorBits =
        ToBits(config.divisor, config.wordBits);

    auto wideProduct = detail::EvalPublicMultiply(
        cc,
        numerator,
        reciprocalBits,
        config.internalBits(),
        config,
        &report,
        &report.reciprocalMultiply);
    report.setupExcludedMs +=
        report.reciprocalMultiply.setupExcludedMs;

    detail::Synchronize(cc);
    auto begin = detail::Clock::now();
    auto quotientBase = cc->EvalRotate(
        wideProduct,
        static_cast<int32_t>(config.wordBits));
    quotientBase = detail::MaskAtCurrentLevel(
        cc,
        std::move(quotientBase),
        config,
        config.wordBits,
        &report.setupExcludedMs);
    detail::Synchronize(cc);
    report.quotientExtractMs =
        detail::Milliseconds(begin, detail::Clock::now());

    quotientBase = detail::PurifyThenBootstrap(
        cc,
        std::move(quotientBase),
        config,
        config.wordBits,
        &report,
        &report.quotientRefreshMs);

    auto quotientForProduct = quotientBase;
    auto scaledQuotient = detail::EvalPublicMultiply(
        cc,
        quotientForProduct,
        divisorBits,
        config.wordBits,
        config,
        &report,
        &report.divisorMultiply);
    report.setupExcludedMs +=
        report.divisorMultiply.setupExcludedMs;
    scaledQuotient = detail::PurifyThenBootstrap(
        cc,
        std::move(scaledQuotient),
        config,
        config.wordBits,
        &report,
        &report.productRefreshMs);

    auto numeratorForSubtract = detail::PurifyThenBootstrap(
        cc,
        numerator,
        config,
        config.wordBits,
        &report,
        &report.numeratorRefreshMs);
    auto remainder = detail::EvalLatestSubtract(
        cc,
        std::move(numeratorForSubtract),
        std::move(scaledQuotient),
        config,
        &report,
        &report.remainderSubtractMs);
    remainder = detail::PurifyThenBootstrap(
        cc,
        std::move(remainder),
        config,
        config.wordBits,
        &report,
        &report.remainderRefreshMs);

    auto divisorCipher = detail::MakePublicCiphertext(
        cc,
        remainder,
        config,
        divisorBits,
        &report);
    auto borrow = detail::EvalCorrectionBorrow(
        cc,
        remainder,
        std::move(divisorCipher),
        config,
        &report);
    auto quotient = detail::EvalCorrectionAdd(
        cc,
        quotientBase,
        borrow,
        config,
        &report);
    quotient = detail::PurifyThenBootstrap(
        cc,
        std::move(quotient),
        config,
        config.wordBits,
        &report,
        &report.finalRefreshMs);

    detail::Synchronize(cc);
    report.totalWallMs =
        detail::Milliseconds(totalBegin, detail::Clock::now());
    return {std::move(quotient), std::move(borrow)};
}

}  // namespace ksdivscalar
