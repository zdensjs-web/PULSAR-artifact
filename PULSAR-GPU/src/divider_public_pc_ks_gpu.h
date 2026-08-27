#pragma once

// Version: public-u256-reciprocal-divider-v40-highonly-unit-rotate-rollback-3gpu
// Previous version: public-u256-reciprocal-divider-v12-correction-preboot
// Internal filenames intentionally keep the original names used by the CMake
// target.  This header implements a public-uint256 reciprocal constant
// multiplier by specializing the existing 128 x uint256 multiplier layout:
//   128 words/ciphertext, 512 slots/word = 256 active bits + 256 guard bits.
//
// Key v32 change:
//   Restore the public-specialized semantics: public zero-bits are skipped,
//   so cost follows popcount(public constant).  To avoid the original dense
//   reciprocal failure for divisors such as 1,000,000, reciprocal one-bits are
//   processed in popcount-aware chunks.  Each chunk uses Dadda compression for
//   row-level parallelism, then bootstraps before chunk-level accumulation.

#include <fideslib.hpp>

#include "multiplier_ks_gpu.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ks256gpu {


namespace public_reciprocal_locate {

inline bool Enabled() {
    const char* raw = std::getenv("KS256_LOCATE_TRACE");
    // Release build: trace is OFF by default. Set KS256_LOCATE_TRACE=1 to enable.
    return raw && std::string(raw) != "0";
}

inline bool ForceSync() {
    const char* raw = std::getenv("KS256_FORCE_SYNC");
    // Release build: diagnostic synchronizations are OFF by default.
    return raw && std::string(raw) != "0";
}

inline void Mark(const std::string& label) {
    if (!Enabled()) {
        return;
    }
    std::cerr << "[LOCATE] " << label << std::endl;
}

inline void Depth(const std::string& label, const Ciphertext<DCRTPoly>& ct) {
    if (!Enabled()) {
        return;
    }
    if (!ct) {
        std::cerr << "[LOCATE-DEPTH] " << label << ": <null>" << std::endl;
        return;
    }
    std::cerr << "[LOCATE-DEPTH] " << label
              << ": level=" << ct->GetLevel()
              << ", noiseScaleDeg=" << ct->GetNoiseScaleDeg()
              << std::endl;
}

inline void Sync(const CryptoContext<DCRTPoly>& cc, const std::string& label) {
    if (!ForceSync() && !Enabled()) {
        return;
    }
    Mark("SYNC-BEGIN " + label);
    cc->Synchronize();
    Mark("SYNC-END " + label);
}

}  // namespace public_reciprocal_locate

struct PublicReciprocalTimingReport {
    MultiplierTimingReport multiplier{};
    double rowGenerationMs = 0.0;
    double outputMaskMs = 0.0;
    double lowCarryMs = 0.0;
    double addLowCarryMs = 0.0;
    double q0BootstrapMs = 0.0;
    double mulDivisorMs = 0.0;
    double tBootstrapMs = 0.0;
    double subtractMs = 0.0;
    double rBootstrapMs = 0.0;
    double correctionCompareMs = 0.0;
    double correctionBootstrapMs = 0.0;
    double finalIncrementMs = 0.0;
    double finalBootstrapMs = 0.0;
    double totalMs = 0.0;

    uint32_t reciprocalOneBits = 0;
    uint32_t reciprocalZeroBits = 0;
    uint64_t publicRowRotations = 0;
    uint64_t publicRowMaskMults = 0;
    uint64_t pushedRows = 0;
    uint32_t dynamicDaddaStages = 0;
    uint32_t dynamicDaddaInitialRows = 0;
    uint32_t divisorOneBits = 0;
    uint32_t mergedHighRows = 0;
    uint32_t mergedLowRows = 0;
    uint64_t mergedHighOverflowRotations = 0;
};

struct PublicReciprocalDivisionMasks {
    MultiplierMasks multiplier;
    Plaintext active264Mask;
    Plaintext carry264Mask;
    Plaintext low8Mask;
    Plaintext divisorMask;
};

// Forward declaration: EvalPublicConstMulLowKS is defined before the full
// implementation of this helper, but v36 needs to align the two final Dadda
// rows before the final KS add.
inline void AlignBinaryCiphertextsToSameLevelByMask(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly>& a,
    Ciphertext<DCRTPoly>& b,
    Plaintext& activeMask,
    const char* label);

namespace public_reciprocal_detail {

using Clock = std::chrono::steady_clock;

inline double Milliseconds(const Clock::time_point& begin,
                           const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

inline void Synchronize(const CryptoContext<DCRTPoly>& cc) {
    cc->Synchronize();
}

inline uint32_t CountOneBits(const std::array<uint8_t, kWordBits>& bits) {
    uint32_t count = 0;
    for (uint8_t bit : bits) {
        count += static_cast<uint32_t>(bit != 0);
    }
    return count;
}

inline std::vector<uint32_t> BuildDaddaTargets(uint32_t inputRows) {
    if (inputRows <= 2) {
        return {};
    }

    std::vector<uint32_t> sequence;
    sequence.push_back(2);
    while (sequence.back() < inputRows) {
        uint32_t next = (3U * sequence.back()) / 2U;
        if (next <= sequence.back()) {
            next = sequence.back() + 1U;
        }
        sequence.push_back(next);
    }

    if (!sequence.empty() && sequence.back() >= inputRows) {
        sequence.pop_back();
    }
    std::reverse(sequence.begin(), sequence.end());
    return sequence;
}

struct DynamicStageState {
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

class DynamicDaddaPipeline {
public:
    DynamicDaddaPipeline(const CryptoContext<DCRTPoly>& cc,
                         int directionSign,
                         uint32_t inputRows,
                         uint32_t compressorBatchSize,
                         MultiplierTimingReport* timing)
        : cc_(cc),
          directionSign_(directionSign),
          inputRows_(inputRows),
          compressorBatchSize_(compressorBatchSize == 0 ? 1 : compressorBatchSize),
          timing_(timing) {
        if (!cc_) {
            throw std::invalid_argument("DynamicDaddaPipeline received null context");
        }
        if (directionSign_ != -1) {
            throw std::invalid_argument("DynamicDaddaPipeline currently requires directionSign=-1");
        }
        if (inputRows_ == 0) {
            throw std::invalid_argument("DynamicDaddaPipeline requires at least one row");
        }

        const auto targets = BuildDaddaTargets(inputRows_);
        stages_.reserve(targets.size());

        uint32_t rows = inputRows_;
        for (uint32_t target : targets) {
            if (target >= rows || target < ((2U * rows + 2U) / 3U)) {
                throw std::runtime_error("invalid dynamic Dadda target sequence");
            }

            DynamicStageState state;
            state.inputRows = rows;
            state.targetRows = target;
            state.fullAdders = rows - target;
            state.passthroughRows = 3U * target - 2U * rows;

            if (3U * state.fullAdders + state.passthroughRows != rows ||
                2U * state.fullAdders + state.passthroughRows != target) {
                throw std::runtime_error("dynamic Dadda row-count identity failed");
            }

            if (timing_ && stages_.size() < timing_->daddaStages.size()) {
                auto& item = timing_->daddaStages[stages_.size()];
                item.inputRows = rows;
                item.targetRows = target;
                item.fullAdders = state.fullAdders;
                item.passthroughRows = state.passthroughRows;
            }

            stages_.push_back(std::move(state));
            rows = target;
        }
    }

    void Push(Ciphertext<DCRTPoly> row) {
        if (!row) {
            throw std::invalid_argument("DynamicDaddaPipeline received null row");
        }
        ++rowsPushed_;
        if (stages_.empty()) {
            finalRows_.push_back(std::move(row));
        }
        else {
            PushToStage(0, std::move(row));
        }
        UpdatePeakBufferedRows();
    }

    std::vector<Ciphertext<DCRTPoly>> Finish() {
        Synchronize(cc_);
        if (timing_) {
            ++timing_->streamingSynchronizations;
        }
        if (rowsPushed_ != inputRows_) {
            throw std::runtime_error("DynamicDaddaPipeline finished with wrong pushed row count");
        }

        for (const auto& state : stages_) {
            if (state.receivedRows != state.inputRows ||
                state.emittedRows != state.targetRows ||
                state.completedFullAdders != state.fullAdders ||
                state.pendingCount != 0) {
                throw std::runtime_error("Dynamic Dadda stage finished with invalid state");
            }
        }

        if (finalRows_.empty() || finalRows_.size() > 2) {
            throw std::runtime_error("Dynamic Dadda tree did not finish with one or two rows");
        }
        return std::move(finalRows_);
    }

private:
    void PushToStage(uint32_t stageIndex, Ciphertext<DCRTPoly> row) {
        if (stageIndex >= stages_.size()) {
            if (finalRows_.size() >= 2) {
                throw std::runtime_error("Dynamic Dadda emitted more than two final rows");
            }
            finalRows_.push_back(std::move(row));
            return;
        }

        auto& state = stages_[stageIndex];
        if (state.receivedRows >= state.inputRows) {
            throw std::runtime_error("Dynamic Dadda stage received too many rows");
        }

        const uint32_t compressedInputRows = 3U * state.fullAdders;
        ++state.receivedRows;

        if (state.receivedRows <= compressedInputRows) {
            state.pending[state.pendingCount++] = std::move(row);
            if (state.pendingCount != 3) {
                return;
            }

            const auto begin = Clock::now();
            auto compressed = multiplier_detail::EvalCompressor3To2(
                cc_,
                state.pending[0],
                state.pending[1],
                state.pending[2],
                directionSign_,
                timing_);

            state.pending = {};
            state.pendingCount = 0;
            ++state.completedFullAdders;
            state.emittedRows += 2;
            ++compressorsSinceSync_;

            MaybeSynchronize();

            if (timing_ && stageIndex < timing_->daddaStages.size()) {
                timing_->daddaStages[stageIndex].wallMs +=
                    Milliseconds(begin, Clock::now());
            }

            PushToStage(stageIndex + 1, std::move(compressed.sum));
            PushToStage(stageIndex + 1, std::move(compressed.alignedCarry));
        }
        else {
            ++state.emittedRows;
            PushToStage(stageIndex + 1, std::move(row));
        }
    }

    void MaybeSynchronize() {
        if (compressorsSinceSync_ < compressorBatchSize_) {
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
        uint32_t buffered = static_cast<uint32_t>(finalRows_.size());
        for (const auto& stage : stages_) {
            buffered += stage.pendingCount;
        }
        timing_->peakBufferedRows = std::max(timing_->peakBufferedRows, buffered);
    }

    CryptoContext<DCRTPoly> cc_;
    int directionSign_;
    uint32_t inputRows_;
    uint32_t compressorBatchSize_;
    MultiplierTimingReport* timing_;
    std::vector<DynamicStageState> stages_;
    std::vector<Ciphertext<DCRTPoly>> finalRows_;
    uint32_t rowsPushed_ = 0;
    uint32_t compressorsSinceSync_ = 0;
};

}  // namespace public_reciprocal_detail

/**
 * Evaluate a public-constant reciprocal multiplication core.
 *
 * The public reciprocal M is supplied as 256 little-endian Boolean bits.
 * Only one-bits of M generate rows.  This is the public-constant specialization
 * of the existing Dadda + KS multiplier: no encrypted multiplier-bit extraction,
 * no broadcast, no ciphertext gating, and no zero-row padding.
 *
 * Current arithmetic semantics:
 *   row j with M_j=1 contributes N >> (256-j), j=1..255.
 *   This implements the same truncated fixed-point reciprocal row schedule as
 *   the CPU verifier in the paired .cpp file.  D=1 is handled by the caller.
 */
inline Ciphertext<DCRTPoly> EvalPublicReciprocalConstMulKS(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cN,
    const std::array<uint8_t, kWordBits>& reciprocalBits,
    int directionSign,
    MultiplierMasks& masks,
    PublicReciprocalTimingReport* timing = nullptr,
    uint32_t compressorBatchSize = kDefaultCompressorBatchSize) {

    if (!cc || !cN) {
        throw std::invalid_argument(
            "EvalPublicReciprocalConstMulKS received null context/input");
    }
    if (directionSign != -1) {
        throw std::invalid_argument(
            "EvalPublicReciprocalConstMulKS currently requires directionSign=-1");
    }
    if (!masks.ksMasks.activeMask || !masks.ksMasks.carryMask ||
        !masks.bitZeroMask) {
        throw std::invalid_argument(
            "EvalPublicReciprocalConstMulKS received invalid masks");
    }

    PublicReciprocalTimingReport local;
    PublicReciprocalTimingReport& report = timing ? *timing : local;
    report = PublicReciprocalTimingReport{};

    report.reciprocalOneBits =
        public_reciprocal_detail::CountOneBits(reciprocalBits);
    report.reciprocalZeroBits = kWordBits - report.reciprocalOneBits;

    // j=0 would contribute N>>256, which is identically zero.  Do not push it.
    uint32_t effectiveRows = 0;
    for (uint32_t j = 1; j < kWordBits; ++j) {
        if (reciprocalBits[j] != 0) {
            ++effectiveRows;
        }
    }
    report.dynamicDaddaInitialRows = effectiveRows;

    if (effectiveRows == 0) {
        auto zero = cc->EvalSub(cN, cN);
        zero = cc->EvalMult(zero, masks.ksMasks.activeMask);
        return zero;
    }

    public_reciprocal_detail::Synchronize(cc);
    const auto totalBegin = public_reciprocal_detail::Clock::now();

    public_reciprocal_detail::DynamicDaddaPipeline pipeline(
        cc,
        directionSign,
        effectiveRows,
        compressorBatchSize,
        &report.multiplier);
    report.dynamicDaddaStages = static_cast<uint32_t>(
        public_reciprocal_detail::BuildDaddaTargets(effectiveRows).size());

    // Incremental right-shift stream: N>>1, N>>2, ..., N>>255.  Only rows whose
    // public reciprocal bit is 1 are pushed into the dynamic Dadda pipeline.
    auto shiftedRight = cc->EvalRotate(cN, 1);
    report.publicRowRotations += 1;

    for (uint32_t shiftRight = 1; shiftRight < kWordBits; ++shiftRight) {
        const uint32_t j = kWordBits - shiftRight;

        public_reciprocal_detail::Synchronize(cc);
        const auto rowBegin = public_reciprocal_detail::Clock::now();

        if (reciprocalBits[j] != 0) {
            auto row = cc->EvalMult(shiftedRight, masks.ksMasks.activeMask);
            ++report.publicRowMaskMults;
            ++report.pushedRows;
            pipeline.Push(std::move(row));
        }

        public_reciprocal_detail::Synchronize(cc);
        report.rowGenerationMs += public_reciprocal_detail::Milliseconds(
            rowBegin, public_reciprocal_detail::Clock::now());

        if (shiftRight + 1 < kWordBits) {
            shiftedRight = cc->EvalRotate(shiftedRight, 1);
            report.publicRowRotations += 1;
        }
    }

    auto rows = pipeline.Finish();

    Ciphertext<DCRTPoly> product;
    if (rows.size() == 1) {
        product = std::move(rows[0]);
        report.multiplier.finalAdder.totalMs = 0.0;
    }
    else if (rows.size() == 2) {
        product = EvalAddKS(
            cc,
            rows[0],
            rows[1],
            directionSign,
            masks.ksMasks,
            &report.multiplier.finalAdder);
    }
    else {
        throw std::runtime_error("unexpected dynamic Dadda output row count");
    }

    public_reciprocal_detail::Synchronize(cc);
    const auto maskBegin = public_reciprocal_detail::Clock::now();
    product = cc->EvalMult(product, masks.ksMasks.activeMask);
    public_reciprocal_detail::Synchronize(cc);
    report.outputMaskMs = public_reciprocal_detail::Milliseconds(
        maskBegin, public_reciprocal_detail::Clock::now());
    report.multiplier.outputMaskMs = report.outputMaskMs;

    report.totalMs = public_reciprocal_detail::Milliseconds(
        totalBegin, public_reciprocal_detail::Clock::now());
    report.multiplier.totalMs = report.totalMs;
    return product;
}


namespace public_reciprocal_masks_detail {

inline std::vector<double> MakeWidthMaskValues(uint32_t width, bool clearBitZero) {
    if (width == 0 || width > kStride) {
        throw std::invalid_argument("invalid mask width");
    }
    std::vector<double> values(kMultiplierTotalSlots, 0.0);
    for (uint32_t word = 0; word < kMultiplierNumWords; ++word) {
        const uint32_t base = word * kStride;
        const uint32_t begin = clearBitZero ? 1U : 0U;
        for (uint32_t bit = begin; bit < width; ++bit) {
            values[base + bit] = 1.0;
        }
    }
    return values;
}

inline std::vector<double> MakeLowBitsMaskValues(uint32_t bits) {
    if (bits == 0 || bits > kWordBits) {
        throw std::invalid_argument("invalid low-bit mask width");
    }
    std::vector<double> values(kMultiplierTotalSlots, 0.0);
    for (uint32_t word = 0; word < kMultiplierNumWords; ++word) {
        const uint32_t base = word * kStride;
        for (uint32_t bit = 0; bit < bits; ++bit) {
            values[base + bit] = 1.0;
        }
    }
    return values;
}

inline std::vector<double> MakeConstantMaskValues(
    const std::array<uint8_t, kWordBits>& bits) {
    std::vector<double> values(kMultiplierTotalSlots, 0.0);
    for (uint32_t word = 0; word < kMultiplierNumWords; ++word) {
        const uint32_t base = word * kStride;
        for (uint32_t bit = 0; bit < kWordBits; ++bit) {
            values[base + bit] = bits[bit] ? 1.0 : 0.0;
        }
    }
    return values;
}

} // namespace public_reciprocal_masks_detail

inline PublicReciprocalDivisionMasks BuildAndLoadPublicReciprocalDivisionMasks(
    const CryptoContext<DCRTPoly>& cc,
    int directionSign,
    const std::array<uint8_t, kWordBits>& divisorBits) {

    if (!cc) {
        throw std::invalid_argument("BuildAndLoadPublicReciprocalDivisionMasks received null context");
    }
    if (directionSign != -1) {
        throw std::invalid_argument("public reciprocal divider currently requires directionSign=-1");
    }

    PublicReciprocalDivisionMasks masks;
    masks.multiplier = BuildAndLoadMultiplierMasks(cc, directionSign);

    masks.active264Mask = cc->MakeCKKSPackedPlaintext(
        public_reciprocal_masks_detail::MakeWidthMaskValues(264, false),
        1, 0, nullptr, kMultiplierTotalSlots);
    masks.carry264Mask = cc->MakeCKKSPackedPlaintext(
        public_reciprocal_masks_detail::MakeWidthMaskValues(264, true),
        1, 0, nullptr, kMultiplierTotalSlots);
    masks.low8Mask = cc->MakeCKKSPackedPlaintext(
        public_reciprocal_masks_detail::MakeLowBitsMaskValues(8),
        1, 0, nullptr, kMultiplierTotalSlots);
    masks.divisorMask = cc->MakeCKKSPackedPlaintext(
        public_reciprocal_masks_detail::MakeConstantMaskValues(divisorBits),
        1, 0, nullptr, kMultiplierTotalSlots);

    masks.active264Mask->SetLength(kMultiplierTotalSlots);
    masks.carry264Mask->SetLength(kMultiplierTotalSlots);
    masks.low8Mask->SetLength(kMultiplierTotalSlots);
    masks.divisorMask->SetLength(kMultiplierTotalSlots);

    cc->LoadPlaintext(masks.active264Mask);
    cc->LoadPlaintext(masks.carry264Mask);
    cc->LoadPlaintext(masks.low8Mask);
    cc->LoadPlaintext(masks.divisorMask);
    public_reciprocal_detail::Synchronize(cc);
    return masks;
}

struct AddWidthResult {
    Ciphertext<DCRTPoly> sum;
};

inline Ciphertext<DCRTPoly> EvalAddKSWidth(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    uint32_t width,
    int directionSign,
    Plaintext& activeMask,
    Plaintext& carryMask) {

    if (!cc || !cA || !cB) {
        throw std::invalid_argument("EvalAddKSWidth received null input");
    }
    if (directionSign != -1) {
        throw std::invalid_argument("EvalAddKSWidth currently requires directionSign=-1");
    }
    if (width == 0 || width > kStride) {
        throw std::invalid_argument("EvalAddKSWidth invalid width");
    }

    const uint32_t layers = width <= kWordBits ? kNumLayers : 9U;

    auto g = cc->EvalMult(cA, cB);
    auto p = cc->EvalSub(cc->EvalAdd(cA, cB), cc->EvalAdd(g, g));
    auto G = g;
    auto Pk = p;

    for (uint32_t layer = 0; layer < layers; ++layer) {
        const int32_t step = static_cast<int32_t>(1U << layer);
        auto Gold = G;
        auto Pold = Pk;
        auto Grot = cc->EvalRotate(Gold, directionSign * step);
        auto Prot = cc->EvalRotate(Pold, directionSign * step);
        auto term = cc->EvalMult(Grot, Pold);
        if (layer + 1 < layers) {
            Pk = cc->EvalMult(Prot, Pold);
        }
        G = cc->EvalAdd(Gold, term);
    }

    auto C = cc->EvalRotate(G, directionSign);
    C = cc->EvalMult(C, carryMask);
    G = cc->EvalMult(G, activeMask);

    auto result = cc->EvalAdd(cA, cB);
    result = cc->EvalAdd(result, C);
    result = cc->EvalSub(result, cc->EvalAdd(G, G));
    public_reciprocal_detail::Synchronize(cc);
    return result;
}

inline Ciphertext<DCRTPoly> BootstrapProjectActive256(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& input,
    PublicReciprocalDivisionMasks& masks) {

    auto x = cc->EvalBootstrap(input);
    public_reciprocal_detail::Synchronize(cc);
    auto x2 = cc->EvalMult(x, x);
    auto x3 = cc->EvalMult(x2, x);
    auto threeX2 = cc->EvalAdd(x2, cc->EvalAdd(x2, x2));
    auto twoX3 = cc->EvalAdd(x3, x3);
    auto projected = cc->EvalSub(threeX2, twoX3);
    projected = cc->EvalMult(projected, masks.multiplier.ksMasks.activeMask);
    public_reciprocal_detail::Synchronize(cc);
    return projected;
}

inline Ciphertext<DCRTPoly> BootstrapProjectActive264(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& input,
    PublicReciprocalDivisionMasks& masks) {

    auto x = cc->EvalBootstrap(input);
    public_reciprocal_detail::Synchronize(cc);
    auto x2 = cc->EvalMult(x, x);
    auto x3 = cc->EvalMult(x2, x);
    auto threeX2 = cc->EvalAdd(x2, cc->EvalAdd(x2, x2));
    auto twoX3 = cc->EvalAdd(x3, x3);
    auto projected = cc->EvalSub(threeX2, twoX3);
    projected = cc->EvalMult(projected, masks.active264Mask);
    public_reciprocal_detail::Synchronize(cc);
    return projected;
}

inline Ciphertext<DCRTPoly> EvalPublicConstMulLowKS(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cN,
    const std::array<uint8_t, kWordBits>& constantBits,
    int directionSign,
    PublicReciprocalDivisionMasks& masks,
    PublicReciprocalTimingReport* timing,
    uint32_t compressorBatchSize) {

    uint32_t effectiveRows = public_reciprocal_detail::CountOneBits(constantBits);
    public_reciprocal_locate::Mark("BEGIN EvalPublicConstMulLowKS v36 final-add-preboot effectiveRows=" + std::to_string(effectiveRows));
    public_reciprocal_locate::Depth("EvalPublicConstMulLowKS input", cN);

    if (effectiveRows == 0) {
        public_reciprocal_locate::Mark("BEGIN EvalPublicConstMulLowKS zero constant");
        auto zero = cc->EvalSub(cN, cN);
        zero = cc->EvalMult(zero, masks.multiplier.ksMasks.activeMask);
        public_reciprocal_locate::Sync(cc, "EvalPublicConstMulLowKS zero constant mask");
        zero = BootstrapProjectActive256(cc, zero, masks);
        public_reciprocal_locate::Depth("END EvalPublicConstMulLowKS zero constant", zero);
        return zero;
    }

    int32_t highestSetBit = -1;
    for (int32_t j = static_cast<int32_t>(kWordBits) - 1; j >= 0; --j) {
        if (constantBits[static_cast<uint32_t>(j)] != 0) {
            highestSetBit = j;
            break;
        }
    }
    if (highestSetBit < 0) {
        throw std::runtime_error("EvalPublicConstMulLowKS internal error: effectiveRows>0 but highestSetBit<0");
    }
    public_reciprocal_locate::Mark("EvalPublicConstMulLowKS highestSetBit=" + std::to_string(highestSetBit));

    // v36 static fix for q0D:
    //   v35 located the crash at the final EvalAddKS after DynamicDaddaPipeline::Finish().
    //   The pipeline rows can be too deep for another KS prefix add.  Boot/project both
    //   final rows before the final add, and boot/project the product before returning.
    //   This keeps the public-constant specialization while removing the depth hazard.
    public_reciprocal_detail::DynamicDaddaPipeline pipeline(
        cc, directionSign, effectiveRows, compressorBatchSize,
        timing ? &timing->multiplier : nullptr);

    auto shiftedLeft = cN;
    for (uint32_t j = 0; j <= static_cast<uint32_t>(highestSetBit); ++j) {
        if (constantBits[j] != 0) {
            public_reciprocal_locate::Mark("BEGIN EvalPublicConstMulLowKS row j=" + std::to_string(j));
            auto row = cc->EvalMult(shiftedLeft, masks.multiplier.ksMasks.activeMask);
            if (timing) {
                ++timing->publicRowMaskMults;
            }
            public_reciprocal_locate::Sync(cc, "EvalPublicConstMulLowKS row j=" + std::to_string(j) + " EvalMult");
            pipeline.Push(std::move(row));
            public_reciprocal_locate::Sync(cc, "EvalPublicConstMulLowKS row j=" + std::to_string(j) + " Push");
            public_reciprocal_locate::Mark("END EvalPublicConstMulLowKS row j=" + std::to_string(j));
        }

        if (j < static_cast<uint32_t>(highestSetBit)) {
            public_reciprocal_locate::Mark("BEGIN EvalPublicConstMulLowKS shift j=" + std::to_string(j) + " -> " + std::to_string(j + 1));
            shiftedLeft = cc->EvalRotate(shiftedLeft, directionSign);
            if (timing) {
                ++timing->publicRowRotations;
            }
            public_reciprocal_locate::Sync(cc, "EvalPublicConstMulLowKS shift j=" + std::to_string(j) + " -> " + std::to_string(j + 1));
            public_reciprocal_locate::Mark("END EvalPublicConstMulLowKS shift j=" + std::to_string(j) + " -> " + std::to_string(j + 1));
        }
    }

    shiftedLeft = nullptr;
    public_reciprocal_locate::Sync(cc, "EvalPublicConstMulLowKS shiftedLeft released before Finish");

    public_reciprocal_locate::Mark("BEGIN EvalPublicConstMulLowKS pipeline Finish");
    auto rows = pipeline.Finish();
    public_reciprocal_locate::Mark("END EvalPublicConstMulLowKS pipeline Finish rowsOut=" + std::to_string(rows.size()));

    Ciphertext<DCRTPoly> product;
    if (rows.size() == 1) {
        public_reciprocal_locate::Depth("EvalPublicConstMulLowKS single row before bootstrap", rows[0]);
        product = BootstrapProjectActive256(cc, rows[0], masks);
        rows.clear();
        public_reciprocal_locate::Depth("END EvalPublicConstMulLowKS single row bootstrapped", product);
        return product;
    }

    if (rows.size() != 2) {
        throw std::runtime_error("EvalPublicConstMulLowKS expected one or two final rows");
    }

    public_reciprocal_locate::Mark("BEGIN EvalPublicConstMulLowKS preboot final row 0");
    rows[0] = BootstrapProjectActive256(cc, rows[0], masks);
    public_reciprocal_locate::Depth("END EvalPublicConstMulLowKS preboot final row 0", rows[0]);

    public_reciprocal_locate::Mark("BEGIN EvalPublicConstMulLowKS preboot final row 1");
    rows[1] = BootstrapProjectActive256(cc, rows[1], masks);
    public_reciprocal_locate::Depth("END EvalPublicConstMulLowKS preboot final row 1", rows[1]);

    public_reciprocal_locate::Mark("BEGIN EvalPublicConstMulLowKS final rows level align");
    AlignBinaryCiphertextsToSameLevelByMask(
        cc, rows[0], rows[1], masks.multiplier.ksMasks.activeMask,
        "EvalPublicConstMulLowKS final rows level-align");
    public_reciprocal_locate::Sync(cc, "EvalPublicConstMulLowKS final rows level align");
    public_reciprocal_locate::Mark("END EvalPublicConstMulLowKS final rows level align");

    public_reciprocal_locate::Mark("BEGIN EvalPublicConstMulLowKS final EvalAddKS");
    product = EvalAddKS(cc, rows[0], rows[1], directionSign,
                        masks.multiplier.ksMasks,
                        timing ? &timing->multiplier.finalAdder : nullptr);
    public_reciprocal_locate::Sync(cc, "EvalPublicConstMulLowKS final EvalAddKS");
    public_reciprocal_locate::Depth("END EvalPublicConstMulLowKS final EvalAddKS", product);

    rows.clear();
    public_reciprocal_locate::Sync(cc, "EvalPublicConstMulLowKS rows cleared before product bootstrap");

    public_reciprocal_locate::Mark("BEGIN EvalPublicConstMulLowKS product bootstrap before return");
    product = BootstrapProjectActive256(cc, product, masks);
    public_reciprocal_locate::Depth("END EvalPublicConstMulLowKS product bootstrap before return", product);
    return product;
}

inline Ciphertext<DCRTPoly> EvalPublicConstMulLowWide264KS(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cN,
    const std::array<uint8_t, kWordBits>& constantBits,
    int directionSign,
    PublicReciprocalDivisionMasks& masks,
    PublicReciprocalTimingReport* timing,
    uint32_t compressorBatchSize) {

    uint32_t effectiveRows = public_reciprocal_detail::CountOneBits(constantBits);
    if (effectiveRows == 0) {
        auto zero = cc->EvalSub(cN, cN);
        zero = cc->EvalMult(zero, masks.active264Mask);
        return zero;
    }

    public_reciprocal_detail::DynamicDaddaPipeline pipeline(
        cc, directionSign, effectiveRows, compressorBatchSize,
        timing ? &timing->multiplier : nullptr);

    auto shiftedLeft = cN;
    for (uint32_t j = 0; j < kWordBits; ++j) {
        if (constantBits[j] != 0) {
            // Low-half row: keep only low 256 bits of (N << j).  Carries produced
            // by the compression tree are allowed to enter bits 256..263.
            auto row = cc->EvalMult(shiftedLeft, masks.multiplier.ksMasks.activeMask);
            pipeline.Push(std::move(row));
        }
        if (j + 1 < kWordBits) {
            shiftedLeft = cc->EvalRotate(shiftedLeft, directionSign);
        }
    }

    auto rows = pipeline.Finish();
    Ciphertext<DCRTPoly> product;
    if (rows.size() == 1) {
        product = std::move(rows[0]);
    }
    else {
        product = EvalAddKSWidth(cc, rows[0], rows[1], 264, directionSign,
                                 masks.active264Mask, masks.carry264Mask);
    }
    product = cc->EvalMult(product, masks.active264Mask);
    public_reciprocal_detail::Synchronize(cc);
    return product;
}



struct AddCarryOutResult {
    Ciphertext<DCRTPoly> sum;
    Ciphertext<DCRTPoly> carryOutBitAtZero;
};

inline AddCarryOutResult EvalAddKSWithCarryOut256(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    int directionSign,
    PublicReciprocalDivisionMasks& masks) {

    if (!cc || !cA || !cB) {
        throw std::invalid_argument("EvalAddKSWithCarryOut256 received null input");
    }
    if (directionSign != -1) {
        throw std::invalid_argument("EvalAddKSWithCarryOut256 requires directionSign=-1");
    }

    auto g = cc->EvalMult(cA, cB);
    auto p = cc->EvalSub(cc->EvalAdd(cA, cB), cc->EvalAdd(g, g));
    auto G = g;
    auto Pk = p;

    for (uint32_t layer = 0; layer < kNumLayers; ++layer) {
        const int32_t step = static_cast<int32_t>(1U << layer);
        auto Gold = G;
        auto Pold = Pk;
        auto Grot = cc->EvalRotate(Gold, directionSign * step);
        auto Prot = cc->EvalRotate(Pold, directionSign * step);
        auto term = cc->EvalMult(Grot, Pold);
        if (layer + 1 < kNumLayers) {
            Pk = cc->EvalMult(Prot, Pold);
        }
        G = cc->EvalAdd(Gold, term);
    }

    auto C = cc->EvalRotate(G, directionSign);
    C = cc->EvalMult(C, masks.multiplier.ksMasks.carryMask);
    G = cc->EvalMult(G, masks.multiplier.ksMasks.activeMask);

    auto carryOut = cc->EvalRotate(G, static_cast<int32_t>(kWordBits - 1));
    carryOut = cc->EvalMult(carryOut, masks.multiplier.bitZeroMask);

    auto result = cc->EvalAdd(cA, cB);
    result = cc->EvalAdd(result, C);
    result = cc->EvalSub(result, cc->EvalAdd(G, G));
    result = cc->EvalMult(result, masks.multiplier.ksMasks.activeMask);
    public_reciprocal_detail::Synchronize(cc);
    return {std::move(result), std::move(carryOut)};
}


struct PublicReciprocalHiLowResult {
    Ciphertext<DCRTPoly> highNoCarry;
    Ciphertext<DCRTPoly> carryDown;
};

namespace public_reciprocal_chunked_detail {

constexpr uint32_t kDefaultReciprocalChunkRows = 16U;

inline uint32_t RuntimeReciprocalChunkRows() {
    const char* raw = std::getenv("KS256_RECIP_CHUNK_ROWS");
    if (!raw || !*raw) {
        return kDefaultReciprocalChunkRows;
    }
    char* end = nullptr;
    unsigned long value = std::strtoul(raw, &end, 10);
    if (end == raw || value == 0UL) {
        return kDefaultReciprocalChunkRows;
    }
    if (value > 64UL) {
        value = 64UL;
    }
    return static_cast<uint32_t>(value);
}

inline uint32_t CountHighRowsInRange(
    const std::vector<uint32_t>& positions,
    uint32_t begin,
    uint32_t end) {
    uint32_t count = 0;
    for (uint32_t i = begin; i < end; ++i) {
        count += static_cast<uint32_t>(positions[i] != 0U);
    }
    return count;
}

inline Ciphertext<DCRTPoly> MakeZeroActive256(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& ref,
    PublicReciprocalDivisionMasks& masks) {
    auto zero = cc->EvalSub(ref, ref);
    zero = cc->EvalMult(zero, masks.multiplier.ksMasks.activeMask);
    public_reciprocal_detail::Synchronize(cc);
    return BootstrapProjectActive256(cc, zero, masks);
}

inline Ciphertext<DCRTPoly> FinishDaddaRowsActive256(
    const CryptoContext<DCRTPoly>& cc,
    std::vector<Ciphertext<DCRTPoly>> rows,
    int directionSign,
    PublicReciprocalDivisionMasks& masks,
    PublicReciprocalTimingReport* timing) {

    if (rows.empty()) {
        throw std::invalid_argument("FinishDaddaRowsActive256 received no rows");
    }

    Ciphertext<DCRTPoly> out;
    if (rows.size() == 1) {
        out = std::move(rows[0]);
    }
    else if (rows.size() == 2) {
        out = EvalAddKS(cc, rows[0], rows[1], directionSign,
                        masks.multiplier.ksMasks,
                        timing ? &timing->multiplier.finalAdder : nullptr);
    }
    else {
        throw std::runtime_error("active256 Dadda finish returned invalid row count");
    }
    out = cc->EvalMult(out, masks.multiplier.ksMasks.activeMask);
    public_reciprocal_detail::Synchronize(cc);
    return BootstrapProjectActive256(cc, out, masks);
}

inline Ciphertext<DCRTPoly> FinishDaddaRowsActive264(
    const CryptoContext<DCRTPoly>& cc,
    std::vector<Ciphertext<DCRTPoly>> rows,
    int directionSign,
    PublicReciprocalDivisionMasks& masks) {

    if (rows.empty()) {
        throw std::invalid_argument("FinishDaddaRowsActive264 received no rows");
    }

    Ciphertext<DCRTPoly> out;
    if (rows.size() == 1) {
        out = std::move(rows[0]);
    }
    else if (rows.size() == 2) {
        out = EvalAddKSWidth(cc, rows[0], rows[1], 264, directionSign,
                             masks.active264Mask, masks.carry264Mask);
    }
    else {
        throw std::runtime_error("active264 Dadda finish returned invalid row count");
    }
    out = cc->EvalMult(out, masks.active264Mask);
    public_reciprocal_detail::Synchronize(cc);
    return BootstrapProjectActive264(cc, out, masks);
}

}  // namespace public_reciprocal_chunked_detail

/**
 * Public-specialized reciprocal high/low construction.
 *
 * This is the performance-oriented public-divisor path.  It deliberately skips
 * public zero-bits and therefore exposes/capitalizes on the public divisor's
 * bit pattern.  Dense reciprocal constants are split into chunks so a single
 * DynamicDaddaPipeline never receives too many rows before a bootstrap.
 *
 * The returned quotient approximation components satisfy:
 *   q0 = highNoCarry + carryDown,
 * where highNoCarry is the sum of high256(chunkProduct_i) and carryDown is the
 * carry generated by adding all chunk low halves, including each chunk's
 * internal low carry and inter-chunk low-256 carry.
 */
inline PublicReciprocalHiLowResult EvalPublicReciprocalHiLowMergedKS(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cN,
    const std::array<uint8_t, kWordBits>& reciprocalBits,
    int directionSign,
    PublicReciprocalDivisionMasks& masks,
    PublicReciprocalTimingReport* timing,
    uint32_t compressorBatchSize) {

    if (!cc || !cN) {
        throw std::invalid_argument("EvalPublicReciprocalHiLowMergedKS received null input");
    }
    if (directionSign != -1) {
        throw std::invalid_argument("EvalPublicReciprocalHiLowMergedKS requires directionSign=-1");
    }

    public_reciprocal_locate::Mark("BEGIN EvalPublicReciprocalHiLowMergedKS");
    public_reciprocal_locate::Depth("input cN", cN);

    std::vector<uint32_t> positions;
    positions.reserve(kWordBits);
    for (uint32_t j = 0; j < kWordBits; ++j) {
        if (reciprocalBits[j] != 0) {
            positions.push_back(j);
        }
    }

    const uint32_t lowRows = static_cast<uint32_t>(positions.size());
    const uint32_t chunkLimit = public_reciprocal_chunked_detail::RuntimeReciprocalChunkRows();
    public_reciprocal_locate::Mark("reciprocal selected one-bit rows=" + std::to_string(lowRows)
        + ", chunkLimit=" + std::to_string(chunkLimit));
    const uint32_t highRows = public_reciprocal_chunked_detail::CountHighRowsInRange(
        positions, 0U, lowRows);

    if (timing) {
        timing->mergedLowRows = lowRows;
        timing->mergedHighRows = highRows;
        timing->pushedRows += static_cast<uint64_t>(lowRows) + highRows;
        timing->dynamicDaddaInitialRows = std::min(
            chunkLimit,
            std::max(lowRows, highRows));
        timing->dynamicDaddaStages = static_cast<uint32_t>(
            public_reciprocal_detail::BuildDaddaTargets(
                timing->dynamicDaddaInitialRows).size());
    }

    public_reciprocal_locate::Mark("BEGIN make zero active256");
    auto zero = public_reciprocal_chunked_detail::MakeZeroActive256(cc, cN, masks);
    public_reciprocal_locate::Depth("END make zero active256", zero);
    auto lowAccumulator = zero;
    auto carryCounter = zero;
    carryCounter = cc->EvalMult(carryCounter, masks.low8Mask);
    public_reciprocal_locate::Sync(cc, "init carryCounter low8");
    auto highAccumulator = zero;

    if (positions.empty()) {
        public_reciprocal_locate::Mark("RETURN EvalPublicReciprocalHiLowMergedKS empty positions");
        return {std::move(highAccumulator), std::move(carryCounter)};
    }

    uint32_t totalProcessed = 0;
    uint32_t rowsInChunk = 0;
    uint32_t chunkBeginIndex = 0;
    uint32_t currentChunkLowRows = std::min(
        chunkLimit,
        lowRows);
    uint32_t currentChunkHighRows = public_reciprocal_chunked_detail::CountHighRowsInRange(
        positions, chunkBeginIndex, chunkBeginIndex + currentChunkLowRows);

    public_reciprocal_locate::Mark("BEGIN create first low/high pipelines");
    std::unique_ptr<public_reciprocal_detail::DynamicDaddaPipeline> lowPipeline(
        new public_reciprocal_detail::DynamicDaddaPipeline(
            cc, directionSign, currentChunkLowRows, compressorBatchSize,
            timing ? &timing->multiplier : nullptr));
    std::unique_ptr<public_reciprocal_detail::DynamicDaddaPipeline> highPipeline;
    if (currentChunkHighRows != 0) {
        highPipeline.reset(new public_reciprocal_detail::DynamicDaddaPipeline(
            cc, directionSign, currentChunkHighRows, compressorBatchSize,
            timing ? &timing->multiplier : nullptr));
    }

    public_reciprocal_locate::Mark("END create first low/high pipelines");

    auto startNextChunk = [&]() {
        public_reciprocal_locate::Mark("BEGIN startNextChunk totalProcessed=" + std::to_string(totalProcessed));
        chunkBeginIndex = totalProcessed;
        rowsInChunk = 0;
        const uint32_t remaining = lowRows - totalProcessed;
        currentChunkLowRows = std::min(
            chunkLimit,
            remaining);
        currentChunkHighRows = public_reciprocal_chunked_detail::CountHighRowsInRange(
            positions, chunkBeginIndex, chunkBeginIndex + currentChunkLowRows);

        lowPipeline.reset(new public_reciprocal_detail::DynamicDaddaPipeline(
            cc, directionSign, currentChunkLowRows, compressorBatchSize,
            timing ? &timing->multiplier : nullptr));
        highPipeline.reset();
        if (currentChunkHighRows != 0) {
            highPipeline.reset(new public_reciprocal_detail::DynamicDaddaPipeline(
                cc, directionSign, currentChunkHighRows, compressorBatchSize,
                timing ? &timing->multiplier : nullptr));
        }
        public_reciprocal_locate::Mark("END startNextChunk chunkBeginIndex=" + std::to_string(chunkBeginIndex)
            + " lowRows=" + std::to_string(currentChunkLowRows)
            + " highRows=" + std::to_string(currentChunkHighRows));
    };

    auto finalizeCurrentChunk = [&]() {
        const uint32_t chunkId = chunkBeginIndex / chunkLimit;
        public_reciprocal_locate::Mark("BEGIN finalize chunk=" + std::to_string(chunkId)
            + " beginIndex=" + std::to_string(chunkBeginIndex)
            + " rows=" + std::to_string(currentChunkLowRows)
            + " highRows=" + std::to_string(currentChunkHighRows));
        public_reciprocal_locate::Mark("BEGIN chunk=" + std::to_string(chunkId) + " lowPipeline Finish");
        auto lowRowsOut = lowPipeline->Finish();
        public_reciprocal_locate::Mark("END chunk=" + std::to_string(chunkId) + " lowPipeline Finish rowsOut=" + std::to_string(lowRowsOut.size()));
        public_reciprocal_locate::Mark("BEGIN chunk=" + std::to_string(chunkId) + " FinishDaddaRowsActive264 lowWide");
        auto chunkLowWide = public_reciprocal_chunked_detail::FinishDaddaRowsActive264(
            cc, std::move(lowRowsOut), directionSign, masks);
        public_reciprocal_locate::Depth("END chunk=" + std::to_string(chunkId) + " lowWide", chunkLowWide);

        public_reciprocal_locate::Mark("BEGIN chunk=" + std::to_string(chunkId) + " chunkLow mask");
        auto chunkLow = cc->EvalMult(chunkLowWide, masks.multiplier.ksMasks.activeMask);
        public_reciprocal_locate::Sync(cc, "chunk=" + std::to_string(chunkId) + " chunkLow mask");
        public_reciprocal_locate::Mark("BEGIN chunk=" + std::to_string(chunkId) + " chunkLow bootstrap");
        chunkLow = BootstrapProjectActive256(cc, chunkLow, masks);
        public_reciprocal_locate::Depth("END chunk=" + std::to_string(chunkId) + " chunkLow bootstrap", chunkLow);

        public_reciprocal_locate::Mark("BEGIN chunk=" + std::to_string(chunkId) + " chunkCarry rotate");
        auto chunkCarry = cc->EvalRotate(chunkLowWide, static_cast<int32_t>(kWordBits));
        chunkCarry = cc->EvalMult(chunkCarry, masks.low8Mask);
        public_reciprocal_locate::Sync(cc, "chunk=" + std::to_string(chunkId) + " chunkCarry rotate+mask");
        public_reciprocal_locate::Mark("BEGIN chunk=" + std::to_string(chunkId) + " chunkCarry bootstrap");
        chunkCarry = BootstrapProjectActive256(cc, chunkCarry, masks);
        public_reciprocal_locate::Depth("END chunk=" + std::to_string(chunkId) + " chunkCarry bootstrap", chunkCarry);
        chunkCarry = cc->EvalMult(chunkCarry, masks.low8Mask);
        public_reciprocal_locate::Sync(cc, "chunk=" + std::to_string(chunkId) + " chunkCarry final low8 mask");

        Ciphertext<DCRTPoly> chunkHigh = zero;
        if (highPipeline) {
            public_reciprocal_locate::Mark("BEGIN chunk=" + std::to_string(chunkId) + " highPipeline Finish");
            auto highRowsOut = highPipeline->Finish();
            public_reciprocal_locate::Mark("END chunk=" + std::to_string(chunkId) + " highPipeline Finish rowsOut=" + std::to_string(highRowsOut.size()));
            public_reciprocal_locate::Mark("BEGIN chunk=" + std::to_string(chunkId) + " FinishDaddaRowsActive256 high");
            chunkHigh = public_reciprocal_chunked_detail::FinishDaddaRowsActive256(
                cc, std::move(highRowsOut), directionSign, masks, timing);
            public_reciprocal_locate::Depth("END chunk=" + std::to_string(chunkId) + " high chunk", chunkHigh);
        }
        else {
            public_reciprocal_locate::Mark("BEGIN chunk=" + std::to_string(chunkId) + " high chunk zero bootstrap");
            chunkHigh = BootstrapProjectActive256(cc, chunkHigh, masks);
            public_reciprocal_locate::Depth("END chunk=" + std::to_string(chunkId) + " high chunk zero bootstrap", chunkHigh);
        }

        // Accumulate low halves.  Both operands are bootstrapped before the KS add.
        public_reciprocal_locate::Mark("BEGIN chunk=" + std::to_string(chunkId) + " lowAccumulator bootstrap");
        lowAccumulator = BootstrapProjectActive256(cc, lowAccumulator, masks);
        public_reciprocal_locate::Depth("END chunk=" + std::to_string(chunkId) + " lowAccumulator bootstrap", lowAccumulator);
        public_reciprocal_locate::Mark("BEGIN chunk=" + std::to_string(chunkId) + " lowAccumulator + chunkLow with carryOut");
        auto lowAdd = EvalAddKSWithCarryOut256(
            cc, lowAccumulator, chunkLow, directionSign, masks);
        public_reciprocal_locate::Depth("END chunk=" + std::to_string(chunkId) + " lowAdd sum before boot", lowAdd.sum);
        lowAccumulator = BootstrapProjectActive256(cc, lowAdd.sum, masks);
        public_reciprocal_locate::Depth("END chunk=" + std::to_string(chunkId) + " lowAccumulator after lowAdd boot", lowAccumulator);

        public_reciprocal_locate::Mark("BEGIN chunk=" + std::to_string(chunkId) + " interChunkCarry bootstrap");
        auto interChunkCarry = BootstrapProjectActive256(cc, lowAdd.carryOutBitAtZero, masks);
        interChunkCarry = cc->EvalMult(interChunkCarry, masks.low8Mask);
        public_reciprocal_locate::Sync(cc, "chunk=" + std::to_string(chunkId) + " interChunkCarry low8 mask");

        // carryCounter += chunkCarry + interChunkCarry, kept in low 8 bits.
        public_reciprocal_locate::Mark("BEGIN chunk=" + std::to_string(chunkId) + " carryCounter bootstrap");
        carryCounter = BootstrapProjectActive256(cc, carryCounter, masks);
        public_reciprocal_locate::Depth("END chunk=" + std::to_string(chunkId) + " carryCounter bootstrap", carryCounter);
        public_reciprocal_locate::Mark("BEGIN chunk=" + std::to_string(chunkId) + " carryCounter + chunkCarry");
        auto carryPlusChunk = EvalAddKS(
            cc, carryCounter, chunkCarry, directionSign,
            masks.multiplier.ksMasks, nullptr);
        public_reciprocal_locate::Depth("END chunk=" + std::to_string(chunkId) + " carryPlusChunk before boot", carryPlusChunk);
        carryCounter = BootstrapProjectActive256(cc, carryPlusChunk, masks);
        carryCounter = cc->EvalMult(carryCounter, masks.low8Mask);
        public_reciprocal_locate::Sync(cc, "chunk=" + std::to_string(chunkId) + " carryCounter low8 after chunkCarry");

        public_reciprocal_locate::Mark("BEGIN chunk=" + std::to_string(chunkId) + " carryCounter + interChunkCarry");
        auto carryPlusInter = EvalAddKS(
            cc, carryCounter, interChunkCarry, directionSign,
            masks.multiplier.ksMasks, nullptr);
        public_reciprocal_locate::Depth("END chunk=" + std::to_string(chunkId) + " carryPlusInter before boot", carryPlusInter);
        carryCounter = BootstrapProjectActive256(cc, carryPlusInter, masks);
        carryCounter = cc->EvalMult(carryCounter, masks.low8Mask);
        public_reciprocal_locate::Sync(cc, "chunk=" + std::to_string(chunkId) + " carryCounter low8 after interCarry");

        // Accumulate high halves at chunk granularity.
        public_reciprocal_locate::Mark("BEGIN chunk=" + std::to_string(chunkId) + " highAccumulator bootstrap");
        highAccumulator = BootstrapProjectActive256(cc, highAccumulator, masks);
        public_reciprocal_locate::Depth("END chunk=" + std::to_string(chunkId) + " highAccumulator bootstrap", highAccumulator);
        public_reciprocal_locate::Mark("BEGIN chunk=" + std::to_string(chunkId) + " highAccumulator + chunkHigh");
        auto highPlusChunk = EvalAddKS(
            cc, highAccumulator, chunkHigh, directionSign,
            masks.multiplier.ksMasks, nullptr);
        public_reciprocal_locate::Depth("END chunk=" + std::to_string(chunkId) + " highPlusChunk before boot", highPlusChunk);
        highAccumulator = BootstrapProjectActive256(cc, highPlusChunk, masks);
        public_reciprocal_locate::Depth("END chunk=" + std::to_string(chunkId) + " highAccumulator after add boot", highAccumulator);

        public_reciprocal_locate::Mark("BEGIN chunk=" + std::to_string(chunkId) + " reset pipelines");
        lowPipeline.reset();
        highPipeline.reset();
        public_reciprocal_locate::Sync(cc, "chunk=" + std::to_string(chunkId) + " after reset pipelines");
        public_reciprocal_locate::Mark("END finalize chunk=" + std::to_string(chunkId));
    };

    auto shiftedLeft = cN;
    uint32_t nextPositionIndex = 0;
    for (uint32_t j = 0; j < kWordBits; ++j) {
        if (nextPositionIndex < positions.size() && positions[nextPositionIndex] == j) {
            public_reciprocal_locate::Mark("BEGIN selected row j=" + std::to_string(j) + " lowRow EvalMult");
            auto lowRow = cc->EvalMult(shiftedLeft, masks.multiplier.ksMasks.activeMask);
            if (timing) {
                ++timing->publicRowMaskMults;
            }
            // Drain the row mask multiplication before shiftedLeft can be overwritten.
            // This avoids keeping an extra vector of row/input ciphertexts alive while
            // preserving the public-specialized row count.
            public_reciprocal_locate::Sync(cc, "selected row j=" + std::to_string(j) + " lowRow EvalMult");
            public_reciprocal_locate::Mark("BEGIN selected row j=" + std::to_string(j) + " lowPipeline Push");
            lowPipeline->Push(std::move(lowRow));
            public_reciprocal_locate::Mark("END selected row j=" + std::to_string(j) + " lowPipeline Push");

            if (j != 0 && highPipeline) {
                public_reciprocal_locate::Mark("BEGIN selected row j=" + std::to_string(j) + " highSource rotate256");
                auto highSource = cc->EvalRotate(
                    shiftedLeft, static_cast<int32_t>(kWordBits));
                if (timing) {
                    ++timing->publicRowRotations;
                    ++timing->mergedHighOverflowRotations;
                }
                public_reciprocal_locate::Sync(cc, "selected row j=" + std::to_string(j) + " highSource rotate256");
                public_reciprocal_locate::Mark("BEGIN selected row j=" + std::to_string(j) + " highRow EvalMult");
                auto highRow = cc->EvalMult(highSource, masks.multiplier.ksMasks.activeMask);
                if (timing) {
                    ++timing->publicRowMaskMults;
                }
                // Drain before highSource leaves scope; this replaces the large
                // per-chunk keepAlive vector used in v32.
                public_reciprocal_locate::Sync(cc, "selected row j=" + std::to_string(j) + " highRow EvalMult");
                public_reciprocal_locate::Mark("BEGIN selected row j=" + std::to_string(j) + " highPipeline Push");
                highPipeline->Push(std::move(highRow));
                public_reciprocal_locate::Mark("END selected row j=" + std::to_string(j) + " highPipeline Push");
            }

            ++rowsInChunk;
            ++totalProcessed;
            ++nextPositionIndex;

            if (rowsInChunk == currentChunkLowRows) {
                finalizeCurrentChunk();
                if (totalProcessed < lowRows) {
                    startNextChunk();
                }
            }
        }

        if (j + 1 < kWordBits) {
            public_reciprocal_locate::Mark("BEGIN shift stream rotate j=" + std::to_string(j) + " -> " + std::to_string(j + 1));
            shiftedLeft = cc->EvalRotate(shiftedLeft, directionSign);
            public_reciprocal_locate::Sync(cc, "shift stream rotate j=" + std::to_string(j) + " -> " + std::to_string(j + 1));
            if (timing) {
                ++timing->publicRowRotations;
            }
        }
    }

    public_reciprocal_locate::Mark("BEGIN final highAccumulator bootstrap before return");
    highAccumulator = BootstrapProjectActive256(cc, highAccumulator, masks);
    public_reciprocal_locate::Depth("END final highAccumulator bootstrap before return", highAccumulator);
    public_reciprocal_locate::Mark("BEGIN final carryCounter bootstrap before return");
    carryCounter = BootstrapProjectActive256(cc, carryCounter, masks);
    public_reciprocal_locate::Depth("END final carryCounter bootstrap before return", carryCounter);
    carryCounter = cc->EvalMult(carryCounter, masks.low8Mask);
    public_reciprocal_locate::Sync(cc, "final carryCounter low8 before return");

    public_reciprocal_locate::Mark("RETURN EvalPublicReciprocalHiLowMergedKS");
    return {std::move(highAccumulator), std::move(carryCounter)};
}

struct SubPlainConstResult {
    Ciphertext<DCRTPoly> difference;
    Ciphertext<DCRTPoly> borrowBitAtZero;
};

inline SubPlainConstResult EvalSubPlainConst256(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& a,
    Plaintext& constantMask,
    int directionSign,
    PublicReciprocalDivisionMasks& masks) {

    if (directionSign != -1) {
        throw std::invalid_argument("EvalSubPlainConst256 requires directionSign=-1");
    }

    auto ac = cc->EvalMult(a, constantMask);
    auto xorAB = cc->EvalSub(cc->EvalAdd(a, constantMask), cc->EvalAdd(ac, ac));
    auto G = cc->EvalSub(constantMask, ac);
    auto Pk = cc->EvalSub(masks.multiplier.ksMasks.activeMask, xorAB);

    for (uint32_t layer = 0; layer < kNumLayers; ++layer) {
        const int32_t step = static_cast<int32_t>(1U << layer);
        auto Gold = G;
        auto Pold = Pk;
        auto Grot = cc->EvalRotate(Gold, directionSign * step);
        auto Prot = cc->EvalRotate(Pold, directionSign * step);
        auto term = cc->EvalMult(Grot, Pold);
        if (layer + 1 < kNumLayers) {
            Pk = cc->EvalMult(Prot, Pold);
        }
        G = cc->EvalAdd(Gold, term);
    }

    auto borrowIn = cc->EvalRotate(G, directionSign);
    borrowIn = cc->EvalMult(borrowIn, masks.multiplier.ksMasks.carryMask);
    G = cc->EvalMult(G, masks.multiplier.ksMasks.activeMask);

    auto difference = cc->EvalSub(a, constantMask);
    difference = cc->EvalSub(difference, borrowIn);
    difference = cc->EvalAdd(difference, cc->EvalAdd(G, G));
    difference = cc->EvalMult(difference, masks.multiplier.ksMasks.activeMask);

    auto borrow = cc->EvalRotate(G, static_cast<int32_t>(kWordBits - 1));
    borrow = cc->EvalMult(borrow, masks.multiplier.bitZeroMask);
    public_reciprocal_detail::Synchronize(cc);
    return {std::move(difference), std::move(borrow)};
}

inline Ciphertext<DCRTPoly> EvalSubCipher256(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& a,
    const Ciphertext<DCRTPoly>& b,
    int directionSign,
    PublicReciprocalDivisionMasks& masks) {

    auto ab = cc->EvalMult(a, b);
    auto xorAB = cc->EvalSub(cc->EvalAdd(a, b), cc->EvalAdd(ab, ab));
    auto G = cc->EvalSub(b, ab);
    auto Pk = cc->EvalSub(masks.multiplier.ksMasks.activeMask, xorAB);

    for (uint32_t layer = 0; layer < kNumLayers; ++layer) {
        const int32_t step = static_cast<int32_t>(1U << layer);
        auto Gold = G;
        auto Pold = Pk;
        auto Grot = cc->EvalRotate(Gold, directionSign * step);
        auto Prot = cc->EvalRotate(Pold, directionSign * step);
        auto term = cc->EvalMult(Grot, Pold);
        if (layer + 1 < kNumLayers) {
            Pk = cc->EvalMult(Prot, Pold);
        }
        G = cc->EvalAdd(Gold, term);
    }

    auto borrowIn = cc->EvalRotate(G, directionSign);
    borrowIn = cc->EvalMult(borrowIn, masks.multiplier.ksMasks.carryMask);
    G = cc->EvalMult(G, masks.multiplier.ksMasks.activeMask);

    auto difference = cc->EvalSub(a, b);
    difference = cc->EvalSub(difference, borrowIn);
    difference = cc->EvalAdd(difference, cc->EvalAdd(G, G));
    difference = cc->EvalMult(difference, masks.multiplier.ksMasks.activeMask);
    public_reciprocal_detail::Synchronize(cc);
    return difference;
}


inline Ciphertext<DCRTPoly> EvalLowCarryCounterSequentialKS(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cN,
    const std::array<uint8_t, kWordBits>& constantBits,
    int directionSign,
    PublicReciprocalDivisionMasks& masks,
    PublicReciprocalTimingReport* timing) {

    auto zero = cc->EvalSub(cN, cN);
    zero = cc->EvalMult(zero, masks.multiplier.ksMasks.activeMask);
    auto lowAccumulator = zero;
    auto carryCounter = zero;

    auto shiftedLeft = cN;
    uint32_t pushed = 0;
    constexpr uint32_t kRefreshEveryRows = 8;

    for (uint32_t j = 0; j < kWordBits; ++j) {
        if (constantBits[j] != 0) {
            auto row = cc->EvalMult(shiftedLeft, masks.multiplier.ksMasks.activeMask);
            auto add = EvalAddKSWithCarryOut256(
                cc, lowAccumulator, row, directionSign, masks);
            lowAccumulator = std::move(add.sum);
            carryCounter = EvalAddKS(
                cc, carryCounter, add.carryOutBitAtZero, directionSign,
                masks.multiplier.ksMasks, nullptr);
            carryCounter = cc->EvalMult(carryCounter, masks.low8Mask);
            ++pushed;

            if ((pushed % kRefreshEveryRows) == 0U) {
                lowAccumulator = BootstrapProjectActive256(cc, lowAccumulator, masks);
                carryCounter = BootstrapProjectActive256(cc, carryCounter, masks);
                carryCounter = cc->EvalMult(carryCounter, masks.low8Mask);
                public_reciprocal_detail::Synchronize(cc);
            }
        }
        if (j + 1 < kWordBits) {
            shiftedLeft = cc->EvalRotate(shiftedLeft, directionSign);
        }
    }

    carryCounter = cc->EvalMult(carryCounter, masks.low8Mask);
    public_reciprocal_detail::Synchronize(cc);
    return carryCounter;
}


inline Ciphertext<DCRTPoly> PadCiphertextToLevelWithMask(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> value,
    uint32_t targetLevel,
    Plaintext& activeMask,
    const char* label) {

    if (!cc || !value) {
        throw std::invalid_argument("PadCiphertextToLevelWithMask received null input");
    }
    const uint32_t startLevel = value->GetLevel();
    if (startLevel > targetLevel) {
        std::ostringstream oss;
        oss << "cannot pad ciphertext level down for " << label
            << ": startLevel=" << startLevel
            << " targetLevel=" << targetLevel;
        throw std::runtime_error(oss.str());
    }
    if (startLevel == targetLevel) {
        return value;
    }

    while (value->GetLevel() < targetLevel) {
        value = cc->EvalMult(value, activeMask);
        public_reciprocal_detail::Synchronize(cc);
        if (value->GetLevel() > targetLevel) {
            std::ostringstream oss;
            oss << "level padding overshot for " << label
                << ": now=" << value->GetLevel()
                << " target=" << targetLevel;
            throw std::runtime_error(oss.str());
        }
    }
    return value;
}

inline void AlignBinaryCiphertextsToSameLevelByMask(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly>& a,
    Ciphertext<DCRTPoly>& b,
    Plaintext& activeMask,
    const char* label) {

    if (!cc || !a || !b) {
        throw std::invalid_argument("AlignBinaryCiphertextsToSameLevelByMask received null input");
    }
    const uint32_t levelA = a->GetLevel();
    const uint32_t levelB = b->GetLevel();
    if (levelA == levelB) {
        return;
    }
    const uint32_t target = std::max(levelA, levelB);
    if (levelA < target) {
        a = PadCiphertextToLevelWithMask(cc, a, target, activeMask, label);
    }
    if (levelB < target) {
        b = PadCiphertextToLevelWithMask(cc, b, target, activeMask, label);
    }
    public_reciprocal_detail::Synchronize(cc);
    if (a->GetLevel() != b->GetLevel()) {
        std::ostringstream oss;
        oss << "failed to align levels for " << label
            << ": A=" << a->GetLevel()
            << " B=" << b->GetLevel();
        throw std::runtime_error(oss.str());
    }
}


// -----------------------------------------------------------------------------
// v38 high-only reciprocal + bounded correction experiment.
//
// This path deliberately does not compute the low-half carryDown of N*M.
// It computes qBase = sum high256(N << j) over public one-bits of M, then
// corrects the resulting under-estimate by a public bounded correction loop.
// The bound used here is popcount(M)+2, which safely covers the omitted low
// carry plus the usual reciprocal off-by-one correction.
// -----------------------------------------------------------------------------
namespace public_reciprocal_highonly_detail {

inline bool ShiftPublicBitsLeftNoOverflow(
    const std::array<uint8_t, kWordBits>& in,
    uint32_t shift,
    std::array<uint8_t, kWordBits>* out) {
    if (!out) {
        throw std::invalid_argument("ShiftPublicBitsLeftNoOverflow received null output");
    }
    out->fill(0);
    for (uint32_t i = 0; i < kWordBits; ++i) {
        if (in[i] == 0) {
            continue;
        }
        if (i + shift >= kWordBits) {
            return false;
        }
        (*out)[i + shift] = 1;
    }
    return true;
}

inline Plaintext BuildAndLoadPublicConstantMask(
    const CryptoContext<DCRTPoly>& cc,
    const std::array<uint8_t, kWordBits>& bits) {
    auto mask = cc->MakeCKKSPackedPlaintext(
        public_reciprocal_masks_detail::MakeConstantMaskValues(bits),
        1, 0, nullptr, kMultiplierTotalSlots);
    mask->SetLength(kMultiplierTotalSlots);
    cc->LoadPlaintext(mask);
    public_reciprocal_detail::Synchronize(cc);
    return mask;
}

inline uint32_t HighestPowerOfTwoAtMost(uint32_t value) {
    if (value == 0) {
        return 0;
    }
    uint32_t p = 1;
    while (p <= (value >> 1U) && p <= (1U << 30U)) {
        p <<= 1U;
    }
    return p;
}

inline uint32_t Log2PowerOfTwo(uint32_t value) {
    uint32_t s = 0;
    while ((1U << s) < value) {
        ++s;
    }
    return s;
}


inline Ciphertext<DCRTPoly> RotateByUnitSteps(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> input,
    uint32_t distance,
    int directionSign,
    PublicReciprocalTimingReport* timing,
    const char* labelPrefix) {

    if (!cc || !input) {
        throw std::invalid_argument("RotateByUnitSteps received null input");
    }
    if (directionSign != -1) {
        throw std::invalid_argument("RotateByUnitSteps currently requires directionSign=-1");
    }

    for (uint32_t i = 0; i < distance; ++i) {
        input = cc->EvalRotate(input, directionSign);
        if (timing) {
            ++timing->publicRowRotations;
        }
        if (public_reciprocal_locate::Enabled()) {
            std::string label = std::string(labelPrefix ? labelPrefix : "unit rotate")
                + " step=" + std::to_string(i + 1U)
                + "/" + std::to_string(distance);
            public_reciprocal_locate::Sync(cc, label);
        }
    }
    return input;
}

inline Ciphertext<DCRTPoly> MakeConditionalPublicConstantFromBit0(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& flagBit0,
    const std::array<uint8_t, kWordBits>& constantBits,
    int directionSign,
    PublicReciprocalDivisionMasks& masks) {

    if (!cc || !flagBit0) {
        throw std::invalid_argument("MakeConditionalPublicConstantFromBit0 received null input");
    }
    if (directionSign != -1) {
        throw std::invalid_argument("MakeConditionalPublicConstantFromBit0 requires directionSign=-1");
    }

    Ciphertext<DCRTPoly> out;
    bool any = false;
    for (uint32_t bit = 0; bit < kWordBits; ++bit) {
        if (constantBits[bit] == 0) {
            continue;
        }
        Ciphertext<DCRTPoly> term;
        if (bit == 0) {
            term = flagBit0;
        }
        else {
            term = RotateByUnitSteps(
                cc, flagBit0, bit, directionSign, nullptr,
                "conditional public constant unit rotate");
        }
        out = any ? cc->EvalAdd(out, term) : term;
        any = true;
    }

    if (!any) {
        out = cc->EvalSub(flagBit0, flagBit0);
    }
    out = cc->EvalMult(out, masks.multiplier.ksMasks.activeMask);
    public_reciprocal_detail::Synchronize(cc);
    return BootstrapProjectActive256(cc, out, masks);
}

}  // namespace public_reciprocal_highonly_detail

inline Ciphertext<DCRTPoly> EvalPublicReciprocalHighOnlyChunkedKS(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cN,
    const std::array<uint8_t, kWordBits>& reciprocalBits,
    int directionSign,
    PublicReciprocalDivisionMasks& masks,
    PublicReciprocalTimingReport* timing,
    uint32_t compressorBatchSize) {

    if (!cc || !cN) {
        throw std::invalid_argument("EvalPublicReciprocalHighOnlyChunkedKS received null input");
    }
    if (directionSign != -1) {
        throw std::invalid_argument("EvalPublicReciprocalHighOnlyChunkedKS requires directionSign=-1");
    }

    std::vector<uint32_t> positions;
    positions.reserve(kWordBits);
    for (uint32_t j = 1; j < kWordBits; ++j) {
        if (reciprocalBits[j] != 0) {
            positions.push_back(j);
        }
    }

    if (timing) {
        timing->mergedHighRows = static_cast<uint32_t>(positions.size());
        timing->mergedLowRows = 0;
        timing->pushedRows += static_cast<uint64_t>(positions.size());
    }

    auto zero = public_reciprocal_chunked_detail::MakeZeroActive256(cc, cN, masks);
    auto highAccumulator = zero;
    if (positions.empty()) {
        return highAccumulator;
    }

    const uint32_t chunkLimit = public_reciprocal_chunked_detail::RuntimeReciprocalChunkRows();
    const uint32_t totalRows = static_cast<uint32_t>(positions.size());
    uint32_t processed = 0;
    uint32_t rowsInChunk = 0;
    uint32_t currentChunkRows = std::min(chunkLimit, totalRows);

    std::unique_ptr<public_reciprocal_detail::DynamicDaddaPipeline> highPipeline(
        new public_reciprocal_detail::DynamicDaddaPipeline(
            cc, directionSign, currentChunkRows, compressorBatchSize,
            timing ? &timing->multiplier : nullptr));

    auto startNextChunk = [&]() {
        rowsInChunk = 0;
        const uint32_t remaining = totalRows - processed;
        currentChunkRows = std::min(chunkLimit, remaining);
        highPipeline.reset(new public_reciprocal_detail::DynamicDaddaPipeline(
            cc, directionSign, currentChunkRows, compressorBatchSize,
            timing ? &timing->multiplier : nullptr));
    };

    auto finalizeCurrentChunk = [&]() {
        auto rowsOut = highPipeline->Finish();
        auto chunkHigh = public_reciprocal_chunked_detail::FinishDaddaRowsActive256(
            cc, std::move(rowsOut), directionSign, masks, timing);

        highAccumulator = BootstrapProjectActive256(cc, highAccumulator, masks);
        AlignBinaryCiphertextsToSameLevelByMask(
            cc, highAccumulator, chunkHigh,
            masks.multiplier.ksMasks.activeMask,
            "high-only reciprocal highAccumulator + chunkHigh");
        auto highPlusChunk = EvalAddKS(
            cc, highAccumulator, chunkHigh, directionSign,
            masks.multiplier.ksMasks, nullptr);
        public_reciprocal_detail::Synchronize(cc);
        highAccumulator = BootstrapProjectActive256(cc, highPlusChunk, masks);
        highPipeline.reset();
    };

    auto shiftedLeft = cN;
    uint32_t currentShift = 0;
    for (uint32_t idx = 0; idx < totalRows; ++idx) {
        const uint32_t j = positions[idx];
        if (j < currentShift) {
            throw std::runtime_error("high-only reciprocal positions are not sorted");
        }
        if (j > currentShift) {
            shiftedLeft = public_reciprocal_highonly_detail::RotateByUnitSteps(
                cc, shiftedLeft, j - currentShift, directionSign, timing,
                "high-only selected-position unit rotate");
            currentShift = j;
        }

        auto highSource = cc->EvalRotate(shiftedLeft, static_cast<int32_t>(kWordBits));
        if (timing) {
            ++timing->publicRowRotations;
            ++timing->mergedHighOverflowRotations;
        }
        public_reciprocal_locate::Sync(cc, "high-only highSource rotate256 j=" + std::to_string(j));
        auto highRow = cc->EvalMult(highSource, masks.multiplier.ksMasks.activeMask);
        if (timing) {
            ++timing->publicRowMaskMults;
        }
        public_reciprocal_locate::Sync(cc, "high-only highRow EvalMult j=" + std::to_string(j));
        highPipeline->Push(std::move(highRow));

        ++rowsInChunk;
        ++processed;
        if (rowsInChunk == currentChunkRows) {
            finalizeCurrentChunk();
            if (processed < totalRows) {
                startNextChunk();
            }
        }
    }

    shiftedLeft = nullptr;
    public_reciprocal_detail::Synchronize(cc);
    return BootstrapProjectActive256(cc, highAccumulator, masks);
}

inline void ApplyHighOnlyBoundedCorrectionKS(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly>& quotient,
    Ciphertext<DCRTPoly>& remainder,
    const std::array<uint8_t, kWordBits>& divisorBits,
    uint32_t correctionBound,
    int directionSign,
    PublicReciprocalDivisionMasks& masks,
    PublicReciprocalTimingReport& report) {

    if (!cc || !quotient || !remainder) {
        throw std::invalid_argument("ApplyHighOnlyBoundedCorrectionKS received null input");
    }
    if (directionSign != -1) {
        throw std::invalid_argument("ApplyHighOnlyBoundedCorrectionKS requires directionSign=-1");
    }

    if (correctionBound == 0) {
        correctionBound = 1;
    }
    uint32_t topPower = public_reciprocal_highonly_detail::HighestPowerOfTwoAtMost(correctionBound);
    if (topPower == 0) {
        topPower = 1;
    }

    for (uint32_t step = topPower; step != 0; step >>= 1U) {
        const uint32_t shift = public_reciprocal_highonly_detail::Log2PowerOfTwo(step);
        std::array<uint8_t, kWordBits> stepDivisorBits{};
        const bool fits = public_reciprocal_highonly_detail::ShiftPublicBitsLeftNoOverflow(
            divisorBits, shift, &stepDivisorBits);
        if (!fits) {
            continue;
        }

        const auto cmpBegin = public_reciprocal_detail::Clock::now();
        auto stepDivisorMask = public_reciprocal_highonly_detail::BuildAndLoadPublicConstantMask(
            cc, stepDivisorBits);
        auto cmp = EvalSubPlainConst256(
            cc, remainder, stepDivisorMask, directionSign, masks);
        auto flag = cc->EvalSub(masks.multiplier.bitZeroMask, cmp.borrowBitAtZero);
        cmp.difference = nullptr;
        cmp.borrowBitAtZero = nullptr;
        flag = cc->EvalMult(flag, masks.multiplier.bitZeroMask);
        public_reciprocal_detail::Synchronize(cc);
        flag = BootstrapProjectActive256(cc, flag, masks);
        flag = cc->EvalMult(flag, masks.multiplier.bitZeroMask);
        public_reciprocal_detail::Synchronize(cc);
        report.correctionCompareMs += public_reciprocal_detail::Milliseconds(
            cmpBegin, public_reciprocal_detail::Clock::now());

        const auto updateBegin = public_reciprocal_detail::Clock::now();
        Ciphertext<DCRTPoly> increment;
        if (shift == 0) {
            increment = flag;
        }
        else {
            increment = public_reciprocal_highonly_detail::RotateByUnitSteps(
                cc, flag, shift, directionSign, nullptr,
                "correction increment unit rotate");
        }
        increment = cc->EvalMult(increment, masks.multiplier.ksMasks.activeMask);
        public_reciprocal_detail::Synchronize(cc);
        increment = BootstrapProjectActive256(cc, increment, masks);

        AlignBinaryCiphertextsToSameLevelByMask(
            cc, quotient, increment, masks.multiplier.ksMasks.activeMask,
            "high-only correction quotient + increment");
        quotient = EvalAddKS(
            cc, quotient, increment, directionSign,
            masks.multiplier.ksMasks, nullptr);
        public_reciprocal_detail::Synchronize(cc);
        quotient = BootstrapProjectActive256(cc, quotient, masks);

        auto conditionalDivisor = public_reciprocal_highonly_detail::MakeConditionalPublicConstantFromBit0(
            cc, flag, stepDivisorBits, directionSign, masks);
        AlignBinaryCiphertextsToSameLevelByMask(
            cc, remainder, conditionalDivisor, masks.multiplier.ksMasks.activeMask,
            "high-only correction remainder - conditionalDivisor");
        remainder = EvalSubCipher256(cc, remainder, conditionalDivisor, directionSign, masks);
        public_reciprocal_detail::Synchronize(cc);
        remainder = BootstrapProjectActive256(cc, remainder, masks);

        report.finalIncrementMs += public_reciprocal_detail::Milliseconds(
            updateBegin, public_reciprocal_detail::Clock::now());
    }
}


inline Ciphertext<DCRTPoly> EvalPublicReciprocalQuotientKS(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cN,
    const std::array<uint8_t, kWordBits>& reciprocalBits,
    const std::array<uint8_t, kWordBits>& divisorBits,
    int directionSign,
    PublicReciprocalDivisionMasks& masks,
    PublicReciprocalTimingReport* timing = nullptr,
    uint32_t compressorBatchSize = kDefaultCompressorBatchSize) {

    PublicReciprocalTimingReport local;
    PublicReciprocalTimingReport& report = timing ? *timing : local;
    report = PublicReciprocalTimingReport{};
    report.reciprocalOneBits = public_reciprocal_detail::CountOneBits(reciprocalBits);
    report.reciprocalZeroBits = kWordBits - report.reciprocalOneBits;
    report.divisorOneBits = public_reciprocal_detail::CountOneBits(divisorBits);

    const auto totalBegin = public_reciprocal_detail::Clock::now();

    const auto highBegin = public_reciprocal_detail::Clock::now();
    auto qBase = EvalPublicReciprocalHighOnlyChunkedKS(
        cc, cN, reciprocalBits, directionSign, masks, &report,
        compressorBatchSize);
    report.rowGenerationMs += public_reciprocal_detail::Milliseconds(
        highBegin, public_reciprocal_detail::Clock::now());

    // EvalPublicReciprocalHighOnlyChunkedKS already returns a bootstrapped/projected qBase.
    report.q0BootstrapMs = 0.0;

    const auto mulDBegin = public_reciprocal_detail::Clock::now();
    auto qBaseD = EvalPublicConstMulLowKS(
        cc, qBase, divisorBits, directionSign, masks, &report,
        compressorBatchSize);
    report.mulDivisorMs = public_reciprocal_detail::Milliseconds(
        mulDBegin, public_reciprocal_detail::Clock::now());
    report.tBootstrapMs = 0.0;

    const auto subBegin = public_reciprocal_detail::Clock::now();
    auto cNForSub = BootstrapProjectActive256(cc, cN, masks);
    AlignBinaryCiphertextsToSameLevelByMask(
        cc, cNForSub, qBaseD, masks.multiplier.ksMasks.activeMask,
        "high-only initial r = N - qBaseD");
    auto remainder = EvalSubCipher256(cc, cNForSub, qBaseD, directionSign, masks);
    cNForSub = nullptr;
    qBaseD = nullptr;
    public_reciprocal_detail::Synchronize(cc);
    report.subtractMs = public_reciprocal_detail::Milliseconds(
        subBegin, public_reciprocal_detail::Clock::now());

    const auto bootRBegin = public_reciprocal_detail::Clock::now();
    remainder = BootstrapProjectActive256(cc, remainder, masks);
    report.rBootstrapMs = public_reciprocal_detail::Milliseconds(
        bootRBegin, public_reciprocal_detail::Clock::now());

    const auto corrBegin = public_reciprocal_detail::Clock::now();
    const uint32_t correctionBound = report.reciprocalOneBits + 2U;
    ApplyHighOnlyBoundedCorrectionKS(
        cc, qBase, remainder, divisorBits, correctionBound,
        directionSign, masks, report);
    report.correctionBootstrapMs += public_reciprocal_detail::Milliseconds(
        corrBegin, public_reciprocal_detail::Clock::now());

    // The last correction round already bootstraps/projects quotient.  The caller
    // still has an optional final bootstrap stage for reporting/decryption.
    report.finalBootstrapMs = 0.0;

    report.totalMs = public_reciprocal_detail::Milliseconds(
        totalBegin, public_reciprocal_detail::Clock::now());
    return qBase;
}

}  // namespace ks256gpu

