#pragma once

#include <fideslib.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ks256gpu {

using fideslib::Ciphertext;
using fideslib::CryptoContext;
using fideslib::DCRTPoly;
using fideslib::Plaintext;

// -----------------------------------------------------------------------------
// Standalone 256-bit Boolean/SIMD layout used only by the multiplier target.
// This header is self-contained and used only by the multiplier target.
// -----------------------------------------------------------------------------
constexpr uint32_t kWordBits  = 256;
constexpr uint32_t kGuardBits = 256;
constexpr uint32_t kStride    = kWordBits + kGuardBits;  // 512 slots/word
constexpr uint32_t kNumLayers = 8;                       // log2(256)

struct LayerTiming {
    double rotateGMs    = 0.0;
    double rotatePMs    = 0.0;
    double termMultMs   = 0.0;
    double prefixMultMs = 0.0;
    double updateGMs    = 0.0;
    double wallMs       = 0.0;
};

struct TimingReport {
    double preprocessMs  = 0.0;
    std::array<LayerTiming, kNumLayers> layers{};
    double carryRotateMs = 0.0;
    double carryMaskMs   = 0.0;
    double groupMaskMs   = 0.0;
    double reconstructMs = 0.0;
    double totalMs       = 0.0;
};

struct Masks {
    Plaintext activeMask;  // 1 on data slots, 0 on guard slots
    Plaintext carryMask;   // also clears the carry-in position of each word
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
inline auto TimedGpuCall(const CryptoContext<DCRTPoly>& cc,
                         double* elapsedMs,
                         Fn&& fn) {
    Synchronize(cc);
    const auto begin = Clock::now();
    auto result = fn();
    Synchronize(cc);
    *elapsedMs = Milliseconds(begin, Clock::now());
    return result;
}

}  // namespace detail

/**
 * Standalone final 256-bit Kogge-Stone carry-propagate adder used by the
 * multiplier.  Both inputs and the returned result use the multiplier's
 * bit-per-slot layout.
 */
inline Ciphertext<DCRTPoly> EvalAddKS(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    int directionSign,
    Masks& masks,
    TimingReport* timing = nullptr) {

    if (!cc || !cA || !cB) {
        throw std::invalid_argument(
            "EvalAddKS received a null context or ciphertext");
    }
    if (!masks.activeMask || !masks.carryMask) {
        throw std::invalid_argument("EvalAddKS received invalid masks");
    }
    if (directionSign != -1 && directionSign != 1) {
        throw std::invalid_argument("directionSign must be -1 or +1");
    }

    TimingReport local;
    TimingReport& report = timing ? *timing : local;
    report = TimingReport{};

    detail::Synchronize(cc);
    const auto totalBegin = detail::Clock::now();

    // g = A AND B, p = A XOR B.
    detail::Synchronize(cc);
    auto begin = detail::Clock::now();
    auto g = cc->EvalMult(cA, cB);
    auto p = cc->EvalSub(cc->EvalAdd(cA, cB), cc->EvalAdd(g, g));
    detail::Synchronize(cc);
    report.preprocessMs = detail::Milliseconds(begin, detail::Clock::now());

    auto G  = g;
    auto Pk = p;

    // Eight prefix stages with distances 1, 2, ..., 128.
    for (uint32_t layer = 0; layer < kNumLayers; ++layer) {
        detail::Synchronize(cc);
        const auto layerBegin = detail::Clock::now();

        const int32_t step = static_cast<int32_t>(1U << layer);
        const int32_t rotation = directionSign * step;

        const auto Gold = G;
        const auto Pold = Pk;

        auto Grot = detail::TimedGpuCall(
            cc, &report.layers[layer].rotateGMs,
            [&]() { return cc->EvalRotate(Gold, rotation); });

        auto Prot = detail::TimedGpuCall(
            cc, &report.layers[layer].rotatePMs,
            [&]() { return cc->EvalRotate(Pold, rotation); });

        auto term = detail::TimedGpuCall(
            cc, &report.layers[layer].termMultMs,
            [&]() { return cc->EvalMult(Grot, Pold); });

        if (layer + 1 < kNumLayers) {
            Pk = detail::TimedGpuCall(
                cc, &report.layers[layer].prefixMultMs,
                [&]() { return cc->EvalMult(Prot, Pold); });
        }

        G = detail::TimedGpuCall(
            cc, &report.layers[layer].updateGMs,
            [&]() { return cc->EvalAdd(Gold, term); });

        detail::Synchronize(cc);
        report.layers[layer].wallMs =
            detail::Milliseconds(layerBegin, detail::Clock::now());
    }

    // G[i] is c_{i+1}; shift it once to obtain C[i] = c_i.
    auto C = detail::TimedGpuCall(
        cc, &report.carryRotateMs,
        [&]() { return cc->EvalRotate(G, directionSign); });

    C = detail::TimedGpuCall(
        cc, &report.carryMaskMs,
        [&]() { return cc->EvalMult(C, masks.carryMask); });

    G = detail::TimedGpuCall(
        cc, &report.groupMaskMs,
        [&]() { return cc->EvalMult(G, masks.activeMask); });

    // s_i = a_i + b_i + c_i - 2c_{i+1}.
    detail::Synchronize(cc);
    begin = detail::Clock::now();
    auto result = cc->EvalAdd(cA, cB);
    result = cc->EvalAdd(result, C);
    result = cc->EvalSub(result, cc->EvalAdd(G, G));
    detail::Synchronize(cc);
    report.reconstructMs = detail::Milliseconds(begin, detail::Clock::now());

    report.totalMs = detail::Milliseconds(totalBegin, detail::Clock::now());
    return result;
}

// Dadda target heights for an initial maximum height of 256.
constexpr uint32_t kNumDaddaStages = 13;
constexpr std::array<uint32_t, kNumDaddaStages> kDaddaTargets = {
    211, 141, 94, 63, 42, 28, 19, 13, 9, 6, 4, 3, 2};

// Multiplier-specific SIMD layout: one ciphertext contains
// 128 independent 256-bit words and their guard regions.
constexpr uint32_t kMultiplierNumWords   = 128;
constexpr uint32_t kMultiplierTotalSlots =
    kMultiplierNumWords * kStride;  //  kMultiplierNumWords * 512 slots

static_assert(kMultiplierTotalSlots == 65536,
              "The 128-word multiplier layout must use 65536 slots");
static_assert((kMultiplierTotalSlots & (kMultiplierTotalSlots - 1)) == 0,
              "The multiplier slot count must be a power of two");

// Periodically synchronize after this many 3:2 compressors have been
// submitted. A small value bounds outstanding GPU work and temporary memory.
// The low-memory validation build uses 1 so every compressor is synchronized before more work is submitted. Increase only after correctness and memory are stable.
constexpr uint32_t kDefaultCompressorBatchSize = 16;

struct DaddaStageTiming {
    uint32_t inputRows       = 0;
    uint32_t targetRows      = 0;
    uint32_t fullAdders      = 0;
    uint32_t passthroughRows = 0;
    double wallMs            = 0.0;
};

struct MultiplierTimingReport {
    double partialProductsMs = 0.0;
    std::array<DaddaStageTiming, kNumDaddaStages> daddaStages{};
    TimingReport finalAdder{};
    double outputMaskMs = 0.0;
    double totalMs      = 0.0;

    uint64_t inputAlignmentRotations  = 0;
    uint64_t broadcastRotations       = 0;
    uint64_t selectorPlainMults       = 0;
    uint64_t partialCipherMults       = 0;
    uint64_t compressorCipherMults    = 0;
    uint64_t compressorCarryRotations = 0;
    uint64_t streamingSynchronizations = 0;
    uint32_t peakBufferedRows          = 0;
};

struct MultiplierMasks {
    Plaintext bitZeroMask;
    Masks ksMasks;
};

namespace multiplier_detail {

using Clock = std::chrono::steady_clock;

inline double Milliseconds(const Clock::time_point& begin,
                           const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

inline std::vector<double> MakeBitZeroMaskValues() {
    std::vector<double> mask(kMultiplierTotalSlots, 0.0);
    for (uint32_t word = 0; word < kMultiplierNumWords; ++word) {
        mask[word * kStride] = 1.0;
    }
    return mask;
}

inline std::vector<double> MakeActiveMaskValues() {
    std::vector<double> mask(kMultiplierTotalSlots, 0.0);
    for (uint32_t word = 0; word < kMultiplierNumWords; ++word) {
        const uint32_t base = word * kStride;
        for (uint32_t bit = 0; bit < kWordBits; ++bit) {
            mask[base + bit] = 1.0;
        }
    }
    return mask;
}

inline std::vector<double> MakeCarryMaskValues(int directionSign) {
    if (directionSign != -1 && directionSign != 1) {
        throw std::invalid_argument("directionSign must be -1 or +1");
    }

    std::vector<double> mask(kMultiplierTotalSlots, 0.0);
    for (uint32_t word = 0; word < kMultiplierNumWords; ++word) {
        const uint32_t base = word * kStride;
        if (directionSign == -1) {
            for (uint32_t bit = 1; bit < kWordBits; ++bit) {
                mask[base + bit] = 1.0;
            }
        }
        else {
            for (uint32_t bit = 0; bit + 1 < kWordBits; ++bit) {
                mask[base + bit] = 1.0;
            }
        }
    }
    return mask;
}

struct CompressorRows {
    Ciphertext<DCRTPoly> sum;
    Ciphertext<DCRTPoly> alignedCarry;
};

/**
 * Boolean 3:2 compressor with two ciphertext-multiplication levels.
 * The carry is immediately aligned to the next more-significant bit.
 */
inline CompressorRows EvalCompressor3To2(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& x,
    const Ciphertext<DCRTPoly>& y,
    const Ciphertext<DCRTPoly>& z,
    int directionSign,
    MultiplierTimingReport* timing) {

    auto xy = cc->EvalMult(x, y);
    auto p = cc->EvalSub(cc->EvalAdd(x, y), cc->EvalAdd(xy, xy));

    auto pz = cc->EvalMult(p, z);
    auto sum = cc->EvalSub(cc->EvalAdd(p, z), cc->EvalAdd(pz, pz));
    auto carry = cc->EvalAdd(xy, pz);
    auto alignedCarry = cc->EvalRotate(carry, directionSign);

    if (timing) {
        timing->compressorCipherMults += 2;
        timing->compressorCarryRotations += 1;
    }

    return {std::move(sum), std::move(alignedCarry)};
}

/**
 * Produces one truncated partial-product row at a time.
 * No vector containing all 256 rows is ever materialized.
 */
class PartialProductGenerator {
public:
    PartialProductGenerator(const CryptoContext<DCRTPoly>& cc,
                            const Ciphertext<DCRTPoly>& cA,
                            const Ciphertext<DCRTPoly>& cB,
                            int directionSign,
                            MultiplierMasks& masks,
                            MultiplierTimingReport* timing)
        : cc_(cc),
          cAShifted_(cA),
          cBAligned_(cB),
          directionSign_(directionSign),
          masks_(masks),
          timing_(timing) {}

    Ciphertext<DCRTPoly> Next() {
        if (nextRow_ >= kWordBits) {
            throw std::out_of_range(
                "PartialProductGenerator requested more than 256 rows");
        }

        // Flushing here also completes the previous row's streaming Dadda
        // cascade, bounding the amount of outstanding asynchronous GPU work.
        detail::Synchronize(cc_);
        const auto begin = Clock::now();

        auto selected = cc_->EvalMult(cBAligned_, masks_.bitZeroMask);

        auto broadcast = selected;
        for (uint32_t layer = 0; layer < kNumLayers; ++layer) {
            const int32_t step = static_cast<int32_t>(1U << layer);
            auto rotated = cc_->EvalRotate(
                broadcast, directionSign_ * step);
            broadcast = cc_->EvalAdd(broadcast, rotated);
        }

        // cAShifted_ = A << j and broadcast contains b_j in every active slot.
        auto partial = cc_->EvalMult(cAShifted_, broadcast);

        ++nextRow_;
        if (nextRow_ < kWordBits) {
            cAShifted_ = cc_->EvalRotate(cAShifted_, directionSign_);
            cBAligned_ = cc_->EvalRotate(cBAligned_, -directionSign_);
        }

        detail::Synchronize(cc_);
        if (timing_) {
            timing_->partialProductsMs +=
                Milliseconds(begin, Clock::now());
            timing_->selectorPlainMults += 1;
            timing_->partialCipherMults += 1;
            timing_->broadcastRotations += kNumLayers;
            if (nextRow_ < kWordBits) {
                timing_->inputAlignmentRotations += 2;
            }
        }

        return partial;
    }

    uint32_t RowsGenerated() const noexcept { return nextRow_; }

private:
    CryptoContext<DCRTPoly> cc_;
    Ciphertext<DCRTPoly> cAShifted_;
    Ciphertext<DCRTPoly> cBAligned_;
    int directionSign_;
    MultiplierMasks& masks_;
    MultiplierTimingReport* timing_;
    uint32_t nextRow_ = 0;
};

struct StreamingStageState {
    uint32_t inputRows       = 0;
    uint32_t targetRows      = 0;
    uint32_t fullAdders      = 0;
    uint32_t passthroughRows = 0;

    uint32_t receivedRows       = 0;
    uint32_t emittedRows        = 0;
    uint32_t completedFullAdders = 0;

    std::array<Ciphertext<DCRTPoly>, 3> pending{};
    uint32_t pendingCount = 0;
};

/**
 * Fully streaming Dadda pipeline.
 *
 * The old implementation stored 211 ciphertext rows after stage 0. Here each
 * stage stores at most two pending input rows. As soon as the third row arrives,
 * it is compressed and the two outputs are pushed directly into the next stage.
 * Peak retained rows are therefore O(number of stages), not O(211).
 */
class StreamingDaddaPipeline {
public:
    StreamingDaddaPipeline(const CryptoContext<DCRTPoly>& cc,
                           int directionSign,
                           uint32_t compressorBatchSize,
                           MultiplierTimingReport* timing)
        : cc_(cc),
          directionSign_(directionSign),
          compressorBatchSize_(compressorBatchSize == 0
                                   ? 1
                                   : compressorBatchSize),
          timing_(timing) {

        uint32_t inputRows = kWordBits;
        for (uint32_t stage = 0; stage < kNumDaddaStages; ++stage) {
            const uint32_t targetRows = kDaddaTargets[stage];
            if (targetRows >= inputRows ||
                targetRows < ((2U * inputRows + 2U) / 3U)) {
                throw std::invalid_argument(
                    "Invalid Dadda target sequence");
            }

            auto& state = stages_[stage];
            state.inputRows = inputRows;
            state.targetRows = targetRows;
            state.fullAdders = inputRows - targetRows;
            state.passthroughRows =
                3U * targetRows - 2U * inputRows;

            if (3U * state.fullAdders + state.passthroughRows !=
                    inputRows ||
                2U * state.fullAdders + state.passthroughRows !=
                    targetRows) {
                throw std::runtime_error(
                    "Dadda row-count identity failed");
            }

            if (timing_) {
                auto& item = timing_->daddaStages[stage];
                item.inputRows = inputRows;
                item.targetRows = targetRows;
                item.fullAdders = state.fullAdders;
                item.passthroughRows = state.passthroughRows;
            }
            inputRows = targetRows;
        }
    }

    void Push(Ciphertext<DCRTPoly> row) {
        if (!row) {
            throw std::invalid_argument(
                "StreamingDaddaPipeline received a null row");
        }
        PushToStage(0, std::move(row));
        UpdatePeakBufferedRows();
    }

    std::array<Ciphertext<DCRTPoly>, 2> Finish() {
        detail::Synchronize(cc_);
        if (timing_) {
            ++timing_->streamingSynchronizations;
        }

        for (uint32_t stage = 0; stage < kNumDaddaStages; ++stage) {
            const auto& state = stages_[stage];
            if (state.receivedRows != state.inputRows ||
                state.emittedRows != state.targetRows ||
                state.completedFullAdders != state.fullAdders ||
                state.pendingCount != 0) {
                throw std::runtime_error(
                    "Streaming Dadda stage finished with an invalid state");
            }
        }

        if (finalCount_ != 2 || !finalRows_[0] || !finalRows_[1]) {
            throw std::runtime_error(
                "Streaming Dadda tree did not terminate with two rows");
        }

        return {std::move(finalRows_[0]), std::move(finalRows_[1])};
    }

private:
    void PushToStage(uint32_t stageIndex,
                     Ciphertext<DCRTPoly> row) {
        if (stageIndex == kNumDaddaStages) {
            if (finalCount_ >= finalRows_.size()) {
                throw std::runtime_error(
                    "Streaming Dadda emitted more than two final rows");
            }
            finalRows_[finalCount_++] = std::move(row);
            return;
        }

        auto& state = stages_[stageIndex];
        if (state.receivedRows >= state.inputRows) {
            throw std::runtime_error(
                "Streaming Dadda stage received too many rows");
        }

        const uint32_t compressedInputRows = 3U * state.fullAdders;
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
                directionSign_,
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

            PushToStage(stageIndex + 1,
                        std::move(compressed.sum));
            PushToStage(stageIndex + 1,
                        std::move(compressed.alignedCarry));
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
        detail::Synchronize(cc_);
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
    int directionSign_;
    uint32_t compressorBatchSize_;
    MultiplierTimingReport* timing_;

    std::array<StreamingStageState, kNumDaddaStages> stages_{};
    std::array<Ciphertext<DCRTPoly>, 2> finalRows_{};
    uint32_t finalCount_ = 0;
    uint32_t compressorsSinceSync_ = 0;
};

}  // namespace multiplier_detail

inline MultiplierMasks BuildAndLoadMultiplierMasks(
    const CryptoContext<DCRTPoly>& cc,
    int directionSign) {

    if (!cc) {
        throw std::invalid_argument(
            "BuildAndLoadMultiplierMasks received a null context");
    }
    if (directionSign != -1) {
        throw std::invalid_argument(
            "The multiplier currently requires directionSign = -1");
    }

    MultiplierMasks masks;

    // Build all 16384-slot masks locally for the multiplier layout.
    masks.ksMasks.activeMask = cc->MakeCKKSPackedPlaintext(
        multiplier_detail::MakeActiveMaskValues(),
        1, 0, nullptr, kMultiplierTotalSlots);
    masks.ksMasks.carryMask = cc->MakeCKKSPackedPlaintext(
        multiplier_detail::MakeCarryMaskValues(directionSign),
        1, 0, nullptr, kMultiplierTotalSlots);
    masks.bitZeroMask = cc->MakeCKKSPackedPlaintext(
        multiplier_detail::MakeBitZeroMaskValues(),
        1, 0, nullptr, kMultiplierTotalSlots);

    masks.ksMasks.activeMask->SetLength(kMultiplierTotalSlots);
    masks.ksMasks.carryMask->SetLength(kMultiplierTotalSlots);
    masks.bitZeroMask->SetLength(kMultiplierTotalSlots);

    cc->LoadPlaintext(masks.ksMasks.activeMask);
    cc->LoadPlaintext(masks.ksMasks.carryMask);
    cc->LoadPlaintext(masks.bitZeroMask);
    detail::Synchronize(cc);
    return masks;
}

/**
 * Evaluates kMultiplierNumWords independent low-256-bit products:
 *
 *     result = A * B mod 2^256.
 *
 * Only O(kNumDaddaStages) ciphertext rows are retained by the Dadda tree.
 */
inline Ciphertext<DCRTPoly> EvalMulDaddaKS(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    int directionSign,
    MultiplierMasks& masks,
    MultiplierTimingReport* timing = nullptr,
    uint32_t compressorBatchSize = kDefaultCompressorBatchSize) {

    if (!cc || !cA || !cB) {
        throw std::invalid_argument(
            "EvalMulDaddaKS received a null context or ciphertext");
    }
    if (!masks.bitZeroMask ||
        !masks.ksMasks.activeMask ||
        !masks.ksMasks.carryMask) {
        throw std::invalid_argument(
            "EvalMulDaddaKS received invalid masks");
    }
    if (directionSign != -1) {
        throw std::invalid_argument(
            "EvalMulDaddaKS currently requires directionSign = -1");
    }

    MultiplierTimingReport local;
    MultiplierTimingReport& report = timing ? *timing : local;
    report = MultiplierTimingReport{};

    detail::Synchronize(cc);
    const auto totalBegin = multiplier_detail::Clock::now();

    multiplier_detail::PartialProductGenerator generator(
        cc, cA, cB, directionSign, masks, &report);
    multiplier_detail::StreamingDaddaPipeline pipeline(
        cc, directionSign, compressorBatchSize, &report);

    for (uint32_t row = 0; row < kWordBits; ++row) {
        pipeline.Push(generator.Next());
    }

    if (generator.RowsGenerated() != kWordBits) {
        throw std::runtime_error(
            "Partial-product generator returned an invalid row count");
    }

    auto rows = pipeline.Finish();

    auto product = EvalAddKS(
        cc,
        rows[0],
        rows[1],
        directionSign,
        masks.ksMasks,
        &report.finalAdder);

    product = detail::TimedGpuCall(
        cc,
        &report.outputMaskMs,
        [&]() {
            return cc->EvalMult(product, masks.ksMasks.activeMask);
        });

    report.totalMs = multiplier_detail::Milliseconds(
        totalBegin, multiplier_detail::Clock::now());
    return product;
}

}  // namespace ks256gpu
