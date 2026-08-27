#pragma once

#include <fideslib.hpp>

#include "greaterthan_ks_gpu.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ks256gpu {

constexpr uint32_t kAuctionNumBids = kNumWords;       // 128 packed uint256 bids
constexpr uint32_t kAuctionReductionRounds = 7;       // log2(128)
constexpr bool kAuctionBootstrapEachRound = true;
constexpr bool kAuctionBootstrapConditionEachRound = true;
constexpr bool kAuctionFinalBootstrap = true;

static_assert(kAuctionNumBids == 128,
              "auction_max_128 expects exactly 128 packed bids");
static_assert((kAuctionNumBids & (kAuctionNumBids - 1)) == 0,
              "auction reduction requires a power-of-two number of bids");

struct AuctionRoundTiming {
    double rotateMs     = 0.0;
    double compareMs    = 0.0;
    // Includes EvalBootstrap + smoothstep Boolean projection + final mask.
    double conditionBootstrapMs = 0.0;
    double broadcastMs  = 0.0;
    double cmuxMs       = 0.0;
    double activeMaskMs = 0.0;
    // Includes EvalBootstrap + smoothstep Boolean projection + active mask.
    double bootstrapMs  = 0.0;
    double wallMs       = 0.0;
};

struct AuctionTimingReport {
    std::array<AuctionRoundTiming, kAuctionReductionRounds> rounds{};
    double rotateTotalMs     = 0.0;
    double compareTotalMs    = 0.0;
    double conditionBootstrapTotalMs = 0.0;
    double broadcastTotalMs  = 0.0;
    double cmuxTotalMs       = 0.0;
    double activeMaskTotalMs = 0.0;
    double bootstrapTotalMs  = 0.0;
    // Includes EvalBootstrap + smoothstep Boolean projection + active mask.
    double finalBootstrapMs  = 0.0;
    double finalActiveMaskMs = 0.0;
    double totalMs           = 0.0;
};

inline std::array<uint32_t, kAuctionReductionRounds>
AuctionReductionOffsets() {
    return {64, 32, 16, 8, 4, 2, 1};
}

inline Ciphertext<DCRTPoly> ProjectBooleanSmoothstep(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& value) {

    if (!cc || !value) {
        throw std::invalid_argument(
            "ProjectBooleanSmoothstep received a null input");
    }

    // Smoothstep Boolean projection:
    //     f(x) = 3x^2 - 2x^3
    // It fixes 0 and 1 and has zero derivative at both endpoints.  This is
    // needed because CKKS EvalBootstrap is an identity refresh, not a Boolean
    // functional bootstrap.  Without this projection, the CMUX output is only
    // approximately Boolean and later GT circuits can amplify the deviation.
    auto x2 = cc->EvalMult(value, value);
    auto x3 = cc->EvalMult(x2, value);
    auto threeX2 = cc->EvalAdd(x2, cc->EvalAdd(x2, x2));
    auto twoX3 = cc->EvalAdd(x3, x3);
    return cc->EvalSub(threeX2, twoX3);
}

inline Ciphertext<DCRTPoly> BootstrapProjectBit0(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& value,
    const Masks& masks) {

    if (!cc || !value || !masks.resultMask) {
        throw std::invalid_argument(
            "BootstrapProjectBit0 received a null input");
    }

    auto refreshed = cc->EvalBootstrap(value, 1, 0);
    detail::Synchronize(cc);
    auto projected = ProjectBooleanSmoothstep(cc, refreshed);

    // FIDES EvalMult(ct, pt) takes Plaintext& (non-const).  Since this
    // function receives const Masks&, copy the shared_ptr locally to obtain a
    // non-const Plaintext variable without mutating the underlying plaintext.
    Plaintext resultMask = masks.resultMask;
    auto masked = cc->EvalMult(projected, resultMask);
    detail::Synchronize(cc);
    return masked;
}

inline Ciphertext<DCRTPoly> BootstrapProjectActive(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& value,
    const Masks& masks) {

    if (!cc || !value || !masks.activeMask) {
        throw std::invalid_argument(
            "BootstrapProjectActive received a null input");
    }

    auto refreshed = cc->EvalBootstrap(value, 1, 0);
    detail::Synchronize(cc);
    auto projected = ProjectBooleanSmoothstep(cc, refreshed);

    // FIDES EvalMult(ct, pt) takes Plaintext& (non-const).  Since this
    // function receives const Masks&, copy the shared_ptr locally to obtain a
    // non-const Plaintext variable without mutating the underlying plaintext.
    Plaintext activeMask = masks.activeMask;
    auto masked = cc->EvalMult(projected, activeMask);
    detail::Synchronize(cc);
    return masked;
}

inline Ciphertext<DCRTPoly> BroadcastBit0ToActiveWord(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& bit0) {

    if (!cc || !bit0) {
        throw std::invalid_argument(
            "BroadcastBit0ToActiveWord received a null input");
    }

    // Input has one Boolean condition at base+0 of each 512-slot word.
    // Repeated negative rotations fill base+[0,255] while leaving the
    // 256 guard slots zero.  Required rotation keys: -1,-2,...,-128.
    auto result = bit0;
    for (uint32_t layer = 0; layer < kNumLayers; ++layer) {
        const int32_t rotation =
            -static_cast<int32_t>(1U << layer);
        const auto old = result;
        auto rotated = cc->EvalRotate(old, rotation);
        result = cc->EvalAdd(old, rotated);
    }
    return result;
}

inline Ciphertext<DCRTPoly> EvalAuctionCmux(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& conditionMask,
    const Ciphertext<DCRTPoly>& trueValue,
    const Ciphertext<DCRTPoly>& falseValue) {

    if (!cc || !conditionMask || !trueValue || !falseValue) {
        throw std::invalid_argument("EvalAuctionCmux received a null input");
    }

    // Boolean CMUX in arithmetic form:
    //   cond ? trueValue : falseValue
    // = falseValue + cond * (trueValue - falseValue).
    auto difference = cc->EvalSub(trueValue, falseValue);
    auto selectedDifference = cc->EvalMult(conditionMask, difference);
    return cc->EvalAdd(falseValue, selectedDifference);
}

/**
 * Finds the maximum of 128 encrypted uint256 bids packed in one CKKS ciphertext.
 *
 * Layout:
 *   [256 Boolean data slots][256 guard slots] x 128.
 *
 * The reduction is a SIMD butterfly/tree reduction over packed words:
 *   offset = 64,32,16,8,4,2,1
 *   shifted = Rotate(current, offset * 512)
 *   cond    = GT(shifted, current)
 *   cond    = Bootstrap + smoothstep projection + resultMask
 *   current = cond ? shifted : current
 *   current = Bootstrap + smoothstep projection + activeMask
 *
 * EvalBootstrap only refreshes CKKS noise.  It does not turn approximate
 * arithmetic values into Boolean 0/1 values.  The smoothstep projection is
 * therefore required before a value is reused as input to GT or as a CMUX
 * selector.
 *
 * Required rotation keys:
 *   +1,+2,+4,+8,+16,+32,+64,+128       for GT
 *   -1,-2,-4,-8,-16,-32,-64,-128      for condition broadcast
 *   +512,+1024,+2048,+4096,+8192,+16384,+32768 for word reduction
 */
inline Ciphertext<DCRTPoly> EvalAuctionMax128(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& packedBids,
    Masks& masks,
    AuctionTimingReport* timing = nullptr,
    bool bootstrapEachRound = kAuctionBootstrapEachRound,
    bool bootstrapConditionEachRound = kAuctionBootstrapConditionEachRound,
    bool finalBootstrap = kAuctionFinalBootstrap) {

    if (!cc || !packedBids) {
        throw std::invalid_argument(
            "EvalAuctionMax128 received a null context or ciphertext");
    }
    if (!masks.activeMask || !masks.resultMask) {
        throw std::invalid_argument("EvalAuctionMax128 received invalid masks");
    }

    AuctionTimingReport local;
    AuctionTimingReport& report = timing ? *timing : local;
    report = AuctionTimingReport{};

    detail::Synchronize(cc);
    const auto totalBegin = detail::Clock::now();

    auto current = packedBids;
    const auto offsets = AuctionReductionOffsets();

    for (uint32_t round = 0; round < kAuctionReductionRounds; ++round) {
        detail::Synchronize(cc);
        const auto roundBegin = detail::Clock::now();

        const uint32_t offsetWords = offsets[round];
        const int32_t rotation =
            static_cast<int32_t>(offsetWords * kStride);

        auto shifted = detail::TimedGpuCall(
            cc,
            &report.rounds[round].rotateMs,
            [&]() { return cc->EvalRotate(current, rotation); });

        TimingReport compareTiming;
        auto conditionBit0 = detail::TimedGpuCall(
            cc,
            &report.rounds[round].compareMs,
            [&]() {
                return EvalGreaterThanPOnly(
                    cc, shifted, current, masks, &compareTiming);
            });

        if (compareTiming.totalMs > 0.0) {
            report.rounds[round].compareMs = compareTiming.totalMs;
        }

        if (bootstrapConditionEachRound) {
            conditionBit0 = detail::TimedGpuCall(
                cc,
                &report.rounds[round].conditionBootstrapMs,
                [&]() {
                    return BootstrapProjectBit0(cc, conditionBit0, masks);
                });
        }

        auto conditionMask = detail::TimedGpuCall(
            cc,
            &report.rounds[round].broadcastMs,
            [&]() { return BroadcastBit0ToActiveWord(cc, conditionBit0); });

        auto selected = detail::TimedGpuCall(
            cc,
            &report.rounds[round].cmuxMs,
            [&]() { return EvalAuctionCmux(cc, conditionMask, shifted, current); });

        if (bootstrapEachRound) {
            selected = detail::TimedGpuCall(
                cc,
                &report.rounds[round].bootstrapMs,
                [&]() { return BootstrapProjectActive(cc, selected, masks); });
        }
        else {
            selected = detail::TimedGpuCall(
                cc,
                &report.rounds[round].activeMaskMs,
                [&]() { return cc->EvalMult(selected, masks.activeMask); });
        }

        current = std::move(selected);

        detail::Synchronize(cc);
        report.rounds[round].wallMs =
            detail::Milliseconds(roundBegin, detail::Clock::now());

        report.rotateTotalMs += report.rounds[round].rotateMs;
        report.compareTotalMs += report.rounds[round].compareMs;
        report.conditionBootstrapTotalMs += report.rounds[round].conditionBootstrapMs;
        report.broadcastTotalMs += report.rounds[round].broadcastMs;
        report.cmuxTotalMs += report.rounds[round].cmuxMs;
        report.bootstrapTotalMs += report.rounds[round].bootstrapMs;
        report.activeMaskTotalMs += report.rounds[round].activeMaskMs;
    }

    if (finalBootstrap) {
        current = detail::TimedGpuCall(
            cc,
            &report.finalBootstrapMs,
            [&]() { return BootstrapProjectActive(cc, current, masks); });
    }

    report.totalMs =
        detail::Milliseconds(totalBegin, detail::Clock::now());

    return current;
}

}  // namespace ks256gpu
