#pragma once

// Version: public-u256-scalar-mul-v2-divider-stability-fixes-3gpu
//
// This header extracts the public-constant low-256 multiplication core from
// divider_public_pc_ks_gpu.h and gives it a scalar_mul-facing wrapper.
//
// Arithmetic semantics:
//   input : encrypted uint256 N, public uint256 scalar S
//   output: encrypted uint256 (N * S) mod 2^256
//
// Layout:
//   128 words/ciphertext, 512 slots/word = 256 active bits + 256 guard bits.
//
// This is intentionally conservative: it reuses the same masks, DynamicDadda
// pipeline, compressor, KS final adder, bootstrap projection helper, and GPU
// synchronization style as the public reciprocal divider.

#include "divider_public_pc_ks_gpu.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ks256gpu {

struct PublicScalarMulTimingReport {
    MultiplierTimingReport multiplier{};

    double rowGenerationMs = 0.0;
    double outputMaskMs = 0.0;
    double finalBootstrapMs = 0.0;
    double totalMs = 0.0;

    uint32_t scalarOneBits = 0;
    uint32_t scalarZeroBits = 0;
    uint64_t publicRowRotations = 0;
    uint64_t publicRowMaskMults = 0;
    uint64_t pushedRows = 0;
    uint32_t dynamicDaddaStages = 0;
    uint32_t dynamicDaddaInitialRows = 0;
};

inline PublicReciprocalDivisionMasks BuildAndLoadPublicScalarMulMasks(
    const CryptoContext<DCRTPoly>& cc,
    int directionSign,
    const std::array<uint8_t, kWordBits>& scalarBits) {

    // The scalar multiplication core only needs masks.multiplier.*.  Reusing
    // BuildAndLoadPublicReciprocalDivisionMasks keeps the mask layout identical
    // to the divider and avoids changing the existing multiplier mask builder.
    // The divisorMask field is loaded with scalarBits but is unused here.
    return BuildAndLoadPublicReciprocalDivisionMasks(
        cc, directionSign, scalarBits);
}


inline int32_t PublicScalarMulHighestSetBit(
    const std::array<uint8_t, kWordBits>& bits) {
    for (int32_t bit = static_cast<int32_t>(kWordBits) - 1; bit >= 0; --bit) {
        if (bits[static_cast<uint32_t>(bit)] != 0) {
            return bit;
        }
    }
    return -1;
}

/**
 * Evaluate encrypted uint256 by public uint256 scalar multiplication.
 *
 * The public scalar S is supplied as 256 little-endian Boolean bits.
 * Only one-bits of S generate rows.  Row j contributes N << j.  The Dadda
 * compression tree and final KS adder return only the low 256 active bits,
 * i.e. Solidity/EVM-style uint256 multiplication modulo 2^256.
 */
inline Ciphertext<DCRTPoly> EvalScalarMulPublic256LowKS(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cN,
    const std::array<uint8_t, kWordBits>& scalarBits,
    int directionSign,
    PublicReciprocalDivisionMasks& masks,
    PublicScalarMulTimingReport* timing = nullptr,
    uint32_t compressorBatchSize = kDefaultCompressorBatchSize) {

    if (!cc || !cN) {
        throw std::invalid_argument(
            "EvalScalarMulPublic256LowKS received null context/input");
    }
    if (directionSign != -1) {
        throw std::invalid_argument(
            "EvalScalarMulPublic256LowKS currently requires directionSign=-1");
    }
    if (!masks.multiplier.ksMasks.activeMask ||
        !masks.multiplier.ksMasks.carryMask ||
        !masks.multiplier.bitZeroMask) {
        throw std::invalid_argument(
            "EvalScalarMulPublic256LowKS received invalid masks");
    }

    PublicScalarMulTimingReport local;
    PublicScalarMulTimingReport& report = timing ? *timing : local;
    report = PublicScalarMulTimingReport{};

    report.scalarOneBits = public_reciprocal_detail::CountOneBits(scalarBits);
    report.scalarZeroBits = kWordBits - report.scalarOneBits;
    report.dynamicDaddaInitialRows = report.scalarOneBits;
    report.dynamicDaddaStages = static_cast<uint32_t>(
        public_reciprocal_detail::BuildDaddaTargets(report.scalarOneBits).size());

    public_reciprocal_detail::Synchronize(cc);
    const auto totalBegin = public_reciprocal_detail::Clock::now();

    if (report.scalarOneBits == 0) {
        auto zero = cc->EvalSub(cN, cN);
        zero = cc->EvalMult(zero, masks.multiplier.ksMasks.activeMask);
        public_reciprocal_detail::Synchronize(cc);
        report.outputMaskMs = 0.0;
        report.totalMs = public_reciprocal_detail::Milliseconds(
            totalBegin, public_reciprocal_detail::Clock::now());
        report.multiplier.totalMs = report.totalMs;
        return zero;
    }

    const int32_t highestSetBit = PublicScalarMulHighestSetBit(scalarBits);
    if (highestSetBit < 0) {
        throw std::runtime_error(
            "EvalScalarMulPublic256LowKS inconsistent scalar bit count");
    }

    public_reciprocal_detail::DynamicDaddaPipeline pipeline(
        cc,
        directionSign,
        report.scalarOneBits,
        compressorBatchSize,
        &report.multiplier);

    const auto rowBegin = public_reciprocal_detail::Clock::now();

    auto shiftedLeft = cN;
    for (uint32_t j = 0; j <= static_cast<uint32_t>(highestSetBit); ++j) {
        if (scalarBits[j] != 0) {
            auto row = cc->EvalMult(
                shiftedLeft, masks.multiplier.ksMasks.activeMask);
            ++report.publicRowMaskMults;
            ++report.pushedRows;
            pipeline.Push(std::move(row));
        }

        if (j < static_cast<uint32_t>(highestSetBit)) {
            shiftedLeft = cc->EvalRotate(shiftedLeft, directionSign);
            ++report.publicRowRotations;
        }
    }

    // Drop the streamed shift handle before finishing the Dadda pipeline.
    // This mirrors the low-memory public-divider fix and avoids keeping a long
    // shifted ciphertext chain alive across Finish().
    shiftedLeft = Ciphertext<DCRTPoly>();
    public_reciprocal_detail::Synchronize(cc);

    auto rows = pipeline.Finish();
    report.rowGenerationMs = public_reciprocal_detail::Milliseconds(
        rowBegin, public_reciprocal_detail::Clock::now());

    Ciphertext<DCRTPoly> product;
    if (rows.size() == 1) {
        product = std::move(rows[0]);
        report.multiplier.finalAdder.totalMs = 0.0;
    }
    else if (rows.size() == 2) {
        // Same failure mode as q0D in the public divider: after a Dadda tree,
        // directly feeding two deep rows into EvalAddKS can exhaust depth or
        // trigger a GPU-side failure.  Preboot both rows, align levels, then run
        // the final KS adder.
        rows[0] = BootstrapProjectActive256(cc, rows[0], masks);
        public_reciprocal_detail::Synchronize(cc);
        rows[1] = BootstrapProjectActive256(cc, rows[1], masks);
        public_reciprocal_detail::Synchronize(cc);
        AlignBinaryCiphertextsToSameLevelByMask(
            cc, rows[0], rows[1], masks.multiplier.ksMasks.activeMask,
            "scalar_mul final Dadda rows");

        product = EvalAddKS(
            cc,
            rows[0],
            rows[1],
            directionSign,
            masks.multiplier.ksMasks,
            &report.multiplier.finalAdder);
    }
    else {
        throw std::runtime_error(
            "EvalScalarMulPublic256LowKS unexpected Dadda output row count");
    }

    public_reciprocal_detail::Synchronize(cc);
    const auto maskBegin = public_reciprocal_detail::Clock::now();
    product = cc->EvalMult(product, masks.multiplier.ksMasks.activeMask);
    public_reciprocal_detail::Synchronize(cc);
    report.outputMaskMs = public_reciprocal_detail::Milliseconds(
        maskBegin, public_reciprocal_detail::Clock::now());
    report.multiplier.outputMaskMs = report.outputMaskMs;

    report.totalMs = public_reciprocal_detail::Milliseconds(
        totalBegin, public_reciprocal_detail::Clock::now());
    report.multiplier.totalMs = report.totalMs;
    return product;
}
}  // namespace ks256gpu
