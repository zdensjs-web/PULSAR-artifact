#pragma once

// Version: public-u256-confidential-vault-claim-shares-v6-standalone-header-guarded-division
//
// This header intentionally does NOT include scalar_mul_public_pc_ks_gpu.h or
// divider_public_pc_ks_gpu.h.  The corresponding .cpp includes those headers
// before this header, then provides the function definitions.
//
// It declares the Confidential Vault claim-shares interface only:
//
//   encrypted_shares = floor(((encrypted_deposit * public_exchange_rate)
//                             mod 2^256) / public_rate_scale)
//
// Layout and low-level operator types/constants are expected to come from the
// already-included project operator headers in the .cpp.

#include <array>
#include <cstdint>

namespace ks256gpu {

struct VaultClaimSharesTimingReport {
    PublicScalarMulTimingReport scalarMul{};
    PublicReciprocalTimingReport division{};

    double scalarMulMs = 0.0;
    double productBootstrapMs = 0.0;
    double publicDivideMs = 0.0;
    double totalMs = 0.0;

    uint32_t exchangeRateOneBits = 0;
    uint32_t exchangeRateZeroBits = 0;
    uint32_t rateScaleOneBits = 0;
    uint32_t rateScaleZeroBits = 0;
    uint32_t rateScaleReciprocalOneBits = 0;
    uint32_t rateScaleReciprocalZeroBits = 0;
};

bool IsUint256One(const std::array<uint8_t, kWordBits>& bits);

PublicReciprocalDivisionMasks BuildAndLoadVaultClaimSharesMasks(
    const CryptoContext<DCRTPoly>& cc,
    int directionSign,
    const std::array<uint8_t, kWordBits>& publicRateScaleBits);

Ciphertext<DCRTPoly> EvalVaultClaimSharesPublicKS(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cDeposits,
    const std::array<uint8_t, kWordBits>& publicExchangeRateBits,
    const std::array<uint8_t, kWordBits>& publicRateScaleBits,
    const std::array<uint8_t, kWordBits>& publicRateScaleReciprocalBits,
    int directionSign,
    PublicReciprocalDivisionMasks& masks,
    VaultClaimSharesTimingReport* timing = nullptr,
    uint32_t compressorBatchSize = kDefaultCompressorBatchSize,
    uint32_t configuredDepth = 40U);

}  // namespace ks256gpu
