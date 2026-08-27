#include "divider_public_pc_ks_gpu.h"
#include "scalar_mul_public_pc_ks_gpu.h"
#include "vault_claim_shares_public_pc_ks_gpu.h"

#include <fideslib.hpp>

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


namespace ks256gpu {

namespace vault_stage_detail {

using Clock = std::chrono::steady_clock;

inline double Milliseconds(const Clock::time_point& begin,
                           const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

inline bool Enabled() {
    const char* raw = std::getenv("KS256_VAULT_TRACE");
    if (!raw || !*raw) {
        return true;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return !(value == "0" || value == "false" || value == "off");
}

inline void Mark(const std::string& label) {
    if (Enabled()) {
        std::cerr << "[VAULT-STAGE] " << label << std::endl;
    }
}

inline void Depth(const std::string& label, const Ciphertext<DCRTPoly>& ct) {
    if (!Enabled()) {
        return;
    }
    if (!ct) {
        std::cerr << "[VAULT-DEPTH] " << label << ": <null>" << std::endl;
        return;
    }
    std::cerr << "[VAULT-DEPTH] " << label
              << ": level=" << ct->GetLevel()
              << ", noiseScaleDeg=" << ct->GetNoiseScaleDeg()
              << std::endl;
}

inline uint32_t EnvUint32(const char* name, uint32_t fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return fallback;
    }
    const unsigned long value = std::strtoul(raw, nullptr, 10);
    return value == 0 ? fallback : static_cast<uint32_t>(value);
}

inline uint32_t LevelsRemaining(uint32_t configuredDepth,
                                const Ciphertext<DCRTPoly>& ciphertext) {
    if (!ciphertext) {
        return 0;
    }
    const uint32_t level = ciphertext->GetLevel();
    const uint32_t scalePenalty = ciphertext->GetNoiseScaleDeg() > 0
                                      ? ciphertext->GetNoiseScaleDeg() - 1
                                      : 0;
    if (configuredDepth <= level + scalePenalty) {
        return 0;
    }
    return configuredDepth - level - scalePenalty;
}

inline Ciphertext<DCRTPoly> BootstrapIfNeeded(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> ct,
    PublicReciprocalDivisionMasks& masks,
    uint32_t configuredDepth,
    uint32_t need,
    const std::string& label) {

    if (!ct) {
        throw std::invalid_argument("BootstrapIfNeeded received null ciphertext: " + label);
    }

    const uint32_t remaining = LevelsRemaining(configuredDepth, ct);
    if (remaining >= need) {
        return ct;
    }

    Mark("BEGIN depth-guard boot: " + label +
         " remaining=" + std::to_string(remaining) +
         " need>=" + std::to_string(need));
    ct = BootstrapProjectActive256(cc, ct, masks);
    public_reciprocal_detail::Synchronize(cc);
    Depth("after depth-guard boot: " + label, ct);
    Mark("END depth-guard boot: " + label);
    return ct;
}

inline uint32_t VaultDivisionChunkRows() {
    const uint32_t dividerLimit =
        public_reciprocal_chunked_detail::RuntimeReciprocalChunkRows();
    const uint32_t defaultLimit = std::min<uint32_t>(dividerLimit, 8U);
    return EnvUint32("KS256_VAULT_DIV_CHUNK_ROWS", defaultLimit);
}

inline Ciphertext<DCRTPoly> FinishDaddaRowsActive256Guarded(
    const CryptoContext<DCRTPoly>& cc,
    std::vector<Ciphertext<DCRTPoly>> rows,
    int directionSign,
    PublicReciprocalDivisionMasks& masks,
    PublicReciprocalTimingReport* timing,
    uint32_t configuredDepth,
    const std::string& label) {

    if (rows.empty()) {
        throw std::invalid_argument("FinishDaddaRowsActive256Guarded received no rows: " + label);
    }

    constexpr uint32_t kNeedForFinalKS = kNumLayers + 4U;

    Ciphertext<DCRTPoly> out;
    if (rows.size() == 1) {
        rows[0] = BootstrapIfNeeded(
            cc, std::move(rows[0]), masks, configuredDepth,
            kNeedForFinalKS, label + " single-row before final projection");
        out = std::move(rows[0]);
    }
    else if (rows.size() == 2) {
        Mark("BEGIN guarded Dadda final rows depth check: " + label);
        rows[0] = BootstrapIfNeeded(
            cc, std::move(rows[0]), masks, configuredDepth,
            kNeedForFinalKS, label + " row0 before EvalAddKS");
        rows[1] = BootstrapIfNeeded(
            cc, std::move(rows[1]), masks, configuredDepth,
            kNeedForFinalKS, label + " row1 before EvalAddKS");

        AlignBinaryCiphertextsToSameLevelByMask(
            cc, rows[0], rows[1], masks.multiplier.ksMasks.activeMask,
            (label + " final rows level-align").c_str());
        public_reciprocal_detail::Synchronize(cc);
        Depth(label + " row0 after guard", rows[0]);
        Depth(label + " row1 after guard", rows[1]);

        out = EvalAddKS(cc, rows[0], rows[1], directionSign,
                        masks.multiplier.ksMasks,
                        timing ? &timing->multiplier.finalAdder : nullptr);
        public_reciprocal_detail::Synchronize(cc);
        Mark("END guarded Dadda final rows EvalAddKS: " + label);
    }
    else {
        throw std::runtime_error("FinishDaddaRowsActive256Guarded invalid row count: " + label);
    }

    out = cc->EvalMult(out, masks.multiplier.ksMasks.activeMask);
    public_reciprocal_detail::Synchronize(cc);
    return BootstrapProjectActive256(cc, out, masks);
}

inline Ciphertext<DCRTPoly> EvalPublicReciprocalHighOnlyChunkedVaultGuardedKS(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> cN,
    const std::array<uint8_t, kWordBits>& reciprocalBits,
    int directionSign,
    PublicReciprocalDivisionMasks& masks,
    PublicReciprocalTimingReport* timing,
    uint32_t compressorBatchSize,
    uint32_t configuredDepth) {

    if (!cc || !cN) {
        throw std::invalid_argument("EvalPublicReciprocalHighOnlyChunkedVaultGuardedKS received null input");
    }
    if (directionSign != -1) {
        throw std::invalid_argument("EvalPublicReciprocalHighOnlyChunkedVaultGuardedKS requires directionSign=-1");
    }

    constexpr uint32_t kNeedBeforeChunkDadda = kNumLayers + 7U;
    cN = BootstrapIfNeeded(
        cc, std::move(cN), masks, configuredDepth,
        kNeedBeforeChunkDadda, "division input before high-only reciprocal chunks");

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
        timing->dynamicDaddaInitialRows = static_cast<uint32_t>(positions.size());
    }

    auto zero = public_reciprocal_chunked_detail::MakeZeroActive256(cc, cN, masks);
    auto highAccumulator = zero;
    if (positions.empty()) {
        return highAccumulator;
    }

    const uint32_t chunkLimit = VaultDivisionChunkRows();
    const uint32_t totalRows = static_cast<uint32_t>(positions.size());
    uint32_t processed = 0;
    uint32_t rowsInChunk = 0;
    uint32_t currentChunkRows = std::min(chunkLimit, totalRows);
    uint32_t chunkId = 0;

    Mark("division high-only guarded chunkRows=" + std::to_string(chunkLimit) +
         " totalRows=" + std::to_string(totalRows));

    std::unique_ptr<public_reciprocal_detail::DynamicDaddaPipeline> highPipeline(
        new public_reciprocal_detail::DynamicDaddaPipeline(
            cc, directionSign, currentChunkRows, compressorBatchSize,
            timing ? &timing->multiplier : nullptr));

    auto startNextChunk = [&]() {
        ++chunkId;
        rowsInChunk = 0;
        const uint32_t remaining = totalRows - processed;
        currentChunkRows = std::min(chunkLimit, remaining);
        highPipeline.reset(new public_reciprocal_detail::DynamicDaddaPipeline(
            cc, directionSign, currentChunkRows, compressorBatchSize,
            timing ? &timing->multiplier : nullptr));
    };

    auto finalizeCurrentChunk = [&]() {
        const std::string label = "division high-only chunk " + std::to_string(chunkId);
        Mark("BEGIN " + label + " pipeline Finish");
        auto rowsOut = highPipeline->Finish();
        Mark("END " + label + " pipeline Finish rowsOut=" + std::to_string(rowsOut.size()));

        auto chunkHigh = FinishDaddaRowsActive256Guarded(
            cc, std::move(rowsOut), directionSign, masks, timing,
            configuredDepth, label + " FinishDaddaRowsActive256");

        highAccumulator = BootstrapIfNeeded(
            cc, std::move(highAccumulator), masks, configuredDepth,
            kNumLayers + 4U, label + " highAccumulator before add");
        chunkHigh = BootstrapIfNeeded(
            cc, std::move(chunkHigh), masks, configuredDepth,
            kNumLayers + 4U, label + " chunkHigh before add");

        AlignBinaryCiphertextsToSameLevelByMask(
            cc, highAccumulator, chunkHigh,
            masks.multiplier.ksMasks.activeMask,
            (label + " highAccumulator + chunkHigh").c_str());

        Mark("BEGIN " + label + " highAccumulator + chunkHigh EvalAddKS");
        auto highPlusChunk = EvalAddKS(
            cc, highAccumulator, chunkHigh, directionSign,
            masks.multiplier.ksMasks, nullptr);
        public_reciprocal_detail::Synchronize(cc);
        Mark("END " + label + " highAccumulator + chunkHigh EvalAddKS");

        highAccumulator = BootstrapProjectActive256(cc, highPlusChunk, masks);
        highPipeline.reset();
    };

    auto shiftedLeft = cN;
    uint32_t currentShift = 0;
    for (uint32_t idx = 0; idx < totalRows; ++idx) {
        const uint32_t j = positions[idx];
        if (j < currentShift) {
            throw std::runtime_error("vault high-only reciprocal positions are not sorted");
        }
        if (j > currentShift) {
            shiftedLeft = public_reciprocal_highonly_detail::RotateByUnitSteps(
                cc, shiftedLeft, j - currentShift, directionSign, timing,
                "vault high-only selected-position unit rotate");
            currentShift = j;
        }

        auto highSource = cc->EvalRotate(shiftedLeft, static_cast<int32_t>(kWordBits));
        if (timing) {
            ++timing->publicRowRotations;
            ++timing->mergedHighOverflowRotations;
        }
        public_reciprocal_detail::Synchronize(cc);

        auto highRow = cc->EvalMult(highSource, masks.multiplier.ksMasks.activeMask);
        if (timing) {
            ++timing->publicRowMaskMults;
        }
        public_reciprocal_detail::Synchronize(cc);
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

inline Ciphertext<DCRTPoly> EvalPublicReciprocalQuotientVaultGuardedKS(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> cN,
    const std::array<uint8_t, kWordBits>& reciprocalBits,
    const std::array<uint8_t, kWordBits>& divisorBits,
    int directionSign,
    PublicReciprocalDivisionMasks& masks,
    PublicReciprocalTimingReport* timing,
    uint32_t compressorBatchSize,
    uint32_t configuredDepth) {

    PublicReciprocalTimingReport local;
    PublicReciprocalTimingReport& report = timing ? *timing : local;
    report = PublicReciprocalTimingReport{};
    report.reciprocalOneBits = public_reciprocal_detail::CountOneBits(reciprocalBits);
    report.reciprocalZeroBits = kWordBits - report.reciprocalOneBits;
    report.divisorOneBits = public_reciprocal_detail::CountOneBits(divisorBits);

    const auto totalBegin = Clock::now();

    Mark("BEGIN guarded high-only reciprocal");
    const auto highBegin = Clock::now();
    cN = BootstrapIfNeeded(
        cc, std::move(cN), masks, configuredDepth,
        kNumLayers + 7U, "division input before guarded high-only reciprocal");
    auto qBase = EvalPublicReciprocalHighOnlyChunkedVaultGuardedKS(
        cc, cN, reciprocalBits, directionSign, masks, &report,
        compressorBatchSize, configuredDepth);
    report.rowGenerationMs += Milliseconds(highBegin, Clock::now());
    Mark("END guarded high-only reciprocal");
    Depth("after guarded high-only qBase", qBase);

    report.q0BootstrapMs = 0.0;

    Mark("BEGIN guarded qBase * public divisor");
    const auto mulDBegin = Clock::now();
    qBase = BootstrapIfNeeded(
        cc, std::move(qBase), masks, configuredDepth,
        kNumLayers + 4U, "qBase before qBase * divisor");
    auto qBaseD = EvalPublicConstMulLowKS(
        cc, qBase, divisorBits, directionSign, masks, &report,
        compressorBatchSize);
    report.mulDivisorMs = Milliseconds(mulDBegin, Clock::now());
    Mark("END guarded qBase * public divisor");
    report.tBootstrapMs = 0.0;

    Mark("BEGIN guarded initial remainder");
    const auto subBegin = Clock::now();
    auto cNForSub = BootstrapProjectActive256(cc, cN, masks);
    cNForSub = BootstrapIfNeeded(
        cc, std::move(cNForSub), masks, configuredDepth,
        kNumLayers + 4U, "cNForSub before initial remainder");
    qBaseD = BootstrapIfNeeded(
        cc, std::move(qBaseD), masks, configuredDepth,
        kNumLayers + 4U, "qBaseD before initial remainder");

    AlignBinaryCiphertextsToSameLevelByMask(
        cc, cNForSub, qBaseD, masks.multiplier.ksMasks.activeMask,
        "vault guarded initial r = N - qBaseD");
    auto remainder = EvalSubCipher256(cc, cNForSub, qBaseD, directionSign, masks);
    cNForSub = nullptr;
    qBaseD = nullptr;
    public_reciprocal_detail::Synchronize(cc);
    report.subtractMs = Milliseconds(subBegin, Clock::now());
    Mark("END guarded initial remainder");

    const auto bootRBegin = Clock::now();
    remainder = BootstrapProjectActive256(cc, remainder, masks);
    report.rBootstrapMs = Milliseconds(bootRBegin, Clock::now());

    Mark("BEGIN guarded bounded correction");
    const auto corrBegin = Clock::now();
    qBase = BootstrapIfNeeded(
        cc, std::move(qBase), masks, configuredDepth,
        kNumLayers + 4U, "quotient before bounded correction");
    remainder = BootstrapIfNeeded(
        cc, std::move(remainder), masks, configuredDepth,
        kNumLayers + 4U, "remainder before bounded correction");

    const uint32_t correctionBound = report.reciprocalOneBits + 2U;
    ApplyHighOnlyBoundedCorrectionKS(
        cc, qBase, remainder, divisorBits, correctionBound,
        directionSign, masks, report);
    report.correctionBootstrapMs += Milliseconds(corrBegin, Clock::now());
    Mark("END guarded bounded correction");

    report.finalBootstrapMs = 0.0;
    report.totalMs = Milliseconds(totalBegin, Clock::now());
    return qBase;
}

}  // namespace vault_stage_detail

bool IsUint256One(const std::array<uint8_t, kWordBits>& bits) {
    if (bits[0] == 0) {
        return false;
    }
    for (uint32_t i = 1; i < kWordBits; ++i) {
        if (bits[i] != 0) {
            return false;
        }
    }
    return true;
}

PublicReciprocalDivisionMasks BuildAndLoadVaultClaimSharesMasks(
    const CryptoContext<DCRTPoly>& cc,
    int directionSign,
    const std::array<uint8_t, kWordBits>& publicRateScaleBits) {

    return BuildAndLoadPublicReciprocalDivisionMasks(
        cc, directionSign, publicRateScaleBits);
}

Ciphertext<DCRTPoly> EvalVaultClaimSharesPublicKS(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cDeposits,
    const std::array<uint8_t, kWordBits>& publicExchangeRateBits,
    const std::array<uint8_t, kWordBits>& publicRateScaleBits,
    const std::array<uint8_t, kWordBits>& publicRateScaleReciprocalBits,
    int directionSign,
    PublicReciprocalDivisionMasks& masks,
    VaultClaimSharesTimingReport* timing,
    uint32_t compressorBatchSize,
    uint32_t configuredDepth) {

    if (!cc || !cDeposits) {
        throw std::invalid_argument(
            "EvalVaultClaimSharesPublicKS received null context/input");
    }
    if (directionSign != -1) {
        throw std::invalid_argument(
            "EvalVaultClaimSharesPublicKS currently requires directionSign=-1");
    }
    if (public_reciprocal_detail::CountOneBits(publicRateScaleBits) == 0) {
        throw std::invalid_argument("publicRateScale must be nonzero");
    }

    VaultClaimSharesTimingReport local;
    VaultClaimSharesTimingReport& report = timing ? *timing : local;
    report = VaultClaimSharesTimingReport{};

    report.exchangeRateOneBits =
        public_reciprocal_detail::CountOneBits(publicExchangeRateBits);
    report.exchangeRateZeroBits = kWordBits - report.exchangeRateOneBits;
    report.rateScaleOneBits =
        public_reciprocal_detail::CountOneBits(publicRateScaleBits);
    report.rateScaleZeroBits = kWordBits - report.rateScaleOneBits;
    report.rateScaleReciprocalOneBits =
        public_reciprocal_detail::CountOneBits(publicRateScaleReciprocalBits);
    report.rateScaleReciprocalZeroBits =
        kWordBits - report.rateScaleReciprocalOneBits;

    vault_stage_detail::Mark("BEGIN EvalVaultClaimSharesPublicKS");
    vault_stage_detail::Depth("input deposits", cDeposits);
    public_reciprocal_detail::Synchronize(cc);
    const auto totalBegin = vault_stage_detail::Clock::now();

    const auto mulBegin = vault_stage_detail::Clock::now();
    vault_stage_detail::Mark("BEGIN scalar_mul deposit * public exchangeRate");
    auto cProductLowRaw = EvalScalarMulPublic256LowKS(
        cc,
        cDeposits,
        publicExchangeRateBits,
        directionSign,
        masks,
        &report.scalarMul,
        compressorBatchSize);
    public_reciprocal_detail::Synchronize(cc);
    vault_stage_detail::Depth("after scalar_mul raw productLow", cProductLowRaw);
    vault_stage_detail::Mark("END scalar_mul deposit * public exchangeRate");
    report.scalarMulMs = vault_stage_detail::Milliseconds(
        mulBegin, vault_stage_detail::Clock::now());

    const auto bootBegin = vault_stage_detail::Clock::now();
    vault_stage_detail::Mark("BEGIN productLow BootstrapProjectActive256");
    auto cProductLow = BootstrapProjectActive256(cc, cProductLowRaw, masks);
    cProductLowRaw = Ciphertext<DCRTPoly>();
    public_reciprocal_detail::Synchronize(cc);
    vault_stage_detail::Depth("after productLow bootstrap", cProductLow);
    vault_stage_detail::Mark("END productLow BootstrapProjectActive256");
    report.productBootstrapMs = vault_stage_detail::Milliseconds(
        bootBegin, vault_stage_detail::Clock::now());

    Ciphertext<DCRTPoly> cShares;
    if (IsUint256One(publicRateScaleBits)) {
        vault_stage_detail::Mark("BEGIN division identity scale=1");
        cShares = std::move(cProductLow);
        report.publicDivideMs = 0.0;
        vault_stage_detail::Depth("identity output shares", cShares);
        vault_stage_detail::Mark("END division identity scale=1");
    }
    else {
        const auto divBegin = vault_stage_detail::Clock::now();
        vault_stage_detail::Mark("BEGIN public division productLow / public rateScale");
        cProductLow = vault_stage_detail::BootstrapIfNeeded(
            cc, std::move(cProductLow), masks, configuredDepth,
            kNumLayers + 7U, "productLow before entering guarded divider");
        cShares = vault_stage_detail::EvalPublicReciprocalQuotientVaultGuardedKS(
            cc,
            std::move(cProductLow),
            publicRateScaleReciprocalBits,
            publicRateScaleBits,
            directionSign,
            masks,
            &report.division,
            compressorBatchSize,
            configuredDepth);
        public_reciprocal_detail::Synchronize(cc);
        vault_stage_detail::Depth("after public division shares", cShares);
        vault_stage_detail::Mark("END public division productLow / public rateScale");
        report.publicDivideMs = vault_stage_detail::Milliseconds(
            divBegin, vault_stage_detail::Clock::now());
    }

    report.totalMs = vault_stage_detail::Milliseconds(
        totalBegin, vault_stage_detail::Clock::now());
    vault_stage_detail::Mark("END EvalVaultClaimSharesPublicKS");
    return cShares;
}

}  // namespace ks256gpu



using namespace fideslib;
using boost::multiprecision::cpp_int;

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint32_t kPrintSampleUsers = 6;

struct PhaseTimes {
    double contextMs         = 0.0;
    double keyGenMs          = 0.0;
    double multKeyGenMs      = 0.0;
    double rotateKeyGenMs    = 0.0;
    double bootstrapSetupMs  = 0.0;
    double bootstrapKeyGenMs = 0.0;
    double loadContextMs     = 0.0;
    double maskBuildLoadMs   = 0.0;
    double inputBuildMs      = 0.0;
    double encodingMs        = 0.0;
    double encryptionMs      = 0.0;
    double claimSharesMs     = 0.0;
    double decryptionMs      = 0.0;
    double verificationMs    = 0.0;
    double totalMs           = 0.0;
};

inline double Milliseconds(const Clock::time_point& begin,
                           const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

inline double PrecisionBits(double error) {
    if (error <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return -std::log2(error);
}

uint32_t LevelsRemaining(uint32_t configuredDepth,
                         const Ciphertext<DCRTPoly>& ciphertext) {
    if (!ciphertext) {
        return 0;
    }
    const uint32_t level = ciphertext->GetLevel();
    const uint32_t scalePenalty = ciphertext->GetNoiseScaleDeg() > 0
                                      ? ciphertext->GetNoiseScaleDeg() - 1
                                      : 0;
    if (configuredDepth <= level + scalePenalty) {
        return 0;
    }
    return configuredDepth - level - scalePenalty;
}

cpp_int TwoPow(uint32_t bits) {
    cpp_int value = 1;
    value <<= bits;
    return value;
}

cpp_int Uint256Mask() {
    return TwoPow(256) - 1;
}

std::string Trim(const std::string& input) {
    const auto begin = input.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = input.find_last_not_of(" \t\r\n");
    return input.substr(begin, end - begin + 1);
}

cpp_int ParseUint256(const std::string& raw) {
    const std::string text = Trim(raw);
    if (text.empty()) {
        throw std::invalid_argument("empty uint256 string");
    }

    cpp_int value = 0;
    if (text.size() > 2 && text[0] == '0' &&
        (text[1] == 'x' || text[1] == 'X')) {
        for (size_t i = 2; i < text.size(); ++i) {
            const char ch = text[i];
            uint32_t digit = 0;
            if (ch >= '0' && ch <= '9') {
                digit = static_cast<uint32_t>(ch - '0');
            }
            else if (ch >= 'a' && ch <= 'f') {
                digit = static_cast<uint32_t>(ch - 'a' + 10);
            }
            else if (ch >= 'A' && ch <= 'F') {
                digit = static_cast<uint32_t>(ch - 'A' + 10);
            }
            else {
                throw std::invalid_argument("invalid hex uint256 string");
            }
            value <<= 4;
            value += digit;
        }
    }
    else {
        for (char ch : text) {
            if (ch < '0' || ch > '9') {
                throw std::invalid_argument("invalid decimal uint256 string");
            }
            value *= 10;
            value += static_cast<uint32_t>(ch - '0');
        }
    }

    if (value < 0 || value >= TwoPow(256)) {
        throw std::out_of_range("value is outside uint256 range");
    }
    return value;
}

std::string ToHex256(cpp_int value) {
    value &= Uint256Mask();
    constexpr char kHex[] = "0123456789abcdef";
    std::string out(64, '0');
    for (uint32_t nibble = 0; nibble < 64; ++nibble) {
        const uint32_t digit = static_cast<uint32_t>(value & 0x0f);
        out[63 - nibble] = kHex[digit];
        value >>= 4;
    }
    return "0x" + out;
}

std::array<uint8_t, ks256gpu::kWordBits> BitsFromUint256(cpp_int value) {
    value &= Uint256Mask();
    std::array<uint8_t, ks256gpu::kWordBits> bits{};
    for (uint32_t bit = 0; bit < ks256gpu::kWordBits; ++bit) {
        bits[bit] = static_cast<uint8_t>((value & 1) != 0);
        value >>= 1;
    }
    return bits;
}

uint32_t CountBits(const std::array<uint8_t, ks256gpu::kWordBits>& bits) {
    uint32_t count = 0;
    for (uint8_t b : bits) {
        count += static_cast<uint32_t>(b != 0);
    }
    return count;
}

void WriteUint256ToPackedSlots(cpp_int value,
                               uint32_t word,
                               std::vector<uint8_t>* bits,
                               std::vector<double>* slots) {
    const auto wordBits = BitsFromUint256(value);
    const uint32_t base = word * ks256gpu::kStride;
    for (uint32_t bit = 0; bit < ks256gpu::kWordBits; ++bit) {
        (*bits)[base + bit] = wordBits[bit];
        (*slots)[base + bit] = static_cast<double>(wordBits[bit]);
    }
}

cpp_int ReadPackedBitsAsUint256(const std::vector<uint8_t>& bits,
                                uint32_t word) {
    cpp_int value = 0;
    const uint32_t base = word * ks256gpu::kStride;
    for (int bit = static_cast<int>(ks256gpu::kWordBits) - 1; bit >= 0;
         --bit) {
        value <<= 1;
        value += bits[base + static_cast<uint32_t>(bit)] ? 1 : 0;
    }
    return value;
}

void BuildTrueVaultClaimBits(const std::vector<cpp_int>& deposits,
                             const cpp_int& publicExchangeRate,
                             const cpp_int& publicRateScale,
                             std::vector<uint8_t>* expected) {
    if (publicRateScale == 0) {
        throw std::invalid_argument("publicRateScale must be nonzero");
    }

    expected->assign(ks256gpu::kMultiplierTotalSlots, 0);
    const cpp_int mask = Uint256Mask();
    for (uint32_t user = 0; user < ks256gpu::kMultiplierNumWords; ++user) {
        // Fixed-width uint256 semantics, matching TFHE-rs U256 scalar_mul:
        //   productLow = (deposit * exchangeRate) mod 2^256
        //   shares     = floor(productLow / rateScale)
        const cpp_int productLow = (deposits[user] * publicExchangeRate) & mask;
        const cpp_int shares = productLow / publicRateScale;
        const auto shareBits = BitsFromUint256(shares);
        const uint32_t base = user * ks256gpu::kStride;
        for (uint32_t bit = 0; bit < ks256gpu::kWordBits; ++bit) {
            (*expected)[base + bit] = shareBits[bit];
        }
    }
}

void PrintPhase(const std::string& name, double milliseconds) {
    std::cout << "  " << std::left << std::setw(44) << name
              << std::right << std::setw(14) << std::fixed
              << std::setprecision(3) << milliseconds << " ms"
              << "  (" << std::setprecision(6) << milliseconds / 1000.0
              << " s)\n";
}

void PrintScalarMulProfile(const ks256gpu::PublicScalarMulTimingReport& profile) {
    std::cout << "\n================ GPU public scalar_mul 计数 ================\n";
    PrintPhase("public exchange-rate rows + dynamic Dadda", profile.rowGenerationMs);
    PrintPhase("output active-256 mask", profile.outputMaskMs);
    PrintPhase("total scalar_mul core", profile.totalMs);

    std::cout << "\n  行生成统计:\n";
    std::cout << "    Exchange-rate one bits      : "
              << profile.scalarOneBits << "\n";
    std::cout << "    Exchange-rate zero bits     : "
              << profile.scalarZeroBits << "\n";
    std::cout << "    Pushed dynamic Dadda rows   : "
              << profile.pushedRows << "\n";
    std::cout << "    Dynamic Dadda initial rows  : "
              << profile.dynamicDaddaInitialRows << "\n";
    std::cout << "    Dynamic Dadda stages        : "
              << profile.dynamicDaddaStages << "\n";
    std::cout << "    Public row rotations        : "
              << profile.publicRowRotations << "\n";
    std::cout << "    Public row mask multipliers : "
              << profile.publicRowMaskMults << "\n";

    std::cout << "\n  Dadda / KS 统计:\n";
    std::cout << "    Compressor ciphertext mults : "
              << profile.multiplier.compressorCipherMults << "\n";
    std::cout << "    Compressor carry rotations  : "
              << profile.multiplier.compressorCarryRotations << "\n";
    std::cout << "    Streaming synchronizations  : "
              << profile.multiplier.streamingSynchronizations << "\n";
    std::cout << "    Peak buffered rows          : "
              << profile.multiplier.peakBufferedRows << "\n";
    std::cout << "    Final KS total              : "
              << std::fixed << std::setprecision(3)
              << profile.multiplier.finalAdder.totalMs << " ms\n";
}

void PrintReciprocalProfile(const ks256gpu::PublicReciprocalTimingReport& profile) {
    std::cout << "\n================ GPU public reciprocal divider 计数 ================\n";
    PrintPhase("merged high/low reciprocal Dadda", profile.rowGenerationMs);
    PrintPhase("low-half carry extraction", profile.lowCarryMs);
    PrintPhase("add low carry into q0", profile.addLowCarryMs);
    PrintPhase("q0 bootstrap + projection", profile.q0BootstrapMs);
    PrintPhase("q0 * public scale low product", profile.mulDivisorMs);
    PrintPhase("q0Scale bootstrap + projection", profile.tBootstrapMs);
    PrintPhase("r = productLow - q0Scale", profile.subtractMs);
    PrintPhase("r bootstrap + projection", profile.rBootstrapMs);
    PrintPhase("compare r >= scale", profile.correctionCompareMs);
    PrintPhase("correction bootstrap + projection", profile.correctionBootstrapMs);
    PrintPhase("conditional increment q0", profile.finalIncrementMs);
    PrintPhase("final quotient bootstrap", profile.finalBootstrapMs);
    PrintPhase("total reciprocal quotient", profile.totalMs);

    std::cout << "\n  行生成统计:\n";
    std::cout << "    Reciprocal one bits         : "
              << profile.reciprocalOneBits << "\n";
    std::cout << "    Reciprocal zero bits        : "
              << profile.reciprocalZeroBits << "\n";
    std::cout << "    Rate-scale one bits         : "
              << profile.divisorOneBits << "\n";
    std::cout << "    Merged high Dadda rows      : "
              << profile.mergedHighRows << "\n";
    std::cout << "    Merged low Dadda rows       : "
              << profile.mergedLowRows << "\n";
    std::cout << "    High overflow rotations     : "
              << profile.mergedHighOverflowRotations << "\n";
    std::cout << "    Pushed dynamic Dadda rows   : "
              << profile.pushedRows << "\n";
    std::cout << "    Dynamic Dadda initial rows  : "
              << profile.dynamicDaddaInitialRows << "\n";
    std::cout << "    Dynamic Dadda stages        : "
              << profile.dynamicDaddaStages << "\n";
    std::cout << "    Public row rotations        : "
              << profile.publicRowRotations << "\n";
    std::cout << "    Public row mask multipliers : "
              << profile.publicRowMaskMults << "\n";

    std::cout << "\n  Dadda / KS 统计:\n";
    std::cout << "    Compressor ciphertext mults : "
              << profile.multiplier.compressorCipherMults << "\n";
    std::cout << "    Compressor carry rotations  : "
              << profile.multiplier.compressorCarryRotations << "\n";
    std::cout << "    Streaming synchronizations  : "
              << profile.multiplier.streamingSynchronizations << "\n";
    std::cout << "    Peak buffered rows          : "
              << profile.multiplier.peakBufferedRows << "\n";
    std::cout << "    Final KS total              : "
              << std::fixed << std::setprecision(3)
              << profile.multiplier.finalAdder.totalMs << " ms\n";
}

void PrintVaultProfile(const ks256gpu::VaultClaimSharesTimingReport& profile) {
    std::cout << "\n================ Confidential Vault claim-shares 计数 ================\n";
    PrintPhase("deposit * public exchangeRate", profile.scalarMulMs);
    PrintPhase("product bootstrap + projection", profile.productBootstrapMs);
    PrintPhase("product / public rateScale", profile.publicDivideMs);
    PrintPhase("total claim-shares pipeline", profile.totalMs);

    std::cout << "\n  公开参数 bit 统计:\n";
    std::cout << "    ExchangeRate one bits       : "
              << profile.exchangeRateOneBits << "\n";
    std::cout << "    ExchangeRate zero bits      : "
              << profile.exchangeRateZeroBits << "\n";
    std::cout << "    RateScale one bits          : "
              << profile.rateScaleOneBits << "\n";
    std::cout << "    RateScale zero bits         : "
              << profile.rateScaleZeroBits << "\n";
    std::cout << "    Reciprocal one bits         : "
              << profile.rateScaleReciprocalOneBits << "\n";
    std::cout << "    Reciprocal zero bits        : "
              << profile.rateScaleReciprocalZeroBits << "\n";
}

struct VerificationStats {
    uint64_t activeBits = 0;
    uint64_t bitFailures = 0;
    uint32_t failedWords = 0;
    double maxAbsError = 0.0;
    double meanAbsError = 0.0;
    double rmse = 0.0;
    double maxGuardAbs = 0.0;
    uint32_t worstWord = 0;
    uint32_t worstBit = 0;
    double worstValue = 0.0;
    double worstExpected = 0.0;
};

VerificationStats VerifyBits(const std::vector<double>& values,
                             const std::vector<uint8_t>& expected) {
    VerificationStats stats;
    stats.activeBits = static_cast<uint64_t>(ks256gpu::kMultiplierNumWords) *
                       ks256gpu::kWordBits;

    double sumAbsError = 0.0;
    double sumSquaredError = 0.0;

    for (uint32_t word = 0; word < ks256gpu::kMultiplierNumWords; ++word) {
        bool wordFailed = false;
        const uint32_t base = word * ks256gpu::kStride;
        for (uint32_t bit = 0; bit < ks256gpu::kWordBits; ++bit) {
            const uint32_t index = base + bit;
            const double target = static_cast<double>(expected[index]);
            const double error = std::abs(values[index] - target);
            sumAbsError += error;
            sumSquaredError += error * error;
            if (error > stats.maxAbsError) {
                stats.maxAbsError = error;
                stats.worstWord = word;
                stats.worstBit = bit;
                stats.worstValue = values[index];
                stats.worstExpected = target;
            }
            const uint8_t recovered = values[index] >= 0.5 ? 1 : 0;
            if (recovered != expected[index]) {
                ++stats.bitFailures;
                wordFailed = true;
            }
        }
        if (wordFailed) {
            ++stats.failedWords;
        }
        for (uint32_t guard = 0; guard < ks256gpu::kGuardBits; ++guard) {
            const uint32_t index = base + ks256gpu::kWordBits + guard;
            stats.maxGuardAbs = std::max(stats.maxGuardAbs,
                                         std::abs(values[index]));
        }
    }

    stats.meanAbsError = sumAbsError / static_cast<double>(stats.activeBits);
    stats.rmse = std::sqrt(sumSquaredError /
                           static_cast<double>(stats.activeBits));
    return stats;
}

void PrintVerificationStats(const std::string& title,
                            const VerificationStats& stats,
                            double logPrecision) {
    std::cout << "\n================ " << title << " ================\n";
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "  有效 bit 槽数                 : "
              << stats.activeBits << "\n";
    std::cout << "  失败 packed users             : "
              << stats.failedWords << " / "
              << ks256gpu::kMultiplierNumWords << "\n";
    std::cout << "  0.5 阈值恢复失败数           : "
              << stats.bitFailures << "\n";
    std::cout << "  最大绝对误差                 : "
              << stats.maxAbsError << "\n";
    std::cout << "  平均 L1 误差                 : "
              << stats.meanAbsError << "\n";
    std::cout << "  RMSE                         : "
              << stats.rmse << "\n";
    std::cout << "  Guard 槽最大绝对值           : "
              << stats.maxGuardAbs << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  GetLogPrecision              : "
              << logPrecision << " bits\n";
    std::cout << "  最差精度 -log2(max error)    : "
              << PrecisionBits(stats.maxAbsError) << " bits\n";
    std::cout << "  平均精度 -log2(mean L1)      : "
              << PrecisionBits(stats.meanAbsError) << " bits\n";
    std::cout << "  RMSE 精度 -log2(RMSE)        : "
              << PrecisionBits(stats.rmse) << " bits\n";
    std::cout << "  最差位置                     : user="
              << stats.worstWord << ", bit=" << stats.worstBit << "\n";
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "  最差位置期望/解密值          : "
              << stats.worstExpected << " / " << stats.worstValue << "\n";
    std::cout << std::fixed;
    std::cout << "  整体结果                     : "
              << (stats.bitFailures == 0 ? "PASS" : "FAIL") << "\n";
}

uint32_t EnvUint32(const char* name, uint32_t fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return fallback;
    }
    const unsigned long value = std::strtoul(raw, nullptr, 10);
    return value == 0 ? fallback : static_cast<uint32_t>(value);
}

std::string EnvString(const char* name, const std::string& fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) {
        return fallback;
    }
    return raw;
}

std::string FullWidthDefaultExchangeRateHex() {
    return "0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff00001";
}

std::string FullWidthDefaultRateScaleHex() {
    return "0x8000000000000000000000000000000000000000000000000000000000100001";
}

cpp_int RandomUint256FullWidth(std::mt19937_64& rng) {
    cpp_int value = 0;
    for (uint32_t limb = 0; limb < 4; ++limb) {
        value <<= 64;
        value += rng();
    }
    value &= Uint256Mask();
    value |= TwoPow(255);
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::unitbuf;
        const auto programBegin = Clock::now();
        PhaseTimes phase;

        std::string exchangeRateText = EnvString(
            "KS256_VAULT_EXCHANGE_RATE",
            FullWidthDefaultExchangeRateHex());
        std::string rateScaleText = EnvString(
            "KS256_VAULT_RATE_SCALE",
            FullWidthDefaultRateScaleHex());

        if (argc >= 2) {
            exchangeRateText = argv[1];
        }
        if (argc >= 3) {
            rateScaleText = argv[2];
        }

        const cpp_int publicExchangeRate = ParseUint256(exchangeRateText);
        const cpp_int publicRateScale = ParseUint256(rateScaleText);
        if (publicRateScale == 0) {
            throw std::invalid_argument("public uint256 rateScale must be nonzero");
        }

        const cpp_int publicRateScaleReciprocal =
            publicRateScale == 1 ? cpp_int(0) : (TwoPow(256) / publicRateScale);

        const auto exchangeRateBits = BitsFromUint256(publicExchangeRate);
        const auto rateScaleBits = BitsFromUint256(publicRateScale);
        const auto rateScaleReciprocalBits = BitsFromUint256(publicRateScaleReciprocal);

        constexpr uint32_t ringDim = 1U << 17;
        constexpr uint32_t numSlots = ks256gpu::kMultiplierTotalSlots;
        constexpr uint32_t multiplicativeDepth = 40U;
        constexpr int directionSign = -1;
        constexpr uint64_t randomSeed = 0x5641554c54434831ULL;  // "VAULTCH1"
        const uint32_t compressorBatchSize = EnvUint32(
            "KS256_COMPRESSOR_BATCH", ks256gpu::kDefaultCompressorBatchSize);

        const SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
        const std::vector<uint32_t> levelBudget = {4, 4};
        const std::vector<uint32_t> bsgsDim = {0, 0};

        std::cout << "============================================================\n";
        std::cout << "FIDESlib GPU: " << ks256gpu::kMultiplierNumWords
                  << " x uint256 Confidential Vault claim-shares (v6 guarded division, 3 GPU)\n";
        std::cout << "Boolean encoding; 256 active slots + 256 guard slots per user\n";
        std::cout << "============================================================\n";
        std::cout << "参数:\n";
        std::cout << "  Ring dimension              : " << ringDim << "\n";
        std::cout << "  CKKS slots                  : " << numSlots << "\n";
        std::cout << "  Parallel users              : "
                  << ks256gpu::kMultiplierNumWords << "\n";
        std::cout << "  Slots per user              : "
                  << ks256gpu::kStride
                  << " (256 active + 256 guard)\n";
        std::cout << "  Public exchangeRate          : "
                  << ToHex256(publicExchangeRate) << "\n";
        std::cout << "  Public rateScale             : "
                  << ToHex256(publicRateScale) << "\n";
        std::cout << "  Reciprocal floor(2^256/scale): "
                  << (publicRateScale == 1
                          ? std::string("special-case scale=1")
                          : ToHex256(publicRateScaleReciprocal)) << "\n";
        std::cout << "  ExchangeRate one bits        : "
                  << CountBits(exchangeRateBits) << "\n";
        std::cout << "  RateScale one bits           : "
                  << CountBits(rateScaleBits) << "\n";
        std::cout << "  Configured mult depth        : "
                  << multiplicativeDepth << "\n";
        std::cout << "  Bootstrap level budget       : {4, 4}\n";
        std::cout << "  HYBRID large digits          : 3\n";
        std::cout << "  Compressor batch size        : "
                  << compressorBatchSize << "\n";
        std::cout << "  Output                       : encrypted uint256 shares per user\n";
        std::cout << "  Formula                      : shares = floor(((deposit * exchangeRate) mod 2^256) / rateScale)\n";
        std::cout << "  Correctness validation       : CPU uint256 formula vs decrypted shares\n";
        std::cout << "  Vault stage trace            : set KS256_VAULT_TRACE=0 to silence\n";
        std::cout << "  Vault division chunk rows     : env KS256_VAULT_DIV_CHUNK_ROWS, default min(KS256_RECIP_CHUNK_ROWS,8)\n";

        auto begin = Clock::now();
        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetSecretKeyDist(secretKeyDist);
        parameters.SetSecurityLevel(HEStd_128_classic);
        parameters.SetRingDim(ringDim);
        parameters.SetBatchSize(numSlots);
        parameters.SetMultiplicativeDepth(multiplicativeDepth);
        parameters.SetScalingTechnique(FLEXIBLEAUTO);
        parameters.SetScalingModSize(59);
        parameters.SetFirstModSize(60);
        parameters.SetKeySwitchTechnique(HYBRID);
        parameters.SetNumLargeDigits(3);
        parameters.SetDevices(std::vector<int>{0, 1, 2});
        parameters.SetPlaintextAutoload(false);
        parameters.SetCiphertextAutoload(true);

        CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);
        cc->Enable(FHE);
        phase.contextMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        auto keys = cc->KeyGen();
        phase.keyGenMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        cc->EvalMultKeyGen(keys.secretKey);
        phase.multKeyGenMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        std::vector<int32_t> rotations;
        rotations.reserve((ks256gpu::kNumLayers + 1) * 2 + 4);
        // Same conservative rotation-key set as the public reciprocal divider.
        rotations.push_back(1);
        rotations.push_back(-1);
        rotations.push_back(static_cast<int32_t>(ks256gpu::kWordBits - 1));
        rotations.push_back(static_cast<int32_t>(ks256gpu::kWordBits));
        for (uint32_t layer = 0; layer <= ks256gpu::kNumLayers; ++layer) {
            rotations.push_back(-static_cast<int32_t>(1U << layer));
        }
        std::sort(rotations.begin(), rotations.end());
        rotations.erase(std::unique(rotations.begin(), rotations.end()),
                        rotations.end());
        cc->EvalRotateKeyGen(keys.secretKey, rotations);
        phase.rotateKeyGenMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        cc->EvalBootstrapSetup(levelBudget, bsgsDim, numSlots, 0);
        phase.bootstrapSetupMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);
        phase.bootstrapKeyGenMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        cc->LoadContext(keys.publicKey);
        cc->Synchronize();
        phase.loadContextMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        auto masks = ks256gpu::BuildAndLoadVaultClaimSharesMasks(
            cc, directionSign, rateScaleBits);
        phase.maskBuildLoadMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        std::vector<double> depositSlots(numSlots, 0.0);
        std::vector<uint8_t> depositBits(numSlots, 0);
        std::vector<cpp_int> deposits(ks256gpu::kMultiplierNumWords, 0);

        std::mt19937_64 rng(randomSeed);
        for (uint32_t user = 0; user < ks256gpu::kMultiplierNumWords; ++user) {
            deposits[user] = RandomUint256FullWidth(rng);
        }

        // Deterministic sample users, kept similar to the divider test style.
        deposits[0] = 0;
        deposits[1] = 1;
        deposits[2] = Uint256Mask();
        deposits[3] = TwoPow(255);
        deposits[4] = publicRateScale;
        deposits[5] = publicExchangeRate;
        for (auto& value : deposits) {
            value &= Uint256Mask();
        }

        for (uint32_t user = 0; user < ks256gpu::kMultiplierNumWords; ++user) {
            WriteUint256ToPackedSlots(deposits[user], user, &depositBits, &depositSlots);
        }

        std::vector<uint8_t> expectedShares;
        BuildTrueVaultClaimBits(deposits, publicExchangeRate,
                                publicRateScale, &expectedShares);
        phase.inputBuildMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        Plaintext pDeposits = cc->MakeCKKSPackedPlaintext(
            depositSlots, 1, 0, nullptr, numSlots);
        pDeposits->SetLength(numSlots);
        phase.encodingMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        auto cDeposits = cc->Encrypt(keys.publicKey, pDeposits);
        cc->Synchronize();
        phase.encryptionMs = Milliseconds(begin, Clock::now());

        std::cout << "\n输入样例:\n";
        for (uint32_t user = 0;
             user < std::min(kPrintSampleUsers, ks256gpu::kMultiplierNumWords);
             ++user) {
            const cpp_int productLow =
                (deposits[user] * publicExchangeRate) & Uint256Mask();
            const cpp_int shares = productLow / publicRateScale;
            std::cout << "  user " << user << " deposit        = "
                      << ToHex256(deposits[user]) << "\n";
            std::cout << "  user " << user << " productLow     = "
                      << ToHex256(productLow) << "\n";
            std::cout << "  user " << user << " expectedShares = "
                      << ToHex256(shares) << "\n";
        }

        begin = Clock::now();
        ks256gpu::VaultClaimSharesTimingReport claimTiming;
        auto cShares = ks256gpu::EvalVaultClaimSharesPublicKS(
            cc,
            cDeposits,
            exchangeRateBits,
            rateScaleBits,
            rateScaleReciprocalBits,
            directionSign,
            masks,
            &claimTiming,
            compressorBatchSize,
            multiplicativeDepth);
        cc->Synchronize();
        phase.claimSharesMs = Milliseconds(begin, Clock::now());

        std::cout << "\nConfidential Vault claim-shares 输出密文状态:\n";
        std::cout << "  GetLevel()         : " << cShares->GetLevel() << "\n";
        std::cout << "  GetNoiseScaleDeg() : "
                  << cShares->GetNoiseScaleDeg() << "\n";
        std::cout << "  估算剩余层数      : "
                  << LevelsRemaining(multiplicativeDepth, cShares) << "\n";

        begin = Clock::now();
        Plaintext decrypted;
        const auto decryptResult = cc->Decrypt(keys.secretKey, cShares,
                                               &decrypted);
        if (!decryptResult.isValid) {
            throw std::runtime_error("Decryption returned invalid result");
        }
        decrypted->SetLength(numSlots);
        const auto values = decrypted->GetRealPackedValue();
        phase.decryptionMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        VerificationStats shareStats = VerifyBits(values, expectedShares);
        phase.verificationMs = Milliseconds(begin, Clock::now());
        phase.totalMs = Milliseconds(programBegin, Clock::now());

        PrintVerificationStats(
            "Confidential Vault claim-shares 正确性验证",
            shareStats,
            decrypted->GetLogPrecision());
        std::cout << "\n说明：验证逻辑是在 CPU 明文侧对 128 个用户逐个计算 "
                     "shares = floor(((deposit * exchangeRate) mod 2^256) / rateScale)，"
                     "再逐 bit 对比密文电路解密结果。这里验证的是固定 uint256 "
                     "语义，与 TFHE-rs U256 scalar_mul + scalar_div benchmark 对齐。\n";

        PrintVaultProfile(claimTiming);
        PrintScalarMulProfile(claimTiming.scalarMul);
        if (publicRateScale != 1) {
            PrintReciprocalProfile(claimTiming.division);
        }

        std::cout << "\n================ 端到端时间分析 ================\n";
        PrintPhase("CryptoContext 生成与 Enable", phase.contextMs);
        PrintPhase("KeyGen", phase.keyGenMs);
        PrintPhase("EvalMultKeyGen", phase.multKeyGenMs);
        PrintPhase("旋转密钥生成", phase.rotateKeyGenMs);
        PrintPhase("EvalBootstrapSetup", phase.bootstrapSetupMs);
        PrintPhase("EvalBootstrapKeyGen", phase.bootstrapKeyGenMs);
        PrintPhase("LoadContext", phase.loadContextMs);
        PrintPhase("掩码编码并加载到 GPU", phase.maskBuildLoadMs);
        PrintPhase("输入生成与 CPU 参考结果", phase.inputBuildMs);
        PrintPhase("CKKS encoding", phase.encodingMs);
        PrintPhase("Encryption", phase.encryptionMs);
        PrintPhase("Vault claim-shares pipeline", phase.claimSharesMs);
        PrintPhase("Decryption", phase.decryptionMs);
        PrintPhase("Verification", phase.verificationMs);
        PrintPhase("Whole program", phase.totalMs);

        std::cout << "\n吞吐与摊销:\n";
        std::cout << "  并行用户数量                 : "
                  << ks256gpu::kMultiplierNumWords << "\n";
        std::cout << "  计算阶段摊销时间              : "
                  << std::fixed << std::setprecision(6)
                  << phase.claimSharesMs / ks256gpu::kMultiplierNumWords
                  << " ms / user\n";
        std::cout << "  scalar_mul 摊销时间           : "
                  << claimTiming.scalarMulMs / ks256gpu::kMultiplierNumWords
                  << " ms / user\n";
        std::cout << "  product bootstrap 摊销时间     : "
                  << claimTiming.productBootstrapMs / ks256gpu::kMultiplierNumWords
                  << " ms / user\n";
        std::cout << "  public_div 摊销时间            : "
                  << claimTiming.publicDivideMs / ks256gpu::kMultiplierNumWords
                  << " ms / user\n";

        return shareStats.bitFailures == 0 ? 0 : 1;
    }
    catch (const std::exception& ex) {
        std::cerr << "\n[ERROR] " << ex.what() << "\n";
        return 1;
    }
}

