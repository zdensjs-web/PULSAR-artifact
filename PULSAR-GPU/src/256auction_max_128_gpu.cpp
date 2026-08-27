#include <fideslib.hpp>

#include "auction_max_128_gpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace fideslib;

namespace {

using Clock = std::chrono::steady_clock;

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
    double auctionMaxMs      = 0.0;
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
    const uint32_t scalePenalty =
        ciphertext->GetNoiseScaleDeg() > 0
            ? ciphertext->GetNoiseScaleDeg() - 1
            : 0;

    if (configuredDepth <= level + scalePenalty) {
        return 0;
    }
    return configuredDepth - level - scalePenalty;
}

std::string DevicesToString(const std::vector<int>& devices) {
    std::ostringstream out;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << devices[i];
    }
    return out.str();
}

std::string BitsToHex(const std::vector<uint8_t>& bits,
                      uint32_t wordIndex) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string output(ks256gpu::kWordBits / 4, '0');
    const uint32_t base = wordIndex * ks256gpu::kStride;

    for (uint32_t nibble = 0; nibble < ks256gpu::kWordBits / 4;
         ++nibble) {
        uint8_t value = 0;
        for (uint32_t offset = 0; offset < 4; ++offset) {
            const uint32_t bit = nibble * 4 + offset;
            value |= static_cast<uint8_t>(bits[base + bit] << offset);
        }
        output[output.size() - 1 - nibble] = kHex[value];
    }
    return "0x" + output;
}

std::string WordBitsToHex(const std::vector<uint8_t>& wordBits) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string output(ks256gpu::kWordBits / 4, '0');

    for (uint32_t nibble = 0; nibble < ks256gpu::kWordBits / 4;
         ++nibble) {
        uint8_t value = 0;
        for (uint32_t offset = 0; offset < 4; ++offset) {
            const uint32_t bit = nibble * 4 + offset;
            value |= static_cast<uint8_t>(wordBits[bit] << offset);
        }
        output[output.size() - 1 - nibble] = kHex[value];
    }
    return "0x" + output;
}

bool WordGreaterThan(const std::vector<uint8_t>& bits,
                     uint32_t lhsWord,
                     uint32_t rhsWord) {
    const uint32_t lhsBase = lhsWord * ks256gpu::kStride;
    const uint32_t rhsBase = rhsWord * ks256gpu::kStride;

    for (int bit = static_cast<int>(ks256gpu::kWordBits) - 1;
         bit >= 0;
         --bit) {
        const uint8_t lhs = bits[lhsBase + static_cast<uint32_t>(bit)];
        const uint8_t rhs = bits[rhsBase + static_cast<uint32_t>(bit)];
        if (lhs != rhs) {
            return lhs > rhs;
        }
    }
    return false;
}

uint32_t FindMaxWordIndex(const std::vector<uint8_t>& bits) {
    uint32_t maxWord = 0;
    for (uint32_t word = 1; word < ks256gpu::kNumWords; ++word) {
        if (WordGreaterThan(bits, word, maxWord)) {
            maxWord = word;
        }
    }
    return maxWord;
}

std::vector<uint8_t> ExtractWordBits(const std::vector<uint8_t>& bits,
                                     uint32_t wordIndex) {
    std::vector<uint8_t> wordBits(ks256gpu::kWordBits, 0);
    const uint32_t base = wordIndex * ks256gpu::kStride;
    for (uint32_t bit = 0; bit < ks256gpu::kWordBits; ++bit) {
        wordBits[bit] = bits[base + bit];
    }
    return wordBits;
}

void BuildExpectedBroadcastMax(const std::vector<uint8_t>& bids,
                               std::vector<uint8_t>* expected,
                               uint32_t* maxWordIndex) {
    *maxWordIndex = FindMaxWordIndex(bids);
    const auto maxWordBits = ExtractWordBits(bids, *maxWordIndex);

    expected->assign(ks256gpu::kTotalSlots, 0);
    for (uint32_t word = 0; word < ks256gpu::kNumWords; ++word) {
        const uint32_t base = word * ks256gpu::kStride;
        for (uint32_t bit = 0; bit < ks256gpu::kWordBits; ++bit) {
            (*expected)[base + bit] = maxWordBits[bit];
        }
    }
}

void WriteWordFromBits(std::vector<uint8_t>* bits,
                       std::vector<double>* slots,
                       uint32_t word,
                       const std::vector<uint8_t>& wordBits) {
    const uint32_t base = word * ks256gpu::kStride;
    for (uint32_t bit = 0; bit < ks256gpu::kWordBits; ++bit) {
        const uint8_t value = wordBits[bit];
        (*bits)[base + bit] = value;
        (*slots)[base + bit] = static_cast<double>(value);
    }
    for (uint32_t guard = 0; guard < ks256gpu::kGuardBits; ++guard) {
        const uint32_t index = base + ks256gpu::kWordBits + guard;
        (*bits)[index] = 0;
        (*slots)[index] = 0.0;
    }
}

std::vector<uint8_t> WordAllZeros() {
    return std::vector<uint8_t>(ks256gpu::kWordBits, 0);
}

std::vector<uint8_t> WordAllOnes() {
    return std::vector<uint8_t>(ks256gpu::kWordBits, 1);
}

std::vector<uint8_t> WordAllOnesMinusOne() {
    auto word = WordAllOnes();
    word[0] = 0;
    return word;
}

std::vector<uint8_t> WordPowerOfTwo(uint32_t bitIndex) {
    std::vector<uint8_t> word(ks256gpu::kWordBits, 0);
    word.at(bitIndex) = 1;
    return word;
}

void PrintPhase(const std::string& name, double milliseconds) {
    std::cout << "  " << std::left << std::setw(40) << name
              << std::right << std::setw(14)
              << std::fixed << std::setprecision(3)
              << milliseconds << " ms"
              << "  (" << std::setprecision(6)
              << milliseconds / 1000.0 << " s)\n";
}

void PrintAuctionProfile(const ks256gpu::AuctionTimingReport& profile) {
    std::cout << "\n================ GPU Auction Max-128 内部计时 ================\n";

    double roundsTotal = 0.0;
    const auto offsets = ks256gpu::AuctionReductionOffsets();

    for (uint32_t round = 0; round < ks256gpu::kAuctionReductionRounds;
         ++round) {
        const auto& item = profile.rounds[round];
        roundsTotal += item.wallMs;

        std::cout << "\n  Round " << round
                  << "  (offset=" << offsets[round]
                  << " words, rotate="
                  << offsets[round] * ks256gpu::kStride
                  << " slots)\n";
        PrintPhase("    Word-level Rotate", item.rotateMs);
        PrintPhase("    Packed GT(shifted > current)", item.compareMs);
        PrintPhase("    Condition bit0 boot+project", item.conditionBootstrapMs);
        PrintPhase("    Broadcast condition bit0", item.broadcastMs);
        PrintPhase("    CMUX 选择较大值", item.cmuxMs);
        PrintPhase("    Round boot+project", item.bootstrapMs);
        PrintPhase("    Active 明文掩码乘法(no boot)", item.activeMaskMs);
        PrintPhase("    本轮总墙钟时间", item.wallMs);
    }

    std::cout << "\n";
    PrintPhase("7 轮归约合计", roundsTotal);
    PrintPhase("Word-level Rotate 合计", profile.rotateTotalMs);
    PrintPhase("Packed GT 合计", profile.compareTotalMs);
    PrintPhase("Condition boot+project 合计", profile.conditionBootstrapTotalMs);
    PrintPhase("Broadcast 合计", profile.broadcastTotalMs);
    PrintPhase("CMUX 合计", profile.cmuxTotalMs);
    PrintPhase("Round boot+project 合计", profile.bootstrapTotalMs);
    PrintPhase("Active mask 合计", profile.activeMaskTotalMs);
    PrintPhase("Final boot+project", profile.finalBootstrapMs);
    PrintPhase("Final active mask(no longer used)", profile.finalActiveMaskMs);
    PrintPhase("EvalAuctionMax128 总时间", profile.totalMs);
}

}  // namespace

int main() {
    try {
        const auto programBegin = Clock::now();
        PhaseTimes phase;

        constexpr uint32_t ringDim = 1U << 17;   // 131072
        constexpr uint32_t numSlots = ks256gpu::kTotalSlots;  // 65536
        constexpr uint32_t multiplicativeDepth = 41;
        constexpr uint64_t randomSeed = 0x4B53415543544D58ULL;  // "KSAUCTMX"

        const SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
        const std::vector<uint32_t> levelBudget = {4, 4};
        const std::vector<uint32_t> bsgsDim = {0, 0};
        const std::vector<int> devices{0, 1, 2};

        std::cout << "============================================================\n";
        std::cout << "FIDESlib GPU: 128 x 256-bit Auction Max 归约 + Boolean 投影\n";
        std::cout << "============================================================\n";

        // --------------------------------------------------------------
        // 1. Create the FIDESlib GPU-enabled CKKS context.
        // --------------------------------------------------------------
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
        parameters.SetDevices(std::vector<int>(devices));
        parameters.SetPlaintextAutoload(false);
        parameters.SetCiphertextAutoload(true);

        auto begin = Clock::now();
        CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);
        cc->Enable(FHE);
        phase.contextMs = Milliseconds(begin, Clock::now());

        std::cout << "参数:\n";
        std::cout << "  Ring dimension              : "
                  << cc->GetRingDimension() << "\n";
        std::cout << "  CKKS slots                  : " << numSlots << "\n";
        std::cout << "  Packed encrypted bids       : "
                  << ks256gpu::kAuctionNumBids << "\n";
        std::cout << "  Slots per bid               : "
                  << ks256gpu::kStride
                  << " (256 active + 256 guard)\n";
        std::cout << "  Reduction rounds            : "
                  << ks256gpu::kAuctionReductionRounds << "\n";
        std::cout << "  Boot+project condition bits : "
                  << (ks256gpu::kAuctionBootstrapConditionEachRound ? "yes" : "no") << "\n";
        std::cout << "  Boot+project round output  : "
                  << (ks256gpu::kAuctionBootstrapEachRound ? "yes" : "no") << "\n";
        std::cout << "  Final boot+project output  : "
                  << (ks256gpu::kAuctionFinalBootstrap ? "yes" : "no") << "\n";
        std::cout << "  Configured mult depth       : "
                  << multiplicativeDepth << "\n";
        std::cout << "  Bootstrap level budget      : {"
                  << levelBudget[0] << ", " << levelBudget[1] << "}\n";
        std::cout << "  FIDESlib visible devices    : "
                  << DevicesToString(devices) << "\n";

        // --------------------------------------------------------------
        // 2. Generate every evaluation key BEFORE LoadContext.
        // --------------------------------------------------------------
        begin = Clock::now();
        auto keys = cc->KeyGen();
        phase.keyGenMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        cc->EvalMultKeyGen(keys.secretKey);
        phase.multKeyGenMs = Milliseconds(begin, Clock::now());

        std::vector<int32_t> rotationIndices;
        rotationIndices.reserve(
            3 * ks256gpu::kNumLayers + ks256gpu::kAuctionReductionRounds);

        for (uint32_t layer = 0; layer < ks256gpu::kNumLayers; ++layer) {
            const int32_t step = static_cast<int32_t>(1U << layer);
            rotationIndices.push_back(step);   // GT prefix network
            rotationIndices.push_back(-step);  // Broadcast bit0 -> active word
        }
        for (const uint32_t offset : ks256gpu::AuctionReductionOffsets()) {
            rotationIndices.push_back(
                static_cast<int32_t>(offset * ks256gpu::kStride));
        }
        std::sort(rotationIndices.begin(), rotationIndices.end());
        rotationIndices.erase(
            std::unique(rotationIndices.begin(), rotationIndices.end()),
            rotationIndices.end());

        begin = Clock::now();
        cc->EvalRotateKeyGen(keys.secretKey, rotationIndices);
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
        auto masks = ks256gpu::BuildAndLoadMasks(cc);
        phase.maskBuildLoadMs = Milliseconds(begin, Clock::now());

        // --------------------------------------------------------------
        // 3. Generate 128 packed encrypted bids and the CPU reference max.
        // --------------------------------------------------------------
        begin = Clock::now();
        std::vector<uint8_t> bidBits(ks256gpu::kTotalSlots, 0);
        std::vector<double> bidSlots(ks256gpu::kTotalSlots, 0.0);

        std::mt19937_64 rng(randomSeed);
        for (uint32_t word = 0; word < ks256gpu::kNumWords; ++word) {
            const uint32_t base = word * ks256gpu::kStride;
            for (uint32_t bit = 0; bit < ks256gpu::kWordBits; ++bit) {
                bidBits[base + bit] = static_cast<uint8_t>(rng() & 1ULL);
                bidSlots[base + bit] = static_cast<double>(bidBits[base + bit]);
            }
        }

        // Deterministic corner cases.  word 73 is the unique maximum.
        WriteWordFromBits(&bidBits, &bidSlots, 0, WordAllZeros());
        WriteWordFromBits(&bidBits, &bidSlots, 1, WordPowerOfTwo(255));
        WriteWordFromBits(&bidBits, &bidSlots, 73, WordAllOnes());
        WriteWordFromBits(&bidBits, &bidSlots, 127, WordAllOnesMinusOne());

        std::vector<uint8_t> expected;
        uint32_t expectedMaxWord = 0;
        BuildExpectedBroadcastMax(bidBits, &expected, &expectedMaxWord);
        const auto expectedMaxBits = ExtractWordBits(bidBits, expectedMaxWord);
        phase.inputBuildMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        Plaintext pBids = cc->MakeCKKSPackedPlaintext(
            bidSlots, 1, 0, nullptr, numSlots);
        pBids->SetLength(numSlots);
        phase.encodingMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        auto cBids = cc->Encrypt(keys.publicKey, pBids);
        cc->Synchronize();
        phase.encryptionMs = Milliseconds(begin, Clock::now());

        std::cout << "\n输入样例:\n";
        std::cout << "  word 0 bid            = " << BitsToHex(bidBits, 0) << "\n";
        std::cout << "  word 1 bid            = " << BitsToHex(bidBits, 1) << "\n";
        std::cout << "  word 73 bid           = " << BitsToHex(bidBits, 73) << "\n";
        std::cout << "  word 127 bid          = " << BitsToHex(bidBits, 127) << "\n";
        std::cout << "  CPU max source word   = " << expectedMaxWord << "\n";
        std::cout << "  CPU expected max      = "
                  << WordBitsToHex(expectedMaxBits) << "\n";

        // --------------------------------------------------------------
        // 4. GPU packed auction max reduction.
        // --------------------------------------------------------------
        ks256gpu::AuctionTimingReport auctionTiming;
        cc->Synchronize();
        begin = Clock::now();
        auto cMax = ks256gpu::EvalAuctionMax128(
            cc, cBids, masks, &auctionTiming,
            ks256gpu::kAuctionBootstrapEachRound,
            ks256gpu::kAuctionBootstrapConditionEachRound,
            ks256gpu::kAuctionFinalBootstrap);
        cc->Synchronize();
        phase.auctionMaxMs = Milliseconds(begin, Clock::now());

        std::cout << "\nAuction Max 后密文状态:\n";
        std::cout << "  GetLevel()         : " << cMax->GetLevel() << "\n";
        std::cout << "  GetNoiseScaleDeg() : "
                  << cMax->GetNoiseScaleDeg() << "\n";
        std::cout << "  估算剩余层数      : "
                  << LevelsRemaining(multiplicativeDepth, cMax) << "\n";

        // --------------------------------------------------------------
        // 5. Copy to CPU and decrypt.
        // --------------------------------------------------------------
        begin = Clock::now();
        Plaintext decrypted;
        const auto decryptResult =
            cc->Decrypt(keys.secretKey, cMax, &decrypted);
        if (!decryptResult.isValid) {
            throw std::runtime_error("Decryption returned an invalid result");
        }
        decrypted->SetLength(numSlots);
        const std::vector<double> values = decrypted->GetRealPackedValue();
        phase.decryptionMs = Milliseconds(begin, Clock::now());

        // --------------------------------------------------------------
        // 6. Exact bit recovery and empirical CKKS precision analysis.
        // --------------------------------------------------------------
        begin = Clock::now();
        double maxAbsError = 0.0;
        double sumAbsError = 0.0;
        double sumSquaredError = 0.0;
        double maxGuardAbs = 0.0;
        uint64_t bitFailures = 0;
        uint32_t worstWord = 0;
        uint32_t worstBit = 0;
        double worstValue = 0.0;
        double worstExpected = 0.0;

        const uint64_t activeCount =
            static_cast<uint64_t>(ks256gpu::kNumWords) *
            ks256gpu::kWordBits;

        for (uint32_t word = 0; word < ks256gpu::kNumWords; ++word) {
            const uint32_t base = word * ks256gpu::kStride;

            for (uint32_t bit = 0; bit < ks256gpu::kWordBits; ++bit) {
                const uint32_t index = base + bit;
                const double target = static_cast<double>(expected[index]);
                const double error = std::abs(values[index] - target);

                sumAbsError += error;
                sumSquaredError += error * error;

                if (error > maxAbsError) {
                    maxAbsError = error;
                    worstWord = word;
                    worstBit = bit;
                    worstValue = values[index];
                    worstExpected = target;
                }

                const uint8_t recovered = values[index] >= 0.5 ? 1 : 0;
                if (recovered != expected[index]) {
                    ++bitFailures;
                }
            }

            for (uint32_t guard = 0; guard < ks256gpu::kGuardBits;
                 ++guard) {
                const uint32_t index =
                    base + ks256gpu::kWordBits + guard;
                maxGuardAbs =
                    std::max(maxGuardAbs, std::abs(values[index]));
            }
        }

        const double meanAbsError =
            sumAbsError / static_cast<double>(activeCount);
        const double rmse = std::sqrt(
            sumSquaredError / static_cast<double>(activeCount));

        phase.verificationMs = Milliseconds(begin, Clock::now());
        phase.totalMs = Milliseconds(programBegin, Clock::now());

        std::cout << "\n================ 解密正确性与经验精度 ================\n";
        std::cout << std::scientific << std::setprecision(10);
        std::cout << "  有效 bit 槽数                 : "
                  << activeCount << "\n";
        std::cout << "  最大绝对误差                 : "
                  << maxAbsError << "\n";
        std::cout << "  平均 L1 误差                 : "
                  << meanAbsError << "\n";
        std::cout << "  RMSE                         : " << rmse << "\n";
        std::cout << "  Guard 槽最大绝对值           : "
                  << maxGuardAbs << "\n";

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  GetLogPrecision              : "
                  << decrypted->GetLogPrecision() << " bits\n";
        std::cout << "  最差精度 -log2(max error)    : "
                  << PrecisionBits(maxAbsError) << " bits\n";
        std::cout << "  平均精度 -log2(mean L1)      : "
                  << PrecisionBits(meanAbsError) << " bits\n";
        std::cout << "  RMSE 精度 -log2(RMSE)        : "
                  << PrecisionBits(rmse) << " bits\n";
        std::cout << "  0.5 阈值恢复失败数           : "
                  << bitFailures << "\n";
        std::cout << "  最差位置                     : word="
                  << worstWord << ", bit=" << worstBit << "\n";
        std::cout << std::scientific << std::setprecision(12);
        std::cout << "  最差位置期望/解密值          : "
                  << worstExpected << " / " << worstValue << "\n";
        std::cout << std::fixed;
        std::cout << "  整体结果                     : "
                  << (bitFailures == 0 ? "PASS" : "FAIL") << "\n";
        std::cout << "说明：上述 bit 数是 CKKS 解密误差对应的经验精度，"
                     "不是 BFV/BGV 的精确噪声预算。\n";

        PrintAuctionProfile(auctionTiming);

        std::cout << "\n================ 端到端时间分析 ================\n";
        PrintPhase("CryptoContext 生成与 Enable", phase.contextMs);
        PrintPhase("KeyGen", phase.keyGenMs);
        PrintPhase("EvalMultKeyGen", phase.multKeyGenMs);
        PrintPhase("Auction/GT/Broadcast 旋转密钥生成", phase.rotateKeyGenMs);
        PrintPhase("EvalBootstrapSetup", phase.bootstrapSetupMs);
        PrintPhase("EvalBootstrapKeyGen", phase.bootstrapKeyGenMs);
        PrintPhase("LoadContext 与评估密钥加载到 GPU", phase.loadContextMs);
        PrintPhase("掩码编码并加载到 GPU", phase.maskBuildLoadMs);
        PrintPhase("输入生成与 CPU 参考结果", phase.inputBuildMs);
        PrintPhase("CKKS 编码", phase.encodingMs);
        PrintPhase("输入加密并加载到 GPU", phase.encryptionMs);
        PrintPhase("GPU Auction Max-128", phase.auctionMaxMs);
        PrintPhase("复制回 CPU 并解密", phase.decryptionMs);
        PrintPhase("正确性与误差扫描", phase.verificationMs);
        PrintPhase("程序总墙钟时间", phase.totalMs);

        const double onlineSeconds = phase.auctionMaxMs / 1000.0;

        std::cout << "\n================ 吞吐与摊销 ================\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  在线总时间/128-bid auction : "
                  << onlineSeconds << " s\n";
        std::cout << "  在线总摊销时间/bid         : "
                  << onlineSeconds / ks256gpu::kAuctionNumBids << " s\n";

        if (onlineSeconds > 0.0) {
            std::cout << "  Auction Max 吞吐量         : "
                      << 1.0 / onlineSeconds
                      << " auction/s\n";
            std::cout << "  Bid 摊销吞吐量             : "
                      << ks256gpu::kAuctionNumBids / onlineSeconds
                      << " bid/s\n";
        }

        return bitFailures == 0 ? 0 : 2;
    }
    catch (const std::exception& error) {
        std::cerr << "\n[FATAL] " << error.what() << "\n";
        return 1;
    }
}
