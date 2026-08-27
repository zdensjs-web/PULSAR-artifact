#include "scalar_mul_public_pc_ks_gpu.h"

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

using namespace fideslib;
using boost::multiprecision::cpp_int;

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint32_t kPrintSampleWords = 6;

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
    double scalarMulMs       = 0.0;
    double bootstrapMs       = 0.0;
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

std::string BitsToHex(const std::vector<uint8_t>& bits, uint32_t wordIndex) {
    return ToHex256(ReadPackedBitsAsUint256(bits, wordIndex));
}

void BuildTrueScalarMulBits(const std::vector<cpp_int>& inputs,
                            const cpp_int& scalar,
                            std::vector<uint8_t>* expected) {
    expected->assign(ks256gpu::kMultiplierTotalSlots, 0);
    const cpp_int mask = Uint256Mask();
    for (uint32_t word = 0; word < ks256gpu::kMultiplierNumWords; ++word) {
        const cpp_int product = (inputs[word] * scalar) & mask;
        const auto productBits = BitsFromUint256(product);
        const uint32_t base = word * ks256gpu::kStride;
        for (uint32_t bit = 0; bit < ks256gpu::kWordBits; ++bit) {
            (*expected)[base + bit] = productBits[bit];
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
    PrintPhase("public scalar rows + dynamic Dadda", profile.rowGenerationMs);
    PrintPhase("output active-256 mask", profile.outputMaskMs);
    PrintPhase("final bootstrap + projection", profile.finalBootstrapMs);
    PrintPhase("total scalar_mul core", profile.totalMs);

    std::cout << "\n  行生成统计:\n";
    std::cout << "    Scalar one bits              : "
              << profile.scalarOneBits << "\n";
    std::cout << "    Scalar zero bits             : "
              << profile.scalarZeroBits << "\n";
    std::cout << "    Pushed dynamic Dadda rows    : "
              << profile.pushedRows << "\n";
    std::cout << "    Dynamic Dadda initial rows  : "
              << profile.dynamicDaddaInitialRows << "\n";
    std::cout << "    Dynamic Dadda stages        : "
              << profile.dynamicDaddaStages << "\n";
    std::cout << "    Public row rotations         : "
              << profile.publicRowRotations << "\n";
    std::cout << "    Public row mask multipliers  : "
              << profile.publicRowMaskMults << "\n";

    std::cout << "\n  Dadda / KS 统计:\n";
    std::cout << "    Compressor ciphertext mults  : "
              << profile.multiplier.compressorCipherMults << "\n";
    std::cout << "    Compressor carry rotations   : "
              << profile.multiplier.compressorCarryRotations << "\n";
    std::cout << "    Streaming synchronizations   : "
              << profile.multiplier.streamingSynchronizations << "\n";
    std::cout << "    Peak buffered rows           : "
              << profile.multiplier.peakBufferedRows << "\n";
    std::cout << "    Final KS total               : "
              << std::fixed << std::setprecision(3)
              << profile.multiplier.finalAdder.totalMs << " ms\n";
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
    std::cout << "  失败 packed words             : "
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
    std::cout << "  最差位置                     : word="
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

}  // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::unitbuf;
        const auto programBegin = Clock::now();
        PhaseTimes phase;

        std::string scalarText = "7";
        if (const char* envS = std::getenv("KS256_PUBLIC_SCALAR")) {
            if (*envS) {
                scalarText = envS;
            }
        }
        if (argc >= 2) {
            scalarText = argv[1];
        }

        const cpp_int scalar = ParseUint256(scalarText);
        const auto scalarBits = BitsFromUint256(scalar);

        constexpr uint32_t ringDim = 1U << 17;
        constexpr uint32_t numSlots = ks256gpu::kMultiplierTotalSlots;
        constexpr bool enableBootstrap = true;
        constexpr uint32_t multiplicativeDepth = 40U;
        constexpr int directionSign = -1;
        constexpr uint64_t randomSeed = 0x5343414c41524d55ULL;  // "SCALARMU"
        const uint32_t compressorBatchSize = EnvUint32(
            "KS256_COMPRESSOR_BATCH", ks256gpu::kDefaultCompressorBatchSize);

        const SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
        const std::vector<uint32_t> levelBudget = {4, 4};
        const std::vector<uint32_t> bsgsDim = {0, 0};

        std::cout << "============================================================\n";
        std::cout << "FIDESlib GPU: " << ks256gpu::kMultiplierNumWords
                  << " x uint256 public scalar_mul (low 256-bit)\n";
        std::cout << "Boolean encoding; 256 active slots + 256 guard slots per word\n";
        std::cout << "============================================================\n";
        std::cout << "参数:\n";
        std::cout << "  Ring dimension              : " << ringDim << "\n";
        std::cout << "  CKKS slots                  : " << numSlots << "\n";
        std::cout << "  Parallel uint256 words      : "
                  << ks256gpu::kMultiplierNumWords << "\n";
        std::cout << "  Slots per word              : "
                  << ks256gpu::kStride
                  << " (256 active + 256 guard)\n";
        std::cout << "  Public scalar S             : "
                  << ToHex256(scalar) << "\n";
        std::cout << "  Scalar one bits             : "
                  << ks256gpu::public_reciprocal_detail::CountOneBits(scalarBits)
                  << "\n";
        std::cout << "  Configured mult depth       : "
                  << multiplicativeDepth << "\n";
        std::cout << "  Bootstrap level budget      : {4, 4}\n";
        std::cout << "  HYBRID large digits         : 3\n";
        std::cout << "  Compressor batch size       : "
                  << compressorBatchSize << "\n";
        std::cout << "  Output                      : encrypted uint256 (N * S) mod 2^256\n";
        std::cout << "  Correctness validation      : CPU uint256 ((N*S) mod 2^256) vs decrypted product\n";

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
        parameters.SetDevices(std::vector<int>{0, 1});
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
        // Keep the same conservative rotation-key set as the public divider.
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
        auto masks = ks256gpu::BuildAndLoadPublicScalarMulMasks(
            cc, directionSign, scalarBits);
        phase.maskBuildLoadMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        std::vector<double> slotsN(numSlots, 0.0);
        std::vector<uint8_t> bitsN(numSlots, 0);
        std::vector<cpp_int> inputs(ks256gpu::kMultiplierNumWords, 0);

        std::mt19937_64 rng(randomSeed);
        auto randomUint256 = [&]() {
            cpp_int value = 0;
            for (uint32_t limb = 0; limb < 4; ++limb) {
                value <<= 64;
                value += rng();
            }
            return value & Uint256Mask();
        };

        for (uint32_t word = 0; word < ks256gpu::kMultiplierNumWords; ++word) {
            inputs[word] = randomUint256();
        }
        inputs[0] = 100;
        inputs[1] = Uint256Mask();
        inputs[2] = 0;
        inputs[3] = 1;
        inputs[4] = TwoPow(255);
        inputs[5] = scalar;
        for (auto& value : inputs) {
            value &= Uint256Mask();
        }

        for (uint32_t word = 0; word < ks256gpu::kMultiplierNumWords; ++word) {
            WriteUint256ToPackedSlots(inputs[word], word, &bitsN, &slotsN);
        }

        std::vector<uint8_t> expectedProduct;
        BuildTrueScalarMulBits(inputs, scalar, &expectedProduct);
        phase.inputBuildMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        Plaintext pN = cc->MakeCKKSPackedPlaintext(slotsN, 1, 0, nullptr,
                                                   numSlots);
        pN->SetLength(numSlots);
        phase.encodingMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        auto cN = cc->Encrypt(keys.publicKey, pN);
        cc->Synchronize();
        phase.encryptionMs = Milliseconds(begin, Clock::now());

        std::cout << "\n输入样例:\n";
        for (uint32_t word = 0;
             word < std::min(kPrintSampleWords, ks256gpu::kMultiplierNumWords);
             ++word) {
            std::cout << "  word " << word << " N        = "
                      << ToHex256(inputs[word]) << "\n";
            std::cout << "  word " << word << " N*S mod 2^256 = "
                      << ToHex256((inputs[word] * scalar) & Uint256Mask())
                      << "\n";
            std::cout << "  word " << word << " expected P= "
                      << ToHex256((inputs[word] * scalar) & Uint256Mask())
                      << "\n";
        }

        begin = Clock::now();
        ks256gpu::PublicScalarMulTimingReport scalarMulTiming;
        auto cResult = ks256gpu::EvalScalarMulPublic256LowKS(
            cc, cN, scalarBits, directionSign, masks,
            &scalarMulTiming, compressorBatchSize);
        cc->Synchronize();
        phase.scalarMulMs = Milliseconds(begin, Clock::now());

        std::cout << "\nscalar_mul 输出密文状态:\n";
        std::cout << "  GetLevel()         : " << cResult->GetLevel() << "\n";
        std::cout << "  GetNoiseScaleDeg() : "
                  << cResult->GetNoiseScaleDeg() << "\n";
        std::cout << "  估算剩余层数      : "
                  << LevelsRemaining(multiplicativeDepth, cResult) << "\n";

        auto cBoot = cResult;
        if (enableBootstrap) {
            begin = Clock::now();
            cBoot = ks256gpu::BootstrapProjectActive256(cc, cResult, masks);
            cc->Synchronize();
            phase.bootstrapMs = Milliseconds(begin, Clock::now());
            scalarMulTiming.finalBootstrapMs = phase.bootstrapMs;
            std::cout << "\n最终乘积 Bootstrap + active projection 后密文状态:\n";
            std::cout << "  GetLevel()         : " << cBoot->GetLevel() << "\n";
            std::cout << "  GetNoiseScaleDeg() : "
                      << cBoot->GetNoiseScaleDeg() << "\n";
            std::cout << "  估算剩余层数      : "
                      << LevelsRemaining(multiplicativeDepth, cBoot) << "\n";
        }

        begin = Clock::now();
        Plaintext decrypted;
        const auto decryptResult = cc->Decrypt(keys.secretKey, cBoot,
                                               &decrypted);
        if (!decryptResult.isValid) {
            throw std::runtime_error("Decryption returned invalid result");
        }
        decrypted->SetLength(numSlots);
        const auto values = decrypted->GetRealPackedValue();
        phase.decryptionMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        VerificationStats productStats = VerifyBits(values, expectedProduct);
        phase.verificationMs = Milliseconds(begin, Clock::now());
        phase.totalMs = Milliseconds(programBegin, Clock::now());

        PrintVerificationStats("uint256 明文标量乘法 (N*S mod 2^256) 正确性验证",
                               productStats,
                               decrypted->GetLogPrecision());
        std::cout << "\n说明：验证逻辑是在 CPU 明文侧直接计算每个 256-bit N 与同一个"
                     "public uint256 S 的乘积低 256 bit，即 (N*S) mod 2^256，"
                     "再逐 bit 对比密文电路解密结果。这里验证的是 Solidity/EVM"
                     " 风格的固定 256-bit 标量乘法语义。\n";

        PrintScalarMulProfile(scalarMulTiming);

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
        PrintPhase("Public scalar_mul low-256", phase.scalarMulMs);
        PrintPhase("Final bootstrap", phase.bootstrapMs);
        PrintPhase("Decryption", phase.decryptionMs);
        PrintPhase("Verification", phase.verificationMs);
        PrintPhase("Whole program", phase.totalMs);

        std::cout << "\n吞吐与摊销:\n";
        std::cout << "  并行 uint256 数量             : "
                  << ks256gpu::kMultiplierNumWords << "\n";
        std::cout << "  计算阶段摊销时间              : "
                  << std::fixed << std::setprecision(6)
                  << phase.scalarMulMs / ks256gpu::kMultiplierNumWords
                  << " ms / uint256\n";
        std::cout << "  含 bootstrap 摊销时间          : "
                  << (phase.scalarMulMs + phase.bootstrapMs) /
                         ks256gpu::kMultiplierNumWords
                  << " ms / uint256\n";

        return productStats.bitFailures == 0 ? 0 : 1;
    }
    catch (const std::exception& ex) {
        std::cerr << "\n[ERROR] " << ex.what() << "\n";
        return 1;
    }
}
