#include <fideslib.hpp>

#include "cipher_divider_ks_runtime.h"

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fideslib;

namespace {

using boost::multiprecision::cpp_int;

struct PhaseTimes {
    double contextMs = 0.0;
    double keyGenMs = 0.0;
    double multKeyGenMs = 0.0;
    double rotateKeyGenMs = 0.0;
    double bootstrapSetupMs = 0.0;
    double bootstrapKeyGenMs = 0.0;
    double loadContextMs = 0.0;
    double inputBuildMs = 0.0;
    double encodingMs = 0.0;
    double encryptionMs = 0.0;
    double divisionWallMs = 0.0;
    double divisionOnlineMs = 0.0;
    double decryptionMs = 0.0;
    double verificationMs = 0.0;
    double totalMs = 0.0;
};

struct Inputs {
    std::vector<uint8_t> dividend;
    std::vector<uint8_t> divisor;
    std::vector<uint8_t> expectedQuotient;
    std::vector<uint8_t> expectedRemainder;
    std::vector<double> encodedDividend;
    std::vector<double> encodedDivisor;
};

struct ErrorReport {
    double maxError = 0.0;
    double meanError = 0.0;
    double rmse = 0.0;
    double maxGuard = 0.0;
    double maxImaginary = 0.0;
    uint64_t failures = 0;
    uint32_t worstWord = 0;
    uint32_t worstBit = 0;
    double worstExpected = 0.0;
    double worstActual = 0.0;
};

void PrintPhase(const std::string& name, double milliseconds) {
    std::cout << "  " << std::left << std::setw(50) << name
              << std::right << std::setw(13)
              << std::fixed << std::setprecision(3)
              << milliseconds << " ms  ("
              << std::setprecision(6)
              << milliseconds / 1000.0 << " s)\n";
}

void StoreWord(const ksdivcipher::RuntimeConfig& config,
               uint32_t word,
               cpp_int value,
               std::vector<uint8_t>* bits) {
    for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
        (*bits)[ksword::Slot(config, word, bit)] =
            static_cast<uint8_t>((value & 1) != 0);
        value >>= 1;
    }
}

cpp_int RandomWord(std::mt19937_64* generator,
                   uint32_t bits) {
    cpp_int value = 0;
    for (uint32_t offset = 0; offset < bits; offset += 64U) {
        value |= cpp_int((*generator)()) << offset;
    }
    return value & ((cpp_int(1) << bits) - 1);
}

Inputs BuildInputs(const ksdivcipher::RuntimeConfig& config) {
    Inputs input;
    input.dividend.assign(config.slots, 0);
    input.divisor.assign(config.slots, 0);
    input.expectedQuotient.assign(config.slots, 0);
    input.expectedRemainder.assign(config.slots, 0);
    input.encodedDividend.assign(config.slots, 0.0);
    input.encodedDivisor.assign(config.slots, 0.0);

    const cpp_int maximum =
        (cpp_int(1) << config.wordBits) - 1;
    std::mt19937_64 generator(0x4349504845524449ULL);
    for (uint32_t word = 0; word < config.words; ++word) {
        cpp_int dividend =
            RandomWord(&generator, config.wordBits);
        cpp_int divisor =
            RandomWord(&generator, config.wordBits);
        if (word == 0) {
            dividend = maximum;
            divisor = 1;
        }
        else if (word == 1) {
            dividend = maximum;
            divisor = maximum;
        }
        else if (word == 2) {
            dividend = 1;
            divisor = 0;
        }
        else if (word == 3) {
            dividend = maximum - 1;
            divisor = maximum;
        }

        const cpp_int quotient =
            divisor == 0 ? cpp_int(0) : dividend / divisor;
        const cpp_int remainder =
            divisor == 0 ? dividend : dividend % divisor;
        StoreWord(
            config, word, dividend, &input.dividend);
        StoreWord(
            config, word, divisor, &input.divisor);
        StoreWord(
            config,
            word,
            quotient,
            &input.expectedQuotient);
        StoreWord(
            config,
            word,
            remainder,
            &input.expectedRemainder);
    }

    for (uint32_t slot = 0; slot < config.slots; ++slot) {
        input.encodedDividend[slot] =
            static_cast<double>(input.dividend[slot]);
        input.encodedDivisor[slot] =
            static_cast<double>(input.divisor[slot]);
    }
    return input;
}

ErrorReport CheckResult(
    const ksdivcipher::RuntimeConfig& config,
    const std::vector<std::complex<double>>& values,
    const std::vector<uint8_t>& expected) {
    ErrorReport report;
    long double sumAbsolute = 0.0;
    long double sumSquared = 0.0;
    uint64_t count = 0;
    for (uint32_t word = 0; word < config.words; ++word) {
        for (uint32_t bit = 0;
             bit < config.wordBits;
             ++bit) {
            const uint32_t slot =
                ksword::Slot(config, word, bit);
            const double target =
                static_cast<double>(expected[slot]);
            const double actual = values[slot].real();
            const double error = std::abs(actual - target);
            if (error > report.maxError) {
                report.maxError = error;
                report.worstWord = word;
                report.worstBit = bit;
                report.worstExpected = target;
                report.worstActual = actual;
            }
            report.maxImaginary = std::max(
                report.maxImaginary,
                std::abs(values[slot].imag()));
            sumAbsolute += error;
            sumSquared +=
                static_cast<long double>(error) * error;
            if (!std::isfinite(actual) || error >= 0.5) {
                ++report.failures;
            }
            ++count;
        }
        for (uint32_t position = config.wordBits;
             position < config.stride;
             ++position) {
            report.maxGuard = std::max(
                report.maxGuard,
                std::abs(values[
                    ksword::Slot(config, word, position)]));
        }
    }
    report.meanError =
        static_cast<double>(sumAbsolute / count);
    report.rmse =
        std::sqrt(static_cast<double>(sumSquared / count));
    return report;
}

void PrintError(const char* label,
                const ErrorReport& report) {
    std::cout << "  " << label << "\n";
    std::cout << "    maximum absolute error : "
              << report.maxError << "\n";
    std::cout << "    mean absolute error    : "
              << report.meanError << "\n";
    std::cout << "    RMSE                   : "
              << report.rmse << "\n";
    std::cout << "    maximum guard value    : "
              << report.maxGuard << "\n";
    std::cout << "    maximum imaginary part : "
              << report.maxImaginary << "\n";
    std::cout << "    threshold failures     : "
              << report.failures << "\n";
    std::cout << "    worst location         : word="
              << report.worstWord << ", bit="
              << report.worstBit << "\n";
    std::cout << "    expected / actual      : "
              << report.worstExpected << " / "
              << report.worstActual << "\n";
}

void PrintConfiguration(
    const ksdivcipher::RuntimeConfig& config) {
    std::cout << "Runtime-configurable exact CKKS ciphertext divider\n";
    std::cout << "============================================================\n";
    std::cout << "backend                 : "
              << ksword::BackendName(config.backend) << "\n";
    std::cout << "word / internal bits    : "
              << config.wordBits << " / "
              << config.wideBits() << "\n";
    std::cout << "packed words/ciphertext : "
              << config.words << " (maximum "
              << config.maxWords() << ")\n";
    std::cout << "slots used / available  : "
              << config.usedSlots() << " / "
              << config.slots << "\n";
    std::cout << "algorithm               : exact radix-16 restoring\n";
    std::cout << "radix rounds / trials   : "
              << config.radixRounds() << " / "
              << 4U * config.radixRounds() << "\n";
    std::cout << "parallel setup blocks   : "
              << config.parallelBlocks << "\n";
    std::cout << "multiplicative depth    : "
              << config.multiplicativeDepth << "\n";
    std::cout << "candidate borrow path  : standard G/P with midpoint refresh\n";
    std::cout << "prefix split           : "
              << config.prefixLayersBeforeRefresh() << " + "
              << config.prefixLayersAfterRefresh() << " layers\n";
    std::cout << "trial bottom span      : "
              << config.trialLevelCost() << "\n";
    std::cout << "trial input / raw level: "
              << config.expectedTrialInputLevel() << " / "
              << config.expectedTrialOutputLevel() << "\n";
    std::cout << "Bootstrap budget/output : {"
              << ksdivcipher::kBootstrapEncodingLevelBudget << ","
              << ksdivcipher::kBootstrapDecodingLevelBudget << "} / "
              << ksdivcipher::kBootstrapOutputLevel << "/2\n";
    std::cout << "post-Boot projected state: "
              << ksdivcipher::kCanonicalStateLevel << "/2\n";
    std::cout << "level placement        : bottom before every Bootstrap\n";
    std::cout << "logical GPU devices     : "
              << ksword::DeviceList(config.devices) << "\n";
    std::cout << "division by zero        : quotient=0, remainder=dividend\n";
    std::cout << "Bootstrap policy        : Bootstrap then Boolean projection; all timed\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::unitbuf;
        ksdivcipher::RuntimeConfig config;
        try {
            config =
                ksdivcipher::ParseCommandLine(argc, argv);
        }
        catch (const std::runtime_error& error) {
            if (std::string(error.what()) == "help") {
                std::cout << ksdivcipher::Usage(argv[0]);
                return 0;
            }
            throw;
        }

        PrintConfiguration(config);
        const auto programBegin = ksword::Clock::now();
        PhaseTimes phase;
        const std::vector<uint32_t> levelBudget{
            ksdivcipher::kBootstrapEncodingLevelBudget,
            ksdivcipher::kBootstrapDecodingLevelBudget};
        const std::vector<uint32_t> bsgsDim{0, 0};

        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetSecretKeyDist(UNIFORM_TERNARY);
        parameters.SetSecurityLevel(HEStd_128_classic);
        parameters.SetRingDim(ksword::kRingDimension);
        parameters.SetBatchSize(config.slots);
        parameters.SetMultiplicativeDepth(
            config.multiplicativeDepth);
        parameters.SetScalingTechnique(FLEXIBLEAUTO);
        parameters.SetScalingModSize(59);
        parameters.SetFirstModSize(60);
        parameters.SetKeySwitchTechnique(HYBRID);
        parameters.SetNumLargeDigits(3);
        parameters.SetDevices(
            std::vector<int>(config.devices));
        parameters.SetPlaintextAutoload(false);
        parameters.SetCiphertextAutoload(true);

        auto begin = ksword::Clock::now();
        auto cc = GenCryptoContext(parameters);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);
        cc->Enable(FHE);
        phase.contextMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        begin = ksword::Clock::now();
        auto keys = cc->KeyGen();
        phase.keyGenMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        begin = ksword::Clock::now();
        cc->EvalMultKeyGen(keys.secretKey);
        phase.multKeyGenMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        const auto ordinary =
            ksdivcipher::OrdinaryRotationIndices(config);
        begin = ksword::Clock::now();
        cc->EvalRotateKeyGen(keys.secretKey, ordinary);
        phase.rotateKeyGenMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        begin = ksword::Clock::now();
        cc->EvalBootstrapSetup(
            levelBudget, bsgsDim, config.slots, 0);
        phase.bootstrapSetupMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        begin = ksword::Clock::now();
        cc->EvalBootstrapKeyGen(
            keys.secretKey, config.slots);
        phase.bootstrapKeyGenMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        begin = ksword::Clock::now();
        cc->LoadContext(keys.publicKey);
        cc->Synchronize();
        phase.loadContextMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        begin = ksword::Clock::now();
        const auto input = BuildInputs(config);
        phase.inputBuildMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        std::cout << "\nInput samples\n";
        const uint32_t samples =
            std::min(config.words, uint32_t{4});
        for (uint32_t word = 0; word < samples; ++word) {
            std::cout << "  word " << word << " A = "
                      << ksword::BitsToHex(
                             input.dividend, config, word)
                      << "\n";
            std::cout << "  word " << word << " D = "
                      << ksword::BitsToHex(
                             input.divisor, config, word)
                      << "\n";
            std::cout << "  word " << word << " Q = "
                      << ksword::BitsToHex(
                             input.expectedQuotient, config, word)
                      << "\n";
            std::cout << "  word " << word << " R = "
                      << ksword::BitsToHex(
                             input.expectedRemainder, config, word)
                      << "\n";
        }

        begin = ksword::Clock::now();
        auto pDividend = cc->MakeCKKSPackedPlaintext(
            input.encodedDividend, 1, 0, nullptr, config.slots);
        auto pDivisor = cc->MakeCKKSPackedPlaintext(
            input.encodedDivisor, 1, 0, nullptr, config.slots);
        pDividend->SetLength(config.slots);
        pDivisor->SetLength(config.slots);
        phase.encodingMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        begin = ksword::Clock::now();
        auto cDividend =
            cc->Encrypt(keys.publicKey, pDividend);
        auto cDivisor =
            cc->Encrypt(keys.publicKey, pDivisor);
        cc->Synchronize();
        phase.encryptionMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        ksdivcipher::TimingReport timing;
        cc->Synchronize();
        begin = ksword::Clock::now();
        auto result = ksdivcipher::EvalDivideEncrypted(
            cc,
            cDividend,
            cDivisor,
            config,
            &timing);
        cc->Synchronize();
        phase.divisionWallMs =
            ksword::Milliseconds(begin, ksword::Clock::now());
        phase.divisionOnlineMs = std::max(
            0.0,
            phase.divisionWallMs - timing.setupExcludedMs);

        std::cout << "\nFinal ciphertext state\n";
        std::cout << "  quotient level/scale  : "
                  << result.quotient->GetLevel() << "/"
                  << result.quotient->GetNoiseScaleDeg() << "\n";
        std::cout << "  remainder level/scale : "
                  << result.remainder->GetLevel() << "/"
                  << result.remainder->GetNoiseScaleDeg() << "\n";

        begin = ksword::Clock::now();
        Plaintext pQuotient;
        Plaintext pRemainder;
        try {
            const auto status = cc->Decrypt(
                keys.secretKey, result.quotient, &pQuotient);
            if (!status.isValid) {
                throw std::runtime_error(
                    "quotient decryption returned an invalid status");
            }
        }
        catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("quotient Decode failed: ") + error.what());
        }
        try {
            const auto status = cc->Decrypt(
                keys.secretKey, result.remainder, &pRemainder);
            if (!status.isValid) {
                throw std::runtime_error(
                    "remainder decryption returned an invalid status");
            }
        }
        catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("remainder Decode failed: ") + error.what());
        }
        pQuotient->SetLength(config.slots);
        pRemainder->SetLength(config.slots);
        const auto quotientValues =
            pQuotient->GetCKKSPackedValue();
        const auto remainderValues =
            pRemainder->GetCKKSPackedValue();
        phase.decryptionMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        begin = ksword::Clock::now();
        const auto quotientErrors = CheckResult(
            config,
            quotientValues,
            input.expectedQuotient);
        const auto remainderErrors = CheckResult(
            config,
            remainderValues,
            input.expectedRemainder);
        phase.verificationMs =
            ksword::Milliseconds(begin, ksword::Clock::now());
        phase.totalMs =
            ksword::Milliseconds(
                programBegin, ksword::Clock::now());
        const uint64_t failures =
            quotientErrors.failures +
            remainderErrors.failures;

        std::cout << "\n================ Correctness and precision ================\n";
        std::cout << std::scientific << std::setprecision(10);
        PrintError("quotient", quotientErrors);
        PrintError("remainder", remainderErrors);
        std::cout << "  overall verdict             : "
                  << (failures == 0 ? "PASS" : "FAIL")
                  << "\n";

        std::cout << "\n================ Divider profile ================\n";
        std::cout << "  radix rounds / candidate trials : "
                  << timing.radixRounds << " / "
                  << timing.candidateSubtractions << "\n";
        std::cout << "  prefix / state / total Bootstraps: "
                  << timing.prefixBootstraps << " / "
                  << timing.stateBootstraps << " / "
                  << timing.totalBootstraps << "\n";
        std::cout << "  Boolean purifications            : "
                  << timing.booleanPurifications << "\n";
        std::cout << "  async setup batches              : "
                  << timing.asyncPreparationBatches << "\n";
        PrintPhase("encrypted zero-divisor detection",
                   timing.zeroDetectionMs);
        PrintPhase("input Bootstrap + Boolean projection",
                   timing.inputRefreshMs);
        PrintPhase("D/2D/4D/8D preparation",
                   timing.multiplePreparationMs);
        PrintPhase("radix nibble formation",
                   timing.nibbleFormationMs);
        PrintPhase("midpoint-refreshed candidate subtraction",
                   timing.candidateSubtractionMs);
        PrintPhase("G/P midpoint Bootstrap + Boolean projection (subset)",
                   timing.prefixRefreshMs);
        PrintPhase("borrow extraction + quotient-bit formation",
                   timing.borrowProjectionMs);
        PrintPhase("candidate-state mux",
                   timing.candidateMuxMs);
        PrintPhase("round-state Bootstrap + Boolean projection",
                   timing.stateRefreshMs);
        PrintPhase("quotient digit accumulation",
                   timing.quotientUpdateMs);
        PrintPhase("encrypted division-by-zero result selection",
                   timing.zeroResultSelectionMs);
        PrintPhase("final quotient refresh",
                   timing.finalRefreshMs);
        PrintPhase("level alignment",
                   timing.levelAlignmentMs);
        PrintPhase("public masks + graph/workspace initialization (excluded)",
                   timing.setupExcludedMs);

        std::cout << "\n================ End-to-end timing ================\n";
        PrintPhase("CryptoContext generation and Enable",
                   phase.contextMs);
        PrintPhase("KeyGen", phase.keyGenMs);
        PrintPhase("EvalMultKeyGen", phase.multKeyGenMs);
        PrintPhase("rotation key generation",
                   phase.rotateKeyGenMs);
        PrintPhase("EvalBootstrapSetup",
                   phase.bootstrapSetupMs);
        PrintPhase("EvalBootstrapKeyGen",
                   phase.bootstrapKeyGenMs);
        PrintPhase("LoadContext and evaluation keys",
                   phase.loadContextMs);
        PrintPhase("input/reference generation",
                   phase.inputBuildMs);
        PrintPhase("CKKS encoding", phase.encodingMs);
        PrintPhase("input encryption/loading",
                   phase.encryptionMs);
        PrintPhase("ciphertext division wall time",
                   phase.divisionWallMs);
        PrintPhase("ciphertext division online latency",
                   phase.divisionOnlineMs);
        PrintPhase("copy/decrypt", phase.decryptionMs);
        PrintPhase("correctness scan",
                   phase.verificationMs);
        PrintPhase("full program wall", phase.totalMs);

        std::cout << "\n================ Online result ================\n";
        std::cout << "  quotient+remainder latency : "
                  << std::fixed << std::setprecision(3)
                  << phase.divisionOnlineMs << " ms\n";
        std::cout << "  amortized per word         : "
                  << phase.divisionOnlineMs / config.words
                  << " ms\n";
        std::cout << "  online throughput          : "
                  << 1000.0 * config.words /
                         phase.divisionOnlineMs
                  << " word/s\n";

        std::cout << "\n================ Standard-key accounting ================\n";
        std::cout << "  candidate trials            : "
                  << timing.candidateSubtractions << "\n";
        std::cout << "  borrow-prefix rotations     : "
                  << static_cast<uint64_t>(timing.candidateSubtractions) *
                         config.wideLayers()
                  << "\n";
        std::cout << "  borrow-prefix conjugations  : "
                  << static_cast<uint64_t>(timing.candidateSubtractions) *
                         config.wideLayers()
                  << "\n";
        std::cout << "  borrow-prefix ct-ct products: "
                  << static_cast<uint64_t>(timing.candidateSubtractions) *
                         config.wideLayers()
                  << "\n";
        std::cout << "  candidate mux ct-ct products: "
                  << timing.candidateSubtractions << "\n";
        std::cout << "  zero-result gate ct-ct products: 1\n";
        std::cout << "  per-bit cubic projections   : 0\n";
        std::cout << "  nonstandard evaluation keys : 0\n";
        return failures == 0 ? 0 : 2;
    }
    catch (const std::exception& error) {
        std::cerr << "[fatal] " << error.what() << "\n";
        std::cerr << "Use --help to list supported runtime options.\n";
        return 1;
    }
}
