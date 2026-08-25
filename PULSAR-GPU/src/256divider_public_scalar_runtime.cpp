#include <fideslib.hpp>

#include "scalar_divider_ks_runtime.h"

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
    double twistedKeyGenMs = 0.0;
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
    std::vector<uint8_t> numerator;
    std::vector<uint8_t> expected;
    std::vector<double> encoded;
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
    std::cout << "  " << std::left << std::setw(48) << name
              << std::right << std::setw(13)
              << std::fixed << std::setprecision(3)
              << milliseconds << " ms  ("
              << std::setprecision(6)
              << milliseconds / 1000.0 << " s)\n";
}

void StoreWord(const ksdivscalar::RuntimeConfig& config,
               uint32_t word,
               cpp_int value,
               std::vector<uint8_t>* bits) {
    for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
        (*bits)[ksword::Slot(config, word, bit)] =
            static_cast<uint8_t>((value & 1) != 0);
        value >>= 1;
    }
}

cpp_int LoadWord(const ksdivscalar::RuntimeConfig& config,
                 uint32_t word,
                 const std::vector<uint8_t>& bits) {
    cpp_int value = 0;
    for (uint32_t bit = config.wordBits; bit-- > 0;) {
        value <<= 1;
        value += bits[ksword::Slot(config, word, bit)];
    }
    return value;
}

cpp_int RandomWord(std::mt19937_64* generator,
                   uint32_t bits) {
    cpp_int value = 0;
    for (uint32_t offset = 0; offset < bits; offset += 64U) {
        value |= cpp_int((*generator)()) << offset;
    }
    return value & ((cpp_int(1) << bits) - 1);
}

Inputs BuildInputs(const ksdivscalar::RuntimeConfig& config) {
    Inputs input;
    input.numerator.assign(config.slots, 0);
    input.expected.assign(config.slots, 0);
    input.encoded.assign(config.slots, 0.0);

    const cpp_int maximum =
        (cpp_int(1) << config.wordBits) - 1;
    std::mt19937_64 generator(0x5343414C41524449ULL);
    for (uint32_t word = 0; word < config.words; ++word) {
        cpp_int numerator =
            RandomWord(&generator, config.wordBits);
        if (word == 0) {
            numerator = maximum;
        }
        else if (word == 1) {
            numerator = config.divisor;
        }
        else if (word == 2) {
            numerator = config.divisor - 1;
        }
        else if (word == 3) {
            numerator = 0;
        }
        StoreWord(
            config, word, numerator, &input.numerator);
        StoreWord(
            config,
            word,
            numerator / config.divisor,
            &input.expected);
    }

    for (uint32_t slot = 0; slot < config.slots; ++slot) {
        input.encoded[slot] =
            static_cast<double>(input.numerator[slot]);
    }
    return input;
}

ErrorReport CheckResult(
    const ksdivscalar::RuntimeConfig& config,
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
            report.maxImaginary =
                std::max(
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
            const auto& value =
                values[ksword::Slot(config, word, position)];
            report.maxGuard = std::max(
                report.maxGuard, std::abs(value));
        }
    }

    report.meanError =
        static_cast<double>(sumAbsolute / count);
    report.rmse =
        std::sqrt(static_cast<double>(sumSquared / count));
    return report;
}

void PrintConfiguration(
    const ksdivscalar::RuntimeConfig& config) {
    const cpp_int reciprocal =
        ksdivscalar::Reciprocal(config);
    std::cout << "Runtime-configurable exact CKKS public-scalar divider\n";
    std::cout << "============================================================\n";
    std::cout << "backend                 : "
              << ksword::BackendName(config.backend) << "\n";
    std::cout << "word bits               : "
              << config.wordBits << "\n";
    std::cout << "internal active bits    : "
              << config.internalBits() << "\n";
    std::cout << "packed words/ciphertext : "
              << config.words << " (maximum "
              << config.maxWords() << ")\n";
    std::cout << "slots used / available  : "
              << config.usedSlots() << " / "
              << config.slots << "\n";
    std::cout << "public divisor          : "
              << config.divisor << "\n";
    std::cout << "reciprocal popcount     : "
              << ksdivscalar::Popcount(reciprocal) << "\n";
    std::cout << "divisor popcount        : "
              << ksdivscalar::Popcount(config.divisor) << "\n";
    std::cout << "parallel shift blocks   : "
              << config.parallelBlocks << "\n";
    std::cout << "Dadda merge batch       : "
              << config.mergeBatch << "\n";
    std::cout << "multiplicative depth    : "
              << config.multiplicativeDepth << "\n";
    std::cout << "logical GPU devices     : "
              << ksword::DeviceList(config.devices) << "\n";
    std::cout << "Bootstrap policy        : purify then Bootstrap; "
                 "all calls timed\n";
}

void PrintMultiply(const char* title,
                   const ksdivscalar::PublicMultiplyTiming& item) {
    std::cout << "  " << title << "\n";
    std::cout << "    partial rows / streams  : "
              << item.inputRows << " / "
              << item.shiftStreams << "\n";
    std::cout << "    Dadda stages            : "
              << item.daddaStages << "\n";
    std::cout << "    row rotations/plain mult: "
              << item.rowRotations << " / "
              << item.rowPlainMults << "\n";
    std::cout << "    compressor ct mults     : "
              << item.compressorCipherMults << "\n";
    std::cout << "    compressor carry rotates: "
              << item.compressorCarryRotations << "\n";
    std::cout << "    synchronizations        : "
              << item.synchronizations << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::unitbuf;
        ksdivscalar::RuntimeConfig config;
        try {
            config =
                ksdivscalar::ParseCommandLine(argc, argv);
        }
        catch (const std::runtime_error& error) {
            if (std::string(error.what()) == "help") {
                std::cout << ksdivscalar::Usage(argv[0]);
                return 0;
            }
            throw;
        }

        PrintConfiguration(config);
        const auto programBegin = ksword::Clock::now();
        PhaseTimes phase;
        const std::vector<uint32_t> levelBudget{4, 4};
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

        const auto rotations =
            ksdivscalar::RotationIndices(config);
        begin = ksword::Clock::now();
        cc->EvalRotateKeyGen(keys.secretKey, rotations);
        phase.rotateKeyGenMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        const auto twisted =
            ksdivscalar::TwistedIndices(config);
        begin = ksword::Clock::now();
        cc->EvalTwistedProductKeyGen(
            keys.secretKey, twisted, config.slots);
        phase.twistedKeyGenMs =
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
            std::cout << "  word " << word << " N = "
                      << ksword::BitsToHex(
                             input.numerator, config, word)
                      << "\n";
            std::cout << "  word " << word << " Q = "
                      << ksword::BitsToHex(
                             input.expected, config, word)
                      << "\n";
        }

        begin = ksword::Clock::now();
        auto plaintext = cc->MakeCKKSPackedPlaintext(
            input.encoded, 1, 0, nullptr, config.slots);
        plaintext->SetLength(config.slots);
        phase.encodingMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        begin = ksword::Clock::now();
        auto numerator =
            cc->Encrypt(keys.publicKey, plaintext);
        cc->Synchronize();
        phase.encryptionMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        ksdivscalar::TimingReport timing;
        cc->Synchronize();
        begin = ksword::Clock::now();
        auto result =
            ksdivscalar::EvalDividePublicScalar(
                cc, numerator, config, &timing);
        cc->Synchronize();
        phase.divisionWallMs =
            ksword::Milliseconds(begin, ksword::Clock::now());
        phase.divisionOnlineMs = std::max(
            0.0,
            phase.divisionWallMs - timing.setupExcludedMs);

        std::cout << "\nFinal quotient ciphertext state\n";
        std::cout << "  level / scale degree : "
                  << result.quotient->GetLevel() << " / "
                  << result.quotient->GetNoiseScaleDeg() << "\n";
        std::cout << "  Bootstrap calls      : "
                  << timing.bootstrapCount << "\n";
        std::cout << "  Boolean purifications: "
                  << timing.purificationCount << "\n";

        begin = ksword::Clock::now();
        Plaintext decrypted;
        const auto decryptResult = cc->Decrypt(
            keys.secretKey, result.quotient, &decrypted);
        if (!decryptResult.isValid) {
            throw std::runtime_error("quotient decryption failed");
        }
        decrypted->SetLength(config.slots);
        const auto values =
            decrypted->GetCKKSPackedValue();
        phase.decryptionMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        begin = ksword::Clock::now();
        const auto errors =
            CheckResult(config, values, input.expected);
        phase.verificationMs =
            ksword::Milliseconds(begin, ksword::Clock::now());
        phase.totalMs =
            ksword::Milliseconds(
                programBegin, ksword::Clock::now());

        std::cout << "\n================ Correctness and precision ================\n";
        std::cout << std::scientific << std::setprecision(10);
        std::cout << "  maximum absolute error : "
                  << errors.maxError << "\n";
        std::cout << "  mean absolute error    : "
                  << errors.meanError << "\n";
        std::cout << "  RMSE                   : "
                  << errors.rmse << "\n";
        std::cout << "  maximum guard value    : "
                  << errors.maxGuard << "\n";
        std::cout << "  maximum imaginary part : "
                  << errors.maxImaginary << "\n";
        std::cout << "  worst precision        : "
                  << ksword::PrecisionBits(errors.maxError)
                  << " bits\n";
        std::cout << "  threshold failures     : "
                  << errors.failures << "\n";
        std::cout << "  worst location         : word="
                  << errors.worstWord << ", bit="
                  << errors.worstBit << "\n";
        std::cout << "  expected / actual      : "
                  << errors.worstExpected << " / "
                  << errors.worstActual << "\n";
        std::cout << "  verdict                : "
                  << (errors.failures == 0 ? "PASS" : "FAIL")
                  << "\n";

        std::cout << "\n================ Division profile ================\n";
        PrintMultiply(
            "N * floor(2^w/d)",
            timing.reciprocalMultiply);
        PrintMultiply(
            "q0 * d",
            timing.divisorMultiply);
        PrintPhase("quotient high-half extraction",
                   timing.quotientExtractMs);
        PrintPhase("quotient purification + Bootstrap",
                   timing.quotientRefreshMs);
        PrintPhase("q0*d purification + Bootstrap",
                   timing.productRefreshMs);
        PrintPhase("numerator purification + Bootstrap",
                   timing.numeratorRefreshMs);
        PrintPhase("latest borrow subtract: N-q0*d",
                   timing.remainderSubtractMs);
        PrintPhase("remainder purification + Bootstrap",
                   timing.remainderRefreshMs);
        PrintPhase("latest borrow correction test",
                   timing.correctionBorrowMs);
        PrintPhase("latest complex-twisted q0+correction",
                   timing.correctionAddMs);
        PrintPhase("final purification + Bootstrap",
                   timing.finalRefreshMs);
        PrintPhase("level alignment",
                   timing.levelAlignmentMs);
        PrintPhase("public masks + ModUp graph initialization (excluded)",
                   timing.setupExcludedMs);

        std::cout << "\n================ End-to-end timing ================\n";
        PrintPhase("CryptoContext generation and Enable",
                   phase.contextMs);
        PrintPhase("KeyGen", phase.keyGenMs);
        PrintPhase("EvalMultKeyGen", phase.multKeyGenMs);
        PrintPhase("rotation key generation",
                   phase.rotateKeyGenMs);
        PrintPhase("mixed-basis twisted KSK generation",
                   phase.twistedKeyGenMs);
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
        PrintPhase("scalar division wall time",
                   phase.divisionWallMs);
        PrintPhase("scalar division online latency",
                   phase.divisionOnlineMs);
        PrintPhase("copy/decrypt", phase.decryptionMs);
        PrintPhase("correctness scan",
                   phase.verificationMs);
        PrintPhase("full program wall", phase.totalMs);

        std::cout << "\n================ Online result ================\n";
        std::cout << "  quotient latency      : "
                  << std::fixed << std::setprecision(3)
                  << phase.divisionOnlineMs << " ms\n";
        std::cout << "  amortized per word    : "
                  << phase.divisionOnlineMs / config.words
                  << " ms\n";
        std::cout << "  online throughput     : "
                  << 1000.0 * config.words /
                         phase.divisionOnlineMs
                  << " word/s\n";
        std::cout << "  divisor identity      : q0 is exact or one low; "
                     "one correction\n";

        return errors.failures == 0 ? 0 : 2;
    }
    catch (const std::exception& error) {
        std::cerr << "[fatal] " << error.what() << "\n";
        std::cerr << "Use --help to list supported runtime options.\n";
        return 1;
    }
}
