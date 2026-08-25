#pragma once

#include <fideslib.hpp>

#include "word_operator_common.h"
#include "word_operator_kernels.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fideslib;

namespace kswordapp {

struct PhaseTimes {
    double contextMs = 0.0;
    double keyGenMs = 0.0;
    double multKeyGenMs = 0.0;
    double rotateKeyGenMs = 0.0;
    double bootstrapSetupMs = 0.0;
    double bootstrapKeyGenMs = 0.0;
    double loadContextMs = 0.0;
    double maskBuildLoadMs = 0.0;
    double inputBuildMs = 0.0;
    double encodingMs = 0.0;
    double encryptionMs = 0.0;
    double operationMs = 0.0;
    double bootstrapMs = 0.0;
    double decryptionMs = 0.0;
    double verificationMs = 0.0;
    double totalMs = 0.0;
};

struct Inputs {
    std::vector<uint8_t> a;
    std::vector<uint8_t> b;
    std::vector<uint8_t> expected;
    std::vector<uint32_t> encryptedShifts;
    std::vector<double> encodedA;
    std::vector<double> encodedB;
};

struct ErrorReport {
    double maxError = 0.0;
    double meanError = 0.0;
    double rmse = 0.0;
    double maxGuard = 0.0;
    uint64_t failures = 0;
    uint32_t worstWord = 0;
    uint32_t worstBit = 0;
    double worstExpected = 0.0;
    double worstActual = 0.0;
};

inline void PrintPhase(const std::string& name, double milliseconds) {
    std::cout << "  " << std::left << std::setw(44) << name
              << std::right << std::setw(13)
              << std::fixed << std::setprecision(3)
              << milliseconds << " ms  ("
              << std::setprecision(6) << milliseconds / 1000.0 << " s)\n";
}

inline uint32_t LevelsRemaining(
    const ksword::RuntimeConfig& config,
    const Ciphertext<DCRTPoly>& ciphertext) {
    const uint32_t level = ciphertext->GetLevel();
    const uint32_t scalePenalty =
        ciphertext->GetNoiseScaleDeg() > 0
            ? ciphertext->GetNoiseScaleDeg() - 1
            : 0;
    return config.multiplicativeDepth > level + scalePenalty
        ? config.multiplicativeDepth - level - scalePenalty
        : 0;
}

inline uint32_t TestEncryptedShift(
    const ksword::RuntimeConfig& config,
    uint32_t word,
    std::mt19937_64* generator) {
    const std::vector<uint32_t> boundary{
        0,
        1,
        2,
        config.wordBits / 4U,
        config.wordBits / 2U,
        config.wordBits - 1U,
    };
    if (word < boundary.size()) {
        return boundary[word];
    }
    return static_cast<uint32_t>((*generator)()) &
           (config.wordBits - 1U);
}

inline Inputs BuildInputs(const ksword::RuntimeConfig& config,
                          ksword::OperatorKind kind) {
    Inputs input;
    input.a.assign(config.slots, 0);
    input.b.assign(config.slots, 0);
    input.expected.assign(config.slots, 0);
    input.encodedA.assign(config.slots, 0.0);
    input.encodedB.assign(config.slots, 0.0);
    input.encryptedShifts.assign(config.words, 0);

    std::mt19937_64 generator(0x574F52444F505338ULL);
    for (uint32_t word = 0; word < config.words; ++word) {
        for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
            const uint32_t slot = ksword::Slot(config, word, bit);
            input.a[slot] = static_cast<uint8_t>(generator() & 1U);
            input.b[slot] = static_cast<uint8_t>(generator() & 1U);
        }
    }

    if (config.words >= 1) {
        for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
            const uint32_t slot = ksword::Slot(config, 0, bit);
            input.a[slot] = static_cast<uint8_t>(bit & 1U);
            input.b[slot] = static_cast<uint8_t>((bit + 1U) & 1U);
        }
    }
    if (config.words >= 2) {
        for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
            input.a[ksword::Slot(config, 1, bit)] = 1;
            input.b[ksword::Slot(config, 1, bit)] =
                static_cast<uint8_t>((bit % 3U) == 0U);
        }
    }

    if (ksword::IsEncryptedShift(kind)) {
        std::fill(input.b.begin(), input.b.end(), uint8_t{0});
        for (uint32_t word = 0; word < config.words; ++word) {
            const uint32_t shift =
                TestEncryptedShift(config, word, &generator);
            input.encryptedShifts[word] = shift;
            for (uint32_t bit = 0;
                 bit < config.controlBits();
                 ++bit) {
                input.b[ksword::Slot(config, word, bit)] =
                    static_cast<uint8_t>((shift >> bit) & 1U);
            }
            // High encrypted bits are semantically ignored. Set one when
            // available so the test checks modulo-word-width behavior.
            if ((word & 1U) != 0U &&
                config.controlBits() < config.wordBits) {
                input.b[ksword::Slot(
                    config, word, config.controlBits())] = 1;
            }
        }
    }

    for (uint32_t word = 0; word < config.words; ++word) {
        uint32_t shift = config.scalarShift;
        if (ksword::IsEncryptedShift(kind)) {
            shift = input.encryptedShifts[word];
        }
        for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
            const uint32_t slot = ksword::Slot(config, word, bit);
            const uint8_t a = input.a[slot];
            const uint8_t b = input.b[slot];
            uint8_t value = 0;

            switch (kind) {
                case ksword::OperatorKind::Xor:
                    value = static_cast<uint8_t>(a ^ b);
                    break;
                case ksword::OperatorKind::And:
                    value = static_cast<uint8_t>(a & b);
                    break;
                case ksword::OperatorKind::Or:
                    value = static_cast<uint8_t>(a | b);
                    break;
                case ksword::OperatorKind::Not:
                    value = static_cast<uint8_t>(1U - a);
                    break;
                case ksword::OperatorKind::ScalarLeftShift:
                case ksword::OperatorKind::LeftShift:
                    if (shift < config.wordBits && bit >= shift) {
                        value = input.a[
                            ksword::Slot(config, word, bit - shift)];
                    }
                    break;
                case ksword::OperatorKind::ScalarRightShift:
                case ksword::OperatorKind::RightShift:
                    if (shift < config.wordBits &&
                        bit + shift < config.wordBits) {
                        value = input.a[
                            ksword::Slot(config, word, bit + shift)];
                    }
                    break;
            }
            input.expected[slot] = value;
        }
    }

    for (uint32_t slot = 0; slot < config.slots; ++slot) {
        input.encodedA[slot] = static_cast<double>(input.a[slot]);
        input.encodedB[slot] = static_cast<double>(input.b[slot]);
    }
    return input;
}

inline ErrorReport CheckResult(
    const ksword::RuntimeConfig& config,
    const std::vector<double>& values,
    const std::vector<uint8_t>& expected) {
    ErrorReport report;
    long double sumAbsolute = 0.0;
    long double sumSquared = 0.0;
    uint64_t active = 0;

    for (uint32_t word = 0; word < config.words; ++word) {
        for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
            const uint32_t slot = ksword::Slot(config, word, bit);
            const double actual = values[slot];
            const double target = static_cast<double>(expected[slot]);
            const double error = std::abs(actual - target);
            if (error > report.maxError) {
                report.maxError = error;
                report.worstWord = word;
                report.worstBit = bit;
                report.worstExpected = target;
                report.worstActual = actual;
            }
            sumAbsolute += error;
            sumSquared += static_cast<long double>(error) * error;
            if (!std::isfinite(actual) || error >= 0.5) {
                ++report.failures;
            }
            ++active;
        }
        for (uint32_t guard = 0; guard < config.guardBits; ++guard) {
            report.maxGuard = std::max(
                report.maxGuard,
                std::abs(values[ksword::Slot(
                    config, word, config.wordBits + guard)]));
        }
    }

    report.meanError =
        static_cast<double>(sumAbsolute / active);
    report.rmse =
        std::sqrt(static_cast<double>(sumSquared / active));
    return report;
}

inline void PrintConfiguration(const ksword::RuntimeConfig& config,
                               ksword::OperatorKind kind) {
    std::cout << "Runtime-configurable CKKS "
              << ksword::OperatorName(kind) << "\n";
    std::cout << "============================================================\n";
    std::cout << "backend                 : "
              << ksword::BackendName(config.backend) << "\n";
    std::cout << "word bits               : "
              << config.wordBits << "\n";
    std::cout << "packed words/ciphertext : "
              << config.words << " (maximum " << config.maxWords() << ")\n";
    std::cout << "slots used / available  : "
              << config.usedSlots() << " / " << config.slots << "\n";
    std::cout << "slots per word          : "
              << config.stride << " (" << config.wordBits
              << " active + " << config.guardBits << " guard)\n";
    if (ksword::IsScalarShift(kind)) {
        std::cout << "public shift amount     : "
                  << config.scalarShift << "\n";
    }
    if (ksword::IsEncryptedShift(kind)) {
        std::cout << "encrypted control bits  : "
                  << config.controlBits() << "\n";
        std::cout << "padded control block    : "
                  << ksword::NextPowerOfTwo(config.controlBits()) << "\n";
    }
    std::cout << "multiplicative depth    : "
              << config.multiplicativeDepth << "\n";
    std::cout << "logical GPU devices     : "
              << ksword::DeviceList(config.devices) << "\n";
    std::cout << "bootstrap               : "
              << (ksword::RequiresBootstrap(kind)
                      ? "one required"
                      : "none (NOT)")
              << "\n";
    std::cout << "detailed profile        : "
              << (config.detailedProfile ? "yes" : "no") << "\n";
}

inline void PrintProfile(const ksword::RuntimeConfig& config,
                         ksword::OperatorKind kind,
                         const ksword::TimingReport& report) {
    std::cout << "\n================ Operator internal timing ================\n";
    if (ksword::IsScalarShift(kind)) {
        if (config.scalarShift == 0) {
            PrintPhase("reuse input ciphertext", report.totalMs);
        }
        else if (config.scalarShift >= config.wordBits) {
            PrintPhase("create encrypted zero", report.zeroMs);
        }
        else {
            PrintPhase("fixed-distance rotation", report.rotateMs);
            PrintPhase("boundary mask multiplication", report.maskMs);
        }
    }
    else if (ksword::IsEncryptedShift(kind)) {
        PrintPhase("keep encrypted control bits",
                   report.lowControlMaskMs);
        for (uint32_t layer = 0;
             layer < report.patternLayers.size();
             ++layer) {
            const auto& item = report.patternLayers[layer];
            std::cout << "\n  Pattern layer " << layer << "\n";
            PrintPhase("    rotate pattern", item.rotateMs);
            PrintPhase("    add pattern", item.addMs);
            PrintPhase("    layer wall", item.wallMs);
        }
        for (uint32_t layer = 0;
             layer < report.shiftLayers.size();
             ++layer) {
            const auto& item = report.shiftLayers[layer];
            std::cout << "\n  Barrel layer " << layer
                      << " (distance=" << (1U << layer) << ")\n";
            PrintPhase("    control lane mask", item.controlMaskMs);
            PrintPhase("    control anchor", item.controlAnchorMs);
            PrintPhase("    control broadcast", item.controlWallMs);
            PrintPhase("    rotate data", item.dataRotateMs);
            PrintPhase("    shift boundary mask", item.shiftMaskMs);
            PrintPhase("    candidate-current", item.differenceMs);
            PrintPhase("    encrypted select multiply",
                       item.selectMultMs);
            PrintPhase("    selected update", item.selectAddMs);
            PrintPhase("    layer wall", item.wallMs);
        }
    }
    else {
        if (report.inputAddMs > 0.0) {
            PrintPhase("input addition", report.inputAddMs);
        }
        if (report.productMs > 0.0) {
            PrintPhase("ciphertext product", report.productMs);
        }
        if (report.doubleProductMs > 0.0) {
            PrintPhase("double product", report.doubleProductMs);
        }
        if (report.subtractMs > 0.0) {
            PrintPhase("subtraction", report.subtractMs);
        }
    }
    PrintPhase("operator total", report.totalMs);
}

inline int Run(int argc,
               char** argv,
               ksword::OperatorKind kind) {
    try {
        ksword::RuntimeConfig config;
        try {
            config = ksword::ParseCommandLine(argc, argv);
        }
        catch (const std::runtime_error& error) {
            if (std::string(error.what()) == "help") {
                std::cout << ksword::Usage(
                    argv[0], ksword::IsScalarShift(kind));
                return 0;
            }
            throw;
        }
        if (!ksword::IsScalarShift(kind) && config.shiftExplicit) {
            throw std::invalid_argument(
                "--shift is only valid for scalar shift operators");
        }
        PrintConfiguration(config, kind);

        const auto programBegin = ksword::Clock::now();
        PhaseTimes phase;
        const std::vector<uint32_t> levelBudget{3, 3};
        const std::vector<uint32_t> bsgsDim{0, 0};

        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetSecretKeyDist(UNIFORM_TERNARY);
        parameters.SetSecurityLevel(HEStd_128_classic);
        parameters.SetRingDim(ksword::kRingDimension);
        parameters.SetBatchSize(config.slots);
        parameters.SetMultiplicativeDepth(config.multiplicativeDepth);
        parameters.SetScalingTechnique(FLEXIBLEAUTO);
        parameters.SetScalingModSize(59);
        parameters.SetFirstModSize(60);
        parameters.SetKeySwitchTechnique(HYBRID);
        parameters.SetNumLargeDigits(3);
        parameters.SetDevices(std::vector<int>(config.devices));
        parameters.SetPlaintextAutoload(false);
        parameters.SetCiphertextAutoload(true);

        auto begin = ksword::Clock::now();
        auto cc = GenCryptoContext(parameters);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);
        if (ksword::RequiresBootstrap(kind)) {
            cc->Enable(FHE);
        }
        phase.contextMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        begin = ksword::Clock::now();
        auto keys = cc->KeyGen();
        phase.keyGenMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        if (ksword::NeedsEvalMultKey(kind)) {
            begin = ksword::Clock::now();
            cc->EvalMultKeyGen(keys.secretKey);
            phase.multKeyGenMs =
                ksword::Milliseconds(begin, ksword::Clock::now());
        }

        const auto rotations =
            ksword::RotationIndices(config, kind);
        if (!rotations.empty()) {
            begin = ksword::Clock::now();
            cc->EvalRotateKeyGen(keys.secretKey, rotations);
            phase.rotateKeyGenMs =
                ksword::Milliseconds(begin, ksword::Clock::now());
        }

        if (ksword::RequiresBootstrap(kind)) {
            begin = ksword::Clock::now();
            cc->EvalBootstrapSetup(
                levelBudget, bsgsDim, config.slots, 0);
            phase.bootstrapSetupMs =
                ksword::Milliseconds(begin, ksword::Clock::now());

            begin = ksword::Clock::now();
            cc->EvalBootstrapKeyGen(keys.secretKey, config.slots);
            phase.bootstrapKeyGenMs =
                ksword::Milliseconds(begin, ksword::Clock::now());
        }

        begin = ksword::Clock::now();
        cc->LoadContext(keys.publicKey);
        cc->Synchronize();
        phase.loadContextMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        begin = ksword::Clock::now();
        auto masks =
            ksword::BuildAndLoadMasks(cc, config, kind);
        phase.maskBuildLoadMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        begin = ksword::Clock::now();
        const auto input = BuildInputs(config, kind);
        phase.inputBuildMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        std::cout << "\nInput samples\n";
        const uint32_t samples = std::min(config.words, uint32_t{3});
        for (uint32_t word = 0; word < samples; ++word) {
            std::cout << "  word " << word << " A = "
                      << ksword::BitsToHex(input.a, config, word) << "\n";
            if (!ksword::IsUnary(kind)) {
                if (ksword::IsEncryptedShift(kind)) {
                    std::cout << "  word " << word
                              << " encrypted shift = "
                              << input.encryptedShifts[word] << "\n";
                }
                else {
                    std::cout << "  word " << word << " B = "
                              << ksword::BitsToHex(
                                     input.b, config, word)
                              << "\n";
                }
            }
            std::cout << "  word " << word << " expected = "
                      << ksword::BitsToHex(
                             input.expected, config, word)
                      << "\n";
        }

        begin = ksword::Clock::now();
        auto pA = cc->MakeCKKSPackedPlaintext(
            input.encodedA, 1, 0, nullptr, config.slots);
        pA->SetLength(config.slots);
        Plaintext pB;
        if (!ksword::IsUnary(kind)) {
            pB = cc->MakeCKKSPackedPlaintext(
                input.encodedB, 1, 0, nullptr, config.slots);
            pB->SetLength(config.slots);
        }
        phase.encodingMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        begin = ksword::Clock::now();
        auto cA = cc->Encrypt(keys.publicKey, pA);
        Ciphertext<DCRTPoly> cB;
        if (!ksword::IsUnary(kind)) {
            cB = cc->Encrypt(keys.publicKey, pB);
        }
        cc->Synchronize();
        phase.encryptionMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        ksword::TimingReport timing;
        cc->Synchronize();
        begin = ksword::Clock::now();
        auto result = ksword::EvalConfiguredOperator(
            cc, cA, cB, config, kind, masks,
            config.detailedProfile ? &timing : nullptr);
        cc->Synchronize();
        phase.operationMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        std::cout << "\nOperator ciphertext state\n";
        std::cout << "  level / scale degree : "
                  << result->GetLevel() << " / "
                  << result->GetNoiseScaleDeg() << "\n";
        std::cout << "  estimated levels left: "
                  << LevelsRemaining(config, result) << "\n";

        auto evaluated = result;
        if (ksword::RequiresBootstrap(kind)) {
            cc->Synchronize();
            begin = ksword::Clock::now();
            evaluated = cc->EvalBootstrap(result);
            cc->Synchronize();
            phase.bootstrapMs =
                ksword::Milliseconds(begin, ksword::Clock::now());

            std::cout << "\nFinal ciphertext state\n";
            std::cout << "  level / scale degree : "
                      << evaluated->GetLevel() << " / "
                      << evaluated->GetNoiseScaleDeg() << "\n";
        }

        begin = ksword::Clock::now();
        Plaintext decrypted;
        const auto decryptResult =
            cc->Decrypt(keys.secretKey, evaluated, &decrypted);
        if (!decryptResult.isValid) {
            throw std::runtime_error("decryption failed");
        }
        decrypted->SetLength(config.slots);
        const auto values = decrypted->GetRealPackedValue();
        phase.decryptionMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        begin = ksword::Clock::now();
        const auto errors =
            CheckResult(config, values, input.expected);
        phase.verificationMs =
            ksword::Milliseconds(begin, ksword::Clock::now());
        phase.totalMs =
            ksword::Milliseconds(programBegin, ksword::Clock::now());

        std::cout
            << "\n================ Correctness and precision ================\n";
        std::cout << std::scientific << std::setprecision(10);
        std::cout << "  valid bit slots         : "
                  << static_cast<uint64_t>(config.words) *
                         config.wordBits
                  << "\n";
        std::cout << "  maximum absolute error : "
                  << errors.maxError << "\n";
        std::cout << "  mean absolute error    : "
                  << errors.meanError << "\n";
        std::cout << "  RMSE                   : "
                  << errors.rmse << "\n";
        std::cout << "  maximum guard value    : "
                  << errors.maxGuard << "\n";
        std::cout << "  worst precision        : "
                  << ksword::PrecisionBits(errors.maxError) << " bits\n";
        std::cout << "  threshold failures     : "
                  << errors.failures << "\n";
        std::cout << "  worst location         : word="
                  << errors.worstWord << ", bit="
                  << errors.worstBit << "\n";
        std::cout << "  expected / actual      : "
                  << errors.worstExpected << " / "
                  << errors.worstActual << "\n";
        std::cout << "  verdict                : "
                  << (errors.failures == 0 ? "PASS" : "FAIL") << "\n";

        if (config.detailedProfile) {
            PrintProfile(config, kind, timing);
        }

        std::cout << "\n================ End-to-end timing ================\n";
        PrintPhase("CryptoContext generation and Enable", phase.contextMs);
        PrintPhase("KeyGen", phase.keyGenMs);
        if (ksword::NeedsEvalMultKey(kind)) {
            PrintPhase("EvalMultKeyGen", phase.multKeyGenMs);
        }
        if (!rotations.empty()) {
            PrintPhase("rotation key generation", phase.rotateKeyGenMs);
        }
        if (ksword::RequiresBootstrap(kind)) {
            PrintPhase("EvalBootstrapSetup", phase.bootstrapSetupMs);
            PrintPhase("EvalBootstrapKeyGen", phase.bootstrapKeyGenMs);
        }
        PrintPhase("LoadContext and evaluation keys", phase.loadContextMs);
        PrintPhase("mask encoding/loading", phase.maskBuildLoadMs);
        PrintPhase("input/reference generation", phase.inputBuildMs);
        PrintPhase("CKKS encoding", phase.encodingMs);
        PrintPhase("input encryption/loading", phase.encryptionMs);
        PrintPhase(
            std::string(ksword::OperatorName(kind)) +
                " (one real execution)",
            phase.operationMs);
        if (ksword::RequiresBootstrap(kind)) {
            PrintPhase("one EvalBootstrap (single call)",
                       phase.bootstrapMs);
        }
        PrintPhase("copy/decrypt", phase.decryptionMs);
        PrintPhase("correctness scan", phase.verificationMs);
        PrintPhase("full program wall", phase.totalMs);

        const double onlineMs =
            phase.operationMs + phase.bootstrapMs;
        std::cout << "\n================ Online result ================\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  circuit latency        : "
                  << phase.operationMs << " ms\n";
        if (ksword::RequiresBootstrap(kind)) {
            std::cout << "  bootstrap single call  : "
                      << phase.bootstrapMs << " ms\n";
        }
        std::cout << "  online latency         : "
                  << onlineMs << " ms\n";
        std::cout << "  amortized per word     : "
                  << onlineMs / config.words << " ms\n";
        std::cout << "  online throughput      : "
                  << (onlineMs > 0.0
                          ? config.words * 1000.0 / onlineMs
                          : 0.0)
                  << " word/s\n";

        return errors.failures == 0 ? 0 : 2;
    }
    catch (const std::exception& error) {
        std::cerr << "[fatal] " << error.what() << "\n";
        std::cerr << "Use --help to list supported runtime options.\n";
        return 1;
    }
}

}  // namespace kswordapp
