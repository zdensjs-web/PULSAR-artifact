#pragma once

#include <fideslib.hpp>

#include "greaterthan_ks_complex_hoisted_gpu.h"
#include "ks_adder_common.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace fideslib;

namespace kscomparisonapp {

using ksadder::Clock;
using ksadder::RuntimeConfig;

struct PhaseTimes {
    double contextMs = 0.0;
    double keyGenMs = 0.0;
    double multKeyGenMs = 0.0;
    double rotateKeyGenMs = 0.0;
    double twistedKeyGenMs = 0.0;
    double bootstrapSetupMs = 0.0;
    double bootstrapKeyGenMs = 0.0;
    double loadContextMs = 0.0;
    double maskBuildLoadMs = 0.0;
    double inputBuildMs = 0.0;
    double encodingMs = 0.0;
    double encryptionMs = 0.0;
    double graphInitMs = 0.0;
    double comparisonMs = 0.0;
    double bootstrapMs = 0.0;
    double decryptionMs = 0.0;
    double verificationMs = 0.0;
    double totalMs = 0.0;
};

struct Inputs {
    std::vector<uint8_t> a;
    std::vector<uint8_t> b;
    std::vector<uint8_t> expected;
    std::vector<double> encodedA;
    std::vector<double> encodedB;
};

struct ErrorReport {
    double maxError = 0.0;
    double meanError = 0.0;
    double rmse = 0.0;
    double maxInactive = 0.0;
    double maxImaginary = 0.0;
    uint64_t failures = 0;
    uint32_t worstWord = 0;
    double worstExpected = 0.0;
    double worstActual = 0.0;
};

void PrintPhase(const std::string& name, double milliseconds) {
    std::cout << "  " << std::left << std::setw(42) << name
              << std::right << std::setw(13)
              << std::fixed << std::setprecision(3)
              << milliseconds << " ms  ("
              << std::setprecision(6) << milliseconds / 1000.0 << " s)\n";
}

uint32_t UsableLevelsRemaining(
    const RuntimeConfig& config,
    const Ciphertext<DCRTPoly>& ciphertext) {
    return ksadder::UsableMultiplicationLevels(
        config.multiplicativeDepth,
        static_cast<uint32_t>(ciphertext->GetLevel()),
        static_cast<uint32_t>(ciphertext->GetNoiseScaleDeg()));
}

void ClearWord(std::vector<uint8_t>* bits,
               const RuntimeConfig& config,
               uint32_t word) {
    for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
        (*bits)[ksadder::Slot(config, word, bit)] = 0;
    }
}

Inputs BuildInputs(const RuntimeConfig& config,
                   ksgreaterthan::Relation relation) {
    Inputs input;
    input.a.assign(config.slots, 0);
    input.b.assign(config.slots, 0);
    input.encodedA.assign(config.slots, 0.0);
    input.encodedB.assign(config.slots, 0.0);

    std::mt19937_64 generator(0x4752454154455232ULL);
    for (uint32_t word = 0; word < config.words; ++word) {
        for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
            const uint32_t slot = ksadder::Slot(config, word, bit);
            input.a[slot] = static_cast<uint8_t>(generator() & 1U);
            input.b[slot] = static_cast<uint8_t>(generator() & 1U);
        }
    }

    // The one-word smoke test exercises the true result.
    if (config.words >= 1) {
        ClearWord(&input.a, config, 0);
        ClearWord(&input.b, config, 0);
        for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
            input.a[ksadder::Slot(config, 0, bit)] = 1;
        }
    }
    // Equality must produce false.
    if (config.words >= 2) {
        for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
            const uint8_t value = static_cast<uint8_t>(bit & 1U);
            input.a[ksadder::Slot(config, 1, bit)] = value;
            input.b[ksadder::Slot(config, 1, bit)] = value;
        }
    }
    // Zero is not greater than the maximum word.
    if (config.words >= 3) {
        ClearWord(&input.a, config, 2);
        ClearWord(&input.b, config, 2);
        for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
            input.b[ksadder::Slot(config, 2, bit)] = 1;
        }
    }
    // An MSB-only difference must dominate all lower bits.
    if (config.words >= 4) {
        for (uint32_t bit = 0; bit + 1 < config.wordBits; ++bit) {
            input.a[ksadder::Slot(config, 3, bit)] =
                static_cast<uint8_t>((bit % 3U) == 0U);
            input.b[ksadder::Slot(config, 3, bit)] =
                static_cast<uint8_t>((bit % 5U) == 0U);
        }
        input.a[ksadder::Slot(
            config, 3, config.wordBits - 1U)] = 1;
        input.b[ksadder::Slot(
            config, 3, config.wordBits - 1U)] = 0;
    }

    input.expected.assign(config.words, 0);
    for (uint32_t word = 0; word < config.words; ++word) {
        bool decided = false;
        for (uint32_t bit = config.wordBits; bit-- > 0;) {
            const uint8_t a =
                input.a[ksadder::Slot(config, word, bit)];
            const uint8_t b =
                input.b[ksadder::Slot(config, word, bit)];
            if (a == b) {
                continue;
            }
            const bool greater = a > b;
            switch (relation) {
                case ksgreaterthan::Relation::GreaterThan:
                    input.expected[word] =
                        static_cast<uint8_t>(greater);
                    break;
                case ksgreaterthan::Relation::LessThan:
                    input.expected[word] =
                        static_cast<uint8_t>(!greater);
                    break;
                case ksgreaterthan::Relation::GreaterEqual:
                    input.expected[word] =
                        static_cast<uint8_t>(greater);
                    break;
                case ksgreaterthan::Relation::LessEqual:
                    input.expected[word] =
                        static_cast<uint8_t>(!greater);
                    break;
                case ksgreaterthan::Relation::Equal:
                    input.expected[word] = 0;
                    break;
                case ksgreaterthan::Relation::NotEqual:
                    input.expected[word] = 1;
                    break;
            }
            decided = true;
            break;
        }
        if (!decided) {
            input.expected[word] = static_cast<uint8_t>(
                relation == ksgreaterthan::Relation::GreaterEqual ||
                relation == ksgreaterthan::Relation::LessEqual ||
                relation == ksgreaterthan::Relation::Equal);
        }
    }

    for (uint32_t slot = 0; slot < config.slots; ++slot) {
        input.encodedA[slot] = static_cast<double>(input.a[slot]);
        input.encodedB[slot] = static_cast<double>(input.b[slot]);
    }
    return input;
}

ErrorReport CheckResult(
    const RuntimeConfig& config,
    const std::vector<std::complex<double>>& values,
    const std::vector<uint8_t>& expected) {
    ErrorReport report;
    long double sumAbsolute = 0.0;
    long double sumSquared = 0.0;

    for (uint32_t word = 0; word < config.words; ++word) {
        const uint32_t resultSlot = ksadder::Slot(config, word, 0);
        const double actual = values[resultSlot].real();
        const double target = static_cast<double>(expected[word]);
        const double error = std::abs(actual - target);
        if (error > report.maxError) {
            report.maxError = error;
            report.worstWord = word;
            report.worstExpected = target;
            report.worstActual = actual;
        }
        sumAbsolute += error;
        sumSquared += static_cast<long double>(error) * error;
        report.maxImaginary = std::max(
            report.maxImaginary, std::abs(values[resultSlot].imag()));
        if (!std::isfinite(actual) || error >= 0.5) {
            ++report.failures;
        }

        for (uint32_t offset = 1; offset < config.stride; ++offset) {
            report.maxInactive = std::max(
                report.maxInactive,
                std::abs(values[ksadder::Slot(config, word, offset)]));
        }
    }

    report.meanError =
        static_cast<double>(sumAbsolute / config.words);
    report.rmse = std::sqrt(
        static_cast<double>(sumSquared / config.words));
    return report;
}

void PrintProfile(const RuntimeConfig& config,
                  const ksgreaterthan::TimingReport& report,
                  ksgreaterthan::Relation relation) {
    std::cout << "\n================ "
              << ksgreaterthan::RelationName(relation)
              << " internal timing ================\n";
    if (ksgreaterthan::IsEquality(relation)) {
        PrintPhase("preprocess X=A xor B", report.prefix.preprocessMs);
    }
    else {
        PrintPhase("preprocess borrow G/P", report.prefix.preprocessMs);
    }
    if (config.algorithm != ksadder::Algorithm::Baseline) {
        if (!ksgreaterthan::IsEquality(relation)) {
            PrintPhase("pack Z=Q/2+iG (Q=-P)",
                       report.prefix.packStateMs);
        }
    }

    for (uint32_t layer = 0; layer < config.layers; ++layer) {
        const auto& item = report.prefix.layers[layer];
        std::cout << "\n  Layer " << layer
                  << " (distance=" << (1U << layer) << ")\n";
        if (ksgreaterthan::IsEquality(relation)) {
            PrintPhase("    Rotate(OR state)", item.transformMs);
            PrintPhase("    state * rotated state", item.productMs);
        }
        else if (config.algorithm == ksadder::Algorithm::Baseline) {
            PrintPhase("    Rotate(G)", item.transformMs);
            PrintPhase("    P * Rotate(G)", item.productMs);
            PrintPhase("    P-prefix auxiliary", item.auxiliaryMs);
        }
        else if (config.algorithm ==
                 ksadder::Algorithm::ComplexStandard) {
            PrintPhase("    Rotate(Z)+Conjugate(Z)", item.transformMs);
            PrintPhase("    complex prefix product", item.productMs);
            PrintPhase("    keep generate channel", item.auxiliaryMs);
        }
        else if (config.algorithm ==
                 ksadder::Algorithm::ComplexHoisted) {
            PrintPhase("    hoisted Rotate(Z)+Conjugate(Z)",
                       item.transformMs);
            PrintPhase("    ordinary complex product", item.productMs);
            PrintPhase("    keep generate channel", item.auxiliaryMs);
        }
        else {
            PrintPhase("    Conjugate(Z)", item.transformMs);
            PrintPhase("    mixed-basis twisted product",
                       item.productMs);
            PrintPhase("    keep generate channel", item.auxiliaryMs);
        }
        PrintPhase("    update state", item.updateMs);
        PrintPhase("    layer wall", item.wallMs);
    }

    if (!ksgreaterthan::IsEquality(relation) &&
        config.algorithm != ksadder::Algorithm::Baseline) {
        PrintPhase("extract conjugation", report.conjugateMs);
    }
    PrintPhase(
        ksgreaterthan::IsEquality(relation)
            ? "mask final OR endpoint"
            : "mask final borrow-generate",
        report.extractMs);
    if (ksgreaterthan::IsEquality(relation)) {
        PrintPhase("explicit endpoint level/scale normalization",
                   report.endpointNormalizeMs);
    }
    PrintPhase("move MSB result to bit 0", report.resultRotateMs);
    if (ksgreaterthan::IsComplement(relation)) {
        PrintPhase("add public one-hot complement", report.complementMs);
    }
    PrintPhase(
        std::string(ksgreaterthan::RelationName(relation)) + " total",
        report.totalMs);
}

void PrintConfiguration(const RuntimeConfig& config,
                        ksgreaterthan::Relation relation) {
    std::cout << "Runtime-configurable CKKS Kogge-Stone "
              << ksgreaterthan::RelationName(relation) << "\n";
    std::cout << "============================================================\n";
    std::cout << "backend / mode          : "
              << ksadder::BackendName(config.backend) << " / "
              << (ksgreaterthan::IsEquality(relation)
                      ? "endpoint-or"
                      : ksadder::AlgorithmName(config.algorithm))
              << "\n";
    std::cout << "word bits / layers      : "
              << config.wordBits << " / " << config.layers << "\n";
    std::cout << "packed words/ciphertext : "
              << config.words << " (maximum " << config.maxWords() << ")\n";
    std::cout << "slots used / available  : "
              << config.usedSlots() << " / " << config.slots << "\n";
    std::cout << "slots per word          : " << config.stride;
    if (config.guardFreeLayout) {
        std::cout << " (no guard slots)\n";
        if (ksgreaterthan::IsEquality(relation)) {
            std::cout << "segment isolation       : endpoint dependency only\n";
        }
        else {
            std::cout << "segment delimiter       : Q[word head] = 0\n";
        }
        std::cout << "per-layer boundary masks: 0\n";
    }
    else {
        std::cout << " (" << config.wordBits
                  << " active + " << config.guardBits << " guard)\n";
    }
    std::cout << "result slot / semantics : bit 0 / "
              << ksgreaterthan::RelationExpression(relation) << "\n";
    std::cout << "multiplicative depth    : "
              << config.multiplicativeDepth << "\n";
    std::cout << "logical GPU devices     : "
              << ksadder::DeviceList(config.devices) << "\n";
    std::cout << "bootstrap / profile     : one required / "
              << (config.detailedProfile ? "yes" : "no") << "\n";
    std::cout << "level placement         : "
              << ksadder::LevelPlacementName(config.levelPlacement) << "\n";
    std::cout << "input / raw level       : "
              << config.inputLevel << " / "
              << config.expectedOutputLevel << "\n";
    std::cout << "comparison level cost   : "
              << config.circuitLevelCost << "\n";
    std::cout << "graph/workspace init    : "
              << (config.graphInit ? "yes" : "no") << "\n";
    if (!config.hasFullPrefix()) {
        std::cout << "warning                  : partial prefix selected; "
                     "full-comparison verification may fail\n";
    }
}

RuntimeConfig ParseComparisonCommandLine(
    int argc,
    char** argv,
    ksgreaterthan::Relation relation) {
    for (int index = 1; index < argc; ++index) {
        if (std::string(argv[index]) == "--bootstrap") {
            throw std::invalid_argument(
                std::string("--bootstrap was removed; ") +
                ksgreaterthan::RelationName(relation) +
                " always performs exactly one final Bootstrap");
        }
    }
    auto config = ksadder::ParseCommandLine(
        argc, argv, true,
        ksadder::Algorithm::ComplexStandard, true);
    config.bootstrap = true;
    config.projectOutputToReal = false;
    if (ksgreaterthan::IsEquality(relation) &&
        config.algorithm != ksadder::Algorithm::ComplexStandard) {
        throw std::invalid_argument(
            "EQ/NE use the fixed standard-key endpoint-or kernel; "
            "omit --mode or pass --mode complex-standard");
    }
    ksgreaterthan::FinalizeComparisonLevelPlacement(&config, relation);
    return config;
}

std::string ComparisonUsage(
    const char* executable,
    ksgreaterthan::Relation relation) {
    std::string usage = ksadder::Usage(
        executable, ksadder::Algorithm::ComplexStandard, true);
    const std::string line =
        "  --bootstrap 0|1   final bootstrap (default 1)\n";
    if (const size_t position = usage.find(line);
        position != std::string::npos) {
        usage.erase(position, line.size());
    }
    usage += "\n";
    usage += ksgreaterthan::RelationName(relation);
    usage += " always performs exactly one final Bootstrap.\n";
    return usage;
}

inline int Run(int argc,
               char** argv,
               ksgreaterthan::Relation relation) {
    try {
        RuntimeConfig config;
        try {
            config = ParseComparisonCommandLine(
                argc, argv, relation);
        }
        catch (const std::runtime_error& error) {
            if (std::string(error.what()) == "help") {
                std::cout << ComparisonUsage(argv[0], relation);
                return 0;
            }
            throw;
        }
        ksadder::ConfigureStableTwistedRuntime(config);
        PrintConfiguration(config, relation);

        const auto programBegin = Clock::now();
        PhaseTimes phase;
        const std::vector<uint32_t> levelBudget{3, 3};
        const std::vector<uint32_t> bsgsDim{0, 0};

        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetSecretKeyDist(UNIFORM_TERNARY);
        parameters.SetSecurityLevel(HEStd_128_classic);
        parameters.SetRingDim(ksadder::kDefaultRingDimension);
        parameters.SetBatchSize(config.slots);
        parameters.SetCKKSDataType(COMPLEX);
        parameters.SetMultiplicativeDepth(config.multiplicativeDepth);
        parameters.SetScalingTechnique(FLEXIBLEAUTO);
        parameters.SetScalingModSize(59);
        parameters.SetFirstModSize(60);
        parameters.SetKeySwitchTechnique(HYBRID);
        parameters.SetNumLargeDigits(3);
        parameters.SetDevices(std::vector<int>(config.devices));
        parameters.SetPlaintextAutoload(false);
        parameters.SetCiphertextAutoload(true);

        auto begin = Clock::now();
        auto cc = GenCryptoContext(parameters);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);
        cc->Enable(FHE);
        phase.contextMs = ksadder::Milliseconds(begin, Clock::now());

        begin = Clock::now();
        auto keys = cc->KeyGen();
        phase.keyGenMs = ksadder::Milliseconds(begin, Clock::now());

        begin = Clock::now();
        cc->EvalMultKeyGen(keys.secretKey);
        phase.multKeyGenMs = ksadder::Milliseconds(begin, Clock::now());

        const auto twistedRotations =
            ksadder::RotationIndices(config);
        const auto ordinaryRotations =
            ksgreaterthan::OrdinaryRotationIndices(config);
        begin = Clock::now();
        cc->EvalRotateKeyGen(keys.secretKey, ordinaryRotations);
        phase.rotateKeyGenMs =
            ksadder::Milliseconds(begin, Clock::now());

        if (config.algorithm == ksadder::Algorithm::ComplexTwisted) {
            begin = Clock::now();
            cc->EvalTwistedProductKeyGen(
                keys.secretKey, twistedRotations, config.slots);
            phase.twistedKeyGenMs =
                ksadder::Milliseconds(begin, Clock::now());
        }

        begin = Clock::now();
        cc->EvalBootstrapSetup(
            levelBudget, bsgsDim, config.slots, 0);
        phase.bootstrapSetupMs =
            ksadder::Milliseconds(begin, Clock::now());

        begin = Clock::now();
        cc->EvalBootstrapKeyGen(keys.secretKey, config.slots);
        phase.bootstrapKeyGenMs =
            ksadder::Milliseconds(begin, Clock::now());

        begin = Clock::now();
        cc->LoadContext(keys.publicKey);
        cc->Synchronize();
        phase.loadContextMs =
            ksadder::Milliseconds(begin, Clock::now());

        begin = Clock::now();
        auto masks = ksgreaterthan::IsEquality(relation)
            ? ksgreaterthan::BuildAndLoadEqualityMasks(
                  cc, config,
                  relation == ksgreaterthan::Relation::Equal)
            : ksgreaterthan::BuildAndLoadMasks(
                  cc, config, ksgreaterthan::IsComplement(relation));
        phase.maskBuildLoadMs =
            ksadder::Milliseconds(begin, Clock::now());

        begin = Clock::now();
        const auto input = BuildInputs(config, relation);
        phase.inputBuildMs =
            ksadder::Milliseconds(begin, Clock::now());

        std::cout << "\nInput samples\n";
        const uint32_t samples = std::min(config.words, uint32_t{4});
        for (uint32_t word = 0; word < samples; ++word) {
            std::cout << "  word " << word << " A = "
                      << ksadder::BitsToHex(input.a, config, word) << "\n";
            std::cout << "  word " << word << " B = "
                      << ksadder::BitsToHex(input.b, config, word) << "\n";
            std::cout << "  word " << word << " expected "
                      << ksgreaterthan::RelationName(relation) << " = "
                      << static_cast<uint32_t>(input.expected[word])
                      << "\n";
        }

        begin = Clock::now();
        auto pA = cc->MakeCKKSPackedPlaintext(
            input.encodedA, 1, config.inputLevel, nullptr, config.slots);
        auto pB = cc->MakeCKKSPackedPlaintext(
            input.encodedB, 1, config.inputLevel, nullptr, config.slots);
        pA->SetLength(config.slots);
        pB->SetLength(config.slots);
        phase.encodingMs = ksadder::Milliseconds(begin, Clock::now());

        begin = Clock::now();
        auto cA = cc->Encrypt(keys.publicKey, pA);
        auto cB = cc->Encrypt(keys.publicKey, pB);
        cc->Synchronize();
        phase.encryptionMs =
            ksadder::Milliseconds(begin, Clock::now());

        if (cA->GetLevel() != config.inputLevel ||
            cB->GetLevel() != config.inputLevel) {
            throw std::runtime_error(
                "input ciphertext level does not match the scheduled level");
        }

        std::cout << "\nScheduled input state\n";
        std::cout << "  level / scale degree : "
                  << cA->GetLevel() << " / "
                  << cA->GetNoiseScaleDeg() << "\n";

        if (config.graphInit) {
            begin = Clock::now();
            ksgreaterthan::InitializeConfiguredComparisonRuntime(
                cc, cA, cB, config, masks, relation);
            phase.graphInitMs =
                ksadder::Milliseconds(begin, Clock::now());
        }

        ksgreaterthan::TimingReport timing;
        cc->Synchronize();
        begin = Clock::now();
        auto comparison = ksgreaterthan::EvalConfiguredComparison(
            cc, cA, cB, config, masks, relation,
            config.detailedProfile ? &timing : nullptr);
        cc->Synchronize();
        phase.comparisonMs =
            ksadder::Milliseconds(begin, Clock::now());

        std::cout << "\n"
                  << ksgreaterthan::RelationName(relation)
                  << " ciphertext state\n";
        std::cout << "  level / scale degree : "
                  << comparison->GetLevel() << " / "
                  << comparison->GetNoiseScaleDeg() << "\n";
        const uint32_t usableLevels =
            UsableLevelsRemaining(config, comparison);
        std::cout << "  usable ct-ct levels   : "
                  << usableLevels << "\n";

        if (comparison->GetLevel() != config.expectedOutputLevel) {
            throw std::runtime_error(
                "comparison consumed an unexpected number of modulus levels: "
                "expected raw level " +
                std::to_string(config.expectedOutputLevel) +
                ", got " + std::to_string(comparison->GetLevel()));
        }
        if (config.levelPlacement == ksadder::LevelPlacement::Bottom &&
            usableLevels != 0) {
            throw std::runtime_error(
                "bottom placement did not exhaust all circuit-usable levels");
        }

        cc->Synchronize();
        begin = Clock::now();
        auto evaluated = cc->EvalBootstrap(comparison);
        cc->Synchronize();
        phase.bootstrapMs =
            ksadder::Milliseconds(begin, Clock::now());

        std::cout << "\nFinal ciphertext state\n";
        std::cout << "  level / scale degree : "
                  << evaluated->GetLevel() << " / "
                  << evaluated->GetNoiseScaleDeg() << "\n";
        std::cout << "  restored ct-ct levels : "
                  << UsableLevelsRemaining(config, evaluated) << "\n";

        begin = Clock::now();
        Plaintext decrypted;
        const auto decryptResult =
            cc->Decrypt(keys.secretKey, evaluated, &decrypted);
        if (!decryptResult.isValid) {
            throw std::runtime_error("decryption failed");
        }
        decrypted->SetLength(config.slots);
        const auto values = decrypted->GetCKKSPackedValue();
        phase.decryptionMs =
            ksadder::Milliseconds(begin, Clock::now());

        begin = Clock::now();
        const auto errors =
            CheckResult(config, values, input.expected);
        phase.verificationMs =
            ksadder::Milliseconds(begin, Clock::now());
        phase.totalMs =
            ksadder::Milliseconds(programBegin, Clock::now());

        std::cout
            << "\n================ Correctness and precision ================\n";
        std::cout << std::scientific << std::setprecision(10);
        std::cout << "  valid comparison slots : " << config.words << "\n";
        std::cout << "  maximum absolute error : "
                  << errors.maxError << "\n";
        std::cout << "  mean absolute error    : "
                  << errors.meanError << "\n";
        std::cout << "  RMSE                   : "
                  << errors.rmse << "\n";
        std::cout << "  maximum inactive value : "
                  << errors.maxInactive << "\n";
        std::cout << "  maximum imaginary part : "
                  << errors.maxImaginary << "\n";
        std::cout << "  worst precision        : "
                  << ksadder::PrecisionBits(errors.maxError) << " bits\n";
        std::cout << "  threshold failures     : "
                  << errors.failures << "\n";
        std::cout << "  worst location         : word="
                  << errors.worstWord << ", result bit=0\n";
        std::cout << "  expected / actual      : "
                  << errors.worstExpected << " / "
                  << errors.worstActual << "\n";
        std::cout << "  verdict                : "
                  << (errors.failures == 0 ? "PASS" : "FAIL") << "\n";

        if (config.detailedProfile) {
            PrintProfile(config, timing, relation);
        }

        std::cout << "\n================ End-to-end timing ================\n";
        PrintPhase("CryptoContext generation and Enable", phase.contextMs);
        PrintPhase("KeyGen", phase.keyGenMs);
        PrintPhase("EvalMultKeyGen", phase.multKeyGenMs);
        PrintPhase("ordinary rotation key generation",
                   phase.rotateKeyGenMs);
        if (ksgreaterthan::IsEquality(relation)) {
            std::cout << "\nEndpoint OR reduction accounting\n";
            std::cout << "  XOR ct-ct products     : 1\n";
            std::cout << "  reduction layers       : "
                      << config.layers << "\n";
            std::cout << "  reduction rotations    : "
                      << config.layers << "\n";
            std::cout << "  reduction ct-ct products: "
                      << config.layers << "\n";
            std::cout << "  prefix conjugations    : 0\n";
            std::cout << "  boundary ct-pt masks   : 0\n";
            std::cout << "  exact endpoint mask    : 1 (zero levels)\n";
            std::cout << "  scalar scale lifts     : 2 (multiply by 1)\n";
            std::cout << "  explicit rescale       : 1\n";
            std::cout << "  final result rotation  : 1\n";
            std::cout << "  public complement      : "
                      << (relation == ksgreaterthan::Relation::Equal
                              ? 1 : 0)
                      << " (linear, zero levels)\n";
            std::cout << "  nonstandard eval keys  : 0\n";
            std::cout << "  guard slots            : 0\n";
        }
        else if (config.algorithm == ksadder::Algorithm::ComplexTwisted) {
            PrintPhase("mixed-basis twisted KSK generation",
                       phase.twistedKeyGenMs);
        }
        PrintPhase("EvalBootstrapSetup", phase.bootstrapSetupMs);
        PrintPhase("EvalBootstrapKeyGen", phase.bootstrapKeyGenMs);
        PrintPhase("LoadContext and evaluation keys", phase.loadContextMs);
        PrintPhase("mask encoding/loading", phase.maskBuildLoadMs);
        PrintPhase("input/reference generation", phase.inputBuildMs);
        PrintPhase("CKKS encoding", phase.encodingMs);
        PrintPhase("input encryption/loading", phase.encryptionMs);
        if (config.graphInit) {
            PrintPhase("graph/workspace initialization",
                       phase.graphInitMs);
        }
        PrintPhase(
            std::string("Kogge-Stone ") +
                ksgreaterthan::RelationName(relation) +
                " (one real execution)",
            phase.comparisonMs);
        PrintPhase("one EvalBootstrap (single call)", phase.bootstrapMs);
        PrintPhase("copy/decrypt", phase.decryptionMs);
        PrintPhase("correctness scan", phase.verificationMs);
        PrintPhase("full program wall", phase.totalMs);

        const double onlineMs =
            phase.comparisonMs + phase.bootstrapMs;
        std::cout << "\n================ Online result ================\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  circuit latency        : "
                  << phase.comparisonMs << " ms\n";
        std::cout << "  bootstrap single call  : "
                  << phase.bootstrapMs << " ms\n";
        std::cout << "  circuit + bootstrap    : "
                  << onlineMs << " ms\n";
        std::cout << "  amortized per word     : "
                  << onlineMs / config.words << " ms\n";
        std::cout << "  online throughput      : "
                  << (onlineMs > 0.0
                          ? config.words * 1000.0 / onlineMs
                          : 0.0)
                  << " word/s\n";

        if (config.algorithm == ksadder::Algorithm::ComplexTwisted) {
            std::cout << "\nTwisted prefix accounting\n";
            std::cout << "  prefix layers          : "
                      << config.layers << "\n";
            std::cout << "  standalone rotate KSK  : 0 inside prefix\n";
            std::cout << "  conjugation KSK        : "
                      << config.layers << " inside prefix\n";
            std::cout << "  mixed-basis KSK        : "
                      << 2 * config.layers << "\n";
            std::cout << "  prefix output ModDowns : "
                      << 2 * config.layers << "\n";
            std::cout << "  final result rotation  : 1\n";
        }
        else if (config.algorithm ==
                 ksadder::Algorithm::ComplexStandard) {
            std::cout << "\nGuard-free complex prefix accounting\n";
            std::cout << "  prefix layers          : "
                      << config.layers << "\n";
            std::cout << "  prefix rotations       : "
                      << config.layers << "\n";
            std::cout << "  prefix conjugations    : "
                      << config.layers << "\n";
            std::cout << "  prefix ct-ct products  : "
                      << config.layers << "\n";
            std::cout << "  boundary ct-pt masks   : 0 inside prefix\n";
            std::cout << "  head-zero extra ops    : 0\n";
            std::cout << "  result extraction conj : 1\n";
            std::cout << "  final result rotation  : 1\n";
            std::cout << "  public complement      : "
                      << (ksgreaterthan::IsComplement(relation) ? 1 : 0)
                      << " (linear, zero levels)\n";
            std::cout << "  nonstandard eval keys  : 0\n";
            std::cout << "  guard slots            : 0\n";
        }

        return errors.failures == 0 ? 0 : 2;
    }
    catch (const std::exception& error) {
        std::cerr << "[fatal] " << error.what() << "\n";
        std::cerr << "Use --help to list supported runtime options.\n";
        return 1;
    }
}

}  // namespace kscomparisonapp
