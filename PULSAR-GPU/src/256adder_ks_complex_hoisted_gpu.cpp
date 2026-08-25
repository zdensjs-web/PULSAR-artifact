#include <fideslib.hpp>

#include "adder_ks_complex_hoisted_gpu.h"
#include "ks_adder_common.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace fideslib;

namespace {

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
    double additionMs = 0.0;
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
    double maxGuard = 0.0;
    double maxImaginary = 0.0;
    uint64_t failures = 0;
    uint32_t worstWord = 0;
    uint32_t worstBit = 0;
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

Inputs BuildInputs(const RuntimeConfig& config) {
    Inputs input;
    input.a.assign(config.slots, 0);
    input.b.assign(config.slots, 0);
    input.encodedA.assign(config.slots, 0.0);
    input.encodedB.assign(config.slots, 0.0);

    std::mt19937_64 generator(0x4B53414444323536ULL);
    for (uint32_t word = 0; word < config.words; ++word) {
        for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
            const uint32_t slot = ksadder::Slot(config, word, bit);
            input.a[slot] = static_cast<uint8_t>(generator() & 1U);
            input.b[slot] = static_cast<uint8_t>(generator() & 1U);
        }
    }

    // Carry across the complete word.
    if (config.words >= 1) {
        for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
            input.a[ksadder::Slot(config, 0, bit)] = 1;
            input.b[ksadder::Slot(config, 0, bit)] = 0;
        }
        input.b[ksadder::Slot(config, 0, 0)] = 1;
    }

    // Carry into the most significant bit without unsigned overflow.
    if (config.words >= 2) {
        for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
            input.a[ksadder::Slot(config, 1, bit)] =
                static_cast<uint8_t>(bit + 1 < config.wordBits);
            input.b[ksadder::Slot(config, 1, bit)] = 0;
        }
        input.b[ksadder::Slot(config, 1, 0)] = 1;
    }

    // All-propagate pattern exercises the complex real channel.
    if (config.words >= 3) {
        for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
            input.a[ksadder::Slot(config, 2, bit)] =
                static_cast<uint8_t>(bit & 1U);
            input.b[ksadder::Slot(config, 2, bit)] =
                static_cast<uint8_t>((bit & 1U) == 0);
        }
    }

    ksadder::BuildExpectedSum(
        config, input.a, input.b, &input.expected);
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
    uint64_t activeCount = 0;

    for (uint32_t word = 0; word < config.words; ++word) {
        for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
            const uint32_t slot = ksadder::Slot(config, word, bit);
            const double actual = values[slot].real();
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
            report.maxImaginary =
                std::max(report.maxImaginary, std::abs(values[slot].imag()));
            if (!std::isfinite(actual) || error >= 0.5) {
                ++report.failures;
            }
            ++activeCount;
        }

        for (uint32_t guard = 0; guard < config.guardBits; ++guard) {
            const uint32_t slot =
                ksadder::Slot(config, word, config.wordBits + guard);
            report.maxGuard = std::max(
                report.maxGuard, std::abs(values[slot]));
        }
    }

    report.meanError =
        static_cast<double>(sumAbsolute / activeCount);
    report.rmse =
        std::sqrt(static_cast<double>(sumSquared / activeCount));
    return report;
}

void PrintProfile(const RuntimeConfig& config,
                  const ksadder::TimingReport& report) {
    std::cout << "\n================ Prefix internal timing ================\n";
    PrintPhase("preprocess G/P", report.preprocessMs);
    if (config.algorithm != ksadder::Algorithm::Baseline) {
        PrintPhase("pack Z=P/2+iG", report.packStateMs);
    }

    for (uint32_t layer = 0; layer < config.layers; ++layer) {
        const auto& item = report.layers[layer];
        std::cout << "\n  Layer " << layer
                  << " (distance=" << (1U << layer) << ")\n";
        if (config.algorithm == ksadder::Algorithm::Baseline) {
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
            PrintPhase("    mixed-basis twisted product", item.productMs);
            PrintPhase("    keep generate channel", item.auxiliaryMs);
        }
        PrintPhase("    update state", item.updateMs);
        PrintPhase("    layer wall", item.wallMs);
    }

    PrintPhase("carry-state rotation", report.carryRotateMs);
    PrintPhase("output masks", report.outputTermsMs);
    PrintPhase("sum reconstruction", report.reconstructMs);
    if (report.projectedToReal) {
        PrintPhase("final real projection", report.projectionMs);
    }
    PrintPhase("adder total", report.totalMs);
}

void PrintConfiguration(const RuntimeConfig& config) {
    std::cout << "Runtime-configurable CKKS Kogge-Stone adder\n";
    std::cout << "============================================================\n";
    std::cout << "backend / mode          : "
              << ksadder::BackendName(config.backend) << " / "
              << ksadder::AlgorithmName(config.algorithm) << "\n";
    std::cout << "word bits / layers      : "
              << config.wordBits << " / " << config.layers << "\n";
    std::cout << "packed words/ciphertext : "
              << config.words << " (maximum " << config.maxWords() << ")\n";
    std::cout << "slots used / available  : "
              << config.usedSlots() << " / " << config.slots << "\n";
    std::cout << "slots per word          : " << config.stride;
    if (config.guardFreeLayout) {
        std::cout << " (no guard slots)\n";
        std::cout << "segment delimiter       : P[word head] = 0\n";
        std::cout << "per-layer boundary masks: 0\n";
    }
    else {
        std::cout << " (" << config.wordBits
                  << " active + " << config.guardBits << " guard)\n";
    }
    std::cout << "multiplicative depth    : "
              << config.multiplicativeDepth << "\n";
    std::cout << "logical GPU devices     : "
              << ksadder::DeviceList(config.devices) << "\n";
    std::cout << "bootstrap / profile     : "
              << (config.bootstrap ? "yes" : "no") << " / "
              << (config.detailedProfile ? "yes" : "no") << "\n";
    std::cout << "level placement         : "
              << ksadder::LevelPlacementName(config.levelPlacement) << "\n";
    std::cout << "input / raw level       : "
              << config.inputLevel << " / "
              << config.expectedOutputLevel << "\n";
    std::cout << "adder level cost        : "
              << config.circuitLevelCost << "\n";
    std::cout << "graph/workspace init    : "
              << (config.graphInit ? "yes" : "no") << "\n";
    if (!config.hasFullPrefix()) {
        std::cout << "warning                  : partial prefix selected; "
                     "full-addition verification may fail\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        RuntimeConfig config;
        try {
            config = ksadder::ParseCommandLine(
                argc, argv, true,
                ksadder::Algorithm::ComplexStandard, true);
        }
        catch (const std::runtime_error& error) {
            if (std::string(error.what()) == "help") {
                std::cout << ksadder::Usage(
                    argv[0], ksadder::Algorithm::ComplexStandard, true);
                return 0;
            }
            throw;
        }
        ksadder::FinalizeAdderLevelPlacement(&config);
        ksadder::ConfigureStableTwistedRuntime(config);
        PrintConfiguration(config);

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

        const auto rotations = ksadder::RotationIndices(config);
        const auto ordinaryRotations =
            ksadder::OrdinaryRotationIndices(config);
        begin = Clock::now();
        cc->EvalRotateKeyGen(keys.secretKey, ordinaryRotations);
        phase.rotateKeyGenMs =
            ksadder::Milliseconds(begin, Clock::now());

        if (config.algorithm == ksadder::Algorithm::ComplexTwisted) {
            begin = Clock::now();
            cc->EvalTwistedProductKeyGen(
                keys.secretKey, rotations, config.slots);
            phase.twistedKeyGenMs =
                ksadder::Milliseconds(begin, Clock::now());
        }

        if (config.bootstrap) {
            begin = Clock::now();
            cc->EvalBootstrapSetup(
                levelBudget, bsgsDim, config.slots, 0);
            phase.bootstrapSetupMs =
                ksadder::Milliseconds(begin, Clock::now());

            begin = Clock::now();
            cc->EvalBootstrapKeyGen(keys.secretKey, config.slots);
            phase.bootstrapKeyGenMs =
                ksadder::Milliseconds(begin, Clock::now());
        }

        begin = Clock::now();
        cc->LoadContext(keys.publicKey);
        cc->Synchronize();
        phase.loadContextMs =
            ksadder::Milliseconds(begin, Clock::now());

        begin = Clock::now();
        auto masks = ksadder::BuildAndLoadMasks(cc, config);
        phase.maskBuildLoadMs =
            ksadder::Milliseconds(begin, Clock::now());

        begin = Clock::now();
        const auto input = BuildInputs(config);
        phase.inputBuildMs =
            ksadder::Milliseconds(begin, Clock::now());

        std::cout << "\nInput samples\n";
        const uint32_t samples = std::min(config.words, uint32_t{3});
        for (uint32_t word = 0; word < samples; ++word) {
            std::cout << "  word " << word << " A = "
                      << ksadder::BitsToHex(input.a, config, word) << "\n";
            std::cout << "  word " << word << " B = "
                      << ksadder::BitsToHex(input.b, config, word) << "\n";
            std::cout << "  word " << word << " S = "
                      << ksadder::BitsToHex(
                             input.expected, config, word) << "\n";
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
            ksadder::InitializeConfiguredAdderRuntime(
                cc, cA, cB, config, masks);
            phase.graphInitMs =
                ksadder::Milliseconds(begin, Clock::now());
        }

        ksadder::TimingReport timing;
        cc->Synchronize();
        begin = Clock::now();
        auto sum = ksadder::EvalConfiguredAdder(
            cc, cA, cB, config, masks,
            config.detailedProfile ? &timing : nullptr);
        cc->Synchronize();
        phase.additionMs =
            ksadder::Milliseconds(begin, Clock::now());

        std::cout << "\nAdder ciphertext state\n";
        std::cout << "  level / scale degree : "
                  << sum->GetLevel() << " / "
                  << sum->GetNoiseScaleDeg() << "\n";
        const uint32_t usableLevels =
            UsableLevelsRemaining(config, sum);
        std::cout << "  usable ct-ct levels   : "
                  << usableLevels << "\n";

        if (sum->GetLevel() != config.expectedOutputLevel) {
            throw std::runtime_error(
                "adder consumed an unexpected number of modulus levels: "
                "expected raw level " +
                std::to_string(config.expectedOutputLevel) +
                ", got " + std::to_string(sum->GetLevel()));
        }
        if (config.levelPlacement == ksadder::LevelPlacement::Bottom &&
            usableLevels != 0) {
            throw std::runtime_error(
                "bottom placement did not exhaust all circuit-usable levels");
        }

        auto evaluated = sum;
        if (config.bootstrap) {
            cc->Synchronize();
            begin = Clock::now();
            evaluated = cc->EvalBootstrap(sum);
            cc->Synchronize();
            phase.bootstrapMs =
                ksadder::Milliseconds(begin, Clock::now());
        }

        std::cout << "\nFinal ciphertext state\n";
        std::cout << "  level / scale degree : "
                  << evaluated->GetLevel() << " / "
                  << evaluated->GetNoiseScaleDeg() << "\n";
        if (config.bootstrap) {
            std::cout << "  restored ct-ct levels : "
                      << UsableLevelsRemaining(config, evaluated) << "\n";
        }

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

        std::cout << "\n================ Correctness and precision ================\n";
        std::cout << std::scientific << std::setprecision(10);
        std::cout << "  maximum absolute error : " << errors.maxError << "\n";
        std::cout << "  mean absolute error    : " << errors.meanError << "\n";
        std::cout << "  RMSE                   : " << errors.rmse << "\n";
        if (config.guardBits != 0) {
            std::cout << "  maximum guard value    : "
                      << errors.maxGuard << "\n";
        }
        std::cout << "  maximum imaginary part : "
                  << errors.maxImaginary << "\n";
        std::cout << "  worst precision        : "
                  << ksadder::PrecisionBits(errors.maxError) << " bits\n";
        std::cout << "  threshold failures     : "
                  << errors.failures << "\n";
        std::cout << "  worst location         : word="
                  << errors.worstWord << ", bit=" << errors.worstBit << "\n";
        std::cout << "  expected / actual      : "
                  << errors.worstExpected << " / "
                  << errors.worstActual << "\n";
        std::cout << "  verdict                : "
                  << (errors.failures == 0 ? "PASS" : "FAIL") << "\n";

        if (config.detailedProfile) {
            PrintProfile(config, timing);
        }

        std::cout << "\n================ End-to-end timing ================\n";
        PrintPhase("CryptoContext generation and Enable", phase.contextMs);
        PrintPhase("KeyGen", phase.keyGenMs);
        PrintPhase("EvalMultKeyGen", phase.multKeyGenMs);
        PrintPhase("ordinary rotation key generation",
                   phase.rotateKeyGenMs);
        if (config.algorithm == ksadder::Algorithm::ComplexTwisted) {
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
        PrintPhase("Kogge-Stone adder (one real execution)",
                   phase.additionMs);
        PrintPhase("one EvalBootstrap (single call)", phase.bootstrapMs);
        PrintPhase("copy/decrypt", phase.decryptionMs);
        PrintPhase("correctness scan", phase.verificationMs);
        PrintPhase("full program wall", phase.totalMs);

        const double onlineMs = phase.additionMs + phase.bootstrapMs;
        std::cout << "\n================ Online result ================\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  circuit latency        : "
                  << phase.additionMs << " ms\n";
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
            std::cout << "  carry rotations        : 1 after prefix\n";
            std::cout << "  final projection conj  : "
                      << (config.projectOutputToReal ? 1 : 0) << "\n";
            std::cout << "  nonstandard eval keys  : 0\n";
            std::cout << "  guard slots            : "
                      << config.guardBits << "\n";
        }

        return errors.failures == 0 ? 0 : 2;
    }
    catch (const std::exception& error) {
        std::cerr << "[fatal] " << error.what() << "\n";
        std::cerr << "Use --help to list supported runtime options.\n";
        return 1;
    }
}
