#include <fideslib.hpp>

#include "multiplier_ks_runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fideslib;

namespace {

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
    double multiplicationMs = 0.0;
    double finalAdderGraphInitMs = 0.0;
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
    uint64_t failures = 0;
    uint32_t worstWord = 0;
    uint32_t worstBit = 0;
    double worstExpected = 0.0;
    double worstActual = 0.0;
};

void PrintPhase(const std::string& name, double milliseconds) {
    std::cout << "  " << std::left << std::setw(46) << name
              << std::right << std::setw(13)
              << std::fixed << std::setprecision(3)
              << milliseconds << " ms  ("
              << std::setprecision(6)
              << milliseconds / 1000.0 << " s)\n";
}

uint32_t LevelsRemaining(
    const ksmul::RuntimeConfig& config,
    const Ciphertext<DCRTPoly>& ciphertext) {
    return ksadder::UsableMultiplicationLevels(
        config.multiplicativeDepth,
        ciphertext->GetLevel(),
        ciphertext->GetNoiseScaleDeg());
}

void BuildExpected(const ksmul::RuntimeConfig& config,
                   Inputs* input) {
    input->expected.assign(config.slots, 0);
    for (uint32_t word = 0; word < config.words; ++word) {
        std::vector<uint32_t> columns(config.wordBits, 0);
        for (uint32_t left = 0;
             left < config.wordBits;
             ++left) {
            if (input->a[ksword::Slot(
                    config, word, left)] == 0) {
                continue;
            }
            for (uint32_t right = 0;
                 left + right < config.wordBits;
                 ++right) {
                columns[left + right] +=
                    input->b[ksword::Slot(
                        config, word, right)];
            }
        }

        uint32_t carry = 0;
        for (uint32_t bit = 0;
             bit < config.wordBits;
             ++bit) {
            const uint32_t total = columns[bit] + carry;
            input->expected[ksword::Slot(
                config, word, bit)] =
                static_cast<uint8_t>(total & 1U);
            carry = total >> 1U;
        }
    }
}

Inputs BuildInputs(const ksmul::RuntimeConfig& config) {
    Inputs input;
    input.a.assign(config.slots, 0);
    input.b.assign(config.slots, 0);
    input.encodedA.assign(config.slots, 0.0);
    input.encodedB.assign(config.slots, 0.0);

    std::mt19937_64 generator(0x44414444414D554CULL);
    for (uint32_t word = 0; word < config.words; ++word) {
        for (uint32_t bit = 0;
             bit < config.wordBits;
             ++bit) {
            input.a[ksword::Slot(config, word, bit)] =
                static_cast<uint8_t>(generator() & 1U);
            input.b[ksword::Slot(config, word, bit)] =
                static_cast<uint8_t>(generator() & 1U);
        }
    }

    if (config.words >= 1) {
        for (uint32_t bit = 0;
             bit < config.wordBits;
             ++bit) {
            input.a[ksword::Slot(config, 0, bit)] = 1;
            input.b[ksword::Slot(config, 0, bit)] = 0;
        }
        input.b[ksword::Slot(config, 0, 0)] = 1;
    }
    if (config.words >= 2) {
        for (uint32_t bit = 0;
             bit < config.wordBits;
             ++bit) {
            input.a[ksword::Slot(config, 1, bit)] = 0;
            input.b[ksword::Slot(config, 1, bit)] = 0;
        }
        input.a[ksword::Slot(
            config, 1, config.wordBits / 2U)] = 1;
        input.b[ksword::Slot(
            config, 1, config.wordBits / 2U - 1U)] = 1;
    }
    if (config.words >= 3) {
        for (uint32_t bit = 0;
             bit < config.wordBits;
             ++bit) {
            input.a[ksword::Slot(config, 2, bit)] = 1;
            input.b[ksword::Slot(config, 2, bit)] = 1;
        }
    }

    BuildExpected(config, &input);
    for (uint32_t slot = 0; slot < config.slots; ++slot) {
        input.encodedA[slot] =
            static_cast<double>(input.a[slot]);
        input.encodedB[slot] =
            static_cast<double>(input.b[slot]);
    }
    return input;
}

ErrorReport CheckResult(
    const ksmul::RuntimeConfig& config,
    const std::vector<double>& values,
    const std::vector<uint8_t>& expected) {
    ErrorReport report;
    long double sumAbsolute = 0.0;
    long double sumSquared = 0.0;
    uint64_t active = 0;

    for (uint32_t word = 0; word < config.words; ++word) {
        for (uint32_t bit = 0;
             bit < config.wordBits;
             ++bit) {
            const uint32_t slot =
                ksword::Slot(config, word, bit);
            const double target =
                static_cast<double>(expected[slot]);
            const double actual = values[slot];
            const double error = std::abs(actual - target);
            if (error > report.maxError) {
                report.maxError = error;
                report.worstWord = word;
                report.worstBit = bit;
                report.worstExpected = target;
                report.worstActual = actual;
            }
            sumAbsolute += error;
            sumSquared +=
                static_cast<long double>(error) * error;
            if (!std::isfinite(actual) || error >= 0.5) {
                ++report.failures;
            }
            ++active;
        }
        for (uint32_t guard = 0;
             guard < config.guardBits;
             ++guard) {
            report.maxGuard = std::max(
                report.maxGuard,
                std::abs(values[ksword::Slot(
                    config,
                    word,
                    config.wordBits + guard)]));
        }
    }

    report.meanError =
        static_cast<double>(sumAbsolute / active);
    report.rmse =
        std::sqrt(static_cast<double>(sumSquared / active));
    return report;
}

void PrintBooleanCheckpoint(
    const std::string& label,
    const CryptoContext<DCRTPoly>& cc,
    PrivateKey<DCRTPoly>& secretKey,
    Ciphertext<DCRTPoly> ciphertext,
    const ksmul::RuntimeConfig& config,
    const std::vector<uint8_t>* expected = nullptr) {
    std::cout << "  " << label << "\n";
    try {
        Plaintext plaintext;
        const auto status = cc->Decrypt(
            secretKey, ciphertext, &plaintext);
        if (!status.isValid) {
            std::cout << "    decrypt status       : invalid\n";
            return;
        }
        plaintext->SetLength(config.slots);
        const auto values = plaintext->GetRealPackedValue();
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();
        double maxBooleanDistance = 0.0;
        double maxGuard = 0.0;
        uint64_t outsideBooleanBasin = 0;
        uint64_t nearTwo = 0;
        uint64_t nonfinite = 0;

        for (uint32_t word = 0; word < config.words; ++word) {
            for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
                const double value =
                    values[ksword::Slot(config, word, bit)];
                if (!std::isfinite(value)) {
                    ++nonfinite;
                    continue;
                }
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
                const double distance = std::min(
                    std::abs(value), std::abs(value - 1.0));
                maxBooleanDistance = std::max(
                    maxBooleanDistance, distance);
                if (distance >= 0.25) {
                    ++outsideBooleanBasin;
                }
                if (std::abs(value - 2.0) < 0.25) {
                    ++nearTwo;
                }
            }
            for (uint32_t guard = 0;
                 guard < config.guardBits;
                 ++guard) {
                maxGuard = std::max(
                    maxGuard,
                    std::abs(values[ksword::Slot(
                        config,
                        word,
                        config.wordBits + guard)]));
            }
        }

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "    state                : "
                  << ciphertext->GetLevel() << "/"
                  << ciphertext->GetNoiseScaleDeg() << "\n";
        std::cout << "    active min / max     : "
                  << minimum << " / " << maximum << "\n";
        std::cout << "    max dist to {0,1}    : "
                  << maxBooleanDistance << "\n";
        std::cout << "    outside 0.25 basin   : "
                  << outsideBooleanBasin << "\n";
        std::cout << "    values near 2        : "
                  << nearTwo << "\n";
        std::cout << "    guard maximum        : "
                  << maxGuard << "\n";
        std::cout << "    nonfinite values     : "
                  << nonfinite << "\n";

        if (expected) {
            const auto errors = CheckResult(
                config, values, *expected);
            std::cout << "    expected max error   : "
                      << errors.maxError << "\n";
            std::cout << "    expected failures    : "
                      << errors.failures << "\n";
        }
    }
    catch (const std::exception& error) {
        std::cout << "    checkpoint decode    : FAIL ("
                  << error.what() << ")\n";
    }
}

std::vector<double> DecryptRealCheckpoint(
    const CryptoContext<DCRTPoly>& cc,
    PrivateKey<DCRTPoly>& secretKey,
    Ciphertext<DCRTPoly> ciphertext,
    uint32_t slots) {
    Plaintext plaintext;
    const auto status = cc->Decrypt(
        secretKey, ciphertext, &plaintext);
    if (!status.isValid) {
        throw std::runtime_error(
            "checkpoint decryption returned invalid status");
    }
    plaintext->SetLength(slots);
    return plaintext->GetRealPackedValue();
}

void PrintDaddaTwoRowSemantics(
    const CryptoContext<DCRTPoly>& cc,
    PrivateKey<DCRTPoly>& secretKey,
    const std::array<Ciphertext<DCRTPoly>, 2>& rows,
    const ksmul::RuntimeConfig& config,
    const std::vector<uint8_t>& expected) {
    std::cout << "  thresholded Dadda two-row sum\n";
    try {
        const auto row0 = DecryptRealCheckpoint(
            cc, secretKey, rows[0], config.slots);
        const auto row1 = DecryptRealCheckpoint(
            cc, secretKey, rows[1], config.slots);
        uint64_t failures = 0;
        uint32_t firstWord = 0;
        uint32_t firstBit = 0;
        bool recorded = false;

        for (uint32_t word = 0; word < config.words; ++word) {
            uint32_t carry = 0;
            for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
                const uint32_t slot =
                    ksword::Slot(config, word, bit);
                const uint32_t total =
                    static_cast<uint32_t>(row0[slot] >= 0.5) +
                    static_cast<uint32_t>(row1[slot] >= 0.5) +
                    carry;
                const uint8_t actual =
                    static_cast<uint8_t>(total & 1U);
                carry = total >> 1U;
                if (actual != expected[slot]) {
                    ++failures;
                    if (!recorded) {
                        firstWord = word;
                        firstBit = bit;
                        recorded = true;
                    }
                }
            }
        }

        std::cout << "    expected failures    : "
                  << failures << "\n";
        if (recorded) {
            std::cout << "    first mismatch       : word="
                      << firstWord << ", bit=" << firstBit << "\n";
        }
    }
    catch (const std::exception& error) {
        std::cout << "    semantic audit       : FAIL ("
                  << error.what() << ")\n";
    }
}

void PrintComplexAdderCheckpoint(
    const CryptoContext<DCRTPoly>& cc,
    PrivateKey<DCRTPoly>& secretKey,
    const char* stage,
    int32_t layer,
    const Ciphertext<DCRTPoly>& ciphertext,
    uint32_t slots) {
    auto mutableCiphertext = ciphertext;
    Plaintext plaintext;
    const auto status = cc->Decrypt(
        secretKey, mutableCiphertext, &plaintext);
    if (!status.isValid) {
        throw std::runtime_error(
            std::string("complex checkpoint decryption failed at ") + stage);
    }
    plaintext->SetLength(slots);
    const auto values = plaintext->GetCKKSPackedValue();
    double maxAbs = 0.0;
    double minReal = std::numeric_limits<double>::infinity();
    double maxReal = -std::numeric_limits<double>::infinity();
    double minImaginary = std::numeric_limits<double>::infinity();
    double maxImaginary = -std::numeric_limits<double>::infinity();
    uint64_t nonfinite = 0;
    for (const auto& value : values) {
        if (!std::isfinite(value.real()) ||
            !std::isfinite(value.imag())) {
            ++nonfinite;
            continue;
        }
        maxAbs = std::max(maxAbs, std::abs(value));
        minReal = std::min(minReal, value.real());
        maxReal = std::max(maxReal, value.real());
        minImaginary = std::min(minImaginary, value.imag());
        maxImaginary = std::max(maxImaginary, value.imag());
    }

    if (nonfinite == values.size()) {
        minReal = maxReal = 0.0;
        minImaginary = maxImaginary = 0.0;
    }

    std::cout << "  [complex-audit] stage="
              << std::left << std::setw(15) << stage
              << std::right << " layer=" << std::setw(2) << layer
              << " state=" << ciphertext->GetLevel() << "/"
              << ciphertext->GetNoiseScaleDeg()
              << std::scientific << std::setprecision(4)
              << " max|z|=" << maxAbs
              << " Re=[" << minReal << "," << maxReal << "]"
              << " Im=[" << minImaginary << "," << maxImaginary << "]"
              << " nonfinite=" << nonfinite << "\n";
}

void PrintConfiguration(const ksmul::RuntimeConfig& config) {
    const auto targets = ksmul::DaddaTargets(config.wordBits);

    std::cout << "Runtime-configurable CKKS Dadda + Kogge-Stone multiplier\n";
    std::cout << "============================================================\n";
    std::cout << "backend                 : "
              << ksword::BackendName(config.backend) << "\n";
    std::cout << "word bits               : "
              << config.wordBits << "\n";
    std::cout << "packed words/ciphertext : "
              << config.words << " (maximum "
              << config.maxWords() << ")\n";
    std::cout << "slots used / available  : "
              << config.usedSlots() << " / "
              << config.slots << "\n";
    std::cout << "slots per word          : "
              << config.stride << " ("
              << config.wordBits << " active + "
              << config.guardBits << " guard)\n";
    std::cout << "product semantics       : A * B mod 2^"
              << config.wordBits << "\n";
    std::cout << "partial-product rows    : "
              << config.wordBits << "\n";
    std::cout << "Dadda stages/targets    : "
              << targets.size() << " / ";
    for (uint32_t index = 0;
         index < targets.size();
         ++index) {
        if (index != 0) {
            std::cout << ",";
        }
        std::cout << targets[index];
    }
    std::cout << "\n";
    std::cout << "maximum merge batch     : "
              << config.mergeBatch << "\n";
    std::cout << "final two-row adder     : complex-standard (Z=P/2+iG)\n";
    std::cout << "Dadda purification stages: ";
    for (uint32_t index = 0;
         index < config.daddaPurificationStages.size();
         ++index) {
        if (index != 0) {
            std::cout << ",";
        }
        std::cout << config.daddaPurificationStages[index];
    }
    std::cout << "\n";
    std::cout << "Dadda pre-refresh maximum: "
              << config.expectedDaddaPreRefreshLevel << "\n";
    std::cout << "refreshed Dadda row state: "
              << ksmul::detail::ExpectedDaddaOutputLevel(config)
              << " / 2\n";
    std::cout << "normalized adder input  : "
              << ksmul::detail::ExpectedDaddaOutputLevel(config) + 1U
              << " / 1\n";
    std::cout << "final-adder level demand: "
              << ksmul::detail::FinalAdderRequiredLevels(config)
              << "\n";
    std::cout << "scheduled raw level     : "
              << config.expectedOutputLevel
              << "\n";
    std::cout << "configured depth        : "
              << config.multiplicativeDepth << "\n";
    std::cout << "level placement         : bottom\n";
    std::cout << "input / raw level       : "
              << config.inputLevel << " / "
              << config.expectedOutputLevel << "\n";
    std::cout << "logical GPU devices     : "
              << ksword::DeviceList(config.devices) << "\n";
    std::cout << "bootstrap policy        : refresh 2 Dadda rows + final output\n";
    std::cout << "graph/workspace init    : "
              << (config.graphInit ? "yes" : "no") << "\n";
    std::cout << "detailed profile        : "
              << (config.detailedProfile ? "yes" : "no")
              << "\n";
    std::cout << "precision audit         : "
              << (config.audit ? "yes" : "no") << "\n";
}

void PrintProfile(const ksmul::TimingReport& profile) {
    std::cout
        << "\n================ Multiplier internal timing ================\n";
    PrintPhase("streamed partial-product generation",
               profile.partialProductsMs);

    double daddaTotal = 0.0;
    for (uint32_t stage = 0;
         stage < profile.daddaStages.size();
         ++stage) {
        const auto& item = profile.daddaStages[stage];
        daddaTotal += item.wallMs;
        std::cout << "\n  Dadda stage " << stage
                  << " (" << item.inputRows
                  << " -> " << item.targetRows << " rows)\n";
        std::cout << "    3:2 compressors     : "
                  << item.fullAdders << "\n";
        std::cout << "    passthrough rows    : "
                  << item.passthroughRows << "\n";
        PrintPhase("    stage wall/submit time",
                   item.wallMs);
        if (item.purifiedRows != 0U) {
            std::cout << "    Boolean-projected rows: "
                      << item.purifiedRows << "\n";
            PrintPhase("    boundary Boolean projection",
                       item.purificationMs);
        }
    }
    PrintPhase("Dadda stage timing sum", daddaTotal);
    if (profile.daddaRowRefreshMs > 0.0) {
        PrintPhase("final two-row Bootstrap + projection",
                   profile.daddaRowRefreshMs);
    }

    if (profile.finalAdderMaskReloadMs > 0.0) {
        PrintPhase("level-specific final-adder masks",
                   profile.finalAdderMaskReloadMs);
    }
    if (profile.finalAdderInputRescaleMs > 0.0) {
        PrintPhase("Dadda-row scale normalization",
                   profile.finalAdderInputRescaleMs);
    }
    if (profile.finalAdderGraphInitMs > 0.0) {
        PrintPhase("final-adder graph/workspace initialization",
                   profile.finalAdderGraphInitMs);
    }

    std::cout << "\n  Final complex-standard Kogge-Stone adder\n";
    PrintPhase("    preprocess G/P",
               profile.finalAdder.preprocessMs);
    PrintPhase("    pack Z=P/2+iG",
               profile.finalAdder.packStateMs);
    for (uint32_t layer = 0;
         layer < profile.finalAdder.layers.size();
         ++layer) {
        const auto& item =
            profile.finalAdder.layers[layer];
        std::cout << "    layer " << layer
                  << " distance=" << (1U << layer)
                  << "\n";
        PrintPhase("      rotation + conjugation", item.transformMs);
        PrintPhase("      ciphertext product", item.productMs);
        PrintPhase("      retained generate", item.auxiliaryMs);
        PrintPhase("      state update", item.updateMs);
        PrintPhase("      layer wall", item.wallMs);
    }
    PrintPhase("    carry rotation",
               profile.finalAdder.carryRotateMs);
    PrintPhase("    output terms",
               profile.finalAdder.outputTermsMs);
    PrintPhase("    real projection",
               profile.finalAdder.projectionMs);
    PrintPhase("    final adder total",
               profile.finalAdder.totalMs);
    PrintPhase("multiplier total", profile.totalMs);

    std::cout << "\n  Operation accounting\n";
    std::cout << "    input alignment rotations : "
              << profile.inputAlignmentRotations << "\n";
    std::cout << "    broadcast rotations       : "
              << profile.broadcastRotations << "\n";
    std::cout << "    selector plaintext mults  : "
              << profile.selectorPlainMults << "\n";
    std::cout << "    partial ciphertext mults  : "
              << profile.partialCipherMults << "\n";
    std::cout << "    compressor ciphertext mults: "
              << profile.compressorCipherMults << "\n";
    std::cout << "    compressor carry rotations: "
              << profile.compressorCarryRotations << "\n";
    std::cout << "    Boolean purifications      : "
              << profile.booleanPurifications << "\n";
    std::cout << "    Dadda-row Bootstraps       : "
              << profile.daddaRowBootstraps << "\n";
    std::cout << "    alignment/plain masks      : "
              << profile.alignmentPlainMults << "\n";
    std::cout << "    streaming synchronizations: "
              << profile.streamingSynchronizations << "\n";
    std::cout << "    peak buffered rows        : "
              << profile.peakBufferedRows << "\n";
    std::cout << "    Dadda output level        : "
              << profile.daddaOutputLevel << "\n";
    std::cout << "    Dadda pre-refresh level   : "
              << profile.daddaPreRefreshLevel << "\n";
    std::cout << "    final-adder input level   : "
              << profile.finalAdderInputLevel << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::unitbuf;
        ksmul::RuntimeConfig config;
        try {
            config = ksmul::ParseCommandLine(argc, argv);
        }
        catch (const std::runtime_error& error) {
            if (std::string(error.what()) == "help") {
                std::cout << ksmul::Usage(argv[0]);
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
        // The final adder stores P/2 and G in the real and imaginary
        // components of one CKKS slot. Without COMPLEX mode, the imaginary
        // plaintext masks are not part of the executable encoding contract
        // and the real projection collapses to the uncarried row0 + row1.
        parameters.SetCKKSDataType(COMPLEX);
        parameters.SetMultiplicativeDepth(
            config.multiplicativeDepth);
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

        const auto rotations = ksmul::RotationIndices(config);
        begin = ksword::Clock::now();
        cc->EvalRotateKeyGen(keys.secretKey, rotations);
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
        auto masks = ksmul::BuildAndLoadMasks(cc, config);
        phase.maskBuildLoadMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        begin = ksword::Clock::now();
        const auto input = BuildInputs(config);
        phase.inputBuildMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        std::cout << "\nInput samples\n";
        const uint32_t samples =
            std::min(config.words, uint32_t{3});
        for (uint32_t word = 0; word < samples; ++word) {
            std::cout << "  word " << word << " A = "
                      << ksword::BitsToHex(
                             input.a, config, word)
                      << "\n";
            std::cout << "  word " << word << " B = "
                      << ksword::BitsToHex(
                             input.b, config, word)
                      << "\n";
            std::cout << "  word " << word << " P = "
                      << ksword::BitsToHex(
                             input.expected, config, word)
                      << "\n";
        }

        begin = ksword::Clock::now();
        auto pA = cc->MakeCKKSPackedPlaintext(
            input.encodedA,
            1,
            config.inputLevel,
            nullptr,
            config.slots);
        auto pB = cc->MakeCKKSPackedPlaintext(
            input.encodedB,
            1,
            config.inputLevel,
            nullptr,
            config.slots);
        pA->SetLength(config.slots);
        pB->SetLength(config.slots);
        phase.encodingMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        begin = ksword::Clock::now();
        auto cA = cc->Encrypt(keys.publicKey, pA);
        auto cB = cc->Encrypt(keys.publicKey, pB);
        cc->Synchronize();
        phase.encryptionMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        if (cA->GetLevel() != config.inputLevel ||
            cB->GetLevel() != config.inputLevel) {
            throw std::runtime_error(
                "input ciphertext level does not match the scheduled level");
        }

        std::cout << "\nScheduled input state\n";
        std::cout << "  level / scale degree : "
                  << cA->GetLevel() << " / "
                  << cA->GetNoiseScaleDeg() << "\n";

        ksmul::TimingReport profile;
        ksadder::AuditCallback finalAdderAudit;
        if (config.audit) {
            std::cout
                << "\n================ Complex final-adder audit ================\n";
            finalAdderAudit = [&](const char* stage,
                                  int32_t layer,
                                  const Ciphertext<DCRTPoly>& checkpoint) {
                PrintComplexAdderCheckpoint(
                    cc,
                    keys.secretKey,
                    stage,
                    layer,
                    checkpoint,
                    config.slots);
            };
        }
        cc->Synchronize();
        begin = ksword::Clock::now();
        auto product = ksmul::EvalMultiplyDaddaKS(
            cc,
            cA,
            cB,
            config,
            masks,
            &profile,
            finalAdderAudit);
        cc->Synchronize();
        const double multiplicationWallMs =
            ksword::Milliseconds(begin, ksword::Clock::now());
        phase.finalAdderGraphInitMs =
            profile.finalAdderGraphInitMs;
        phase.multiplicationMs = std::max(
            0.0,
            multiplicationWallMs -
                phase.finalAdderGraphInitMs);

        std::cout << "\nMultiplier ciphertext state\n";
        std::cout << "  Dadda output level     : "
                  << profile.daddaOutputLevel << "\n";
        std::cout << "  final-adder input state: "
                  << profile.finalAdderInputLevel << " / 1\n";
        std::cout << "  level / scale degree : "
                  << product->GetLevel() << " / "
                  << product->GetNoiseScaleDeg() << "\n";
        std::cout << "  usable ct-ct levels   : "
                  << LevelsRemaining(config, product) << "\n";

        if (config.audit) {
            std::cout
                << "\n================ Precision checkpoint audit ================\n";
            PrintBooleanCheckpoint(
                "Dadda row 0 after refresh",
                cc,
                keys.secretKey,
                profile.auditDaddaRows[0],
                config);
            PrintBooleanCheckpoint(
                "Dadda row 1 after refresh",
                cc,
                keys.secretKey,
                profile.auditDaddaRows[1],
                config);
            PrintDaddaTwoRowSemantics(
                cc,
                keys.secretKey,
                profile.auditDaddaRows,
                config,
                input.expected);
            PrintBooleanCheckpoint(
                "original separated-G/P final adder",
                cc,
                keys.secretKey,
                profile.auditBaselineProduct,
                config,
                &input.expected);
            PrintBooleanCheckpoint(
                "raw product before final refresh",
                cc,
                keys.secretKey,
                product,
                config,
                &input.expected);
        }

        phase.bootstrapMs = 0.0;
        auto evaluated = ksmul::EvalFinalizeProduct(
            cc,
            std::move(product),
            config,
            masks,
            &phase.bootstrapMs);

        std::cout << "\nFinal ciphertext state\n";
        std::cout << "  level / scale degree : "
                  << evaluated->GetLevel() << " / "
                  << evaluated->GetNoiseScaleDeg() << "\n";

        begin = ksword::Clock::now();
        Plaintext decrypted;
        const auto decryptResult = cc->Decrypt(
            keys.secretKey, evaluated, &decrypted);
        if (!decryptResult.isValid) {
            throw std::runtime_error("decryption failed");
        }
        decrypted->SetLength(config.slots);
        const auto values =
            decrypted->GetRealPackedValue();
        phase.decryptionMs =
            ksword::Milliseconds(begin, ksword::Clock::now());

        begin = ksword::Clock::now();
        const auto errors = CheckResult(
            config, values, input.expected);
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

        if (config.detailedProfile) {
            PrintProfile(profile);
        }

        std::cout
            << "\n================ End-to-end timing ================\n";
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
        PrintPhase("mask encoding/loading",
                   phase.maskBuildLoadMs);
        PrintPhase("input/reference generation",
                   phase.inputBuildMs);
        PrintPhase("CKKS encoding", phase.encodingMs);
        PrintPhase("input encryption/loading",
                   phase.encryptionMs);
        if (phase.finalAdderGraphInitMs > 0.0) {
            PrintPhase("final-adder graph/workspace initialization",
                       phase.finalAdderGraphInitMs);
        }
        PrintPhase(
            "Dadda + complex-standard multiply (one execution)",
            phase.multiplicationMs);
        PrintPhase("final Bootstrap + Boolean projection",
                   phase.bootstrapMs);
        PrintPhase("copy/decrypt", phase.decryptionMs);
        PrintPhase("correctness scan", phase.verificationMs);
        PrintPhase("full program wall", phase.totalMs);

        const double onlineMs =
            phase.multiplicationMs + phase.bootstrapMs;
        std::cout << "\n================ Online result ================\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  circuit latency        : "
                  << phase.multiplicationMs << " ms\n";
        std::cout << "  final refresh          : "
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

        return errors.failures == 0 ? 0 : 2;
    }
    catch (const std::exception& error) {
        std::cerr << "[fatal] " << error.what() << "\n";
        std::cerr
            << "Use --help to list supported runtime options.\n";
        return 1;
    }
}
