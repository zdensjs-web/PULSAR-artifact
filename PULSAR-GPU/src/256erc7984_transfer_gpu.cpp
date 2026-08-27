#include <fideslib.hpp>

#include "erc7984_transfer_gpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fideslib;

namespace {

using Clock = std::chrono::steady_clock;

struct SetupTiming {
    double contextMs = 0.0;
    double keyGenMs = 0.0;
    double multKeyGenMs = 0.0;
    double rotateKeyGenMs = 0.0;
    double bootstrapSetupMs = 0.0;
    double bootstrapKeyGenMs = 0.0;
    double loadContextMs = 0.0;
    double maskLoadMs = 0.0;
    double inputBuildMs = 0.0;
    double encodeMs = 0.0;
    double encryptMs = 0.0;
};

struct PlainTransferBatch {
    std::vector<uint8_t> sender;
    std::vector<uint8_t> receiver;
    std::vector<uint8_t> amount;
    std::vector<uint8_t> expectedSender;
    std::vector<uint8_t> expectedReceiver;
};

inline double Milliseconds(const Clock::time_point& begin,
                           const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void PrintTime(const std::string& label, double milliseconds) {
    std::cout << "  " << std::left << std::setw(42) << label
              << std::right << std::setw(14) << std::fixed
              << std::setprecision(3) << milliseconds << " ms"
              << "  (" << std::setprecision(6)
              << milliseconds / 1000.0 << " s)\n";
}

std::vector<int> ParseDevices() {
    // CUDA remaps the selected physical GPUs to logical device IDs 0..N-1.
    // For example, CUDA_VISIBLE_DEVICES=1,2 means FIDESlib must receive {0,1}.
    const char* value = std::getenv("CUDA_VISIBLE_DEVICES");
    if (value == nullptr || *value == '\0') {
        throw std::invalid_argument(
            "CUDA_VISIBLE_DEVICES must be set, for example 0,1 or 0,1,2");
    }

    uint32_t visibleCount = 0;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        const auto first = token.find_first_not_of(" \t");
        const auto last = token.find_last_not_of(" \t");
        if (first == std::string::npos) {
            continue;
        }
        const std::string trimmed = token.substr(first, last - first + 1);
        if (trimmed == "-1") {
            continue;
        }
        ++visibleCount;
    }

    if (visibleCount == 0) {
        throw std::invalid_argument(
            "CUDA_VISIBLE_DEVICES does not expose any GPU");
    }

    std::vector<int> devices;
    devices.reserve(visibleCount);
    for (uint32_t i = 0; i < visibleCount; ++i) {
        devices.push_back(static_cast<int>(i));
    }
    return devices;
}

std::string DevicesToString(const std::vector<int>& devices) {
    std::ostringstream output;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (i != 0) {
            output << ',';
        }
        output << devices[i];
    }
    return output.str();
}

bool GreaterEqualWord(const std::vector<uint8_t>& lhs,
                      const std::vector<uint8_t>& rhs,
                      uint32_t word) {
    const uint32_t base = word * erc7984gpu::kStride;
    for (int32_t bit = static_cast<int32_t>(erc7984gpu::kWordBits) - 1;
         bit >= 0;
         --bit) {
        const uint8_t a = lhs[base + static_cast<uint32_t>(bit)];
        const uint8_t b = rhs[base + static_cast<uint32_t>(bit)];
        if (a != b) {
            return a > b;
        }
    }
    return true;
}

void AddWordMod256(const std::vector<uint8_t>& lhs,
                   const std::vector<uint8_t>& rhs,
                   uint32_t word,
                   std::vector<uint8_t>* output) {
    const uint32_t base = word * erc7984gpu::kStride;
    uint32_t carry = 0;
    for (uint32_t bit = 0; bit < erc7984gpu::kWordBits; ++bit) {
        const uint32_t total =
            lhs[base + bit] + rhs[base + bit] + carry;
        (*output)[base + bit] = static_cast<uint8_t>(total & 1U);
        carry = total >> 1U;
    }
}

void SubWordMod256(const std::vector<uint8_t>& lhs,
                   const std::vector<uint8_t>& rhs,
                   uint32_t word,
                   std::vector<uint8_t>* output) {
    const uint32_t base = word * erc7984gpu::kStride;
    int32_t borrow = 0;
    for (uint32_t bit = 0; bit < erc7984gpu::kWordBits; ++bit) {
        int32_t value = static_cast<int32_t>(lhs[base + bit])
                      - static_cast<int32_t>(rhs[base + bit])
                      - borrow;
        if (value < 0) {
            value += 2;
            borrow = 1;
        }
        else {
            borrow = 0;
        }
        (*output)[base + bit] = static_cast<uint8_t>(value);
    }
}

void SetWordZero(std::vector<uint8_t>* bits, uint32_t word) {
    const uint32_t base = word * erc7984gpu::kStride;
    std::fill(bits->begin() + base,
              bits->begin() + base + erc7984gpu::kWordBits,
              uint8_t{0});
}

void SetWordOne(std::vector<uint8_t>* bits, uint32_t word) {
    SetWordZero(bits, word);
    (*bits)[word * erc7984gpu::kStride] = 1;
}

void SetWordMax(std::vector<uint8_t>* bits, uint32_t word) {
    const uint32_t base = word * erc7984gpu::kStride;
    std::fill(bits->begin() + base,
              bits->begin() + base + erc7984gpu::kWordBits,
              uint8_t{1});
}

PlainTransferBatch BuildInputs(uint64_t seed) {
    PlainTransferBatch batch;
    batch.sender.assign(erc7984gpu::kTotalSlots, 0);
    batch.receiver.assign(erc7984gpu::kTotalSlots, 0);
    batch.amount.assign(erc7984gpu::kTotalSlots, 0);
    batch.expectedSender.assign(erc7984gpu::kTotalSlots, 0);
    batch.expectedReceiver.assign(erc7984gpu::kTotalSlots, 0);

    std::mt19937_64 rng(seed);
    for (uint32_t word = 0; word < erc7984gpu::kNumWords; ++word) {
        const uint32_t base = word * erc7984gpu::kStride;
        for (uint32_t bit = 0; bit < erc7984gpu::kWordBits; ++bit) {
            batch.sender[base + bit] = static_cast<uint8_t>(rng() & 1ULL);
            batch.receiver[base + bit] = static_cast<uint8_t>(rng() & 1ULL);
            batch.amount[base + bit] = static_cast<uint8_t>(rng() & 1ULL);
        }
    }

    // Deterministic edge cases.
    SetWordZero(&batch.sender, 0);
    SetWordZero(&batch.amount, 0);       // valid: 0 sends 0

    SetWordZero(&batch.sender, 1);
    SetWordOne(&batch.amount, 1);        // invalid: 0 sends 1

    SetWordMax(&batch.sender, 2);
    SetWordOne(&batch.amount, 2);        // valid: max sends 1

    SetWordOne(&batch.sender, 3);
    SetWordMax(&batch.amount, 3);        // invalid: 1 sends max

    std::vector<uint8_t> transfer(erc7984gpu::kTotalSlots, 0);
    for (uint32_t word = 0; word < erc7984gpu::kNumWords; ++word) {
        const uint32_t base = word * erc7984gpu::kStride;
        if (GreaterEqualWord(batch.sender, batch.amount, word)) {
            std::copy_n(batch.amount.begin() + base,
                        erc7984gpu::kWordBits,
                        transfer.begin() + base);
        }
    }

    for (uint32_t word = 0; word < erc7984gpu::kNumWords; ++word) {
        SubWordMod256(
            batch.sender, transfer, word, &batch.expectedSender);
        AddWordMod256(
            batch.receiver, transfer, word, &batch.expectedReceiver);
    }

    return batch;
}

std::vector<double> ToDoubleSlots(const std::vector<uint8_t>& bits) {
    std::vector<double> values(bits.size(), 0.0);
    std::transform(bits.begin(), bits.end(), values.begin(),
                   [](uint8_t bit) { return static_cast<double>(bit); });
    return values;
}

std::string BitsToHex(const std::vector<uint8_t>& bits, uint32_t word) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string output(erc7984gpu::kWordBits / 4, '0');
    const uint32_t base = word * erc7984gpu::kStride;

    for (uint32_t nibble = 0; nibble < erc7984gpu::kWordBits / 4; ++nibble) {
        uint8_t value = 0;
        for (uint32_t offset = 0; offset < 4; ++offset) {
            value |= static_cast<uint8_t>(
                bits[base + nibble * 4 + offset] << offset);
        }
        output[output.size() - 1 - nibble] = kHex[value];
    }
    return "0x" + output;
}

struct VerifyResult {
    uint64_t bitErrors = 0;
    double maxActiveError = 0.0;
    double maxGuardAbs = 0.0;
};

template <class SecretKeyType>
VerifyResult VerifyCiphertext(
    const CryptoContext<DCRTPoly>& cc,
    const SecretKeyType& secretKey,
    Ciphertext<DCRTPoly> ciphertext,
    const std::vector<uint8_t>& expected,
    const std::string& label) {

    Plaintext decrypted;
    const auto result = cc->Decrypt(secretKey, ciphertext, &decrypted);
    if (!result.isValid) {
        throw std::runtime_error("Decrypt failed for " + label);
    }
    decrypted->SetLength(erc7984gpu::kTotalSlots);
    const auto values = decrypted->GetRealPackedValue();

    VerifyResult report;
    uint32_t printed = 0;
    for (uint32_t word = 0; word < erc7984gpu::kNumWords; ++word) {
        const uint32_t base = word * erc7984gpu::kStride;
        for (uint32_t bit = 0; bit < erc7984gpu::kWordBits; ++bit) {
            const double value = values[base + bit];
            const double target = static_cast<double>(expected[base + bit]);
            report.maxActiveError =
                std::max(report.maxActiveError, std::abs(value - target));
            const uint8_t rounded = value >= 0.5 ? 1U : 0U;
            if (rounded != expected[base + bit]) {
                ++report.bitErrors;
                if (printed < 8) {
                    std::cout << "  [Mismatch] " << label
                              << " word=" << word
                              << " bit=" << bit
                              << " expected="
                              << static_cast<uint32_t>(expected[base + bit])
                              << " value=" << std::setprecision(10)
                              << value << "\n";
                    ++printed;
                }
            }
        }
        for (uint32_t slot = erc7984gpu::kWordBits;
             slot < erc7984gpu::kStride;
             ++slot) {
            report.maxGuardAbs =
                std::max(report.maxGuardAbs, std::abs(values[base + slot]));
        }
    }
    return report;
}

void PrintTransferTiming(erc7984gpu::Scenario scenario,
                         const erc7984gpu::TransferTiming& timing) {
    std::cout << "\n================ "
              << erc7984gpu::ScenarioName(scenario)
              << " online timing ================\n";

    if (scenario == erc7984gpu::Scenario::Overflow) {
        PrintTime("overflowing_sub (difference + borrow prefix)",
                  timing.compareOrOverflowSubMs);
        PrintTime("extract/broadcast underflow and Boolean NOT",
                  timing.conditionPrepareMs);
    }
    else {
        PrintTime("greater_equal", timing.compareOrOverflowSubMs);
        PrintTime("broadcast condition to 256 bits",
                  timing.conditionPrepareMs);
    }

    if (scenario == erc7984gpu::Scenario::Whitepaper) {
        PrintTime("CMUX(condition, amount, 0)", timing.transferSelectMs);
    }
    else {
        PrintTime("Boolean mask multiplication", timing.transferSelectMs);
    }

    PrintTime("sender balance update", timing.senderUpdateMs);
    PrintTime("receiver balance update", timing.receiverUpdateMs);
    PrintTime("final bootstrap (sender + receiver)",
              timing.conditionalBootstrapMs);
    PrintTime("TOTAL online transfer time", timing.totalMs);

    const double seconds = timing.totalMs / 1000.0;
    const double perTransactionMs = timing.totalMs / erc7984gpu::kNumWords;
    const double tps = seconds > 0.0
        ? static_cast<double>(erc7984gpu::kNumWords) / seconds
        : std::numeric_limits<double>::infinity();

    std::cout << "  " << std::left << std::setw(42)
              << "Bootstraps actually executed"
              << std::right << std::setw(14)
              << timing.bootstrapCount << "\n";
    PrintTime("Amortized time per transaction", perTransactionMs);
    std::cout << "  " << std::left << std::setw(42)
              << "Throughput"
              << std::right << std::setw(14)
              << std::fixed << std::setprecision(6)
              << tps << " tx/s\n";
}

std::vector<erc7984gpu::Scenario> SelectScenarios(const std::string& name) {
    if (name == "all") {
        return {
            erc7984gpu::Scenario::Whitepaper,
            erc7984gpu::Scenario::NoCmux,
            erc7984gpu::Scenario::Overflow,
        };
    }
    return {erc7984gpu::ParseScenario(name)};
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    try {
        const std::string scenarioArg = argc >= 2 ? argv[1] : "all";
        const auto scenarios = SelectScenarios(scenarioArg);
        const auto devices = ParseDevices();

        constexpr uint32_t ringDim = 1U << 17;
        constexpr uint32_t numSlots = erc7984gpu::kTotalSlots;

        // The successful standalone add/sub benchmarks use depth 25 and then
        // bootstrap their output.  The deepest transfer output is estimated
        // at level 22, so it is still a valid final-bootstrap input when the
        // latest accepted input level is depth - 2 = 23.
        constexpr uint32_t multiplicativeDepth = 25;
        constexpr uint64_t randomSeed = 0x4552433739383455ULL;

        erc7984gpu::BootstrapPolicy bootstrapPolicy;
        bootstrapPolicy.multiplicativeDepth = multiplicativeDepth;
        bootstrapPolicy.safetyMargin = 1;
        bootstrapPolicy.bootstrapInputMaxLevel = 23;
        bootstrapPolicy.bootstrapOutputLevelHint = 18;
        bootstrapPolicy.requiredOutputDepth = 0;
        bootstrapPolicy.bootstrapPrepared = true;
        bootstrapPolicy.allowBootstrap = true;
        bootstrapPolicy.verboseLevels = true;

        const SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
        const std::vector<uint32_t> levelBudget = {3, 3};
        const std::vector<uint32_t> bsgsDim = {0, 0};

        const uint32_t usableMaxLevel =
            erc7984gpu::detail::UsableMaxLevel(bootstrapPolicy);
        uint32_t maxFreshOutputEstimate = 0;
        for (const auto scenario : scenarios) {
            maxFreshOutputEstimate = std::max(
                maxFreshOutputEstimate,
                erc7984gpu::FreshOutputLevelEstimate(scenario));
        }

        // A final refresh is part of every measured transfer, so bootstrap
        // precomputation and keys are always prepared.
        constexpr bool prepareBootstrap = true;

        SetupTiming setup;

        std::cout << "============================================================\n";
        std::cout << "FIDESlib CKKS Boolean ERC-7984 transfer benchmark\n";
        std::cout << "  Packed transactions : " << erc7984gpu::kNumWords << "\n";
        std::cout << "  Integer width       : 256 bits\n";
        std::cout << "  Slots per value     : " << erc7984gpu::kStride
                  << " (256 active + 256 guard)\n";
        std::cout << "  GPU devices         : " << DevicesToString(devices) << "\n";
        std::cout << "  Scenario selection  : " << scenarioArg << "\n";
        std::cout << "  Multiplicative depth: " << multiplicativeDepth << "\n";
        std::cout << "  Final bootstrap     : enabled (2 output ciphertexts)\n";
        std::cout << "  Fresh output estimate: "
                  << maxFreshOutputEstimate << "\n";
        std::cout << "============================================================\n";

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
        setup.contextMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        auto keys = cc->KeyGen();
        setup.keyGenMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        cc->EvalMultKeyGen(keys.secretKey);
        setup.multKeyGenMs = Milliseconds(begin, Clock::now());

        std::vector<int32_t> rotationIndices;
        rotationIndices.reserve(2 * erc7984gpu::kNumLayers + 1);
        for (uint32_t layer = 0; layer < erc7984gpu::kNumLayers; ++layer) {
            const int32_t step = static_cast<int32_t>(1U << layer);
            rotationIndices.push_back(step);   // GE suffix network
            rotationIndices.push_back(-step);  // add/sub and broadcast
        }
        const bool needsOverflowRotation = std::any_of(
            scenarios.begin(), scenarios.end(),
            [](erc7984gpu::Scenario scenario) {
                return scenario == erc7984gpu::Scenario::Overflow;
            });
        if (needsOverflowRotation) {
            rotationIndices.push_back(
                static_cast<int32_t>(erc7984gpu::kWordBits - 1U)); // +255
        }
        std::sort(rotationIndices.begin(), rotationIndices.end());
        rotationIndices.erase(
            std::unique(rotationIndices.begin(), rotationIndices.end()),
            rotationIndices.end());

        std::cout << "  Rotation indices    : " << rotationIndices.size() << "\n";

        std::cout << "[setup] EvalRotateKeyGen..." << std::endl;
        begin = Clock::now();
        cc->EvalRotateKeyGen(keys.secretKey, rotationIndices);
        setup.rotateKeyGenMs = Milliseconds(begin, Clock::now());
        std::cout << "[setup] EvalRotateKeyGen complete" << std::endl;

        if (prepareBootstrap) {
            std::cout << "[setup] EvalBootstrapSetup..." << std::endl;
            begin = Clock::now();
            cc->EvalBootstrapSetup(levelBudget, bsgsDim, numSlots, 0);
            setup.bootstrapSetupMs = Milliseconds(begin, Clock::now());
            std::cout << "[setup] EvalBootstrapSetup complete" << std::endl;

            std::cout << "[setup] EvalBootstrapKeyGen..." << std::endl;
            begin = Clock::now();
            cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);
            setup.bootstrapKeyGenMs = Milliseconds(begin, Clock::now());
            std::cout << "[setup] EvalBootstrapKeyGen complete" << std::endl;
        }
        else {
            std::cout << "  Bootstrap setup/keygen skipped: fresh transfer fits the "
                         "configured level budget.\n";
        }

        begin = Clock::now();
        cc->LoadContext(keys.publicKey);
        cc->Synchronize();
        setup.loadContextMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        auto masks = erc7984gpu::BuildAndLoadMasks(cc);
        setup.maskLoadMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        const auto plain = BuildInputs(randomSeed);
        const auto senderSlots = ToDoubleSlots(plain.sender);
        const auto receiverSlots = ToDoubleSlots(plain.receiver);
        const auto amountSlots = ToDoubleSlots(plain.amount);
        setup.inputBuildMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        Plaintext pSender = cc->MakeCKKSPackedPlaintext(
            senderSlots, 1, 0, nullptr, numSlots);
        Plaintext pReceiver = cc->MakeCKKSPackedPlaintext(
            receiverSlots, 1, 0, nullptr, numSlots);
        Plaintext pAmount = cc->MakeCKKSPackedPlaintext(
            amountSlots, 1, 0, nullptr, numSlots);
        pSender->SetLength(numSlots);
        pReceiver->SetLength(numSlots);
        pAmount->SetLength(numSlots);
        setup.encodeMs = Milliseconds(begin, Clock::now());

        begin = Clock::now();
        auto cSender = cc->Encrypt(keys.publicKey, pSender);
        auto cReceiver = cc->Encrypt(keys.publicKey, pReceiver);
        auto cAmount = cc->Encrypt(keys.publicKey, pAmount);
        cc->Synchronize();
        setup.encryptMs = Milliseconds(begin, Clock::now());

        std::cout << "\nInput samples:\n";
        for (uint32_t word = 0; word < 4; ++word) {
            std::cout << "  word " << word << " sender   = "
                      << BitsToHex(plain.sender, word) << "\n";
            std::cout << "  word " << word << " receiver = "
                      << BitsToHex(plain.receiver, word) << "\n";
            std::cout << "  word " << word << " amount   = "
                      << BitsToHex(plain.amount, word) << "\n";
            std::cout << "  word " << word << " enough   = "
                      << (GreaterEqualWord(
                              plain.sender, plain.amount, word)
                              ? "true" : "false")
                      << "\n";
        }

        bool allPassed = true;
        for (const auto scenario : scenarios) {
            erc7984gpu::TransferTiming timing;
            std::cerr << "[ERC7984] Starting scenario "
                      << erc7984gpu::ScenarioName(scenario)
                      << std::endl;
            const auto result = erc7984gpu::EvalTransfer(
                cc,
                cSender,
                cReceiver,
                cAmount,
                masks,
                scenario,
                bootstrapPolicy,
                &timing);

            PrintTransferTiming(scenario, timing);
            std::cout << "  new sender level    : "
                      << result.newSender->GetLevel() << "\n";
            std::cout << "  new receiver level  : "
                      << result.newReceiver->GetLevel() << "\n";

            std::cout << "\nVerification for "
                      << erc7984gpu::ScenarioName(scenario) << ":\n";
            const auto senderVerification = VerifyCiphertext(
                cc,
                keys.secretKey,
                result.newSender,
                plain.expectedSender,
                "new_sender");
            const auto receiverVerification = VerifyCiphertext(
                cc,
                keys.secretKey,
                result.newReceiver,
                plain.expectedReceiver,
                "new_receiver");

            std::cout << "  sender bit errors    : "
                      << senderVerification.bitErrors << "\n";
            std::cout << "  receiver bit errors  : "
                      << receiverVerification.bitErrors << "\n";
            std::cout << "  sender max bit error : "
                      << std::setprecision(10)
                      << senderVerification.maxActiveError << "\n";
            std::cout << "  receiver max bit error: "
                      << receiverVerification.maxActiveError << "\n";
            std::cout << "  sender max guard abs : "
                      << senderVerification.maxGuardAbs << "\n";
            std::cout << "  receiver max guard abs: "
                      << receiverVerification.maxGuardAbs << "\n";

            const bool passed =
                senderVerification.bitErrors == 0 &&
                receiverVerification.bitErrors == 0;
            allPassed = allPassed && passed;
            std::cout << "  result               : "
                      << (passed ? "PASS" : "FAIL") << "\n";
        }

        std::cout << "\n================ One-time setup timing ================\n";
        PrintTime("GenCryptoContext", setup.contextMs);
        PrintTime("KeyGen", setup.keyGenMs);
        PrintTime("EvalMultKeyGen", setup.multKeyGenMs);
        PrintTime("EvalRotateKeyGen", setup.rotateKeyGenMs);
        PrintTime("EvalBootstrapSetup", setup.bootstrapSetupMs);
        PrintTime("EvalBootstrapKeyGen", setup.bootstrapKeyGenMs);
        PrintTime("LoadContext", setup.loadContextMs);
        PrintTime("Build/load masks", setup.maskLoadMs);
        PrintTime("Build plaintext inputs", setup.inputBuildMs);
        PrintTime("Encode three input ciphertexts", setup.encodeMs);
        PrintTime("Encrypt three input ciphertexts", setup.encryptMs);

        std::cout << "\nOverall verification: "
                  << (allPassed ? "PASS" : "FAIL") << "\n";
        return allPassed ? 0 : 2;
    }
    catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
