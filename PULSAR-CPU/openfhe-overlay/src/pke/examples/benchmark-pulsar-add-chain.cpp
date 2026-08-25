#include "openfhe.h"
#include "encoding/z-encoding.h"
#include "scheme/ckksrns/ckksrns-fhe.h"
#include "scheme/ckksrns/z-advancedshe.h"
#include "scheme/ckksrns/z-dense-boolean.h"
#include "scheme/ckksrns/z-fhe.h"
#include "scheme/ckksrns/z-pke.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <malloc.h>
#if defined(__GNUC__) || defined(__clang__)
extern "C" void MallocExtension_ReleaseFreeMemory(void)
    __attribute__((weak));
#endif
#endif

using namespace lbcrypto;

namespace {

constexpr uint32_t kLogN = 16;
constexpr uint32_t kRingDimension = 1U << kLogN;
constexpr uint32_t kComplexSlots = kRingDimension / 2;
constexpr uint32_t kScalingBits = 43;
constexpr uint32_t kFirstModulusBits = 52;
constexpr uint32_t kMultiplicativeDepth = 32;
constexpr uint32_t kAuxiliaryPrimeBits = 60;
constexpr uint32_t kHybridLargeDigits = 6;
constexpr uint32_t kExpectedQCount = kMultiplicativeDepth + 1;
constexpr uint32_t kExpectedPCount = 5;
constexpr uint32_t kDenseMetadataZN = 256;
constexpr uint32_t kDenseMetadataZSlots = 256;
constexpr uint32_t kClassic128LogQPBound = 1747;
constexpr double kBooleanTolerance = 0.125;
constexpr double kPrefixMapTolerance = 0.05;

using CipherPair = std::array<Ciphertext<DCRTPoly>, 2>;

struct DirectPrefixOutput {
    CipherPair packed;
    CipherPair localPropagate;
};

struct ProcessMemory {
    uint64_t rssKiB = 0;
    uint64_t swapKiB = 0;
};

ProcessMemory ReadProcessMemory() {
    ProcessMemory memory;
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        auto readKiB = [&](const char* field, uint64_t& value) {
            if (line.rfind(field, 0) == 0) {
                std::istringstream input(line.substr(std::char_traits<char>::length(field)));
                input >> value;
            }
        };
        readKiB("VmRSS:", memory.rssKiB);
        readKiB("VmSwap:", memory.swapKiB);
    }
    return memory;
}

void PrintProcessMemory(const std::string& boundary) {
    const auto memory = ReadProcessMemory();
    constexpr double kiBPerGiB = 1024.0 * 1024.0;
    std::cout << "[memory] boundary=" << boundary
              << " rss_gib=" << std::fixed << std::setprecision(2)
              << (static_cast<double>(memory.rssKiB) / kiBPerGiB)
              << " swap_gib="
              << (static_cast<double>(memory.swapKiB) / kiBPerGiB)
              << std::defaultfloat << '\n';
}

void ReleaseAllocatorMemory(const std::string& boundary) {
    const auto start = std::chrono::steady_clock::now();
    const char* allocator = "unavailable";
#if defined(__linux__)
    if (MallocExtension_ReleaseFreeMemory != nullptr) {
        MallocExtension_ReleaseFreeMemory();
        allocator = "tcmalloc";
    }
    else {
        malloc_trim(0);
        allocator = "glibc";
    }
#endif
    const auto end = std::chrono::steady_clock::now();
    std::cout << "[memory-release] boundary=" << boundary
              << " allocator=" << allocator
              << " seconds="
              << std::chrono::duration<double>(end - start).count() << '\n';
    PrintProcessMemory(boundary + "_after_release");
}

struct AuditResult {
    size_t failures = 0;
    size_t firstFailure = 0;
    double firstValue = 0.0;
    double firstExpected = 0.0;
    double maxError = 0.0;
    double maxImaginary = 0.0;
};

struct DirectPrefixAuditResult {
    size_t failures = 0;
    double maxStateError = 0.0;
    double maxLocalError = 0.0;
};

DCRTPoly DecryptCore(const std::vector<DCRTPoly>& ciphertext,
                     const PrivateKey<DCRTPoly>& privateKey) {
    const DCRTPoly& secret = privateKey->GetPrivateElement();
    const size_t sizeQ = secret.GetParams()->GetParams().size();
    const size_t sizeQl = ciphertext[0].GetParams()->GetParams().size();
    auto secretAtLevel(secret);
    secretAtLevel.DropLastElements(sizeQ - sizeQl);

    DCRTPoly secretPower(secretAtLevel);
    DCRTPoly message(ciphertext[0]);
    message.SetFormat(Format::EVALUATION);
    for (size_t index = 1; index < ciphertext.size(); ++index) {
        DCRTPoly component(ciphertext[index]);
        component.SetFormat(Format::EVALUATION);
        message += secretPower * component;
        secretPower *= secretAtLevel;
    }
    return message;
}

std::vector<BigComplex> DecodeDense(ConstCiphertext<DCRTPoly> ciphertext,
                                    const PrivateKey<DCRTPoly>& privateKey) {
    auto message = DecryptCore(ciphertext->GetElements(), privateKey);
    auto encoding = std::make_shared<ZEncodingImpl>(message.GetParams(), message,
                                                     ciphertext->GetScalingFactorBFP(),
                                                     ciphertext->GetZEncodingParams());
    return ZEncodingImpl::decodeR(encoding).toCSlots().getSlots();
}

AuditResult AuditDense(const std::string& label,
                       ConstCiphertext<DCRTPoly> ciphertext,
                       const PrivateKey<DCRTPoly>& privateKey,
                       const std::vector<uint8_t>& expected) {
    const auto decoded = DecodeDense(ciphertext, privateKey);
    if (decoded.size() != expected.size()) {
        throw std::runtime_error(label + ": decoded slot count mismatch");
    }

    AuditResult result;
    for (size_t slot = 0; slot < expected.size(); ++slot) {
        const auto value = decoded[slot].convertToComplex();
        const double expectedValue = static_cast<double>(expected[slot]);
        const double error = std::abs(value - std::complex<double>(expectedValue, 0.0));
        result.maxError = std::max(result.maxError, error);
        result.maxImaginary = std::max(result.maxImaginary, std::abs(value.imag()));
        if (error > kBooleanTolerance) {
            if (result.failures == 0) {
                result.firstFailure = slot;
                result.firstValue = value.real();
                result.firstExpected = expectedValue;
            }
            ++result.failures;
        }
    }

    std::cout << "[gate] " << std::left << std::setw(30) << label << std::right
              << " failures=" << result.failures
              << " max_error=" << std::scientific << std::setprecision(3) << result.maxError
              << " max_imag=" << result.maxImaginary << std::defaultfloat << '\n';
    if (result.failures != 0) {
        std::cout << "[gate] first failure slot=" << result.firstFailure
                  << " value=" << std::scientific << result.firstValue
                  << " expected=" << result.firstExpected << std::defaultfloat << '\n';
    }
    return result;
}

AuditResult AuditDenseActiveWords(const std::string& label,
                                  ConstCiphertext<DCRTPoly> ciphertext,
                                  const PrivateKey<DCRTPoly>& privateKey,
                                  const std::vector<uint8_t>& expected,
                                  uint32_t wordBits,
                                  uint32_t words,
                                  uint32_t activeWords) {
    const auto decoded = DecodeDense(ciphertext, privateKey);
    if (expected.size() != static_cast<size_t>(wordBits) * words ||
        decoded.size() != expected.size() || activeWords == 0 || activeWords > words) {
        throw std::runtime_error(label + ": invalid active-word audit layout");
    }
    AuditResult result;
    for (uint32_t bit = 0; bit < wordBits; ++bit) {
        for (uint32_t word = 0; word < activeWords; ++word) {
            const size_t slot = static_cast<size_t>(bit) * words + word;
            const auto value = decoded[slot].convertToComplex();
            const double expectedValue = static_cast<double>(expected[slot]);
            const double error = std::abs(value - std::complex<double>(expectedValue, 0.0));
            result.maxError = std::max(result.maxError, error);
            result.maxImaginary = std::max(result.maxImaginary, std::abs(value.imag()));
            if (error > kBooleanTolerance) {
                if (result.failures == 0) {
                    result.firstFailure = slot;
                    result.firstValue = value.real();
                    result.firstExpected = expectedValue;
                }
                ++result.failures;
            }
        }
    }
    std::cout << "[gate] " << std::left << std::setw(30) << label << std::right
              << " active_words=" << activeWords << " failures=" << result.failures
              << " max_error=" << std::scientific << std::setprecision(3) << result.maxError
              << " max_imag=" << result.maxImaginary << std::defaultfloat << '\n';
    if (result.failures != 0) {
        std::cout << "[gate] first failure slot=" << result.firstFailure
                  << " value=" << std::scientific << result.firstValue
                  << " expected=" << result.firstExpected << std::defaultfloat << '\n';
    }
    return result;
}

void RequireAudit(const std::string& label, const AuditResult& result) {
    if (result.failures != 0) {
        std::ostringstream message;
        message << label << " correctness failed: failures=" << result.failures
                << " max_error=" << result.maxError;
        throw std::runtime_error(message.str());
    }
}

std::vector<uint8_t> MakeOperand(uint32_t wordBits,
                                 uint32_t words,
                                 uint64_t seed,
                                 uint32_t pattern) {
    std::vector<uint8_t> result(static_cast<size_t>(wordBits) * words);
    std::mt19937_64 random(seed);
    for (uint32_t word = 0; word < words; ++word) {
        for (uint32_t bit = 0; bit < wordBits; ++bit) {
            bool value = (random() & 1U) != 0;
            if (word == 0) {
                value = pattern == 0 ? true : bit == 0;
            }
            else if (word == 1) {
                value = pattern == 0 ? bit + 1 < wordBits : bit == 0;
            }
            else if (word == 2) {
                value = pattern == 0 ? (bit & 1U) == 0 : (bit & 1U) != 0;
            }
            else if (word == 3) {
                value = pattern != 0;
            }
            result[static_cast<size_t>(bit) * words + word] = value ? 1 : 0;
        }
    }
    return result;
}

std::vector<uint8_t> PlainAdd(const std::vector<uint8_t>& left,
                              const std::vector<uint8_t>& right,
                              uint32_t wordBits,
                              uint32_t words) {
    std::vector<uint8_t> result(left.size());
    for (uint32_t word = 0; word < words; ++word) {
        uint8_t carry = 0;
        for (uint32_t bit = 0; bit < wordBits; ++bit) {
            const size_t slot = static_cast<size_t>(bit) * words + word;
            const uint8_t total = left[slot] + right[slot] + carry;
            result[slot] = total & 1U;
            carry = total >> 1;
        }
    }
    return result;
}

std::vector<uint8_t> PlainSub(const std::vector<uint8_t>& left,
                              const std::vector<uint8_t>& right,
                              uint32_t wordBits,
                              uint32_t words) {
    std::vector<uint8_t> result(left.size());
    for (uint32_t word = 0; word < words; ++word) {
        uint8_t borrow = 0;
        for (uint32_t bit = 0; bit < wordBits; ++bit) {
            const size_t slot = static_cast<size_t>(bit) * words + word;
            const int value = static_cast<int>(left[slot]) -
                              static_cast<int>(right[slot]) - borrow;
            if (value < 0) {
                result[slot] = static_cast<uint8_t>(value + 2);
                borrow = 1;
            }
            else {
                result[slot] = static_cast<uint8_t>(value);
                borrow = 0;
            }
        }
    }
    return result;
}

std::vector<uint8_t> PlainGreaterEqual(
    const std::vector<uint8_t>& left,
    const std::vector<uint8_t>& right,
    uint32_t wordBits,
    uint32_t words) {
    std::vector<uint8_t> result(left.size());
    for (uint32_t word = 0; word < words; ++word) {
        bool greaterEqual = true;
        for (uint32_t bit = wordBits; bit-- > 0;) {
            const size_t slot = static_cast<size_t>(bit) * words + word;
            if (left[slot] != right[slot]) {
                greaterEqual = left[slot] > right[slot];
                break;
            }
        }
        for (uint32_t bit = 0; bit < wordBits; ++bit) {
            result[static_cast<size_t>(bit) * words + word] =
                greaterEqual ? 1 : 0;
        }
    }
    return result;
}

std::vector<uint8_t> PlainSelect(const std::vector<uint8_t>& value,
                                 const std::vector<uint8_t>& condition) {
    if (value.size() != condition.size()) {
        throw std::invalid_argument("plain selection size mismatch");
    }
    std::vector<uint8_t> result(value.size());
    for (size_t slot = 0; slot < value.size(); ++slot) {
        result[slot] = value[slot] & condition[slot];
    }
    return result;
}

std::vector<uint8_t> PlainConditionalSelect(
    const std::vector<uint8_t>& falseValue,
    const std::vector<uint8_t>& trueValue,
    const std::vector<uint8_t>& condition) {
    if (falseValue.size() != trueValue.size() ||
        falseValue.size() != condition.size()) {
        throw std::invalid_argument("plain conditional selection size mismatch");
    }
    std::vector<uint8_t> result(falseValue.size());
    for (size_t slot = 0; slot < result.size(); ++slot) {
        result[slot] = condition[slot] ? trueValue[slot] : falseValue[slot];
    }
    return result;
}

std::vector<uint8_t> PlainRotateWords(const std::vector<uint8_t>& input,
                                      uint32_t wordBits,
                                      uint32_t words,
                                      uint32_t offset) {
    if (input.size() != static_cast<size_t>(wordBits) * words) {
        throw std::invalid_argument("plain word rotation size mismatch");
    }
    std::vector<uint8_t> result(input.size());
    for (uint32_t bit = 0; bit < wordBits; ++bit) {
        for (uint32_t word = 0; word < words; ++word) {
            const uint32_t source = (word + offset) % words;
            result[static_cast<size_t>(bit) * words + word] =
                input[static_cast<size_t>(bit) * words + source];
        }
    }
    return result;
}

std::vector<uint8_t> PlainMaximum(const std::vector<uint8_t>& left,
                                  const std::vector<uint8_t>& right,
                                  uint32_t wordBits,
                                  uint32_t words) {
    const auto chooseLeft = PlainGreaterEqual(left, right, wordBits, words);
    return PlainConditionalSelect(right, left, chooseLeft);
}

std::array<std::vector<uint8_t>, 2> MakeAuctionBids(uint32_t wordBits,
                                                    uint32_t words) {
    std::array<std::vector<uint8_t>, 2> bids{
        MakeOperand(wordBits, words, 0xD701U, 0),
        MakeOperand(wordBits, words, 0xD702U, 1)};
    for (auto& group : bids) {
        for (uint32_t word = 0; word < words; ++word) {
            group[static_cast<size_t>(wordBits - 1) * words + word] = 0;
        }
    }
    for (uint32_t bit = 0; bit + 1 < wordBits; ++bit) {
        bids[1][static_cast<size_t>(bit) * words + (words - 1)] = 1;
    }
    return bids;
}

ZEncoding EncodeDense(const std::vector<uint8_t>& bits,
                      const std::shared_ptr<DCRTPoly::Params>& elementParams,
                      const BigFixedPoint& scale) {
    if (bits.size() != kComplexSlots) {
        throw std::runtime_error("dense Boolean encoding must fill all complex slots");
    }
    std::vector<BigComplex> slots(bits.size());
    for (size_t slot = 0; slot < bits.size(); ++slot) {
        slots[slot] = BigComplex(bits[slot] ? BigFixedPoint::one() : BigFixedPoint::zero());
    }
    const ZEncodingParams params(BModeFull, kDenseMetadataZN, kDenseMetadataZSlots);
    return ZEncodingImpl::encodeC(CSlots(params, slots), elementParams, scale);
}

ZEncoding EncodePrefixWarmupResidues(
    const std::vector<uint8_t>& bits,
    const std::shared_ptr<DCRTPoly::Params>& elementParams,
    const BigFixedPoint& scale) {
    if (bits.size() != kComplexSlots) {
        throw std::runtime_error(
            "Prefix Boot warmup encoding must fill all complex slots");
    }
    std::vector<BigComplex> slots(bits.size());
    for (size_t slot = 0; slot < bits.size(); ++slot) {
        const double residue = static_cast<double>(slot % 5);
        slots[slot] = BigComplex(BigFixedPoint::fromDouble(residue));
    }
    const ZEncodingParams params(
        BModeFull, kDenseMetadataZN, kDenseMetadataZSlots);
    return ZEncodingImpl::encodeC(
        CSlots(params, slots), elementParams, scale);
}

template <typename PrimeVector>
std::string PrimeBits(const PrimeVector& primes) {
    std::ostringstream result;
    for (size_t index = 0; index < primes.size(); ++index) {
        if (index != 0) {
            result << ',';
        }
        result << primes[index]->GetModulus().GetMSB();
    }
    return result.str();
}

DirectPrefixAuditResult AuditDirectPrefixMap(
    const std::string& label,
    ConstCiphertext<DCRTPoly> residue,
    ConstCiphertext<DCRTPoly> packed,
    ConstCiphertext<DCRTPoly> localPropagate,
    const PrivateKey<DCRTPoly>& privateKey) {
    const auto decodedResidue = DecodeDense(residue, privateKey);
    const auto decodedPacked = DecodeDense(packed, privateKey);
    const auto decodedLocal = DecodeDense(localPropagate, privateKey);
    if (decodedResidue.size() != decodedPacked.size() ||
        decodedResidue.size() != decodedLocal.size()) {
        throw std::runtime_error(label + ": decoded slot count mismatch");
    }

    DirectPrefixAuditResult result;
    bool reportedFirstFailure = false;
    for (size_t slot = 0; slot < decodedResidue.size(); ++slot) {
        const auto input = decodedResidue[slot].convertToComplex();
        const int64_t residueValue = std::llround(input.real());
        const double residueError = std::abs(
            input - std::complex<double>(
                        static_cast<double>(residueValue), 0.0));
        std::complex<double> expectedPacked(0.0, 0.0);
        double expectedLocal = 0.0;
        switch (residueValue) {
            case 0:
                expectedPacked = {0.5, 0.0};
                expectedLocal = 1.0;
                break;
            case 1:
                break;
            case 2:
                expectedLocal = 1.0;
                break;
            case 3:
                expectedPacked = {0.0, 1.0};
                break;
            case 4:
                expectedPacked = {0.0, 1.0};
                expectedLocal = 1.0;
                break;
            default:
                ++result.failures;
                if (!reportedFirstFailure) {
                    std::cout << "[gate] first failure slot=" << slot
                              << " invalid_residue=" << input << '\n';
                    reportedFirstFailure = true;
                }
                continue;
        }

        const auto actualPacked = decodedPacked[slot].convertToComplex();
        const auto actualLocal = decodedLocal[slot].convertToComplex();
        const double stateError = std::abs(actualPacked - expectedPacked);
        const double localError = std::abs(
            actualLocal - std::complex<double>(expectedLocal, 0.0));
        result.maxStateError = std::max(result.maxStateError, stateError);
        result.maxLocalError = std::max(result.maxLocalError, localError);
        if (residueError > kBooleanTolerance ||
            stateError > kPrefixMapTolerance ||
            localError > kPrefixMapTolerance) {
            ++result.failures;
            if (!reportedFirstFailure) {
                std::cout << "[gate] first failure slot=" << slot
                          << " residue=" << input
                          << " packed=" << actualPacked
                          << " expected_packed=" << expectedPacked
                          << " local=" << actualLocal
                          << " expected_local=" << expectedLocal << '\n';
                reportedFirstFailure = true;
            }
        }
    }
    std::cout << "[gate] " << std::left << std::setw(30) << label << std::right
              << " failures=" << result.failures
              << " state_error=" << std::scientific << std::setprecision(3)
              << result.maxStateError
              << " local_error=" << result.maxLocalError
              << std::defaultfloat << '\n';
    return result;
}

CipherPair BooleanBoundaryRefresh(
    const FHEZ& fhe,
    ConstCiphertext<DCRTPoly> first,
    ConstCiphertext<DCRTPoly> second,
    double* elapsedSeconds = nullptr) {
    const ZEncodingParams params(BModeFull, kDenseMetadataZN, kDenseMetadataZSlots);
    auto firstInput = first->Clone();
    auto secondInput = second->Clone();
    firstInput->SetZEncodingParams(params);
    secondInput->SetZEncodingParams(params);
    firstInput->SetSlots(kComplexSlots);
    secondInput->SetSlots(kComplexSlots);
    std::cout << "[bootstrap-stage] native B2B begin"
              << " input_level=" << firstInput->GetLevel() << '\n';
    PrintProcessMemory("native_B2B_begin");
    const auto bootStart = std::chrono::steady_clock::now();
    auto refreshed = fhe->EvalBooleanToBooleanFull(
        CiphertextGroup({firstInput, secondInput}));
    const auto bootEnd = std::chrono::steady_clock::now();
    if (refreshed.size() != 2) {
        throw std::runtime_error("native B2B did not return two ciphertexts");
    }
    const double seconds =
        std::chrono::duration<double>(bootEnd - bootStart).count();
    if (elapsedSeconds != nullptr) {
        *elapsedSeconds += seconds;
    }
    CipherPair output{refreshed[0], refreshed[1]};
    for (auto& ciphertext : output) {
        ciphertext->SetZEncodingParams(params);
        ciphertext->SetSlots(kComplexSlots);
    }
    std::cout << "[bootstrap-stage] native B2B ready"
              << " output_level=" << output[0]->GetLevel()
              << " boot_seconds=" << seconds
              << '\n';
    PrintProcessMemory("native_B2B_ready");
    return output;
}

DirectPrefixOutput DirectPrefixFunctionalRefresh(
    const FHEZ& fhe,
    ConstCiphertext<DCRTPoly> first,
    ConstCiphertext<DCRTPoly> second) {
    const ZEncodingParams params(
        BModeFull, kDenseMetadataZN, kDenseMetadataZSlots);
    auto firstInput = first->Clone();
    auto secondInput = second->Clone();
    firstInput->SetZEncodingParams(params);
    secondInput->SetZEncodingParams(params);
    firstInput->SetSlots(kComplexSlots);
    secondInput->SetSlots(kComplexSlots);
    std::cout << "[bootstrap-stage] direct Prefix Boot begin"
              << " input_level=" << firstInput->GetLevel() << '\n';
    PrintProcessMemory("direct_Prefix_Boot_begin");

    const auto bootStart = std::chrono::steady_clock::now();
    auto refreshed = fhe->EvalFiveStatePrefixFull(
        CiphertextGroup({firstInput, secondInput}));
    const auto bootEnd = std::chrono::steady_clock::now();
    if (refreshed.size() != 4) {
        throw std::runtime_error(
            "direct Prefix Boot did not return four ciphertexts");
    }
    DirectPrefixOutput output{
        {refreshed[0], refreshed[2]},
        {refreshed[1], refreshed[3]}};
    for (auto& ciphertext : output.packed) {
        ciphertext->SetZEncodingParams(params);
        ciphertext->SetSlots(kComplexSlots);
    }
    for (auto& ciphertext : output.localPropagate) {
        ciphertext->SetZEncodingParams(params);
        ciphertext->SetSlots(kComplexSlots);
    }
    std::cout << "[bootstrap-stage] direct Prefix Boot ready"
              << " state_level=" << output.packed[0]->GetLevel()
              << " local_level=" << output.localPropagate[0]->GetLevel()
              << " boot_seconds="
              << std::chrono::duration<double>(bootEnd - bootStart).count()
              << '\n';
    PrintProcessMemory("direct_Prefix_Boot_ready");
    return output;
}

bool ParseSwitch(const char* value, const char* name) {
    const std::string text(value);
    if (text == "0") {
        return false;
    }
    if (text == "1") {
        return true;
    }
    throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}

uint32_t ParsePositiveUint32(const char* value, const char* name) {
    size_t parsed = 0;
    const uint64_t result = std::stoull(value, &parsed);
    if (parsed != std::char_traits<char>::length(value) || result == 0 ||
        result > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(
            std::string(name) + " must be a positive 32-bit integer");
    }
    return static_cast<uint32_t>(result);
}

double ParseErrorThreshold(const char* value) {
    size_t parsed = 0;
    const double result = std::stod(value, &parsed);
    if (parsed != std::char_traits<char>::length(value) ||
        !std::isfinite(result) || result < 0.0 ||
        result >= kBooleanTolerance) {
        throw std::invalid_argument(
            "bit_clean_threshold must be in [0, 0.125)");
    }
    return result;
}

bool IsSupportedWordBits(uint32_t wordBits) {
    return wordBits >= 16 && wordBits <= 256 &&
           (wordBits & (wordBits - 1)) == 0;
}

uint32_t PrefixLayerCount(uint32_t wordBits) {
    uint32_t layers = 0;
    while ((1U << layers) < wordBits) {
        ++layers;
    }
    return layers;
}

enum class ScheduleRoute {
    Direct,
    DirectPrefixBoot,
    BoundaryB2B,
};

struct ScheduleStep {
    ScheduleRoute route = ScheduleRoute::Direct;
    uint32_t inputLevel = 0;
    uint32_t outputLevel = 0;
    uint32_t completedPrefixLayers = 0;
};

struct SchedulePlan {
    bool valid = false;
    uint32_t bootstraps = std::numeric_limits<uint32_t>::max();
    uint32_t boundaryBootstraps = std::numeric_limits<uint32_t>::max();
    uint64_t levelSum = std::numeric_limits<uint64_t>::max();
    std::vector<ScheduleStep> steps;
};

struct ScheduleCost {
    bool valid = false;
    uint32_t bootstraps = std::numeric_limits<uint32_t>::max();
    uint32_t boundaryBootstraps = std::numeric_limits<uint32_t>::max();
    uint64_t levelSum = std::numeric_limits<uint64_t>::max();
};

const char* ScheduleRouteName(ScheduleRoute route) {
    switch (route) {
        case ScheduleRoute::Direct:
            return "direct";
        case ScheduleRoute::DirectPrefixBoot:
            return "direct_prefix_Boot";
        case ScheduleRoute::BoundaryB2B:
            return "endpoint_native_B2B_then_direct";
    }
    return "unknown";
}

bool BetterSchedule(const ScheduleCost& candidate,
                    const ScheduleCost& current) {
    if (!candidate.valid) {
        return false;
    }
    if (!current.valid) {
        return true;
    }
    return std::tie(candidate.bootstraps,
                    candidate.boundaryBootstraps,
                    candidate.levelSum) <
           std::tie(current.bootstraps,
                    current.boundaryBootstraps,
                    current.levelSum);
}

SchedulePlan BuildDirectPrefixSchedule(
    uint32_t rounds,
    uint32_t initialLevel,
    uint32_t multiplicativeDepth,
    uint32_t circuitDepth,
    uint32_t prefixLayers,
    uint32_t prefixOutputLevel,
    uint32_t boundaryOutputLevel,
    uint32_t maximumPrefixInputLevel,
    uint32_t maximumBoundaryInputLevel,
    uint32_t refreshedPrefixFinishDepth,
    bool finalRefresh) {
    constexpr uint32_t prefixBeginDepth = 2;
    if (initialLevel > multiplicativeDepth) {
        return {};
    }

    std::vector<ScheduleCost> suffix(multiplicativeDepth + 1);
    std::vector<ScheduleCost> current(multiplicativeDepth + 1);
    for (uint32_t level = 0; level <= multiplicativeDepth; ++level) {
        if (!finalRefresh || level <= maximumBoundaryInputLevel) {
            suffix[level] = {true, 0, 0, 0};
        }
    }
    std::vector<std::vector<ScheduleStep>> choices(
        rounds, std::vector<ScheduleStep>(multiplicativeDepth + 1));

    for (uint32_t remaining = rounds; remaining > 0; --remaining) {
        const uint32_t round = remaining - 1;
        std::fill(current.begin(), current.end(), ScheduleCost{});
        for (uint32_t inputLevel = 0;
             inputLevel <= multiplicativeDepth; ++inputLevel) {
            ScheduleCost best;
            ScheduleStep bestStep;
            const auto consider = [&](ScheduleRoute route,
                                      uint32_t outputLevel,
                                      uint32_t completedPrefixLayers) {
                if (outputLevel > multiplicativeDepth ||
                    !suffix[outputLevel].valid) {
                    return;
                }
                ScheduleCost candidate = suffix[outputLevel];
                candidate.bootstraps +=
                    route == ScheduleRoute::Direct ? 0U : 1U;
                candidate.boundaryBootstraps +=
                    route == ScheduleRoute::BoundaryB2B ? 1U : 0U;
                candidate.levelSum += outputLevel;
                if (BetterSchedule(candidate, best)) {
                    best = candidate;
                    bestStep = {route, inputLevel, outputLevel,
                                completedPrefixLayers};
                }
            };

            consider(ScheduleRoute::Direct,
                     inputLevel + circuitDepth, 0);
            for (uint32_t completed = 0;
                 completed <= prefixLayers; ++completed) {
                const uint32_t checkpointLevel =
                    inputLevel + prefixBeginDepth + completed;
                if (checkpointLevel > maximumPrefixInputLevel) {
                    break;
                }
                const uint32_t outputLevel =
                    prefixOutputLevel + (prefixLayers - completed) +
                    refreshedPrefixFinishDepth;
                consider(ScheduleRoute::DirectPrefixBoot,
                         outputLevel, completed);
            }
            if (inputLevel <= maximumBoundaryInputLevel) {
                consider(ScheduleRoute::BoundaryB2B,
                         boundaryOutputLevel + circuitDepth, 0);
            }
            current[inputLevel] = best;
            choices[round][inputLevel] = bestStep;
        }
        suffix.swap(current);
    }

    const ScheduleCost& initialCost = suffix[initialLevel];
    if (!initialCost.valid) {
        return {};
    }
    SchedulePlan plan{true, initialCost.bootstraps,
                      initialCost.boundaryBootstraps,
                      initialCost.levelSum, {}};
    plan.steps.reserve(rounds);
    uint32_t level = initialLevel;
    for (uint32_t round = 0; round < rounds; ++round) {
        const auto step = choices[round][level];
        plan.steps.push_back(step);
        level = step.outputLevel;
    }
    return plan;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        std::cout << std::unitbuf;
        if (argc > 7) {
            std::cerr << "Usage: " << argv[0]
                      << " [word_bits:16|32|64|128|256] [rounds]"
                      << " [initial_refresh:0|1] [final_refresh:0|1]"
                      << " [scheduler|direct-prefix-probe|transfer|auction|sha256]"
                      << " [bit_clean_threshold]\n";
            return 2;
        }
        const uint32_t wordBits = argc >= 2
            ? ParsePositiveUint32(argv[1], "word_bits") : 128;
        const uint32_t rounds = argc >= 3
            ? ParsePositiveUint32(argv[2], "rounds") : 10;
        const bool initialRefresh = argc >= 4 ? ParseSwitch(argv[3], "initial_refresh") : true;
        const bool finalRefresh = argc >= 5 ? ParseSwitch(argv[4], "final_refresh") : false;
        const std::string mode = argc >= 6 ? argv[5] : "scheduler";
        const bool directPrefixProbe = mode == "direct-prefix-probe";
        const bool transferMode = mode == "transfer";
        const bool auctionMode = mode == "auction";
        const bool sha256Mode = mode == "sha256";
        const double bitCleanThreshold = argc >= 7
            ? ParseErrorThreshold(argv[6]) : 0.01;
        if (mode != "scheduler" && !directPrefixProbe && !transferMode &&
            !auctionMode && !sha256Mode) {
            throw std::invalid_argument(
                "mode must be scheduler, direct-prefix-probe, transfer, auction, or sha256");
        }
        if (!IsSupportedWordBits(wordBits) ||
            (directPrefixProbe && rounds != 2) ||
            ((transferMode || auctionMode) && rounds != 1) ||
            (sha256Mode && (wordBits != 32 || rounds > 64))) {
            throw std::invalid_argument(
                "word_bits must be one of 16, 32, 64, 128, or 256; "
                "the direct probe requires two rounds, transfer and auction require one, "
                "and sha256 requires 32-bit words and 1 to 64 rounds");
        }
        const uint32_t nativeBatch = kComplexSlots / wordBits;
        const uint32_t aggregateBatch = 2 * nativeBatch;
        const uint32_t prefixLayers = PrefixLayerCount(wordBits);
        const uint32_t circuitDepth = DenseBooleanZImpl::KoggeStoneDepth(wordBits);
        constexpr uint32_t prefixEncoderDepth = 0;
        constexpr uint32_t activePrefixDecoderDepth = 0;
        constexpr uint32_t refreshedPrefixFinishDepth = 2;
        const std::vector<uint32_t> levelBudget{3, 2};
        const uint32_t maximumBoundaryInputLevel =
            kMultiplicativeDepth - levelBudget[1];
        const uint32_t maximumPrefixInputLevel =
            kMultiplicativeDepth - levelBudget[1];
        const auto setupStart = std::chrono::steady_clock::now();

        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetSecretKeyDist(SPARSE_ENCAPSULATED);
        parameters.SetSecurityLevel(HEStd_128_classic);
        parameters.SetRingDim(kRingDimension);
        parameters.SetScalingTechnique(FLEXIBLEMANUAL);
        parameters.SetScalingModSize(kScalingBits);
        parameters.SetFirstModSize(kFirstModulusBits);
        parameters.SetNumLargeDigits(kHybridLargeDigits);
        parameters.SetMultiplicativeDepth(kMultiplicativeDepth);
        AUXMODSIZE_FLEXIBLEMANUAL = kAuxiliaryPrimeBits;

        std::cout << "[setup-stage] creating context\n";
        auto cc = GenCryptoContext(parameters);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);
        cc->Enable(FHE);

        const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
        const auto elementParams = cryptoParams->GetElementParams();
        const auto paramsP = cryptoParams->GetParamsP();
        const uint32_t q0Bits = elementParams->GetParams()[0]->GetModulus().GetMSB();
        const uint32_t p0Bits = paramsP->GetParams()[0]->GetModulus().GetMSB();
        const uint32_t logQP =
            (elementParams->GetModulus() * paramsP->GetModulus()).GetMSB();
        if (q0Bits != kFirstModulusBits) {
            std::ostringstream message;
            message << "generated Q0 has " << q0Bits
                    << " bits, expected the Lattigo-aligned "
                    << kFirstModulusBits << " bits";
            throw std::runtime_error(message.str());
        }
        if (elementParams->GetParams().size() != kExpectedQCount ||
            paramsP->GetParams().size() != kExpectedPCount) {
            std::ostringstream message;
            message << "generated Q/P counts are "
                    << elementParams->GetParams().size() << '/'
                    << paramsP->GetParams().size() << ", expected "
                    << kExpectedQCount << '/' << kExpectedPCount;
            throw std::runtime_error(message.str());
        }
        for (size_t index = 0; index < paramsP->GetParams().size(); ++index) {
            const uint32_t bits = paramsP->GetParams()[index]->GetModulus().GetMSB();
            if (bits != kAuxiliaryPrimeBits) {
                std::ostringstream message;
                message << "generated P[" << index << "] has " << bits
                        << " bits, expected " << kAuxiliaryPrimeBits;
                throw std::runtime_error(message.str());
            }
        }
        if (logQP > kClassic128LogQPBound) {
            std::ostringstream message;
            message << "generated logQP=" << logQP
                    << " exceeds the configured classic-128 bound "
                    << kClassic128LogQPBound;
            throw std::runtime_error(message.str());
        }
        if (p0Bits < q0Bits + 6) {
            std::ostringstream message;
            message << "sparse encapsulated key switching requires P0 >= Q0 + 6 bits, got P0/Q0="
                    << p0Bits << '/' << q0Bits;
            throw std::runtime_error(message.str());
        }

        std::cout << "[setup-stage] context ready\n";
        LeveledZ leveled = std::make_shared<LeveledZImpl>();
        AdvancedZ advanced = std::make_shared<AdvancedZImpl>(leveled);
        FHEZ fhe = std::make_shared<FHEZImpl>(leveled, advanced);
        DenseBooleanZ dense = std::make_shared<DenseBooleanZImpl>(leveled);

        std::cout << "[setup-stage] precomputing native B2B and Prefix Boot transforms\n";
        fhe->EvalBootstrapSetup(
            *cc, kDenseMetadataZN, kDenseMetadataZSlots,
            levelBudget, {0, 0}, 4, -24, 1);
        std::cout << "[setup-stage] native B2B and Prefix Boot transforms ready\n";
        std::cout << "[setup-stage] releasing unused generic Bootstrap transforms\n";
        fhe->ReleaseUnusedBooleanRadixPrecom(kComplexSlots);

        auto keyPair = cc->KeyGen();
        std::cout << "[setup-stage] generating relinearization key\n";
        cc->EvalMultKeyGen(keyPair.secretKey);
        std::cout << "[setup-stage] generating native B2B and Prefix Boot keys\n";
        fhe->EvalBootstrapKeyGenBooleanRadixOnly(
            keyPair.secretKey, kDenseMetadataZN, kDenseMetadataZSlots);
        std::cout << "[setup-stage] native B2B and Prefix Boot keys ready\n";
        const size_t refreshKeyCount =
            CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(
                keyPair.secretKey->GetKeyTag()).size();
        std::cout << "[setup-stage] available Bootstrap keys: "
                  << refreshKeyCount << '\n';
        PKEZ pke = std::make_shared<PKEZImpl>(keyPair.publicKey, keyPair.secretKey);
        pkeZ_global = pke;

        std::vector<int32_t> rotations;
        for (uint32_t distance = 1; distance < wordBits; distance <<= 1) {
            rotations.push_back(-static_cast<int32_t>(distance * nativeBatch));
            if (transferMode || auctionMode || sha256Mode) {
                rotations.push_back(
                    static_cast<int32_t>(distance * nativeBatch));
            }
        }
        if (sha256Mode) {
            for (uint32_t amount :
                 {2U, 3U, 6U, 7U, 10U, 11U, 13U, 17U, 18U, 19U, 22U, 25U}) {
                rotations.push_back(
                    static_cast<int32_t>(amount * nativeBatch));
            }
        }
        if (auctionMode) {
            for (uint32_t offset = 1; offset < nativeBatch; offset <<= 1) {
                rotations.push_back(static_cast<int32_t>(offset));
            }
        }
        std::sort(rotations.begin(), rotations.end());
        rotations.erase(std::unique(rotations.begin(), rotations.end()), rotations.end());
        cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
        std::cout << "[setup-stage] all evaluation keys ready\n";

        const auto scale = cryptoParams->GetScalingFactorBFP(0);
        std::array<std::vector<uint8_t>, 2> expected{
            MakeOperand(wordBits, nativeBatch, 0xA501U, 0),
            MakeOperand(wordBits, nativeBatch, 0xA502U, 0)};
        CipherPair accumulator{
            pke->Encrypt(EncodeDense(expected[0], elementParams, scale)),
            pke->Encrypt(EncodeDense(expected[1], elementParams, scale))};

        const auto setupEnd = std::chrono::steady_clock::now();

        const auto warmupStart = std::chrono::steady_clock::now();
        std::cout << "[setup] excluded native B2B warmup\n";
        CipherPair stableAccumulator = BooleanBoundaryRefresh(
            fhe, accumulator[0], accumulator[1]);
        uint32_t boundaryRefreshOutputLevel =
            stableAccumulator[0]->GetLevel();

        uint32_t prefixRefreshOutputLevel = 0;
        Ciphertext<DCRTPoly> prefixWarmupFirst;
        Ciphertext<DCRTPoly> prefixWarmupSecond;
        DirectPrefixOutput prefixWarmupOutput{};
        if (!auctionMode) {
            prefixWarmupFirst = pke->Encrypt(
                EncodePrefixWarmupResidues(expected[0], elementParams, scale));
            prefixWarmupSecond = pke->Encrypt(
                EncodePrefixWarmupResidues(expected[1], elementParams, scale));
            prefixWarmupFirst = dense->AdjustToLevel(
                prefixWarmupFirst, maximumPrefixInputLevel);
            prefixWarmupSecond = dense->AdjustToLevel(
                prefixWarmupSecond, maximumPrefixInputLevel);
            std::cout << "[setup] excluded direct Prefix Boot warmup\n";
            prefixWarmupOutput = DirectPrefixFunctionalRefresh(
                fhe, prefixWarmupFirst, prefixWarmupSecond);
            const auto prefixWarmupAudit0 = AuditDirectPrefixMap(
                "direct Prefix Boot warmup 0", prefixWarmupFirst,
                prefixWarmupOutput.packed[0],
                prefixWarmupOutput.localPropagate[0], keyPair.secretKey);
            const auto prefixWarmupAudit1 = AuditDirectPrefixMap(
                "direct Prefix Boot warmup 1", prefixWarmupSecond,
                prefixWarmupOutput.packed[1],
                prefixWarmupOutput.localPropagate[1], keyPair.secretKey);
            if (prefixWarmupAudit0.failures != 0 ||
                prefixWarmupAudit1.failures != 0) {
                throw std::runtime_error(
                    "direct Prefix Boot failed its warmup mapping audit");
            }
            prefixRefreshOutputLevel =
                prefixWarmupOutput.packed[0]->GetLevel();
        }
        else {
            std::cout << "[setup] direct Prefix Boot warmup omitted for auction\n";
        }
        if (initialRefresh) {
            accumulator = std::move(stableAccumulator);
        }
        else {
            stableAccumulator = CipherPair{};
        }
        ReleaseAllocatorMemory("after_warmup");
        const auto warmupEnd = std::chrono::steady_clock::now();

        RequireAudit("initial accumulator group 0",
                     AuditDense("initial accumulator group 0", accumulator[0], keyPair.secretKey, expected[0]));
        RequireAudit("initial accumulator group 1",
                     AuditDense("initial accumulator group 1", accumulator[1], keyPair.secretKey, expected[1]));

        const uint32_t stableInputLevel = accumulator[0]->GetLevel();
        if (!auctionMode &&
                prefixRefreshOutputLevel + activePrefixDecoderDepth +
                refreshedPrefixFinishDepth > maximumBoundaryInputLevel) {
            throw std::runtime_error(
                "direct Prefix Boot leaves insufficient recovery runway");
        }

#include "pulsar-sha256-benchmark.inc"
#include "pulsar-auction-benchmark.inc"

        if (transferMode) {
            std::array<std::vector<uint8_t>, 2> senderPlain = expected;
            std::array<std::vector<uint8_t>, 2> receiverPlain{
                MakeOperand(wordBits, nativeBatch, 0xC101U, 1),
                MakeOperand(wordBits, nativeBatch, 0xC102U, 1)};
            std::array<std::vector<uint8_t>, 2> amountPlain{
                MakeOperand(wordBits, nativeBatch, 0xC201U, 1),
                MakeOperand(wordBits, nativeBatch, 0xC202U, 1)};
            std::array<std::vector<uint8_t>, 2> conditionPlain{
                PlainGreaterEqual(senderPlain[0], amountPlain[0],
                                  wordBits, nativeBatch),
                PlainGreaterEqual(senderPlain[1], amountPlain[1],
                                  wordBits, nativeBatch)};
            std::array<std::vector<uint8_t>, 2> movedPlain{
                PlainSelect(amountPlain[0], conditionPlain[0]),
                PlainSelect(amountPlain[1], conditionPlain[1])};
            std::array<std::vector<uint8_t>, 2> senderOutputPlain{
                PlainSub(senderPlain[0], movedPlain[0],
                         wordBits, nativeBatch),
                PlainSub(senderPlain[1], movedPlain[1],
                         wordBits, nativeBatch)};
            std::array<std::vector<uint8_t>, 2> receiverOutputPlain{
                PlainAdd(receiverPlain[0], movedPlain[0],
                         wordBits, nativeBatch),
                PlainAdd(receiverPlain[1], movedPlain[1],
                         wordBits, nativeBatch)};

            CipherPair sender = accumulator;
            CipherPair receiver{
                pke->Encrypt(EncodeDense(
                    receiverPlain[0], elementParams, scale)),
                pke->Encrypt(EncodeDense(
                    receiverPlain[1], elementParams, scale))};
            CipherPair amount{
                pke->Encrypt(EncodeDense(
                    amountPlain[0], elementParams, scale)),
                pke->Encrypt(EncodeDense(
                    amountPlain[1], elementParams, scale))};

            double inputB2BSeconds = 0.0;
            uint32_t inputB2Bs = 0;
            std::cout << "[input-refresh] sender native B2B excluded\n";
            sender = BooleanBoundaryRefresh(
                fhe, sender[0], sender[1], &inputB2BSeconds);
            ++inputB2Bs;
            ReleaseAllocatorMemory("after_input_sender_B2B");
            std::cout << "[input-refresh] receiver native B2B excluded\n";
            receiver = BooleanBoundaryRefresh(
                fhe, receiver[0], receiver[1], &inputB2BSeconds);
            ++inputB2Bs;
            ReleaseAllocatorMemory("after_input_receiver_B2B");
            std::cout << "[input-refresh] amount native B2B excluded\n";
            amount = BooleanBoundaryRefresh(
                fhe, amount[0], amount[1], &inputB2BSeconds);
            ++inputB2Bs;
            ReleaseAllocatorMemory("after_input_amount_B2B");
            boundaryRefreshOutputLevel = sender[0]->GetLevel();
            if (receiver[0]->GetLevel() != boundaryRefreshOutputLevel ||
                amount[0]->GetLevel() != boundaryRefreshOutputLevel) {
                throw std::runtime_error(
                    "input B2B refreshes returned inconsistent levels");
            }
            RequireAudit(
                "refreshed sender group 0",
                AuditDense("refreshed sender group 0", sender[0],
                           keyPair.secretKey, senderPlain[0]));
            RequireAudit(
                "refreshed sender group 1",
                AuditDense("refreshed sender group 1", sender[1],
                           keyPair.secretKey, senderPlain[1]));
            RequireAudit(
                "refreshed receiver group 0",
                AuditDense("refreshed receiver group 0", receiver[0],
                           keyPair.secretKey, receiverPlain[0]));
            RequireAudit(
                "refreshed receiver group 1",
                AuditDense("refreshed receiver group 1", receiver[1],
                           keyPair.secretKey, receiverPlain[1]));
            RequireAudit(
                "refreshed amount group 0",
                AuditDense("refreshed amount group 0", amount[0],
                           keyPair.secretKey, amountPlain[0]));
            RequireAudit(
                "refreshed amount group 1",
                AuditDense("refreshed amount group 1", amount[1],
                           keyPair.secretKey, amountPlain[1]));

            if (prefixRefreshOutputLevel + activePrefixDecoderDepth +
                    refreshedPrefixFinishDepth > maximumBoundaryInputLevel) {
                throw std::runtime_error(
                    "direct Prefix Boot leaves insufficient recovery runway");
            }
            const auto transferPreparationEnd =
                std::chrono::steady_clock::now();

            double comparisonSeconds = 0.0;
            double selectionSeconds = 0.0;
            double arithmeticSeconds = 0.0;
            double bitCleanSeconds = 0.0;
            double observedMaxError = 0.0;
            uint32_t bitCleanEvents = 0;
            uint32_t bitCleanCiphertexts = 0;
            uint32_t prefixBoots = 0;
            double prefixBootSeconds = 0.0;
            uint32_t onlineBoundaryB2Bs = 0;
            double onlineBoundaryB2BSeconds = 0.0;

            auto maybeBitClean = [&](
                const std::string& boundary,
                CipherPair values,
                const std::array<std::vector<uint8_t>, 2>& expectedValues,
                uint32_t requiredLevelsAfterClean) {
                const auto before0 = AuditDense(
                    boundary + " before clean group 0", values[0],
                    keyPair.secretKey, expectedValues[0]);
                const auto before1 = AuditDense(
                    boundary + " before clean group 1", values[1],
                    keyPair.secretKey, expectedValues[1]);
                RequireAudit(boundary + " before clean group 0", before0);
                RequireAudit(boundary + " before clean group 1", before1);
                const double currentError =
                    std::max(before0.maxError, before1.maxError);
                observedMaxError = std::max(observedMaxError, currentError);
                bool apply = currentError > bitCleanThreshold;
                std::cout << "[bit-clean-plan] boundary=" << boundary
                          << " current_error=" << std::scientific
                          << currentError
                          << " threshold=" << bitCleanThreshold
                          << " action=" << (apply ? "apply" : "skip")
                          << std::defaultfloat << '\n';
                if (!apply) {
                    return values;
                }
                const uint32_t cleanLimit = requiredLevelsAfterClean == 0
                    ? kMultiplicativeDepth
                    : maximumPrefixInputLevel;
                if (values[0]->GetLevel() + 2 + requiredLevelsAfterClean >
                        cleanLimit ||
                    values[1]->GetLevel() + 2 + requiredLevelsAfterClean >
                        cleanLimit) {
                    std::cout << "[schedule] boundary=" << boundary
                              << " action=native_B2B_before_Bit_Clean"
                              << " input_level=" << values[0]->GetLevel()
                              << " required_levels_after_clean="
                              << requiredLevelsAfterClean << '\n';
                    values = BooleanBoundaryRefresh(
                        fhe, values[0], values[1],
                        &onlineBoundaryB2BSeconds);
                    ++onlineBoundaryB2Bs;
                    const auto refreshed0 = AuditDense(
                        boundary + " after B2B group 0", values[0],
                        keyPair.secretKey, expectedValues[0]);
                    const auto refreshed1 = AuditDense(
                        boundary + " after B2B group 1", values[1],
                        keyPair.secretKey, expectedValues[1]);
                    RequireAudit(
                        boundary + " after B2B group 0", refreshed0);
                    RequireAudit(
                        boundary + " after B2B group 1", refreshed1);
                    const double refreshedError =
                        std::max(refreshed0.maxError, refreshed1.maxError);
                    observedMaxError = std::max(
                        observedMaxError, refreshedError);
                    apply = refreshedError > bitCleanThreshold;
                    std::cout << "[bit-clean-plan] boundary=" << boundary
                              << " post_b2b_error=" << std::scientific
                              << refreshedError
                              << " threshold=" << bitCleanThreshold
                              << " action=" << (apply ? "apply" : "skip")
                              << std::defaultfloat << '\n';
                    if (!apply) {
                        return values;
                    }
                }
                if (values[0]->GetLevel() + 2 + requiredLevelsAfterClean >
                        cleanLimit ||
                    values[1]->GetLevel() + 2 + requiredLevelsAfterClean >
                        cleanLimit) {
                    throw std::runtime_error(
                        "B2B did not leave enough levels for Bit Clean and its successor");
                }
                const auto start = std::chrono::steady_clock::now();
                CipherPair cleaned{
                    dense->EvalBitClean(values[0]),
                    dense->EvalBitClean(values[1])};
                const auto end = std::chrono::steady_clock::now();
                const double seconds =
                    std::chrono::duration<double>(end - start).count();
                bitCleanSeconds += seconds;
                ++bitCleanEvents;
                bitCleanCiphertexts += 2;
                const auto after0 = AuditDense(
                    boundary + " after clean group 0", cleaned[0],
                    keyPair.secretKey, expectedValues[0]);
                const auto after1 = AuditDense(
                    boundary + " after clean group 1", cleaned[1],
                    keyPair.secretKey, expectedValues[1]);
                RequireAudit(boundary + " after clean group 0", after0);
                RequireAudit(boundary + " after clean group 1", after1);
                observedMaxError = std::max(
                    {observedMaxError, after0.maxError, after1.maxError});
                std::cout << "[bit-clean] boundary=" << boundary
                          << " input_level=" << values[0]->GetLevel()
                          << " output_level=" << cleaned[0]->GetLevel()
                          << " seconds=" << seconds << '\n';
                return cleaned;
            };

            const double setupSeconds =
                std::chrono::duration<double>(
                    transferPreparationEnd - setupStart).count();
            const double warmupSeconds =
                std::chrono::duration<double>(
                    transferPreparationEnd - setupEnd).count();
            std::cout << "============================================================\n"
                      << "PULSAR Boolean transfer over OpenFHE\n"
                      << "implementation_build=v3.2.1-transfer-clear-stale-local-sum\n"
                      << "semantics=moved=(sender>=amount)?amount:0;_sender-=moved;_receiver+=moved\n"
                      << "word_bits=" << wordBits << '\n'
                      << "native_batch=" << nativeBatch << '\n'
                      << "aggregate_batch=" << aggregateBatch << '\n'
                      << "instruction_sequence=GE,AND,SUB+ADD\n"
                      << "representation=Boolean_input_output_for_every_instruction\n"
                      << "scheduler=remaining-level-and-observed-error-aware\n"
                      << "bit_clean=3x^2-2x^3\n"
                      << "bit_clean_depth=2\n"
                      << "bit_clean_threshold=" << bitCleanThreshold << '\n'
                      << "error_source=profiling_decryption_excluded_from_latency\n"
                      << "refresh_primitives=native_B2B,Prefix_Boot\n"
                      << "input_refresh=three_native_B2B_calls_excluded\n"
                      << "terminal_output_refresh=scheduler_decision\n"
                      << "input_b2b_excluded=" << inputB2Bs << '\n'
                      << "input_b2b_s_excluded=" << inputB2BSeconds << '\n'
                      << "input_level=" << boundaryRefreshOutputLevel << '\n'
                      << "circuit_depth_add_sub_compare=" << circuitDepth << '\n'
                      << "multiplicative_depth=" << kMultiplicativeDepth << '\n'
                      << "log_scale=" << kScalingBits << '\n'
                      << "logQP=" << logQP << '\n'
                      << "security=HEStd_128_classic\n"
                      << "correctness_claim=empirical_threshold_only\n"
                      << "setup_seconds_excluded=" << setupSeconds << '\n'
                      << "warmup_seconds_excluded=" << warmupSeconds << '\n'
                      << "============================================================\n";

            const uint32_t comparisonInputLevel = sender[0]->GetLevel();
            if (comparisonInputLevel + circuitDepth > kMultiplicativeDepth) {
                throw std::runtime_error(
                    "refreshed transfer inputs leave insufficient levels for GE");
            }
            std::cout << "[schedule] instruction=GE action=direct"
                      << " input_level=" << comparisonInputLevel
                      << " projected_output_level="
                      << (comparisonInputLevel + circuitDepth) << '\n';
            const auto comparisonStart = std::chrono::steady_clock::now();
            CipherPair condition{
                dense->EvalPulsarGreaterEqual(
                    sender[0], amount[0], wordBits, nativeBatch),
                dense->EvalPulsarGreaterEqual(
                    sender[1], amount[1], wordBits, nativeBatch)};
            const auto comparisonEnd = std::chrono::steady_clock::now();
            comparisonSeconds = std::chrono::duration<double>(
                comparisonEnd - comparisonStart).count();
            if (condition[0]->GetLevel() !=
                    comparisonInputLevel + circuitDepth ||
                condition[1]->GetLevel() !=
                    comparisonInputLevel + circuitDepth) {
                throw std::runtime_error(
                    "GE consumed an unexpected number of levels");
            }
            condition = maybeBitClean(
                "condition", std::move(condition), conditionPlain, 3);

            const uint32_t selectionInputLevel = condition[0]->GetLevel();
            auto amountForSelection0 = dense->AdjustToLevel(
                amount[0], selectionInputLevel);
            auto amountForSelection1 = dense->AdjustToLevel(
                amount[1], selectionInputLevel);
            std::cout << "[schedule] instruction=AND_SELECT action=direct"
                      << " input_level=" << selectionInputLevel
                      << " projected_output_level="
                      << (selectionInputLevel + 1) << '\n';
            const auto selectionStart = std::chrono::steady_clock::now();
            CipherPair moved{
                dense->EvalBooleanAnd(amountForSelection0, condition[0]),
                dense->EvalBooleanAnd(amountForSelection1, condition[1])};
            const auto selectionEnd = std::chrono::steady_clock::now();
            selectionSeconds = std::chrono::duration<double>(
                selectionEnd - selectionStart).count();
            moved = maybeBitClean(
                "selected_amount", std::move(moved), movedPlain, 2);

            const uint32_t arithmeticInputLevel = moved[0]->GetLevel();
            auto senderAtLevel0 = dense->AdjustToLevel(
                sender[0], arithmeticInputLevel);
            auto senderAtLevel1 = dense->AdjustToLevel(
                sender[1], arithmeticInputLevel);
            auto receiverAtLevel0 = dense->AdjustToLevel(
                receiver[0], arithmeticInputLevel);
            auto receiverAtLevel1 = dense->AdjustToLevel(
                receiver[1], arithmeticInputLevel);
            auto evaluateArithmetic = [&] (
                const std::string& instruction,
                bool subtraction,
                const CipherPair& left,
                const CipherPair& right) {
                const uint32_t inputLevel = left[0]->GetLevel();
                if (left[1]->GetLevel() != inputLevel ||
                    right[0]->GetLevel() != inputLevel ||
                    right[1]->GetLevel() != inputLevel) {
                    throw std::runtime_error(
                        instruction + " inputs have inconsistent levels");
                }
                const bool direct =
                    inputLevel + circuitDepth <= kMultiplicativeDepth;
                if (direct) {
                    std::cout << "[schedule] instruction=" << instruction
                              << " action=direct input_level=" << inputLevel
                              << " projected_output_level="
                              << (inputLevel + circuitDepth) << '\n';
                    const auto start = std::chrono::steady_clock::now();
                    CipherPair output = subtraction
                        ? CipherPair{
                              dense->EvalPulsarSub(
                                  left[0], right[0], wordBits, nativeBatch),
                              dense->EvalPulsarSub(
                                  left[1], right[1], wordBits, nativeBatch)}
                        : CipherPair{
                              dense->EvalPulsarAdd(
                                  left[0], right[0], wordBits, nativeBatch),
                              dense->EvalPulsarAdd(
                                  left[1], right[1], wordBits, nativeBatch)};
                    const auto end = std::chrono::steady_clock::now();
                    arithmeticSeconds +=
                        std::chrono::duration<double>(end - start).count();
                    return output;
                }

                constexpr uint32_t prefixBeginDepth = 2;
                if (inputLevel + prefixBeginDepth > maximumPrefixInputLevel) {
                    throw std::runtime_error(
                        instruction + " cannot reach a Prefix Boot checkpoint");
                }
                const uint32_t completedPrefixLayers = std::min(
                    prefixLayers,
                    maximumPrefixInputLevel - inputLevel - prefixBeginDepth);
                const uint32_t remainingPrefixLayers =
                    prefixLayers - completedPrefixLayers;
                const uint32_t projectedOutputLevel =
                    prefixRefreshOutputLevel + remainingPrefixLayers +
                    refreshedPrefixFinishDepth;
                if (projectedOutputLevel > kMultiplicativeDepth) {
                    throw std::runtime_error(
                        instruction + " has insufficient levels after Prefix Boot");
                }
                std::cout << "[schedule] instruction=" << instruction
                          << " action=Prefix_Boot input_level=" << inputLevel
                          << " completed_prefix_layers="
                          << completedPrefixLayers
                          << " remaining_prefix_layers="
                          << remainingPrefixLayers
                          << " projected_output_level="
                          << projectedOutputLevel << '\n';

                const auto preBootStart = std::chrono::steady_clock::now();
                std::array<DenseBooleanZImpl::PrefixState, 2> prefix{
                    subtraction
                        ? dense->EvalPulsarBorrowBegin(
                              left[0], right[0], wordBits, nativeBatch)
                        : dense->EvalPulsarAddBegin(
                              left[0], right[0], wordBits, nativeBatch),
                    subtraction
                        ? dense->EvalPulsarBorrowBegin(
                              left[1], right[1], wordBits, nativeBatch)
                        : dense->EvalPulsarAddBegin(
                              left[1], right[1], wordBits, nativeBatch)};
                for (uint32_t layer = 0;
                     layer < completedPrefixLayers; ++layer) {
                    dense->EvalPulsarPrefixAdvance(
                        prefix[0], wordBits, nativeBatch);
                    dense->EvalPulsarPrefixAdvance(
                        prefix[1], wordBits, nativeBatch);
                }
                auto encoded0 = dense->EvalPrefixBootEncode(
                    prefix[0], wordBits, nativeBatch);
                auto encoded1 = dense->EvalPrefixBootEncode(
                    prefix[1], wordBits, nativeBatch);
                const uint32_t nextDistance = prefix[0].nextDistance;
                const auto preBootEnd = std::chrono::steady_clock::now();
                arithmeticSeconds += std::chrono::duration<double>(
                    preBootEnd - preBootStart).count();

                const auto bootStart = std::chrono::steady_clock::now();
                auto bootOutput = DirectPrefixFunctionalRefresh(
                    fhe, encoded0, encoded1);
                const auto bootEnd = std::chrono::steady_clock::now();
                const double bootSeconds = std::chrono::duration<double>(
                    bootEnd - bootStart).count();
                prefixBootSeconds += bootSeconds;
                ++prefixBoots;

                const auto directAudit0 = AuditDirectPrefixMap(
                    instruction + " Prefix Boot map 0", encoded0,
                    bootOutput.packed[0],
                    bootOutput.localPropagate[0], keyPair.secretKey);
                const auto directAudit1 = AuditDirectPrefixMap(
                    instruction + " Prefix Boot map 1", encoded1,
                    bootOutput.packed[1],
                    bootOutput.localPropagate[1], keyPair.secretKey);
                if (directAudit0.failures != 0 ||
                    directAudit1.failures != 0) {
                    throw std::runtime_error(
                        instruction + " Prefix Boot mapping audit failed");
                }

                const auto continuationStart =
                    std::chrono::steady_clock::now();
                for (size_t group = 0; group < prefix.size(); ++group) {
                    prefix[group].packed = bootOutput.packed[group];
                    prefix[group].localPropagate =
                        bootOutput.localPropagate[group];
                    prefix[group].localSum = Ciphertext<DCRTPoly>{};
                    prefix[group].nextDistance = nextDistance;
                }
                for (uint32_t layer = completedPrefixLayers;
                     layer < prefixLayers; ++layer) {
                    dense->EvalPulsarPrefixAdvance(
                        prefix[0], wordBits, nativeBatch);
                    dense->EvalPulsarPrefixAdvance(
                        prefix[1], wordBits, nativeBatch);
                }
                CipherPair output = subtraction
                    ? CipherPair{
                          dense->EvalPulsarSubFinish(
                              prefix[0], wordBits, nativeBatch),
                          dense->EvalPulsarSubFinish(
                              prefix[1], wordBits, nativeBatch)}
                    : CipherPair{
                          dense->EvalPulsarAddFinish(
                              prefix[0], wordBits, nativeBatch),
                          dense->EvalPulsarAddFinish(
                              prefix[1], wordBits, nativeBatch)};
                const auto continuationEnd =
                    std::chrono::steady_clock::now();
                arithmeticSeconds += std::chrono::duration<double>(
                    continuationEnd - continuationStart).count();
                if (output[0]->GetLevel() != projectedOutputLevel ||
                    output[1]->GetLevel() != projectedOutputLevel) {
                    throw std::runtime_error(
                        instruction + " Prefix Boot output level differs from plan");
                }
                std::cout << "[refresh] instruction=" << instruction
                          << " kind=Prefix_Boot output_level="
                          << output[0]->GetLevel()
                          << " seconds=" << bootSeconds << '\n';
                return output;
            };

            std::cout << "[schedule] dependency_group=SUB,ADD action=independent_branches\n";
            CipherPair senderOutput = evaluateArithmetic(
                "SUB", true,
                CipherPair{senderAtLevel0, senderAtLevel1}, moved);
            CipherPair receiverOutput = evaluateArithmetic(
                "ADD", false,
                CipherPair{receiverAtLevel0, receiverAtLevel1}, moved);
            senderOutput = maybeBitClean(
                "sender_output", std::move(senderOutput),
                senderOutputPlain, 0);
            receiverOutput = maybeBitClean(
                "receiver_output", std::move(receiverOutput),
                receiverOutputPlain, 0);

            const double circuitSeconds = comparisonSeconds +
                selectionSeconds + arithmeticSeconds;
            const double totalSeconds = circuitSeconds + bitCleanSeconds +
                prefixBootSeconds + onlineBoundaryB2BSeconds;
            const double precisionBits = observedMaxError == 0.0
                ? std::numeric_limits<double>::infinity()
                : -std::log2(observedMaxError);
            std::cout << std::fixed << std::setprecision(9)
                      << "RESULT app=transfer implementation=PULSAR_Boolean_OpenFHE"
                      << " comparison_s=" << comparisonSeconds
                      << " selection_s=" << selectionSeconds
                      << " arithmetic_s=" << arithmeticSeconds
                      << " circuit_s=" << circuitSeconds
                      << " bit_clean_s=" << bitCleanSeconds
                      << " prefix_boot_s=" << prefixBootSeconds
                      << " boundary_b2b_s=" << onlineBoundaryB2BSeconds
                      << " total_s=" << totalSeconds
                      << " bit_clean_events=" << bitCleanEvents
                      << " bit_clean_ciphertexts=" << bitCleanCiphertexts
                      << " prefix_boots=" << prefixBoots
                      << " online_boundary_b2b=" << onlineBoundaryB2Bs
                      << " input_b2b_excluded=" << inputB2Bs
                      << " batch=" << aggregateBatch
                      << " amortized_ms="
                      << (1000.0 * totalSeconds / aggregateBatch)
                      << " failures=0"
                      << " max_error=" << observedMaxError
                      << " worst_precision_bits=" << precisionBits
                      << " sender_level=" << senderOutput[0]->GetLevel()
                      << " receiver_level=" << receiverOutput[0]->GetLevel()
                      << " status=PASS\n"
                      << "OVERALL status=PASS\n";
            return 0;
        }

        if (directPrefixProbe) {
            const uint32_t probeSecondInputLevel =
                stableInputLevel + circuitDepth;
            constexpr uint32_t prefixBeginDepth = 2;
            if (probeSecondInputLevel + prefixBeginDepth >
                maximumPrefixInputLevel) {
                throw std::runtime_error(
                    "the selected width has no valid Direct Prefix Boot checkpoint");
            }
            const uint32_t probeCompletedPrefixLayers = std::min(
                prefixLayers,
                maximumPrefixInputLevel - probeSecondInputLevel -
                    prefixBeginDepth);
            const double setupSeconds = std::chrono::duration<double>(
                setupEnd - setupStart).count();
            const double warmupSeconds = std::chrono::duration<double>(
                warmupEnd - warmupStart).count();
            std::cout << "============================================================\n"
                      << "PULSAR direct Prefix Boot feasibility probe\n"
                      << "implementation_build=v2.1.2-stable-prefix-halfscale-direct-prefix-probe\n"
                      << "word_bits=" << wordBits << '\n'
                      << "probe_additions=2\n"
                      << "native_batch=" << nativeBatch << '\n'
                      << "aggregate_batch=" << aggregateBatch << '\n'
                      << "prefix_layers=" << prefixLayers << '\n'
                      << "prefix_state=Z=P/2+iG\n"
                      << "prefix_boot_input_residue=u+1_in_[0,4],_u=2(G-P)+p\n"
                      << "prefix_boot_outputs=Z_and_local_p\n"
                      << "external_prefix_decoder_depth=0\n"
                      << "checkpoint_completed_prefix_layers="
                      << probeCompletedPrefixLayers << '\n'
                      << "checkpoint_remaining_prefix_layers="
                      << (prefixLayers - probeCompletedPrefixLayers) << '\n'
                      << "initial_refresh="
                      << (initialRefresh ? "enabled_excluded" : "disabled") << '\n'
                      << "final_refresh=disabled\n"
                      << "logN=" << kLogN << '\n'
                      << "log_scale=" << kScalingBits << '\n'
                      << "requested_first_modulus_bits="
                      << kFirstModulusBits << '\n'
                      << "multiplicative_depth=" << kMultiplicativeDepth << '\n'
                      << "Q_count=" << elementParams->GetParams().size() << '\n'
                      << "Q_bits=" << PrimeBits(elementParams->GetParams()) << '\n'
                      << "P_count=" << paramsP->GetParams().size() << '\n'
                      << "P_bits=" << PrimeBits(paramsP->GetParams()) << '\n'
                      << "logQP=" << logQP << '\n'
                      << "classic_128_logQP_bound="
                      << kClassic128LogQPBound << '\n'
                      << "stable_input_level=" << stableInputLevel << '\n'
                      << "direct_prefix_output_level="
                      << prefixRefreshOutputLevel << '\n'
                      << "setup_seconds_excluded=" << setupSeconds << '\n'
                      << "warmup_seconds_excluded=" << warmupSeconds << '\n'
                      << "============================================================\n";

            double circuitSeconds = 0.0;
            const std::array<std::vector<uint8_t>, 2> firstRightPlain{
                MakeOperand(wordBits, nativeBatch, 0xB601U, 1),
                MakeOperand(wordBits, nativeBatch, 0xB602U, 1)};
            CipherPair firstRight{
                dense->AdjustToLevel(
                    pke->Encrypt(EncodeDense(
                        firstRightPlain[0], elementParams, scale)),
                    stableInputLevel),
                dense->AdjustToLevel(
                    pke->Encrypt(EncodeDense(
                        firstRightPlain[1], elementParams, scale)),
                    stableInputLevel)};
            const auto firstCircuitStart = std::chrono::steady_clock::now();
            CipherPair firstOutput{
                dense->EvalPulsarAdd(
                    accumulator[0], firstRight[0], wordBits, nativeBatch),
                dense->EvalPulsarAdd(
                    accumulator[1], firstRight[1], wordBits, nativeBatch)};
            const auto firstCircuitEnd = std::chrono::steady_clock::now();
            circuitSeconds += std::chrono::duration<double>(
                firstCircuitEnd - firstCircuitStart).count();
            firstRight = CipherPair{};

            std::array<std::vector<uint8_t>, 2> firstExpected{
                PlainAdd(expected[0], firstRightPlain[0], wordBits, nativeBatch),
                PlainAdd(expected[1], firstRightPlain[1], wordBits, nativeBatch)};
            RequireAudit(
                "probe first addition group 0",
                AuditDense("probe first addition group 0", firstOutput[0],
                           keyPair.secretKey, firstExpected[0]));
            RequireAudit(
                "probe first addition group 1",
                AuditDense("probe first addition group 1", firstOutput[1],
                           keyPair.secretKey, firstExpected[1]));

            const uint32_t secondInputLevel = firstOutput[0]->GetLevel();
            if (secondInputLevel != probeSecondInputLevel) {
                throw std::runtime_error(
                    "the first probe addition consumed an unexpected number of levels");
            }
            const std::array<std::vector<uint8_t>, 2> secondRightPlain{
                MakeOperand(wordBits, nativeBatch, 0xB603U, 1),
                MakeOperand(wordBits, nativeBatch, 0xB604U, 1)};
            CipherPair secondRight{
                dense->AdjustToLevel(
                    pke->Encrypt(EncodeDense(
                        secondRightPlain[0], elementParams, scale)),
                    secondInputLevel),
                dense->AdjustToLevel(
                    pke->Encrypt(EncodeDense(
                        secondRightPlain[1], elementParams, scale)),
                    secondInputLevel)};

            const auto preBootStart = std::chrono::steady_clock::now();
            std::array<DenseBooleanZImpl::PrefixState, 2> prefix{
                dense->EvalPulsarAddBegin(
                    firstOutput[0], secondRight[0], wordBits, nativeBatch),
                dense->EvalPulsarAddBegin(
                    firstOutput[1], secondRight[1], wordBits, nativeBatch)};
            for (uint32_t layer = 0;
                 layer < probeCompletedPrefixLayers; ++layer) {
                dense->EvalPulsarPrefixAdvance(
                    prefix[0], wordBits, nativeBatch);
                dense->EvalPulsarPrefixAdvance(
                    prefix[1], wordBits, nativeBatch);
            }
            const uint32_t nextDistance = prefix[0].nextDistance;
            auto encoded0 = dense->EvalPrefixBootEncode(
                prefix[0], wordBits, nativeBatch);
            auto encoded1 = dense->EvalPrefixBootEncode(
                prefix[1], wordBits, nativeBatch);
            const uint32_t checkpointLevel = encoded0->GetLevel();
            if (checkpointLevel > maximumPrefixInputLevel ||
                encoded1->GetLevel() != checkpointLevel) {
                throw std::runtime_error(
                    "direct Prefix Boot probe reached an invalid checkpoint level");
            }
            const auto preBootEnd = std::chrono::steady_clock::now();
            circuitSeconds += std::chrono::duration<double>(
                preBootEnd - preBootStart).count();

            const auto bootStart = std::chrono::steady_clock::now();
            auto directOutput = DirectPrefixFunctionalRefresh(
                fhe, encoded0, encoded1);
            const auto bootEnd = std::chrono::steady_clock::now();
            const double bootSeconds = std::chrono::duration<double>(
                bootEnd - bootStart).count();
            const auto directAudit0 = AuditDirectPrefixMap(
                "direct Prefix Boot map 0", encoded0,
                directOutput.packed[0], directOutput.localPropagate[0],
                keyPair.secretKey);
            const auto directAudit1 = AuditDirectPrefixMap(
                "direct Prefix Boot map 1", encoded1,
                directOutput.packed[1], directOutput.localPropagate[1],
                keyPair.secretKey);
            if (directAudit0.failures != 0 || directAudit1.failures != 0) {
                throw std::runtime_error(
                    "direct Prefix Boot failed its mapping audit");
            }

            const uint32_t stateOutputLevel =
                directOutput.packed[0]->GetLevel();
            const uint32_t localOutputLevel =
                directOutput.localPropagate[0]->GetLevel();
            std::array<DenseBooleanZImpl::PrefixState, 2> refreshedPrefix;
            for (size_t group = 0; group < refreshedPrefix.size(); ++group) {
                refreshedPrefix[group].packed = directOutput.packed[group];
                refreshedPrefix[group].localPropagate =
                    directOutput.localPropagate[group];
                refreshedPrefix[group].nextDistance = nextDistance;
            }

            const auto continuationStart = std::chrono::steady_clock::now();
            for (uint32_t layer = probeCompletedPrefixLayers;
                 layer < prefixLayers; ++layer) {
                dense->EvalPulsarPrefixAdvance(
                    refreshedPrefix[0], wordBits, nativeBatch);
                dense->EvalPulsarPrefixAdvance(
                    refreshedPrefix[1], wordBits, nativeBatch);
            }
            CipherPair finalOutput{
                dense->EvalPulsarAddFinish(
                    refreshedPrefix[0], wordBits, nativeBatch),
                dense->EvalPulsarAddFinish(
                    refreshedPrefix[1], wordBits, nativeBatch)};
            const auto continuationEnd = std::chrono::steady_clock::now();
            circuitSeconds += std::chrono::duration<double>(
                continuationEnd - continuationStart).count();
            secondRight = CipherPair{};
            firstOutput = CipherPair{};

            std::array<std::vector<uint8_t>, 2> finalExpected{
                PlainAdd(firstExpected[0], secondRightPlain[0],
                         wordBits, nativeBatch),
                PlainAdd(firstExpected[1], secondRightPlain[1],
                         wordBits, nativeBatch)};
            const auto finalAudit0 = AuditDense(
                "probe final addition group 0", finalOutput[0],
                keyPair.secretKey, finalExpected[0]);
            const auto finalAudit1 = AuditDense(
                "probe final addition group 1", finalOutput[1],
                keyPair.secretKey, finalExpected[1]);
            RequireAudit("probe final addition group 0", finalAudit0);
            RequireAudit("probe final addition group 1", finalAudit1);

            ReleaseAllocatorMemory("after_direct_prefix_probe");
            std::cout << "============================================================\n"
                      << "Direct Prefix Boot feasibility result\n"
                      << "checkpoint_level=" << checkpointLevel << '\n'
                      << "completed_prefix_layers="
                      << probeCompletedPrefixLayers << '\n'
                      << "remaining_prefix_layers="
                      << (prefixLayers - probeCompletedPrefixLayers) << '\n'
                      << "state_output_level=" << stateOutputLevel << '\n'
                      << "local_output_level=" << localOutputLevel << '\n'
                      << "final_output_level="
                      << finalOutput[0]->GetLevel() << '\n'
                      << "external_decoder_depth=0\n"
                      << "physical_prefix_boots=1\n"
                      << "prefix_boot_seconds=" << bootSeconds << '\n'
                      << "circuit_seconds=" << circuitSeconds << '\n'
                      << "max_map_error="
                      << std::max({directAudit0.maxStateError,
                                   directAudit0.maxLocalError,
                                   directAudit1.maxStateError,
                                   directAudit1.maxLocalError}) << '\n'
                      << "final_max_error="
                      << std::max(finalAudit0.maxError,
                                  finalAudit1.maxError) << '\n'
                      << "OVERALL status=PASS\n";
            return 0;
        }

        const auto schedule = BuildDirectPrefixSchedule(
            rounds, stableInputLevel, kMultiplicativeDepth,
            circuitDepth, prefixLayers, prefixRefreshOutputLevel,
            boundaryRefreshOutputLevel, maximumPrefixInputLevel,
            maximumBoundaryInputLevel, refreshedPrefixFinishDepth,
            finalRefresh);
        if (!schedule.valid || schedule.steps.size() != rounds) {
            throw std::runtime_error(
                "lookahead scheduler could not construct a valid execution plan");
        }
        const uint32_t predictedPrefixBoots =
            schedule.bootstraps - schedule.boundaryBootstraps;
        const uint32_t predictedEndpointBootstraps =
            schedule.boundaryBootstraps + (finalRefresh ? 1U : 0U);
        const uint32_t predictedTotalBoots =
            schedule.bootstraps + (finalRefresh ? 1U : 0U);
        std::cout << "[schedule-plan] rounds=" << rounds
                  << " prefix_boots=" << predictedPrefixBoots
                  << " boundary_b2b=" << predictedEndpointBootstraps
                  << " total_boots=" << predictedTotalBoots << '\n';
        for (uint32_t round = 0; round < rounds; ++round) {
            const auto& step = schedule.steps[round];
            std::cout << "[schedule-plan] round=" << (round + 1)
                      << " route=" << ScheduleRouteName(step.route)
                      << " input_level=" << step.inputLevel
                      << " output_level=" << step.outputLevel;
            if (step.route == ScheduleRoute::DirectPrefixBoot) {
                std::cout << " completed_prefix_layers="
                          << step.completedPrefixLayers
                          << " remaining_prefix_layers="
                          << (prefixLayers - step.completedPrefixLayers);
            }
            std::cout << '\n';
        }

        const double setupSeconds = std::chrono::duration<double>(setupEnd - setupStart).count();
        const double warmupSeconds = std::chrono::duration<double>(warmupEnd - warmupStart).count();
        std::cout << "============================================================\n"
                  << "PULSAR chained Boolean addition over OpenFHE\n"
                  << "implementation_build=v3.2.1-native-B2B-direct-prefix\n"
                  << "semantics=accumulator=(accumulator+operand[r]) mod 2^w\n"
                  << "word_bits=" << wordBits << '\n'
                  << "rounds=" << rounds << '\n'
                  << "ring_dimension=" << cc->GetRingDimension() << '\n'
                  << "complex_slots=" << kComplexSlots << '\n'
                  << "layout=bit_major_dense_no_guard_slots\n"
                  << "prefix_state=Z=P/2+iG\n"
                  << "prefix_products_per_layer=1\n"
                  << "prefix_rotations_per_layer=1\n"
                  << "native_batch=" << nativeBatch << '\n'
                  << "aggregate_batch=" << aggregateBatch << '\n'
                  << "prefix_layers=" << prefixLayers << '\n'
                  << "circuit_depth=" << circuitDepth << '\n'
                  << "scheduler=iterative_global_minimum-Boot_lookahead\n"
                  << "supported_word_bits=16,32,64,128,256\n"
                  << "round_count=runtime_positive_uint32\n"
                  << "prefix_boot=FHEZ_EvalFiveStatePrefixFull\n"
                  << "prefix_boot_interpolation=trigonometric_Hermite_order_3\n"
                  << "prefix_boot_input_residue=u+1_in_[0,4],_u=2(G-P)+p\n"
                  << "prefix_boot_output=Z=P/2+iG_and_local_p\n"
                  << "prefix_boot_encoder_depth=" << prefixEncoderDepth << '\n'
                  << "prefix_boot_decoder=integrated_into_functional_Boot\n"
                  << "prefix_boot_external_decoder_depth=0\n"
                  << "refreshed_prefix_finish_depth="
                  << refreshedPrefixFinishDepth << '\n'
                  << "bootstrap_level_budget="
                  << levelBudget[0] << ',' << levelBudget[1] << '\n'
                  << "boundary_refresh=native_B2B\n"
                  << "initial_refresh=" << (initialRefresh ? "enabled_excluded" : "disabled") << '\n'
                  << "final_refresh=" << (finalRefresh ? "enabled_included" : "disabled") << '\n'
                  << "bootstrap_resources=one_exact_Z_transform_and_key_set\n"
                  << "allocator_release=after_warmup_and_each_round\n"
                  << "transform_plaintext_cache=one_scale_modulus_pair_per_entry\n"
                  << "input_encryption=per-round_excluded_from_operator_timing\n"
                  << "logN=" << kLogN << '\n'
                  << "scaling_technique=FLEXIBLEMANUAL_separate_first_modulus\n"
                  << "log_scale=" << kScalingBits << '\n'
                  << "requested_first_modulus_bits=" << kFirstModulusBits << '\n'
                  << "multiplicative_depth=" << kMultiplicativeDepth << '\n'
                  << "hybrid_large_digits=" << kHybridLargeDigits << '\n'
                  << "Q_count=" << elementParams->GetParams().size() << '\n'
                  << "Q_bits=" << PrimeBits(elementParams->GetParams()) << '\n'
                  << "logQ=" << elementParams->GetModulus().GetMSB() << '\n'
                  << "P_count=" << paramsP->GetParams().size() << '\n'
                  << "P_bits=" << PrimeBits(paramsP->GetParams()) << '\n'
                  << "logP=" << paramsP->GetModulus().GetMSB() << '\n'
                  << "logQP=" << logQP << '\n'
                  << "classic_128_logQP_bound=" << kClassic128LogQPBound << '\n'
                  << "classic_128_headroom_bits=" << (kClassic128LogQPBound - logQP) << '\n'
                  << "security=HEStd_128_classic\n"
                  << "stable_input_level=" << stableInputLevel << '\n'
                  << "boundary_refresh_output_level=" << boundaryRefreshOutputLevel << '\n'
                  << "prefix_refresh_output_level=" << prefixRefreshOutputLevel << '\n'
                  << "predicted_prefix_boots=" << predictedPrefixBoots << '\n'
                  << "predicted_boundary_b2b="
                  << predictedEndpointBootstraps << '\n'
                  << "predicted_total_boots=" << predictedTotalBoots << '\n'
                  << "maximum_boundary_input_level=" << maximumBoundaryInputLevel << '\n'
                  << "maximum_prefix_input_level=" << maximumPrefixInputLevel << '\n'
                  << "setup_seconds_excluded=" << setupSeconds << '\n'
                  << "warmup_seconds_excluded=" << warmupSeconds << '\n'
                  << "============================================================\n";

        double circuitSeconds = 0.0;
        double prefixBootSeconds = 0.0;
        double endpointBootstrapSeconds = 0.0;
        uint32_t prefixBoots = 0;
        uint32_t endpointBootstraps = 0;
        double maxError = 0.0;
        for (uint32_t round = 0; round < rounds; ++round) {
            const auto& step = schedule.steps[round];
            uint32_t inputLevel = accumulator[0]->GetLevel();
            if (inputLevel != accumulator[1]->GetLevel()) {
                throw std::runtime_error("accumulator groups have different levels");
            }
            if (inputLevel != step.inputLevel) {
                std::ostringstream message;
                message << "schedule level mismatch before round "
                        << (round + 1) << ": actual=" << inputLevel
                        << " planned=" << step.inputLevel;
                throw std::runtime_error(message.str());
            }

            const bool prefixRoute =
                step.route == ScheduleRoute::DirectPrefixBoot;
            const std::string roundRoute = ScheduleRouteName(step.route);
            if (step.route == ScheduleRoute::BoundaryB2B) {
                std::cout << "[schedule] round=" << (round + 1)
                          << " action=endpoint_native_B2B"
                          << " input_level=" << inputLevel
                          << " planned_output_level="
                          << step.outputLevel << '\n';
                const auto refreshStart = std::chrono::steady_clock::now();
                accumulator = BooleanBoundaryRefresh(
                    fhe, accumulator[0], accumulator[1]);
                const auto refreshEnd = std::chrono::steady_clock::now();
                const double seconds = std::chrono::duration<double>(refreshEnd - refreshStart).count();
                endpointBootstrapSeconds += seconds;
                ++endpointBootstraps;
                std::cout << "[refresh] round=" << (round + 1)
                          << " kind=endpoint_native_B2B"
                          << " output_level="
                          << accumulator[0]->GetLevel()
                          << " seconds=" << seconds << '\n';
                inputLevel = accumulator[0]->GetLevel();
                if (inputLevel != boundaryRefreshOutputLevel) {
                    throw std::runtime_error(
                        "native B2B output level differs from its warmup level");
                }
            }
            else if (prefixRoute) {
                std::cout << "[schedule] round=" << (round + 1)
                          << " action=direct_prefix_Boot"
                          << " input_level=" << inputLevel
                          << " completed_prefix_layers="
                          << step.completedPrefixLayers
                          << " remaining_prefix_layers="
                          << (prefixLayers -
                              step.completedPrefixLayers) << '\n';
            }
            else {
                std::cout << "[schedule] round=" << (round + 1)
                          << " action=continue input_level=" << inputLevel
                          << " output_level=" << step.outputLevel << '\n';
            }

            const uint32_t addInputLevel = accumulator[0]->GetLevel();
            const std::array<std::vector<uint8_t>, 2> roundRightPlain{
                MakeOperand(wordBits, nativeBatch,
                            0xB601U + 2U * round, 1),
                MakeOperand(wordBits, nativeBatch,
                            0xB602U + 2U * round, 1)};
            auto encryptedRight0 = pke->Encrypt(
                EncodeDense(roundRightPlain[0], elementParams, scale));
            auto encryptedRight1 = pke->Encrypt(
                EncodeDense(roundRightPlain[1], elementParams, scale));
            auto right0 = dense->AdjustToLevel(encryptedRight0, addInputLevel);
            auto right1 = dense->AdjustToLevel(encryptedRight1, addInputLevel);
            encryptedRight0.reset();
            encryptedRight1.reset();
            const auto circuitStart = std::chrono::steady_clock::now();
            if (prefixRoute) {
                std::array<DenseBooleanZImpl::PrefixState, 2> prefix{
                    dense->EvalPulsarAddBegin(
                        accumulator[0], right0, wordBits, nativeBatch),
                    dense->EvalPulsarAddBegin(
                        accumulator[1], right1, wordBits, nativeBatch)};
                for (uint32_t layer = 0;
                     layer < step.completedPrefixLayers; ++layer) {
                    dense->EvalPulsarPrefixAdvance(
                        prefix[0], wordBits, nativeBatch);
                    dense->EvalPulsarPrefixAdvance(
                        prefix[1], wordBits, nativeBatch);
                }

                auto encoded0 = dense->EvalPrefixBootEncode(
                    prefix[0], wordBits, nativeBatch);
                auto encoded1 = dense->EvalPrefixBootEncode(
                    prefix[1], wordBits, nativeBatch);
                if (encoded0->GetLevel() > maximumPrefixInputLevel ||
                    encoded1->GetLevel() > maximumPrefixInputLevel) {
                    throw std::runtime_error(
                        "encoded Prefix Boot state is below the exact Z input level");
                }
                const uint32_t nextDistance = prefix[0].nextDistance;
                const auto preBootEnd = std::chrono::steady_clock::now();
                circuitSeconds +=
                    std::chrono::duration<double>(preBootEnd - circuitStart).count();

                const auto bootStart = std::chrono::steady_clock::now();
                auto bootOutput = DirectPrefixFunctionalRefresh(
                    fhe, encoded0, encoded1);
                const auto bootEnd = std::chrono::steady_clock::now();
                const double bootSeconds =
                    std::chrono::duration<double>(bootEnd - bootStart).count();
                prefixBootSeconds += bootSeconds;
                ++prefixBoots;

                const auto directAudit0 = AuditDirectPrefixMap(
                    "direct Prefix Boot map 0", encoded0,
                    bootOutput.packed[0],
                    bootOutput.localPropagate[0], keyPair.secretKey);
                const auto directAudit1 = AuditDirectPrefixMap(
                    "direct Prefix Boot map 1", encoded1,
                    bootOutput.packed[1],
                    bootOutput.localPropagate[1], keyPair.secretKey);
                if (directAudit0.failures != 0 ||
                    directAudit1.failures != 0) {
                    throw std::runtime_error(
                        "direct Prefix Boot failed its mapping audit");
                }

                const auto continuationStart = std::chrono::steady_clock::now();
                std::array<DenseBooleanZImpl::PrefixState, 2> refreshedPrefix;
                for (size_t group = 0; group < refreshedPrefix.size(); ++group) {
                    refreshedPrefix[group].packed = bootOutput.packed[group];
                    refreshedPrefix[group].localPropagate =
                        bootOutput.localPropagate[group];
                    refreshedPrefix[group].nextDistance = nextDistance;
                }
                prefix = std::move(refreshedPrefix);
                const uint32_t remainingPrefixLayers =
                    prefixLayers - step.completedPrefixLayers;
                const uint32_t projectedFromActualOutput =
                    prefix[0].packed->GetLevel() + remainingPrefixLayers +
                    refreshedPrefixFinishDepth;
                if (projectedFromActualOutput != step.outputLevel) {
                    throw std::runtime_error(
                        "direct Prefix Boot output differs from the planned level");
                }
                for (uint32_t layer = step.completedPrefixLayers;
                     layer < prefixLayers; ++layer) {
                    dense->EvalPulsarPrefixAdvance(
                        prefix[0], wordBits, nativeBatch);
                    dense->EvalPulsarPrefixAdvance(
                        prefix[1], wordBits, nativeBatch);
                }
                accumulator = {
                    dense->EvalPulsarAddFinish(
                        prefix[0], wordBits, nativeBatch),
                    dense->EvalPulsarAddFinish(
                        prefix[1], wordBits, nativeBatch)};
                const auto continuationEnd = std::chrono::steady_clock::now();
                circuitSeconds += std::chrono::duration<double>(
                    continuationEnd - continuationStart).count();
                std::cout << "[refresh] round=" << (round + 1)
                          << " kind=direct_prefix_Boot"
                          << " state_output_level="
                          << bootOutput.packed[0]->GetLevel()
                          << " local_output_level="
                          << bootOutput.localPropagate[0]->GetLevel()
                          << " output_level=" << accumulator[0]->GetLevel()
                          << " seconds=" << bootSeconds << '\n';
            }
            else {
                accumulator = {
                    dense->EvalPulsarAdd(
                        accumulator[0], right0, wordBits, nativeBatch),
                    dense->EvalPulsarAdd(
                        accumulator[1], right1, wordBits, nativeBatch)};
                const auto circuitEnd = std::chrono::steady_clock::now();
                circuitSeconds += std::chrono::duration<double>(
                    circuitEnd - circuitStart).count();
            }
            if (accumulator[0]->GetLevel() != step.outputLevel ||
                accumulator[1]->GetLevel() != step.outputLevel) {
                std::ostringstream message;
                message << "schedule level mismatch after round "
                        << (round + 1) << ": actual="
                        << accumulator[0]->GetLevel()
                        << " planned=" << step.outputLevel;
                throw std::runtime_error(message.str());
            }
            right0.reset();
            right1.reset();
            expected[0] = PlainAdd(
                expected[0], roundRightPlain[0], wordBits, nativeBatch);
            expected[1] = PlainAdd(
                expected[1], roundRightPlain[1], wordBits, nativeBatch);

            const std::string label0 = "round " + std::to_string(round + 1) + " group 0";
            const std::string label1 = "round " + std::to_string(round + 1) + " group 1";
            const auto audit0 = AuditDense(label0, accumulator[0], keyPair.secretKey, expected[0]);
            const auto audit1 = AuditDense(label1, accumulator[1], keyPair.secretKey, expected[1]);
            RequireAudit(label0, audit0);
            RequireAudit(label1, audit1);
            maxError = std::max({maxError, audit0.maxError, audit1.maxError});
            std::cout << "[round] index=" << (round + 1)
                       << " input_level=" << addInputLevel
                       << " output_level=" << accumulator[0]->GetLevel()
                       << " route=" << roundRoute
                       << " status=PASS\n";
            ReleaseAllocatorMemory(
                "after_round_" + std::to_string(round + 1));
        }

        if (finalRefresh) {
            if (accumulator[0]->GetLevel() > maximumBoundaryInputLevel) {
                throw std::runtime_error("final output is below the native B2B input level");
            }
            const auto refreshStart = std::chrono::steady_clock::now();
            accumulator = BooleanBoundaryRefresh(
                fhe, accumulator[0], accumulator[1]);
            const auto refreshEnd = std::chrono::steady_clock::now();
            const double seconds = std::chrono::duration<double>(refreshEnd - refreshStart).count();
            endpointBootstrapSeconds += seconds;
            ++endpointBootstraps;
            std::cout << "[refresh] final=true kind=endpoint_native_B2B output_level="
                       << accumulator[0]->GetLevel()
                       << " seconds=" << seconds << '\n';
        }

        const auto finalAudit0 = AuditDense("final group 0", accumulator[0], keyPair.secretKey, expected[0]);
        const auto finalAudit1 = AuditDense("final group 1", accumulator[1], keyPair.secretKey, expected[1]);
        RequireAudit("final group 0", finalAudit0);
        RequireAudit("final group 1", finalAudit1);
        maxError = std::max({maxError, finalAudit0.maxError, finalAudit1.maxError});
        if (prefixBoots != predictedPrefixBoots ||
            endpointBootstraps != predictedEndpointBootstraps) {
            std::ostringstream message;
            message << "actual refresh counts differ from the lookahead plan: "
                    << "prefix=" << prefixBoots << '/' << predictedPrefixBoots
                    << " endpoint=" << endpointBootstraps << '/'
                    << predictedEndpointBootstraps;
            throw std::runtime_error(message.str());
        }

        const double totalSeconds =
            circuitSeconds + prefixBootSeconds + endpointBootstrapSeconds;
        const double precisionBits = maxError == 0.0
            ? std::numeric_limits<double>::infinity() : -std::log2(maxError);
        const uint32_t finalLevel = accumulator[0]->GetLevel();
        const auto nextRoundPlan = BuildDirectPrefixSchedule(
            1, finalLevel, kMultiplicativeDepth, circuitDepth,
            prefixLayers, prefixRefreshOutputLevel,
            boundaryRefreshOutputLevel, maximumPrefixInputLevel,
            maximumBoundaryInputLevel, refreshedPrefixFinishDepth, false);
        const bool nextRoundReady = nextRoundPlan.valid;
        std::cout << std::fixed << std::setprecision(9)
                  << "RESULT op=chained_add implementation=PULSAR_complex_prefix_OpenFHE"
                  << " rounds=" << rounds
                  << " circuit_s=" << circuitSeconds
                  << " prefix_boot_s=" << prefixBootSeconds
                  << " endpoint_bootstrap_s=" << endpointBootstrapSeconds
                  << " refresh_s=" << (prefixBootSeconds + endpointBootstrapSeconds)
                  << " total_s=" << totalSeconds
                  << " prefix_boots=" << prefixBoots
                  << " endpoint_bootstraps=" << endpointBootstraps
                  << " online_refreshes=" << (prefixBoots + endpointBootstraps)
                  << " batch=" << aggregateBatch
                  << " amortized_ms=" << (1000.0 * totalSeconds / (aggregateBatch * rounds))
                  << " failures=0"
                  << " max_error=" << maxError
                  << " worst_precision_bits=" << precisionBits
                  << " level=" << finalLevel
                  << " next_round=" << (nextRoundReady ? "READY" : "ENDPOINT_BOOTSTRAP_REQUIRED")
                  << " status=PASS\n"
                  << "OVERALL status=PASS\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 2;
    }
}
