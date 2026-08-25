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
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

using CipherPair = std::array<Ciphertext<DCRTPoly>, 2>;
using PlainPair = std::array<std::vector<uint8_t>, 2>;

enum class Operation {
    Add,
    GreaterThan,
    Equal,
    Xor,
};

struct AuditResult {
    size_t failures = 0;
    size_t firstFailure = 0;
    double firstValue = 0.0;
    double firstExpected = 0.0;
    double maxError = 0.0;
    double maxImaginary = 0.0;
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
    auto encoding = std::make_shared<ZEncodingImpl>(
        message.GetParams(), message, ciphertext->GetScalingFactorBFP(),
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
        const double error = std::abs(
            value - std::complex<double>(expectedValue, 0.0));
        result.maxError = std::max(result.maxError, error);
        result.maxImaginary = std::max(
            result.maxImaginary, std::abs(value.imag()));
        if (error > kBooleanTolerance) {
            if (result.failures == 0) {
                result.firstFailure = slot;
                result.firstValue = value.real();
                result.firstExpected = expectedValue;
            }
            ++result.failures;
        }
    }

    std::cout << "[gate] " << std::left << std::setw(28) << label
              << std::right << " failures=" << result.failures
              << " max_error=" << std::scientific << std::setprecision(3)
              << result.maxError << " max_imag=" << result.maxImaginary
              << std::defaultfloat << '\n';
    if (result.failures != 0) {
        std::cout << "[gate] first failure slot=" << result.firstFailure
                  << " value=" << std::scientific << result.firstValue
                  << " expected=" << result.firstExpected
                  << std::defaultfloat << '\n';
    }
    return result;
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

std::vector<uint8_t> PlainGreaterThan(
    const std::vector<uint8_t>& left,
    const std::vector<uint8_t>& right,
    uint32_t wordBits,
    uint32_t words) {
    std::vector<uint8_t> result(left.size());
    for (uint32_t word = 0; word < words; ++word) {
        bool greater = false;
        for (uint32_t bit = wordBits; bit-- > 0;) {
            const size_t slot = static_cast<size_t>(bit) * words + word;
            if (left[slot] != right[slot]) {
                greater = left[slot] > right[slot];
                break;
            }
        }
        for (uint32_t bit = 0; bit < wordBits; ++bit) {
            result[static_cast<size_t>(bit) * words + word] =
                greater ? 1 : 0;
        }
    }
    return result;
}

std::vector<uint8_t> PlainXor(const std::vector<uint8_t>& left,
                              const std::vector<uint8_t>& right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("plain XOR size mismatch");
    }
    std::vector<uint8_t> result(left.size());
    for (size_t slot = 0; slot < result.size(); ++slot) {
        result[slot] = left[slot] ^ right[slot];
    }
    return result;
}

std::vector<uint8_t> PlainEqual(const std::vector<uint8_t>& left,
                                const std::vector<uint8_t>& right,
                                uint32_t wordBits,
                                uint32_t words) {
    std::vector<uint8_t> result(left.size());
    for (uint32_t word = 0; word < words; ++word) {
        bool equal = true;
        for (uint32_t bit = 0; bit < wordBits; ++bit) {
            const size_t slot = static_cast<size_t>(bit) * words + word;
            if (left[slot] != right[slot]) {
                equal = false;
                break;
            }
        }
        for (uint32_t bit = 0; bit < wordBits; ++bit) {
            result[static_cast<size_t>(bit) * words + word] =
                equal ? 1 : 0;
        }
    }
    return result;
}

ZEncoding EncodeDense(const std::vector<uint8_t>& bits,
                      const std::shared_ptr<DCRTPoly::Params>& elementParams,
                      const BigFixedPoint& scale) {
    if (bits.size() != kComplexSlots) {
        throw std::runtime_error(
            "dense Boolean encoding must fill all complex slots");
    }
    std::vector<BigComplex> slots(bits.size());
    for (size_t slot = 0; slot < bits.size(); ++slot) {
        slots[slot] = BigComplex(
            bits[slot] ? BigFixedPoint::one() : BigFixedPoint::zero());
    }
    const ZEncodingParams params(
        BModeFull, kDenseMetadataZN, kDenseMetadataZSlots);
    return ZEncodingImpl::encodeC(
        CSlots(params, slots), elementParams, scale);
}

CipherPair BooleanBoundaryRefresh(const FHEZ& fhe,
                                  ConstCiphertext<DCRTPoly> first,
                                  ConstCiphertext<DCRTPoly> second,
                                  double* elapsedSeconds = nullptr) {
    const ZEncodingParams params(
        BModeFull, kDenseMetadataZN, kDenseMetadataZSlots);
    auto firstInput = first->Clone();
    auto secondInput = second->Clone();
    firstInput->SetZEncodingParams(params);
    secondInput->SetZEncodingParams(params);
    firstInput->SetSlots(kComplexSlots);
    secondInput->SetSlots(kComplexSlots);

    const auto start = std::chrono::steady_clock::now();
    auto refreshed = fhe->EvalBooleanToBooleanFull(
        CiphertextGroup({firstInput, secondInput}));
    const auto end = std::chrono::steady_clock::now();
    if (refreshed.size() != 2) {
        throw std::runtime_error("native B2B returned the wrong group size");
    }
    const double seconds = std::chrono::duration<double>(end - start).count();
    if (elapsedSeconds != nullptr) {
        *elapsedSeconds += seconds;
    }

    CipherPair output{refreshed[0], refreshed[1]};
    for (auto& ciphertext : output) {
        ciphertext->SetZEncodingParams(params);
        ciphertext->SetSlots(kComplexSlots);
    }
    std::cout << "[refresh] input_level=" << first->GetLevel()
              << " output_level=" << output[0]->GetLevel()
              << " seconds=" << seconds << '\n';
    return output;
}

Operation ParseOperation(const std::string& value) {
    if (value == "add") {
        return Operation::Add;
    }
    if (value == "gt") {
        return Operation::GreaterThan;
    }
    if (value == "eq") {
        return Operation::Equal;
    }
    if (value == "xor") {
        return Operation::Xor;
    }
    throw std::invalid_argument("operation must be add, gt, eq, or xor");
}

const char* OperationName(Operation operation) {
    switch (operation) {
        case Operation::Add:
            return "ADD";
        case Operation::GreaterThan:
            return "GT";
        case Operation::Equal:
            return "EQ";
        case Operation::Xor:
            return "XOR";
    }
    throw std::logic_error("invalid operation");
}

uint32_t ParseWordBits(const char* value) {
    size_t parsed = 0;
    const uint64_t result = std::stoull(value, &parsed);
    if (parsed != std::char_traits<char>::length(value) ||
        result < 16 || result > 256 || (result & (result - 1)) != 0) {
        throw std::invalid_argument(
            "word_bits must be one of 16, 32, 64, 128, or 256");
    }
    return static_cast<uint32_t>(result);
}

bool ParseSwitch(const char* value) {
    const std::string parsed(value);
    if (parsed == "0" || parsed == "false" || parsed == "off") {
        return false;
    }
    if (parsed == "1" || parsed == "true" || parsed == "on") {
        return true;
    }
    throw std::invalid_argument(
        "output_refresh must be 0/1, false/true, or off/on");
}

PlainPair EvaluatePlain(Operation operation,
                        const PlainPair& left,
                        const PlainPair& right,
                        uint32_t wordBits,
                        uint32_t words) {
    PlainPair output;
    for (size_t group = 0; group < output.size(); ++group) {
        switch (operation) {
            case Operation::Add:
                output[group] = PlainAdd(
                    left[group], right[group], wordBits, words);
                break;
            case Operation::GreaterThan:
                output[group] = PlainGreaterThan(
                    left[group], right[group], wordBits, words);
                break;
            case Operation::Equal:
                output[group] = PlainEqual(
                    left[group], right[group], wordBits, words);
                break;
            case Operation::Xor:
                output[group] = PlainXor(left[group], right[group]);
                break;
        }
    }
    return output;
}

CipherPair EvaluateCipher(Operation operation,
                          const DenseBooleanZ& dense,
                          const CipherPair& left,
                          const CipherPair& right,
                          uint32_t wordBits,
                          uint32_t words) {
    CipherPair output;
    for (size_t group = 0; group < output.size(); ++group) {
        switch (operation) {
            case Operation::Add:
                output[group] = dense->EvalPulsarAdd(
                    left[group], right[group], wordBits, words);
                break;
            case Operation::GreaterThan: {
                auto rightGreaterEqualLeft = dense->EvalPulsarGreaterEqual(
                    right[group], left[group], wordBits, words);
                output[group] = dense->EvalBooleanNot(rightGreaterEqualLeft);
                break;
            }
            case Operation::Equal:
                output[group] = dense->EvalPulsarEqual(
                    left[group], right[group], wordBits, words);
                break;
            case Operation::Xor:
                output[group] = dense->EvalBooleanXor(
                    left[group], right[group]);
                break;
        }
    }
    return output;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        std::cout << std::unitbuf;
        if (argc > 4) {
            std::cerr << "Usage: " << argv[0]
                      << " [add|gt|eq|xor] [word_bits] [output_refresh:0|1]\n";
            return 2;
        }

        const Operation operation = ParseOperation(argc >= 2 ? argv[1] : "add");
        const uint32_t wordBits = argc >= 3 ? ParseWordBits(argv[2]) : 128;
        const bool outputRefresh = argc >= 4 ? ParseSwitch(argv[3]) : true;
        const uint32_t nativeBatch = kComplexSlots / wordBits;
        const uint32_t aggregateBatch = 2 * nativeBatch;
        const std::vector<uint32_t> levelBudget{3, 2};

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

        const auto cryptoParams =
            std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(
                cc->GetCryptoParameters());
        const auto elementParams = cryptoParams->GetElementParams();
        const auto paramsP = cryptoParams->GetParamsP();
        const uint32_t q0Bits =
            elementParams->GetParams()[0]->GetModulus().GetMSB();
        const uint32_t logQP =
            (elementParams->GetModulus() * paramsP->GetModulus()).GetMSB();
        if (q0Bits != kFirstModulusBits ||
            elementParams->GetParams().size() != kExpectedQCount ||
            paramsP->GetParams().size() != kExpectedPCount ||
            logQP > kClassic128LogQPBound) {
            std::ostringstream message;
            message << "generated parameter chain does not match the PULSAR contract"
                    << " q0_bits=" << q0Bits
                    << " q_count=" << elementParams->GetParams().size()
                    << " p_count=" << paramsP->GetParams().size()
                    << " logQP=" << logQP;
            throw std::runtime_error(message.str());
        }

        LeveledZ leveled = std::make_shared<LeveledZImpl>();
        AdvancedZ advanced = std::make_shared<AdvancedZImpl>(leveled);
        FHEZ fhe = std::make_shared<FHEZImpl>(leveled, advanced);
        DenseBooleanZ dense = std::make_shared<DenseBooleanZImpl>(leveled);

        std::cout << "[setup-stage] precomputing native B2B transforms\n";
        fhe->EvalBootstrapSetup(
            *cc, kDenseMetadataZN, kDenseMetadataZSlots,
            levelBudget, {0, 0}, 4, -24, 1);
        fhe->ReleaseUnusedBooleanRadixPrecom(kComplexSlots);

        auto keyPair = cc->KeyGen();
        cc->EvalMultKeyGen(keyPair.secretKey);
        fhe->EvalBootstrapKeyGenBooleanRadixOnly(
            keyPair.secretKey, kDenseMetadataZN, kDenseMetadataZSlots);

        std::vector<int32_t> rotations;
        for (uint32_t distance = 1; distance < wordBits; distance <<= 1) {
            const int32_t step = static_cast<int32_t>(distance * nativeBatch);
            rotations.push_back(-step);
            rotations.push_back(step);
        }
        std::sort(rotations.begin(), rotations.end());
        rotations.erase(
            std::unique(rotations.begin(), rotations.end()), rotations.end());
        cc->EvalRotateKeyGen(keyPair.secretKey, rotations);

        PKEZ pke = std::make_shared<PKEZImpl>(
            keyPair.publicKey, keyPair.secretKey);
        pkeZ_global = pke;
        const auto scale = cryptoParams->GetScalingFactorBFP(0);

        const PlainPair leftPlain{
            MakeOperand(wordBits, nativeBatch, 0xA501U, 0),
            MakeOperand(wordBits, nativeBatch, 0xA502U, 0)};
        PlainPair rightPlain{
            MakeOperand(wordBits, nativeBatch, 0xB601U, 1),
            MakeOperand(wordBits, nativeBatch, 0xB602U, 1)};
        const std::array<uint32_t, 3> equalWords{
            4, nativeBatch / 2, nativeBatch - 1};
        for (size_t group = 0; group < rightPlain.size(); ++group) {
            for (const uint32_t word : equalWords) {
                for (uint32_t bit = 0; bit < wordBits; ++bit) {
                    const size_t slot =
                        static_cast<size_t>(bit) * nativeBatch + word;
                    rightPlain[group][slot] = leftPlain[group][slot];
                }
            }
        }
        const PlainPair expected = EvaluatePlain(
            operation, leftPlain, rightPlain, wordBits, nativeBatch);

        CipherPair left{
            pke->Encrypt(EncodeDense(leftPlain[0], elementParams, scale)),
            pke->Encrypt(EncodeDense(leftPlain[1], elementParams, scale))};
        CipherPair right{
            pke->Encrypt(EncodeDense(rightPlain[0], elementParams, scale)),
            pke->Encrypt(EncodeDense(rightPlain[1], elementParams, scale))};
        const auto setupEnd = std::chrono::steady_clock::now();

        std::cout << "[input-refresh] left B2B excluded\n";
        left = BooleanBoundaryRefresh(fhe, left[0], left[1]);
        std::cout << "[input-refresh] right B2B excluded\n";
        right = BooleanBoundaryRefresh(fhe, right[0], right[1]);
        const uint32_t inputLevel = left[0]->GetLevel();

        const auto operationStart = std::chrono::steady_clock::now();
        auto output = EvaluateCipher(
            operation, dense, left, right, wordBits, nativeBatch);
        const auto operationEnd = std::chrono::steady_clock::now();
        const uint32_t rawOutputLevel = output[0]->GetLevel();

        double outputRefreshSeconds = 0.0;
        if (outputRefresh) {
            std::cout << "[output-refresh] native B2B included\n";
            output = BooleanBoundaryRefresh(
                fhe, output[0], output[1], &outputRefreshSeconds);
        }
        const auto measuredEnd = std::chrono::steady_clock::now();

        const auto audit0 = AuditDense(
            "output group 0", output[0], keyPair.secretKey, expected[0]);
        const auto audit1 = AuditDense(
            "output group 1", output[1], keyPair.secretKey, expected[1]);
        const size_t failures = audit0.failures + audit1.failures;
        const double maxError = std::max(audit0.maxError, audit1.maxError);
        const double operationSeconds = std::chrono::duration<double>(
            operationEnd - operationStart).count();
        const double totalSeconds = std::chrono::duration<double>(
            measuredEnd - operationStart).count();
        const double setupSeconds = std::chrono::duration<double>(
            setupEnd - setupStart).count();

        std::cout << "============================================================\n"
                  << "PULSAR single-round CPU operator benchmark\n"
                  << "operation=" << OperationName(operation) << '\n'
                  << "word_bits=" << wordBits << '\n'
                  << "native_batch=" << nativeBatch << '\n'
                  << "aggregate_batch=" << aggregateBatch << '\n'
                  << "ring_dimension=" << kRingDimension << '\n'
                  << "log_scale=" << kScalingBits << '\n'
                  << "multiplicative_depth=" << kMultiplicativeDepth << '\n'
                  << "logQP=" << logQP << '\n'
                  << "security=HEStd_128_classic\n"
                  << "input_refresh=two_native_B2B_calls_excluded\n"
                  << "output_refresh="
                  << (outputRefresh ? "native_B2B_included" : "disabled") << '\n'
                  << "input_level=" << inputLevel << '\n'
                  << "raw_output_level=" << rawOutputLevel << '\n'
                  << "final_output_level=" << output[0]->GetLevel() << '\n'
                  << "setup_seconds_excluded=" << setupSeconds << '\n'
                  << "============================================================\n";

        std::cout << "RESULT operation=" << OperationName(operation)
                  << " word_bits=" << wordBits
                  << " batch=" << aggregateBatch
                  << " operation_s=" << std::fixed << std::setprecision(9)
                  << operationSeconds
                  << " output_refresh_s=" << outputRefreshSeconds
                  << " total_s=" << totalSeconds
                  << " amortized_ms="
                  << (1000.0 * totalSeconds / aggregateBatch)
                  << " failures=" << failures
                  << " max_error=" << maxError
                  << " status=" << (failures == 0 ? "PASS" : "FAIL")
                  << '\n';
        std::cout << "OVERALL status="
                  << (failures == 0 ? "PASS" : "FAIL") << '\n';
        return failures == 0 ? 0 : 1;
    }
    catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
