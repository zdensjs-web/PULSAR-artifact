#include "openfhe.h"
#include "encoding/z-encoding.h"
#include "scheme/ckksrns/z-advancedshe.h"
#include "scheme/ckksrns/z-dense-radix-multiplier.h"
#include "scheme/ckksrns/z-fhe.h"
#include "scheme/ckksrns/z-pke.h"

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace lbcrypto;
using boost::multiprecision::cpp_int;

namespace {

constexpr uint32_t kLogN = 16;
constexpr uint32_t kRingDimension = 1U << kLogN;
constexpr uint32_t kComplexSlots = kRingDimension / 2;
uint32_t kWordBits = 256;
uint32_t kNativeBatch = 128;
uint32_t kAggregateBatch = 256;
uint32_t kRadixDigits = 64;
uint32_t kLazyCarryPasses = 3;
uint32_t kPrefixLayers = 6;
uint32_t kMultiplierRawBoundaryLevel = 28;
constexpr uint32_t kScalingBits = 43;
constexpr uint32_t kFirstModulusBits = 52;
constexpr uint32_t kMultiplicativeDepth = 32;
constexpr uint32_t kAuxiliaryPrimeBits = 60;
constexpr uint32_t kHybridLargeDigits = 6;
constexpr uint32_t kClassic128LogQPBound = 1747;
constexpr double kTolerance = 0.125;

void Stage(const char* label) {
    std::cout << "[setup-stage] " << label << std::endl;
}

void ConfigureWidth(uint32_t bits) {
    if (bits < 16 || bits > 256 || (bits & (bits - 1)) != 0 || bits % 4 != 0) {
        throw std::invalid_argument(
            "word_bits must be one of 16,32,64,128,256");
    }
    kWordBits = bits;
    kNativeBatch = kComplexSlots / bits;
    kAggregateBatch = 2 * kNativeBatch;
    kRadixDigits = bits / 4;
    kLazyCarryPasses = 0;
    uint32_t maximumDigit = kRadixDigits * 15 * 15;
    while (maximumDigit >= 31) {
        maximumDigit = maximumDigit / 16 + 15;
        ++kLazyCarryPasses;
    }
    kPrefixLayers = 0;
    for (uint32_t distance = 1; distance < kRadixDigits; distance <<= 1) {
        ++kPrefixLayers;
    }
    kMultiplierRawBoundaryLevel = 18 + kPrefixLayers + 3 + 1;
}

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
        auto component = ciphertext[index];
        component.SetFormat(Format::EVALUATION);
        message += secretPower * component;
        secretPower *= secretAtLevel;
    }
    return message;
}

std::vector<BigComplex> Decode(ConstCiphertext<DCRTPoly> ciphertext,
                               const PrivateKey<DCRTPoly>& privateKey) {
    auto message = DecryptCore(ciphertext->GetElements(), privateKey);
    auto encoding = std::make_shared<ZEncodingImpl>(
        message.GetParams(), message, ciphertext->GetScalingFactorBFP(),
        ciphertext->GetZEncodingParams());
    return ZEncodingImpl::decodeR(encoding).toCSlots().getSlots();
}

AuditResult Audit(const std::string& label,
                  ConstCiphertext<DCRTPoly> ciphertext,
                  const PrivateKey<DCRTPoly>& privateKey,
                  const std::vector<uint8_t>& expected) {
    const auto decoded = Decode(ciphertext, privateKey);
    if (decoded.size() != expected.size()) {
        throw std::runtime_error(label + ": decoded slot count mismatch");
    }
    AuditResult result;
    for (size_t slot = 0; slot < expected.size(); ++slot) {
        const auto value = decoded[slot].convertToComplex();
        const double want = static_cast<double>(expected[slot]);
        const double error = std::abs(value - std::complex<double>(want, 0.0));
        result.maxError = std::max(result.maxError, error);
        result.maxImaginary = std::max(result.maxImaginary,
                                       std::abs(value.imag()));
        if (error > kTolerance) {
            if (result.failures == 0) {
                result.firstFailure = slot;
                result.firstValue = value.real();
                result.firstExpected = want;
            }
            ++result.failures;
        }
    }
    std::cout << "[gate] " << std::left << std::setw(28) << label << std::right
              << " failures=" << result.failures
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

AuditResult AuditIntegerSlots(const std::string& label,
                              ConstCiphertext<DCRTPoly> ciphertext,
                              const PrivateKey<DCRTPoly>& privateKey,
                              const std::vector<int64_t>& expected,
                              double tolerance = kTolerance) {
    const auto decoded = Decode(ciphertext, privateKey);
    if (decoded.size() != expected.size()) {
        throw std::runtime_error(label + ": decoded slot count mismatch");
    }
    AuditResult result;
    for (size_t slot = 0; slot < expected.size(); ++slot) {
        const auto value = decoded[slot].convertToComplex();
        const double want = static_cast<double>(expected[slot]);
        const double error = std::abs(value - std::complex<double>(want, 0.0));
        result.maxError = std::max(result.maxError, error);
        result.maxImaginary = std::max(result.maxImaginary, std::abs(value.imag()));
        if (error > tolerance) {
            if (result.failures == 0) {
                result.firstFailure = slot;
                result.firstValue = value.real();
                result.firstExpected = want;
            }
            ++result.failures;
        }
    }
    std::cout << "[gate] " << std::left << std::setw(28) << label << std::right
              << " failures=" << result.failures
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

void Require(const std::string& label, const AuditResult& result) {
    if (result.failures != 0) {
        std::ostringstream message;
        message << label << " failed: failures=" << result.failures
                << " max_error=" << result.maxError;
        throw std::runtime_error(message.str());
    }
}

std::vector<cpp_int> MakeWords(uint64_t seed, bool right) {
    std::mt19937_64 random(seed);
    std::vector<cpp_int> result(kAggregateBatch);
    const cpp_int modulus = cpp_int(1) << kWordBits;
    for (uint32_t word = 0; word < kAggregateBatch; ++word) {
        cpp_int value = 0;
        for (uint32_t limb = 0; limb < (kWordBits + 63) / 64; ++limb) {
            value |= cpp_int(random()) << (64 * limb);
        }
        result[word] = value & (modulus - 1);
    }
    result[0] = right ? cpp_int(1) : cpp_int(modulus - 1);
    result[1] = right ? cpp_int(1) :
                cpp_int((cpp_int(1) << (kWordBits - 1)) - 1);
    result[2] = right ? cpp_int(modulus - 1) : cpp_int(0);
    result[3] = cpp_int(modulus - 1);
    return result;
}

std::vector<cpp_int> PlainProducts(const std::vector<cpp_int>& left,
                                  const std::vector<cpp_int>& right) {
    const cpp_int mask = (cpp_int(1) << kWordBits) - 1;
    std::vector<cpp_int> result(left.size());
    for (size_t word = 0; word < left.size(); ++word) {
        result[word] = (left[word] * right[word]) & mask;
    }
    return result;
}

std::vector<uint8_t> DenseGroupBits(const std::vector<cpp_int>& words,
                                    uint32_t group) {
    std::vector<uint8_t> bits(kComplexSlots);
    const uint32_t baseWord = group * kNativeBatch;
    for (uint32_t bit = 0; bit < kWordBits; ++bit) {
        for (uint32_t word = 0; word < kNativeBatch; ++word) {
            bits[bit * kNativeBatch + word] =
                static_cast<uint8_t>(
                    ((words[baseWord + word] >> bit) & 1).convert_to<unsigned>());
        }
    }
    return bits;
}

std::vector<int64_t> PackReference(const std::vector<cpp_int>& words) {
    std::vector<int64_t> slots(kComplexSlots, 0);
    for (uint32_t digit = 0; digit < kRadixDigits; ++digit) {
        for (uint32_t word = 0; word < kAggregateBatch; ++word) {
            slots[digit * kAggregateBatch + word] =
                ((words[word] >> (4 * digit)) & 15).convert_to<int64_t>();
        }
    }
    return slots;
}

std::vector<int64_t> ConvolutionReference(
    const std::vector<cpp_int>& left,
    const std::vector<cpp_int>& right) {
    std::vector<int64_t> slots(kComplexSlots, 0);
    for (uint32_t word = 0; word < kAggregateBatch; ++word) {
        for (uint32_t leftDigit = 0; leftDigit < kRadixDigits; ++leftDigit) {
            const int64_t a =
                ((left[word] >> (4 * leftDigit)) & 15).convert_to<int64_t>();
            for (uint32_t rightDigit = 0;
                 leftDigit + rightDigit < kRadixDigits; ++rightDigit) {
                const int64_t b =
                    ((right[word] >> (4 * rightDigit)) & 15).convert_to<int64_t>();
                slots[(leftDigit + rightDigit) * kAggregateBatch + word] += a * b;
            }
        }
    }
    return slots;
}

std::vector<int64_t> LazyReference(std::vector<int64_t> slots,
                                   uint32_t passes) {
    for (uint32_t pass = 0; pass < passes; ++pass) {
        for (uint32_t word = 0; word < kAggregateBatch; ++word) {
            std::vector<int64_t> digits(kRadixDigits);
            for (uint32_t digit = 0; digit < kRadixDigits; ++digit) {
                digits[digit] = slots[digit * kAggregateBatch + word];
            }
            DenseRadixMultiplierZImpl::DecodeLazyCarryReference(digits);
            for (uint32_t digit = 0; digit < kRadixDigits; ++digit) {
                slots[digit * kAggregateBatch + word] = digits[digit];
            }
        }
    }
    return slots;
}

std::vector<int64_t> FBTReference(const std::vector<int64_t>& lazy,
                                  uint32_t output) {
    std::vector<int64_t> slots(kComplexSlots, 0);
    for (uint32_t digit = 0; digit < kRadixDigits; ++digit) {
        for (uint32_t word = 0; word < kAggregateBatch; ++word) {
            const int64_t value = lazy[digit * kAggregateBatch + word];
            int64_t expected = 0;
            if (output < 4) {
                expected = ((value & 15) >> output) & 1;
            }
            else if (output == 4) {
                expected = value >= 16;
            }
            else {
                expected = value == 15;
            }
            slots[digit * kAggregateBatch + word] = expected;
        }
    }
    return slots;
}

std::vector<int64_t> CarryReference(const std::vector<int64_t>& lazy) {
    std::vector<int64_t> slots(kComplexSlots, 0);
    for (uint32_t word = 0; word < kAggregateBatch; ++word) {
        int64_t carry = 0;
        for (uint32_t digit = 0; digit < kRadixDigits; ++digit) {
            slots[digit * kAggregateBatch + word] = carry;
            carry = lazy[digit * kAggregateBatch + word] + carry >= 16;
        }
        // PrefixCarry rotates each word's carry-out into the following radix
        // row. The first guard row temporarily holds the carry past the word
        // boundary before the output map discards it modulo 2^word_bits.
        slots[kRadixDigits * kAggregateBatch + word] = carry;
    }
    return slots;
}

std::vector<int64_t> CorrectedPlaneReference(
    const std::vector<cpp_int>& products,
    const std::vector<int64_t>& carry,
    uint32_t plane) {
    std::vector<int64_t> slots(kComplexSlots, 0);
    for (uint32_t digit = 0; digit < kRadixDigits; ++digit) {
        for (uint32_t word = 0; word < kAggregateBatch; ++word) {
            slots[digit * kAggregateBatch + word] =
                ((products[word] >> (4 * digit + plane)) & 1)
                    .convert_to<int64_t>();
        }
    }
    if (plane == 0) {
        for (uint32_t word = 0; word < kAggregateBatch; ++word) {
            slots[kRadixDigits * kAggregateBatch + word] =
                carry[kRadixDigits * kAggregateBatch + word];
        }
    }
    return slots;
}

ZEncoding EncodeDense(const std::vector<uint8_t>& bits,
                      const std::shared_ptr<DCRTPoly::Params>& elementParams,
                      const BigFixedPoint& scale) {
    std::vector<BigComplex> slots(bits.size());
    for (size_t slot = 0; slot < bits.size(); ++slot) {
        slots[slot] = BigComplex(bits[slot] ? BigFixedPoint::one() :
                                             BigFixedPoint::zero());
    }
    return ZEncodingImpl::encodeC(
        CSlots(ZEncodingParams(BModeFull, kWordBits, kAggregateBatch), slots),
        elementParams, scale);
}

CiphertextGroup RefreshPair(const FHEZ& fhe,
                            Ciphertext<DCRTPoly> first,
                            Ciphertext<DCRTPoly> second) {
    const ZEncodingParams metadata(BModeFull, kWordBits, kAggregateBatch);
    first->SetZEncodingParams(metadata);
    second->SetZEncodingParams(metadata);
    auto result = fhe->EvalBooleanToBooleanFull(
        CiphertextGroup({first, second}));
    if (result.size() != 2) {
        throw std::runtime_error("B2B did not return two Boolean ciphertexts");
    }
    result[0]->SetZEncodingParams(metadata);
    result[1]->SetZEncodingParams(metadata);
    return result;
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

std::vector<int32_t> MissingRotationIndices(
    const CryptoContext<DCRTPoly>& cc,
    const PrivateKey<DCRTPoly>& secretKey,
    const std::vector<int32_t>& required) {
    const auto& existing =
        CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(
            secretKey->GetKeyTag());
    std::vector<int32_t> missing;
    missing.reserve(required.size());
    for (const int32_t rotation : required) {
        const uint32_t automorphism = FindAutomorphismIndex2nComplex(
            rotation, cc->GetCyclotomicOrder());
        if (existing.find(automorphism) == existing.end()) {
            missing.push_back(rotation);
        }
    }
    return missing;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc > 3) {
            std::cerr << "Usage: " << argv[0]
                      << " [word_bits] [repeats]\n";
            return 2;
        }
        uint32_t wordBits = 256;
        uint32_t repeats = 1;
        if (argc == 2) {
            const uint32_t value = static_cast<uint32_t>(std::stoul(argv[1]));
            if (value >= 16 && value <= 256 && (value & (value - 1)) == 0) {
                wordBits = value;
            }
            else {
                repeats = value;
            }
        }
        else if (argc == 3) {
            wordBits = static_cast<uint32_t>(std::stoul(argv[1]));
            repeats = static_cast<uint32_t>(std::stoul(argv[2]));
        }
        ConfigureWidth(wordBits);
        if (repeats == 0) {
            throw std::invalid_argument("repeats must be positive");
        }

        // Context, key and transform preparation are excluded setup. Thread
        // control is intentionally external: OMP_NUM_THREADS applies to setup,
        // input refresh, the complete circuit and final refresh alike.
        const auto setupStart = std::chrono::steady_clock::now();
        Stage("external OpenMP thread policy enabled");
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

        auto cc = GenCryptoContext(parameters);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        const auto cryptoParams =
            std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(
                cc->GetCryptoParameters());
        const auto elementParams = cryptoParams->GetElementParams();
        const auto paramsP = cryptoParams->GetParamsP();
        const uint32_t logQP =
            (elementParams->GetModulus() * paramsP->GetModulus()).GetMSB();
        if (logQP > kClassic128LogQPBound) {
            std::ostringstream message;
            message << "generated logQP=" << logQP
                    << " exceeds classic-128 bound " << kClassic128LogQPBound;
            throw std::runtime_error(message.str());
        }
        const uint32_t q0Bits =
            elementParams->GetParams()[0]->GetModulus().GetMSB();
        const uint32_t p0Bits = paramsP->GetParams()[0]->GetModulus().GetMSB();
        if (p0Bits < q0Bits + 6) {
            throw std::runtime_error(
                "sparse encapsulated key switching P/Q margin is insufficient");
        }

        Stage("context ready");
        auto keyPair = cc->KeyGen();
        Stage("generating relinearization key");
        cc->EvalMultKeyGen(keyPair.secretKey);
        Stage("relinearization key ready");
        LeveledZ leveled = std::make_shared<LeveledZImpl>();
        AdvancedZ advanced = std::make_shared<AdvancedZImpl>(leveled);
        FHEZ fhe = std::make_shared<FHEZImpl>(leveled, advanced);
        DenseRadixMultiplierZ multiplier =
            std::make_shared<DenseRadixMultiplierZImpl>(wordBits, leveled, fhe);
        PKEZ pke = std::make_shared<PKEZImpl>(keyPair.publicKey,
                                              keyPair.secretKey);
        pkeZ_global = pke;

        // The 256-bit circuit ends at level 28; smaller widths retain their
        // unused tail levels. Two final Q primes are reserved for native
        // B2B's pre-ModRaise C2R runway for every width.
        const std::vector<uint32_t> levelBudget{3, 2};
        const uint32_t finalB2BInputLevel =
            kMultiplicativeDepth - levelBudget[1];
        if (kMultiplierRawBoundaryLevel > finalB2BInputLevel) {
            throw std::runtime_error(
                "selected width leaves insufficient native B2B runway");
        }
        Stage("precomputing B2B transforms");
        fhe->EvalBootstrapSetup(*cc, kWordBits, kAggregateBatch, levelBudget,
                                {0, 0}, 4, -24, 1);
        Stage("B2B transforms ready; generating minimal Boolean/radix keys");
        fhe->EvalBootstrapKeyGenBooleanRadixOnly(
            keyPair.secretKey, kWordBits, kAggregateBatch);
        const size_t booleanRadixKeyCount =
            CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(
                keyPair.secretKey->GetKeyTag()).size();
        std::cout << "[setup-stage] minimal Boolean/radix keys ready: "
                  << booleanRadixKeyCount << std::endl;
        Stage("multiplier rotation keys");
        const auto missingRotations = MissingRotationIndices(
            cc, keyPair.secretKey,
            DenseRadixMultiplierZImpl::RotationIndices(wordBits));
        if (!missingRotations.empty()) {
            cc->EvalRotateKeyGen(keyPair.secretKey, missingRotations);
        }
        const size_t totalRotationKeyCount =
            CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(
                keyPair.secretKey->GetKeyTag()).size();
        Stage("releasing unused generic A2B/B2A precomputations");
        fhe->ReleaseUnusedBooleanRadixPrecom(kComplexSlots);
        Stage("all evaluation keys ready");

        Stage("building deterministic plaintext inputs");
        const auto leftWords = MakeWords(0xA501B601ULL, false);
        const auto rightWords = MakeWords(0xC701D801ULL, true);
        const auto productWords = PlainProducts(leftWords, rightWords);
        const auto left0 = DenseGroupBits(leftWords, 0);
        const auto left1 = DenseGroupBits(leftWords, 1);
        const auto right0 = DenseGroupBits(rightWords, 0);
        const auto right1 = DenseGroupBits(rightWords, 1);
        const auto expected0 = DenseGroupBits(productWords, 0);
        const auto expected1 = DenseGroupBits(productWords, 1);
        const auto scale = cryptoParams->GetScalingFactorBFP(0);
        Stage("encrypting four dense Boolean inputs");
        auto ctLeft0 = pke->Encrypt(EncodeDense(left0, elementParams, scale));
        auto ctLeft1 = pke->Encrypt(EncodeDense(left1, elementParams, scale));
        auto ctRight0 = pke->Encrypt(EncodeDense(right0, elementParams, scale));
        auto ctRight1 = pke->Encrypt(EncodeDense(right1, elementParams, scale));
        Stage("four dense Boolean inputs ready");
        const auto setupEnd = std::chrono::steady_clock::now();

        std::cout << "[setup] excluded shared B2B input refresh A\n";
        const auto inputRefreshStart = std::chrono::steady_clock::now();
        auto stableLeft = RefreshPair(fhe, ctLeft0, ctLeft1);
        std::cout << "[setup] excluded shared B2B input refresh B\n";
        auto stableRight = RefreshPair(fhe, ctRight0, ctRight1);
        const auto inputRefreshEnd = std::chrono::steady_clock::now();
        Require("refreshed A group 0",
                Audit("refreshed A group 0", stableLeft[0],
                      keyPair.secretKey, left0));
        Require("refreshed A group 1",
                Audit("refreshed A group 1", stableLeft[1],
                      keyPair.secretKey, left1));
        Require("refreshed B group 0",
                Audit("refreshed B group 0", stableRight[0],
                      keyPair.secretKey, right0));
        Require("refreshed B group 1",
                Audit("refreshed B group 1", stableRight[1],
                      keyPair.secretKey, right1));

        const auto packedLeftReference = PackReference(leftWords);
        const auto packedRightReference = PackReference(rightWords);
        const auto convolutionReference =
            ConvolutionReference(leftWords, rightWords);
        std::vector<std::vector<int64_t>> lazyReferences;
        lazyReferences.reserve(kLazyCarryPasses);
        for (uint32_t pass = 1; pass <= kLazyCarryPasses; ++pass) {
            lazyReferences.push_back(LazyReference(convolutionReference, pass));
        }
        const auto& finalLazyReference = lazyReferences.back();
        const auto carryReference = CarryReference(finalLazyReference);
        std::map<std::string, std::vector<int64_t>> stageReference;
        stageReference.emplace("packed A", packedLeftReference);
        stageReference.emplace("packed B", packedRightReference);
        stageReference.emplace("convolution", convolutionReference);
        for (uint32_t pass = 0; pass < kLazyCarryPasses; ++pass) {
            stageReference.emplace("LazyCarry #" + std::to_string(pass + 1),
                                   lazyReferences[pass]);
        }
        for (uint32_t outputIndex = 0; outputIndex < 6; ++outputIndex) {
            static const char* labels[6] = {"FBT b0", "FBT b1", "FBT b2",
                                             "FBT b3", "FBT G", "FBT P"};
            stageReference.emplace(labels[outputIndex],
                                   FBTReference(finalLazyReference, outputIndex));
        }
        stageReference.emplace("radix carry", carryReference);
        for (uint32_t plane = 0; plane < 4; ++plane) {
            stageReference.emplace(
                "corrected b" + std::to_string(plane),
                CorrectedPlaneReference(productWords, carryReference, plane));
        }
        multiplier->SetStageObserver(
            [&](const std::string& label, ConstCiphertext<DCRTPoly> ciphertext) {
                const auto found = stageReference.find(label);
                if (found == stageReference.end()) {
                    return;
                }
                const auto audit = AuditIntegerSlots(label, ciphertext,
                                                     keyPair.secretKey,
                                                     found->second);
                Require(label, audit);
            });

        std::cout << "============================================================\n"
                  << "PULSAR single-round Boolean multiplication over OpenFHE\n"
                  << "implementation_build=v1.2.0-q33-scale43-dense-radix16\n"
                  << "semantics=(A*B) mod 2^" << kWordBits << '\n'
                  << "input=actual_shared_B2B_output\n"
                  << "layout=two_guard_free_Boolean_ciphertexts_per_operand\n"
                  << "word_bits=" << kWordBits << '\n'
                  << "native_batch=" << kNativeBatch << '\n'
                  << "aggregate_batch=" << kAggregateBatch << '\n'
                  << "input_pack=one_level_Boolean_to_radix16\n"
                  << "convolution=normalized_DFT_ctct_product_masked_inverse_DFT\n"
                  << "radix_digits=" << kRadixDigits << '\n'
                  << "lazy_carry_passes=" << kLazyCarryPasses << '\n'
                  << "functional_transform_factors=C2S_3+S2C_3\n"
                  << "shared_FBT=(b0+i*b1),(b2+i*b3),(G+i*P)\n"
                  << "FBT_P=raw_[x==15];_zero_guard_segments_prefix\n"
                  << "guard_row=" << kRadixDigits
                  << "_modulus_overflow_discarded_by_output_pack\n"
                  << "post_FBT_tail=" << kPrefixLayers
                  << "_prefix+3_correction+1_repack\n"
                  << "prefix_layers=" << kPrefixLayers << '\n'
                  << "nibble_correction=exact_balanced_depth_3\n"
                  << "output_pack=one_level_radix_to_two_Boolean_groups\n"
                  << "final_refresh=one_unchanged_shared_native_B2B_included\n"
                  << "allowed_difference_1=OpenFHE_Q_chain\n"
                  << "allowed_difference_2=boundary_Boot_replaced_by_B2B\n"
                  << "alternative_paths=none\n"
                  << "new_multiplier_rotation_keys="
                  << missingRotations.size() << '\n'
                  << "Boolean_radix_bootstrap_keys="
                  << booleanRadixKeyCount << '\n'
                  << "total_automorphism_keys="
                  << totalRotationKeyCount << '\n'
                  << "unused_A2B_B2A_keys=not_generated\n"
                  << "unused_A2B_B2A_precomputations=released\n"
                  << "linear_transform_plaintexts=fixed_level_no_runtime_cache\n"
                  << "linear_transform_rotations=hoisted_BSGS\n"
                  << "thread_control=external_OMP_environment\n"
                  << "ring_dimension=" << cc->GetRingDimension() << '\n'
                  << "logN=" << kLogN << '\n'
                  << "log_scale=" << kScalingBits << '\n'
                  << "requested_first_modulus_bits=" << kFirstModulusBits << '\n'
                  << "multiplicative_depth=" << kMultiplicativeDepth << '\n'
                  << "multiplier_circuit_max_level="
                  << kMultiplierRawBoundaryLevel << '\n'
                  << "native_B2B_entry_runway_levels="
                  << levelBudget[1] << '\n'
                  << "Q_count=" << elementParams->GetParams().size() << '\n'
                  << "Q_bits=" << PrimeBits(elementParams->GetParams()) << '\n'
                  << "P_count=" << paramsP->GetParams().size() << '\n'
                  << "P_bits=" << PrimeBits(paramsP->GetParams()) << '\n'
                  << "logQP=" << logQP << '\n'
                  << "classic_128_logQP_bound=" << kClassic128LogQPBound << '\n'
                  << "stable_input_level=" << stableLeft[0]->GetLevel() << '\n'
                  << "final_B2B_input_level=" << finalB2BInputLevel << '\n'
                  << "scheduled_shared_FBT_output_level="
                  << 18 << '\n'
                  << "setup_seconds_excluded="
                  << std::chrono::duration<double>(setupEnd - setupStart).count() << '\n'
                  << "input_refresh_seconds_excluded="
                  << std::chrono::duration<double>(inputRefreshEnd - inputRefreshStart).count() << '\n'
                  << "repeats=" << repeats << '\n'
                  << "============================================================\n";

        CiphertextGroup output;
        double totalOnline = 0.0;
        for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
            std::cout << "[run] " << (repeat + 1) << '/' << repeats
                      << " exact multiplier\n";
            output = multiplier->EvalMultiply(stableLeft, stableRight,
                                               finalB2BInputLevel);
            totalOnline += multiplier->GetTiming().OnlineSeconds();
        }
        const auto audit0 = Audit("final product group 0", output[0],
                                  keyPair.secretKey, expected0);
        const auto audit1 = Audit("final product group 1", output[1],
                                  keyPair.secretKey, expected1);
        Require("final product group 0", audit0);
        Require("final product group 1", audit1);

        const auto& timing = multiplier->GetTiming();
        const auto& levels = multiplier->GetLevels();
        const double meanOnline = totalOnline / repeats;
        const double maxError = std::max(audit0.maxError, audit1.maxError);
        const double precision = maxError == 0.0 ?
            std::numeric_limits<double>::infinity() : -std::log2(maxError);
        std::cout << std::fixed << std::setprecision(9)
                  << "\nOnline timing\n"
                  << "  Boolean -> radix pack : " << timing.inputPack << "s\n"
                  << "  DFT convolution       : " << timing.convolution << "s\n";
        for (uint32_t pass = 0; pass < kLazyCarryPasses; ++pass) {
            std::cout << "  LazyCarry #" << (pass + 1)
                      << "          : " << timing.lazyCarry[pass] << "s\n";
        }
        std::cout << "  shared six-output FBT : " << timing.sharedFBT << "s\n"
                  << "  radix carry prefix    : " << timing.prefix << "s\n"
                  << "  nibble correction     : " << timing.correction << "s\n"
                  << "  radix -> Boolean pack : " << timing.outputPack << "s\n"
                  << "  final shared B2B      : " << timing.finalB2B << "s\n"
                  << "  setup excluded        : " << timing.setupExcluded << "s\n"
                  << "\nCiphertext levels\n"
                  << "  stable input          : " << levels.stableInput << '\n'
                  << "  packed radix          : " << levels.packedRadix << '\n'
                  << "  convolution           : " << levels.convolution << '\n'
                  << "  LazyCarry outputs     : ";
        for (uint32_t pass = 0; pass < kLazyCarryPasses; ++pass) {
            if (pass != 0) {
                std::cout << '/';
            }
            std::cout << levels.lazyCarry[pass];
        }
        std::cout << '\n'
                  << "  shared FBT raw        : " << levels.sharedFBTRaw << '\n'
                  << "  shared FBT scheduled  : " << levels.sharedFBT << '\n'
                  << "  prefix                : " << levels.prefix << '\n'
                  << "  corrected planes      : " << levels.correction << '\n'
                  << "  raw Boolean           : " << levels.rawBoolean << '\n'
                  << "  final Boolean         : " << levels.finalBoolean << '\n'
                  << "\nRESULT op=mul implementation=PULSAR_CPU_Q33_scale43"
                  << " circuit_s=" << timing.CircuitSeconds()
                  << " final_refresh_s=" << timing.finalB2B
                  << " mean_s=" << meanOnline
                  << " batch=" << kAggregateBatch
                  << " amortized_ms="
                  << (1000.0 * meanOnline / kAggregateBatch)
                  << " failures=0 max_error=" << maxError
                  << " worst_precision_bits=" << precision
                  << " level=" << levels.finalBoolean
                  << " next_round=READY status=PASS\n"
                  << "OVERALL status=PASS\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 2;
    }
}
