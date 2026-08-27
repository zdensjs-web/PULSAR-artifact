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
#include <utility>
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
uint32_t kScaleDivRawBoundaryLevel = 28;
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
    // Full product: log2(2K)=prefix+1, optimized correction=2,
    // quotient extraction=1. This has the same total tail as multiplication.
    kScaleDivRawBoundaryLevel = 18 + (kPrefixLayers + 1) + 2 + 1;
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

void Require(const std::string& label, const AuditResult& result) {
    if (result.failures != 0) {
        std::ostringstream message;
        message << label << " failed: failures=" << result.failures
                << " max_error=" << result.maxError;
        throw std::runtime_error(message.str());
    }
}

std::vector<cpp_int> MakeWords(uint64_t seed) {
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
    result[0] = cpp_int(0);
    result[1] = cpp_int(1);
    result[2] = cpp_int(modulus - 1);
    result[3] = cpp_int(1) << (kWordBits - 1);
    return result;
}

std::vector<cpp_int> PlainVaultShares(const std::vector<cpp_int>& deposits,
                                      const cpp_int& exchangeRate,
                                      const cpp_int& rateScale) {
    std::vector<cpp_int> result(deposits.size());
    const cpp_int mask = (cpp_int(1) << kWordBits) - 1;
    for (size_t word = 0; word < deposits.size(); ++word) {
        const cpp_int productLow = (deposits[word] * exchangeRate) & mask;
        result[word] = productLow / rateScale;
    }
    return result;
}

struct MagicDescriptor {
    cpp_int magic = 0;
    uint32_t shift = 0;
    bool addMarker = false;

    bool PowerOfTwo() const { return magic == 0; }
};

MagicDescriptor GenerateMagic(const cpp_int& divisor) {
    if (divisor <= 0 || divisor >= (cpp_int(1) << kWordBits)) {
        throw std::invalid_argument("public divisor must be in [1,2^word_bits)");
    }
    MagicDescriptor result;
    cpp_int scan = divisor;
    while (scan > 1) {
        scan >>= 1;
        ++result.shift;
    }
    if ((divisor & (divisor - 1)) == cpp_int(0)) {
        return result;
    }
    const cpp_int numerator = cpp_int(1) << (kWordBits + result.shift);
    cpp_int proposed = numerator / divisor;
    const cpp_int remainder = numerator % divisor;
    const cpp_int errorBound = divisor - remainder;
    result.addMarker = errorBound >= (cpp_int(1) << result.shift);
    if (result.addMarker) {
        proposed <<= 1;
        if ((remainder << 1) >= divisor) {
            ++proposed;
        }
    }
    const cpp_int mask = (cpp_int(1) << kWordBits) - 1;
    result.magic = (proposed + 1) & mask;
    if (result.magic == 0) {
        throw std::runtime_error("non-power-of-two divisor generated zero magic");
    }
    return result;
}

cpp_int MagicQuotient(const cpp_int& input,
                      const MagicDescriptor& descriptor) {
    if (descriptor.PowerOfTwo()) {
        return input >> descriptor.shift;
    }
    cpp_int q = (input * descriptor.magic) >> kWordBits;
    if (descriptor.addMarker) {
        q += (input - q) >> 1;
    }
    return q >> descriptor.shift;
}

cpp_int ParseUnsigned(const std::string& text, const char* label) {
    if (text.empty()) {
        throw std::invalid_argument(std::string(label) + " is empty");
    }
    size_t offset = 0;
    uint32_t base = 10;
    if (text.size() > 2 && text[0] == '0' &&
        (text[1] == 'x' || text[1] == 'X')) {
        offset = 2;
        base = 16;
    }
    if (offset == text.size()) {
        throw std::invalid_argument(std::string(label) + " has no digits");
    }
    cpp_int result = 0;
    for (; offset < text.size(); ++offset) {
        const char character = text[offset];
        uint32_t digit = 0;
        if (character >= '0' && character <= '9') {
            digit = static_cast<uint32_t>(character - '0');
        }
        else if (character >= 'a' && character <= 'f') {
            digit = 10 + static_cast<uint32_t>(character - 'a');
        }
        else if (character >= 'A' && character <= 'F') {
            digit = 10 + static_cast<uint32_t>(character - 'A');
        }
        else {
            throw std::invalid_argument(std::string(label) + " contains a non-digit");
        }
        if (digit >= base) {
            throw std::invalid_argument(std::string(label) + " digit exceeds its base");
        }
        result *= base;
        result += digit;
    }
    return result;
}

std::vector<uint32_t> RadixDigits(cpp_int value) {
    std::vector<uint32_t> result(kRadixDigits);
    for (uint32_t digit = 0; digit < kRadixDigits; ++digit) {
        result[digit] = static_cast<uint32_t>((value & 15).convert_to<unsigned>());
        value >>= 4;
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
        if (argc > 5) {
            std::cerr << "Usage: " << argv[0]
                      << " [word_bits] [public_exchange_rate]"
                      << " [public_rate_scale] [repeats]\n";
            return 2;
        }
        uint32_t wordBits = 256;
        cpp_int publicExchangeRate = 3;
        cpp_int publicRateScale = 3;
        uint32_t repeats = 1;
        if (argc >= 2) {
            wordBits = static_cast<uint32_t>(std::stoul(argv[1]));
        }
        if (argc >= 3) {
            publicExchangeRate = ParseUnsigned(argv[2], "public exchange rate");
        }
        if (argc >= 4) {
            publicRateScale = ParseUnsigned(argv[3], "public rate scale");
        }
        if (argc == 5) {
            repeats = static_cast<uint32_t>(std::stoul(argv[4]));
        }
        ConfigureWidth(wordBits);
        if (repeats == 0) {
            throw std::invalid_argument("repeats must be positive");
        }
        const cpp_int modulus = cpp_int(1) << kWordBits;
        if (publicExchangeRate < 0 || publicExchangeRate >= modulus) {
            throw std::invalid_argument(
                "public exchange rate must be in [0,2^word_bits)");
        }
        if (publicRateScale <= 0 || publicRateScale >= modulus) {
            throw std::invalid_argument(
                "public rate scale must be in [1,2^word_bits)");
        }
        const auto descriptor = GenerateMagic(publicRateScale);
        const auto exchangeRateDigits = RadixDigits(publicExchangeRate);
        const auto divisorDigits = descriptor.PowerOfTwo() ?
            std::vector<uint32_t>{} : RadixDigits(descriptor.magic);

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

        // Two final Q primes are reserved for native B2B's pre-ModRaise C2R
        // runway for every width.
        const std::vector<uint32_t> levelBudget{3, 2};
        const uint32_t finalB2BInputLevel =
            kMultiplicativeDepth - levelBudget[1];
        if (kScaleDivRawBoundaryLevel > finalB2BInputLevel) {
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
        Stage("Vault scale_mul and scale_div rotation keys");
        const auto missingRotations = MissingRotationIndices(
            cc, keyPair.secretKey,
            DenseRadixMultiplierZImpl::ScaleDivisionRotationIndices(
                wordBits, descriptor.shift, descriptor.addMarker));
        if (!missingRotations.empty()) {
            cc->EvalRotateKeyGen(keyPair.secretKey, missingRotations);
        }
        const size_t totalRotationKeyCount =
            CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(
                keyPair.secretKey->GetKeyTag()).size();
        Stage("releasing unused generic A2B/B2A precomputations");
        fhe->ReleaseUnusedBooleanRadixPrecom(kComplexSlots);
        Stage("all evaluation keys ready");

        Stage("building deterministic plaintext input");
        const auto inputWords = MakeWords(0x5641554C54435055ULL);
        const auto productWords = PlainVaultShares(
            inputWords, publicExchangeRate, publicRateScale);
        const cpp_int wordMask = modulus - 1;
        for (uint32_t word = 0; word < kAggregateBatch; ++word) {
            const cpp_int productLow =
                (inputWords[word] * publicExchangeRate) & wordMask;
            if (MagicQuotient(productLow, descriptor) !=
                productWords[word]) {
                throw std::runtime_error(
                    "Vault plaintext descriptor gate failed at word " +
                    std::to_string(word));
            }
        }
        const auto input0 = DenseGroupBits(inputWords, 0);
        const auto input1 = DenseGroupBits(inputWords, 1);
        const auto expected0 = DenseGroupBits(productWords, 0);
        const auto expected1 = DenseGroupBits(productWords, 1);
        const auto scale = cryptoParams->GetScalingFactorBFP(0);
        Stage("encrypting two dense Boolean input groups");
        auto ctInput0 = pke->Encrypt(EncodeDense(input0, elementParams, scale));
        auto ctInput1 = pke->Encrypt(EncodeDense(input1, elementParams, scale));
        Stage("two dense Boolean input groups ready");
        const auto setupEnd = std::chrono::steady_clock::now();

        std::cout << "[setup] excluded shared B2B input refresh\n";
        const auto inputRefreshStart = std::chrono::steady_clock::now();
        auto stableInput = RefreshPair(fhe, ctInput0, ctInput1);
        const auto inputRefreshEnd = std::chrono::steady_clock::now();
        Require("refreshed input group 0",
                Audit("refreshed input group 0", stableInput[0],
                      keyPair.secretKey, input0));
        Require("refreshed input group 1",
                Audit("refreshed input group 1", stableInput[1],
                      keyPair.secretKey, input1));

        std::cout << "============================================================\n"
                  << "PULSAR Vault over the FHE-SIMD-ALU B2B environment\n"
                  << "semantics=floor(((deposit*exchange_rate) mod 2^"
                  << kWordBits << ")/rate_scale)\n"
                  << "public_exchange_rate=" << publicExchangeRate << '\n'
                  << "public_rate_scale=" << publicRateScale << '\n'
                  << "magic=" << descriptor.magic << '\n'
                  << "shift=" << descriptor.shift << '\n'
                  << "add_marker=" << std::boolalpha
                  << descriptor.addMarker << std::noboolalpha << '\n'
                  << "input=actual_shared_B2B_output\n"
                  << "layout=two_guard_free_Boolean_ciphertexts_for_one_aggregate_batch\n"
                  << "word_bits=" << kWordBits << '\n'
                  << "native_batch=" << kNativeBatch << '\n'
                  << "aggregate_batch=" << kAggregateBatch << '\n'
                  << "operators=EvalScaleMultiply+EvalScaleDivide\n"
                  << "mul_div_boundary=shared_native_B2B\n"
                  << "final_refresh=shared_native_B2B\n"
                  << "new_rotation_keys="
                  << missingRotations.size() << '\n'
                  << "Boolean_radix_bootstrap_keys="
                  << booleanRadixKeyCount << '\n'
                  << "total_automorphism_keys="
                  << totalRotationKeyCount << '\n'
                  << "thread_control=external_OMP_environment\n"
                  << "ring_dimension=" << cc->GetRingDimension() << '\n'
                  << "logN=" << kLogN << '\n'
                  << "log_scale=" << kScalingBits << '\n'
                  << "requested_first_modulus_bits=" << kFirstModulusBits << '\n'
                  << "multiplicative_depth=" << kMultiplicativeDepth << '\n'
                  << "scale_div_circuit_max_level="
                  << kScaleDivRawBoundaryLevel << '\n'
                  << "Q_count=" << elementParams->GetParams().size() << '\n'
                  << "Q_bits=" << PrimeBits(elementParams->GetParams()) << '\n'
                  << "P_count=" << paramsP->GetParams().size() << '\n'
                  << "P_bits=" << PrimeBits(paramsP->GetParams()) << '\n'
                  << "logQP=" << logQP << '\n'
                  << "classic_128_logQP_bound=" << kClassic128LogQPBound << '\n'
                  << "stable_input_level=" << stableInput[0]->GetLevel() << '\n'
                  << "final_B2B_input_level=" << finalB2BInputLevel << '\n'
                  << "setup_seconds_excluded="
                  << std::chrono::duration<double>(setupEnd - setupStart).count() << '\n'
                  << "input_refresh_seconds_excluded="
                  << std::chrono::duration<double>(inputRefreshEnd - inputRefreshStart).count() << '\n'
                  << "repeats=" << repeats << '\n'
                  << "============================================================\n";

        CiphertextGroup output;
        double totalOnline = 0.0;
        double totalScaleMul = 0.0;
        double totalScaleDiv = 0.0;
        for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
            std::cout << "[run] " << (repeat + 1) << '/' << repeats
                      << " PULSAR Vault\n";
            auto product = multiplier->EvalScaleMultiply(
                stableInput, exchangeRateDigits, finalB2BInputLevel);
            const double scaleMulSeconds =
                multiplier->GetTiming().OnlineSeconds();
            double scaleDivSeconds = 0.0;
            if (publicRateScale == 1) {
                output = std::move(product);
            }
            else {
                output = multiplier->EvalScaleDivide(
                    product, divisorDigits, descriptor.shift,
                    descriptor.addMarker, finalB2BInputLevel);
                scaleDivSeconds = multiplier->GetTiming().OnlineSeconds();
            }
            totalScaleMul += scaleMulSeconds;
            totalScaleDiv += scaleDivSeconds;
            totalOnline += scaleMulSeconds + scaleDivSeconds;
        }
        const auto audit0 = Audit("Vault shares group 0", output[0],
                                  keyPair.secretKey, expected0);
        const auto audit1 = Audit("Vault shares group 1", output[1],
                                  keyPair.secretKey, expected1);
        Require("Vault shares group 0", audit0);
        Require("Vault shares group 1", audit1);

        const double meanOnline = totalOnline / repeats;
        const double meanScaleMul = totalScaleMul / repeats;
        const double meanScaleDiv = totalScaleDiv / repeats;
        const double maxError = std::max(audit0.maxError, audit1.maxError);
        const double precision = maxError == 0.0 ?
            std::numeric_limits<double>::infinity() : -std::log2(maxError);
        std::cout << std::fixed << std::setprecision(9)
                  << "\nOnline timing\n"
                  << "  public scale_mul      : " << meanScaleMul << "s\n"
                  << "  public scale_div      : " << meanScaleDiv << "s\n"
                  << "  total                 : " << meanOnline << "s\n"
                  << "\nRESULT app=vault implementation=PULSAR_CPU_Q33_scale43"
                  << " mean_s=" << meanOnline
                  << " batch=" << kAggregateBatch
                  << " amortized_ms="
                  << (1000.0 * meanOnline / kAggregateBatch)
                  << " failures=0 max_error=" << maxError
                  << " worst_precision_bits=" << precision
                  << " level=" << output[0]->GetLevel()
                  << " next_round=READY status=PASS\n"
                  << "OVERALL status=PASS\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 2;
    }
}
