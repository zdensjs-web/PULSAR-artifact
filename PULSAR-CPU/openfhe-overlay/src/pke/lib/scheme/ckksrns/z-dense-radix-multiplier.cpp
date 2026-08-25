#include "scheme/ckksrns/z-dense-radix-multiplier.h"

#include "encoding/z-encoding.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <set>
#include <sstream>

namespace lbcrypto {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

int32_t NormalizeRotation(int64_t rotation, uint32_t slots) {
    rotation %= static_cast<int64_t>(slots);
    if (rotation > static_cast<int64_t>(slots / 2)) {
        rotation -= slots;
    }
    if (rotation <= -static_cast<int64_t>(slots / 2)) {
        rotation += slots;
    }
    return static_cast<int32_t>(rotation);
}

}  // namespace

double DenseRadixMultiplierTiming::CircuitSeconds() const {
    return inputPack + convolution + lazyCarry[0] + lazyCarry[1] +
           lazyCarry[2] + sharedFBT + prefix + correction + outputPack;
}

double DenseRadixMultiplierTiming::OnlineSeconds() const {
    return CircuitSeconds() + finalB2B;
}

DenseRadixMultiplierZImpl::DenseRadixMultiplierZImpl(
    uint32_t requestedWordBits,
    LeveledZ leveled,
    FHEZ functional)
    : z(std::move(leveled)), fhe(std::move(functional)),
      wordBits(requestedWordBits) {
    if (wordBits < 8 || wordBits > 256 ||
        (wordBits & (wordBits - 1)) != 0 || wordBits % 4 != 0) {
        OPENFHE_THROW(
            "dense radix multiplier supports 8,16,32,64,128,256 bits");
    }
    booleanBatch = kSlots / wordBits;
    radixDigits = wordBits / 4;
    radixSlice = 2 * radixDigits;
    radixBatch = kSlots / radixSlice;
    uint32_t maximumDigit = radixDigits * 15 * 15;
    while (maximumDigit >= 31) {
        maximumDigit = maximumDigit / 16 + 15;
        ++lazyCarryPasses;
    }
    for (uint32_t distance = 1; distance < radixDigits; distance <<= 1) {
        ++prefixLayers;
    }
    postFBTTailDepth = prefixLayers + 3 + 1;
    rawBoundaryLevel = kSharedFBTOutputLevel + postFBTTailDepth;
    if (radixBatch != 2 * booleanBatch || radixSlice * radixBatch != kSlots) {
        OPENFHE_THROW("word width does not induce a full-slot radix layout");
    }
}

double DenseRadixMultiplierZImpl::Seconds(
    std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double>(duration).count();
}

void DenseRadixMultiplierZImpl::Observe(
    const std::string& label,
    ConstCiphertext<DCRTPoly> ciphertext) const {
    if (stageObserver) {
        stageObserver(label, ciphertext);
    }
}

void DenseRadixMultiplierZImpl::RequirePair(CiphertextGroup pair,
                                             const char* label) {
    if (pair.size() != 2 || !pair[0] || !pair[1]) {
        OPENFHE_THROW(std::string(label) + " must contain two ciphertexts");
    }
    if (pair[0]->GetLevel() != pair[1]->GetLevel()) {
        OPENFHE_THROW(std::string(label) + " has inconsistent levels");
    }
    if (!pair[0]->GetScalingFactorBFP().almostEqual(
            pair[1]->GetScalingFactorBFP())) {
        OPENFHE_THROW(std::string(label) + " has inconsistent scales");
    }
}

std::vector<int32_t> DenseRadixMultiplierZImpl::RotationIndices(
    uint32_t wordBits) {
    DenseRadixMultiplierZImpl dimensions(
        wordBits, LeveledZ{}, FHEZ{});
    std::set<int32_t> unique;
    const auto addPlan = [&](uint32_t logicalLength, uint32_t stride) {
        const uint32_t baby = static_cast<uint32_t>(
            std::ceil(std::sqrt(static_cast<double>(logicalLength))));
        const uint32_t giant = (logicalLength + baby - 1) / baby;
        for (uint32_t index = 1; index < baby; ++index) {
            unique.insert(NormalizeRotation(
                static_cast<int64_t>(index) * stride, kSlots));
        }
        for (uint32_t index = 1; index < giant; ++index) {
            unique.insert(NormalizeRotation(
                static_cast<int64_t>(index) * baby * stride, kSlots));
        }
    };
    addPlan(dimensions.wordBits, dimensions.booleanBatch);
    addPlan(dimensions.radixSlice, dimensions.radixBatch);
    for (uint32_t distance = 1; distance < dimensions.radixDigits;
         distance <<= 1) {
        unique.insert(-static_cast<int32_t>(
            distance * dimensions.radixBatch));
    }
    unique.insert(-static_cast<int32_t>(dimensions.radixBatch));
    unique.erase(0);
    return std::vector<int32_t>(unique.begin(), unique.end());
}

void DenseRadixMultiplierZImpl::DecodeLazyCarryReference(
    std::vector<int64_t>& digits) {
    if (digits.empty()) {
        OPENFHE_THROW("LazyCarry reference requires at least one radix digit");
    }
    std::vector<int64_t> output(digits.size());
    for (size_t digit = 0; digit < digits.size(); ++digit) {
        int64_t value = digits[digit] & 15;
        if (digit != 0) {
            value += digits[digit - 1] >> 4;
        }
        output[digit] = value;
    }
    digits = std::move(output);
}

Plaintext DenseRadixMultiplierZImpl::EncodeDiagonal(
    BigCVector diagonal,
    uint32_t diagonalRotation,
    uint32_t stride,
    ConstCiphertext<DCRTPoly> input) {
    if (!input || diagonal.size() != kSlots) {
        OPENFHE_THROW("invalid fixed-level linear-transform diagonal");
    }
    const auto elementParams = input->GetElements()[0].GetParams();
    const auto& towers = elementParams->GetParams();
    if (towers.size() < 2) {
        OPENFHE_THROW("fixed-level linear transform has no rescale prime");
    }
    const auto cryptoParams =
        std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(
            input->GetCryptoParameters());
    const auto qLast = BigFixedPoint(
        towers.back()->GetModulus(), 0, false).scaleTo(128);
    const auto targetScale =
        cryptoParams->GetScalingFactorBFP(input->GetLevel() + 1);
    const auto plaintextScale =
        targetScale * qLast / input->GetScalingFactorBFP();
    auto shifted = lbcrypto::Rotate(
        diagonal,
        -static_cast<int32_t>(diagonalRotation * stride));
    return ZEncodingImpl::encodeC(
        CSlots(ZEncodingParams(CMode, shifted.size() * 2), shifted),
        elementParams, plaintextScale);
}

DenseRadixMultiplierZImpl::LinearPlan
DenseRadixMultiplierZImpl::BuildInputPlan(
    ConstCiphertext<DCRTPoly> input,
    uint32_t group) const {
    if (group >= 2) {
        OPENFHE_THROW("Boolean input group must be zero or one");
    }
    const uint32_t logicalLength = wordBits;
    const uint32_t baby = static_cast<uint32_t>(
        std::ceil(std::sqrt(static_cast<double>(logicalLength))));
    LinearPlan result(logicalLength);
    // Generate one diagonal at a time. Lattigo's encoded transform retains
    // only the encoded plaintexts, never all clear diagonals beside them.
    for (uint32_t logical = 0; logical < logicalLength; ++logical) {
        BigCVector diagonal(kSlots, BigComplex(BigFixedPoint::zero()));
        bool nonzero = false;
        for (uint32_t digit = 0; digit < radixDigits; ++digit) {
            for (uint32_t residue = 0; residue < 4; ++residue) {
                const auto weight = BigComplex(BigFixedPoint::fromDouble(
                    static_cast<double>(uint32_t{1} << residue)));
                for (uint32_t word = 0; word < booleanBatch; ++word) {
                    const uint32_t from =
                        (4 * digit + residue) * booleanBatch + word;
                    const uint32_t to = digit * radixBatch +
                                        group * booleanBatch + word;
                    const uint32_t delta = (kSlots + from - to) & (kSlots - 1);
                    if (delta == logical * booleanBatch) {
                        diagonal[to] += weight;
                        nonzero = true;
                    }
                }
            }
        }
        if (nonzero) {
            result[logical] = EncodeDiagonal(
                std::move(diagonal), (logical / baby) * baby,
                booleanBatch, input);
        }
    }
    return result;
}

DenseRadixMultiplierZImpl::LinearPlan
DenseRadixMultiplierZImpl::BuildDFTPlan(
    ConstCiphertext<DCRTPoly> input,
    bool inverseMasked) const {
    const uint32_t baby = static_cast<uint32_t>(
        std::ceil(std::sqrt(static_cast<double>(radixSlice))));
    LinearPlan result(radixSlice);
    const double scale = inverseMasked ? 1.0 :
                         1.0 / std::sqrt(static_cast<double>(radixSlice));
    const double sign = inverseMasked ? 1.0 : -1.0;
    for (uint32_t offset = 0; offset < radixSlice; ++offset) {
        BigCVector diagonal(kSlots, BigComplex(BigFixedPoint::zero()));
        bool nonzero = false;
        for (uint32_t row = 0; row < radixSlice; ++row) {
            if (inverseMasked && row >= radixDigits) {
                continue;
            }
            const uint32_t column = (row + offset) % radixSlice;
            const double phase = sign * 2.0 * kPi *
                static_cast<double>(row) * static_cast<double>(column) /
                static_cast<double>(radixSlice);
            const std::complex<double> coefficient =
                scale * std::exp(std::complex<double>(0.0, phase));
            const BigComplex value(
                BigFixedPoint::fromDouble(coefficient.real()),
                BigFixedPoint::fromDouble(coefficient.imag()));
            for (uint32_t word = 0; word < radixBatch; ++word) {
                diagonal[row * radixBatch + word] = value;
                nonzero = true;
            }
        }
        if (nonzero) {
            result[offset] = EncodeDiagonal(
                std::move(diagonal), offset % baby, radixBatch, input);
        }
    }
    return result;
}

DenseRadixMultiplierZImpl::LinearPlan
DenseRadixMultiplierZImpl::BuildOutputPlan(ConstCiphertext<DCRTPoly> input,
                                           uint32_t group,
                                           uint32_t residue) const {
    if (group >= 2 || residue >= 4) {
        OPENFHE_THROW("invalid Boolean output group or residue");
    }
    const uint32_t logicalLength = wordBits;
    const uint32_t baby = static_cast<uint32_t>(
        std::ceil(std::sqrt(static_cast<double>(logicalLength))));
    LinearPlan result(logicalLength);
    const BigComplex one(BigFixedPoint::one());
    for (uint32_t logical = 0; logical < logicalLength; ++logical) {
        BigCVector diagonal(kSlots, BigComplex(BigFixedPoint::zero()));
        bool nonzero = false;
        for (uint32_t digit = 0; digit < radixDigits; ++digit) {
            for (uint32_t word = 0; word < booleanBatch; ++word) {
                const uint32_t from = digit * radixBatch +
                                      group * booleanBatch + word;
                const uint32_t to =
                    (4 * digit + residue) * booleanBatch + word;
                const uint32_t delta = (kSlots + from - to) & (kSlots - 1);
                if (delta == logical * booleanBatch) {
                    diagonal[to] += one;
                    nonzero = true;
                }
            }
        }
        if (nonzero) {
            result[logical] = EncodeDiagonal(
                std::move(diagonal), (logical / baby) * baby,
                booleanBatch, input);
        }
    }
    return result;
}

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::EvalLinearPlan(
    ConstCiphertext<DCRTPoly> input,
    LinearPlan& plan,
    uint32_t logicalLength,
    uint32_t stride) {
    auto result = EvalLinearPlanRaw(input, plan, logicalLength, stride);
    z->ModReduceInPlace(result);
    result->SetZEncodingParams(input->GetZEncodingParams());
    return result;
}

Ciphertext<DCRTPoly>
DenseRadixMultiplierZImpl::EvalLinearPlanFromRotations(
    ConstCiphertext<DCRTPoly> input,
    LinearPlan& plan,
    uint32_t logicalLength,
    uint32_t stride,
    const std::vector<Ciphertext<DCRTPoly>>& giantRotations) {
    if (!input || plan.size() != logicalLength) {
        OPENFHE_THROW("invalid dense linear-transform input or plan");
    }
    const uint32_t baby = static_cast<uint32_t>(
        std::ceil(std::sqrt(static_cast<double>(logicalLength))));
    const uint32_t giant = (logicalLength + baby - 1) / baby;
    if (giantRotations.size() != giant) {
        OPENFHE_THROW("linear-transform giant-step table has the wrong size");
    }

    Ciphertext<DCRTPoly> result;
    for (uint32_t babyIndex = 0; babyIndex < baby; ++babyIndex) {
        Ciphertext<DCRTPoly> inner;
        for (uint32_t outer = 0; outer < giant; ++outer) {
            const uint32_t logical = outer * baby + babyIndex;
            if (logical >= logicalLength || !plan[logical]) {
                continue;
            }
            auto term = z->EvalMult(giantRotations[outer], plan[logical]);
            if (!inner) {
                inner = std::move(term);
            }
            else {
                z->EvalAddInPlace(inner, term);
            }
        }
        if (!inner) {
            continue;
        }
        if (babyIndex != 0) {
            inner = Rotate(
                inner, static_cast<int32_t>(babyIndex * stride));
        }
        if (!result) {
            result = std::move(inner);
        }
        else {
            z->EvalAddInPlace(result, inner);
        }
    }
    if (!result) {
        OPENFHE_THROW("dense linear transform produced no terms");
    }
    result->SetZEncodingParams(input->GetZEncodingParams());
    return result;
}

Ciphertext<DCRTPoly>
DenseRadixMultiplierZImpl::EvalBridgePlanFromRotations(
    ConstCiphertext<DCRTPoly> input,
    LinearPlan& plan,
    const std::vector<Ciphertext<DCRTPoly>>& babyRotations) {
    const uint32_t logicalLength = wordBits;
    const uint32_t stride = booleanBatch;
    const uint32_t baby = static_cast<uint32_t>(
        std::ceil(std::sqrt(static_cast<double>(logicalLength))));
    const uint32_t giant = (logicalLength + baby - 1) / baby;
    if (!input || plan.size() != logicalLength ||
        babyRotations.size() != baby) {
        OPENFHE_THROW("invalid dense bridge plan or baby-step table");
    }

    Ciphertext<DCRTPoly> result;
    for (uint32_t outer = 0; outer < giant; ++outer) {
        Ciphertext<DCRTPoly> inner;
        for (uint32_t babyIndex = 0; babyIndex < baby; ++babyIndex) {
            const uint32_t logical = outer * baby + babyIndex;
            if (logical >= logicalLength) {
                break;
            }
            if (!plan[logical]) {
                continue;
            }
            auto term = z->EvalMult(
                babyRotations[babyIndex], plan[logical]);
            if (!inner) {
                inner = std::move(term);
            }
            else {
                z->EvalAddInPlace(inner, term);
            }
        }
        if (!inner) {
            continue;
        }
        if (outer != 0) {
            inner = Rotate(
                inner,
                static_cast<int32_t>(outer * baby * stride));
        }
        if (!result) {
            result = std::move(inner);
        }
        else {
            z->EvalAddInPlace(result, inner);
        }
    }
    if (!result) {
        OPENFHE_THROW("dense bridge transform produced no terms");
    }
    result->SetZEncodingParams(input->GetZEncodingParams());
    return result;
}

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::EvalBridgePlanRaw(
    ConstCiphertext<DCRTPoly> input,
    LinearPlan& plan) {
    const uint32_t baby = static_cast<uint32_t>(
        std::ceil(std::sqrt(static_cast<double>(wordBits))));
    const uint32_t stride = booleanBatch;
    if (!input || plan.size() != wordBits) {
        OPENFHE_THROW("invalid dense bridge input or plan");
    }
    const auto cc = input->GetCryptoContext();
    const auto digits = cc->EvalFastRotationPrecompute(input);
    std::vector<Ciphertext<DCRTPoly>> babyRotations(baby);
    babyRotations[0] = input->Clone();
    for (uint32_t index = 1; index < baby; ++index) {
        babyRotations[index] = cc->EvalFastRotation(
            input,
            NormalizeRotation(
                static_cast<int64_t>(index) * stride, kSlots),
            digits);
        babyRotations[index]->SetScalingFactorBFP(
            input->GetScalingFactorBFP());
        babyRotations[index]->SetZEncodingParams(
            input->GetZEncodingParams());
    }
    return EvalBridgePlanFromRotations(input, plan, babyRotations);
}

std::vector<Ciphertext<DCRTPoly>>
DenseRadixMultiplierZImpl::EvalBridgePlansRaw(
    ConstCiphertext<DCRTPoly> input,
    const std::vector<LinearPlan*>& plans) {
    const uint32_t baby = static_cast<uint32_t>(
        std::ceil(std::sqrt(static_cast<double>(wordBits))));
    const uint32_t stride = booleanBatch;
    if (!input || plans.empty()) {
        OPENFHE_THROW("invalid multi-output dense bridge input");
    }
    const auto cc = input->GetCryptoContext();
    const auto digits = cc->EvalFastRotationPrecompute(input);
    std::vector<Ciphertext<DCRTPoly>> babyRotations(baby);
    babyRotations[0] = input->Clone();
    for (uint32_t index = 1; index < baby; ++index) {
        babyRotations[index] = cc->EvalFastRotation(
            input,
            NormalizeRotation(
                static_cast<int64_t>(index) * stride, kSlots),
            digits);
        babyRotations[index]->SetScalingFactorBFP(
            input->GetScalingFactorBFP());
        babyRotations[index]->SetZEncodingParams(
            input->GetZEncodingParams());
    }

    std::vector<Ciphertext<DCRTPoly>> outputs;
    outputs.reserve(plans.size());
    for (auto* plan : plans) {
        if (plan == nullptr) {
            OPENFHE_THROW("nil plan in multi-output dense bridge");
        }
        outputs.push_back(EvalBridgePlanFromRotations(
            input, *plan, babyRotations));
    }
    return outputs;
}

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::EvalLinearPlanRaw(
    ConstCiphertext<DCRTPoly> input,
    LinearPlan& plan,
    uint32_t logicalLength,
    uint32_t stride) {
    if (!input || plan.size() != logicalLength) {
        OPENFHE_THROW("invalid dense linear-transform input or plan");
    }
    const uint32_t baby = static_cast<uint32_t>(
        std::ceil(std::sqrt(static_cast<double>(logicalLength))));
    const auto cc = input->GetCryptoContext();
    const uint32_t giant = (logicalLength + baby - 1) / baby;

    // Exact counterpart of Lattigo RotateHoistedNew: decompose the input
    // once, then derive every giant-step automorphism from that decomposition.
    const auto digits = cc->EvalFastRotationPrecompute(input);
    std::vector<Ciphertext<DCRTPoly>> giantRotations(giant);
    giantRotations[0] = input->Clone();
    for (uint32_t index = 1; index < giant; ++index) {
        giantRotations[index] = cc->EvalFastRotation(
            input,
            NormalizeRotation(
                static_cast<int64_t>(index) * baby * stride, kSlots),
            digits);
        giantRotations[index]->SetScalingFactorBFP(
            input->GetScalingFactorBFP());
        giantRotations[index]->SetZEncodingParams(
            input->GetZEncodingParams());
    }
    return EvalLinearPlanFromRotations(
        input, plan, logicalLength, stride, giantRotations);
}

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::Rotate(
    ConstCiphertext<DCRTPoly> input, int32_t offset) {
    auto rotated = input->GetCryptoContext()->EvalRotate(
        input, NormalizeRotation(offset, kSlots));
    rotated->SetScalingFactorBFP(input->GetScalingFactorBFP());
    rotated->SetZEncodingParams(input->GetZEncodingParams());
    return rotated;
}

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::Align(
    ConstCiphertext<DCRTPoly> input, uint32_t level) {
    return z->AdjustCiphertextToLevel(input, level);
}

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::AddAdjusted(
    ConstCiphertext<DCRTPoly> left, ConstCiphertext<DCRTPoly> right) {
    return z->EvalAddWithAdjust(left, right);
}

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::MultiplyReduce(
    ConstCiphertext<DCRTPoly> left, ConstCiphertext<DCRTPoly> right) {
    auto product = z->EvalMultWithAdjust(left, right);
    z->ModReduceInPlace(product);
    product->SetZEncodingParams(left->GetZEncodingParams());
    return product;
}

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::PackBooleanGroups(
    CiphertextGroup input) {
    RequirePair(input, "Boolean/radix pack input");
    if (inputPlans.size() != 2) {
        OPENFHE_THROW("Boolean/radix input plans were not prepared");
    }
    // Lattigo evaluates both group transforms first, adds at product scale,
    // and performs exactly one rescale for the combined radix ciphertext.
    auto first = EvalBridgePlanRaw(input[0], inputPlans.at(0));
    auto second = EvalBridgePlanRaw(input[1], inputPlans.at(1));
    z->EvalAddInPlace(first, second);
    z->ModReduceInPlace(first);
    first->SetZEncodingParams(input[0]->GetZEncodingParams());
    return first;
}

void DenseRadixMultiplierZImpl::PrepareInputPlans(CiphertextGroup input) {
    RequirePair(input, "Boolean/radix input-plan sample");
    inputPlans.clear();
    for (uint32_t group = 0; group < 2; ++group) {
        const auto setupStart = std::chrono::steady_clock::now();
        inputPlans.emplace(group, BuildInputPlan(input[group], group));
        timing.setupExcluded +=
            Seconds(std::chrono::steady_clock::now() - setupStart);
    }
}

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::ForwardDFT(
    ConstCiphertext<DCRTPoly> input) {
    if (forwardPlan.empty()) {
        const auto setupStart = std::chrono::steady_clock::now();
        forwardPlan = BuildDFTPlan(input, false);
        timing.setupExcluded +=
            Seconds(std::chrono::steady_clock::now() - setupStart);
    }
    return EvalLinearPlan(input, forwardPlan, radixSlice, radixBatch);
}

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::InverseDFTMasked(
    ConstCiphertext<DCRTPoly> input) {
    if (inversePlan.empty()) {
        const auto setupStart = std::chrono::steady_clock::now();
        inversePlan = BuildDFTPlan(input, true);
        timing.setupExcluded +=
            Seconds(std::chrono::steady_clock::now() - setupStart);
    }
    return EvalLinearPlan(input, inversePlan, radixSlice, radixBatch);
}

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::PrefixCarry(
    ConstCiphertext<DCRTPoly> propagate,
    ConstCiphertext<DCRTPoly> generate) {
    auto currentP = propagate->Clone();
    auto currentG = generate->Clone();
    for (uint32_t distance = 1; distance < radixDigits; distance <<= 1) {
        auto rotatedP = Rotate(
            currentP, -static_cast<int32_t>(distance * radixBatch));
        auto rotatedG = Rotate(
            currentG, -static_cast<int32_t>(distance * radixBatch));
        auto nextP = MultiplyReduce(currentP, rotatedP);
        auto propagatedG = MultiplyReduce(currentP, rotatedG);
        auto nextG = AddAdjusted(currentG, propagatedG);
        currentP = std::move(nextP);
        currentG = std::move(nextG);
    }
    return Rotate(currentG, -static_cast<int32_t>(radixBatch));
}

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::LinearBit(
    ConstCiphertext<DCRTPoly> bit,
    ConstCiphertext<DCRTPoly> carryIn,
    ConstCiphertext<DCRTPoly> carryOut) {
    const uint32_t target = std::max(
        bit->GetLevel(), std::max(carryIn->GetLevel(), carryOut->GetLevel()));
    auto b = Align(bit, target);
    auto c = Align(carryIn, target);
    auto o = Align(carryOut, target);
    auto doubled = z->EvalAdd(o, o);
    return z->EvalSub(z->EvalAdd(b, c), doubled);
}

std::vector<Ciphertext<DCRTPoly>>
DenseRadixMultiplierZImpl::CorrectNibbles(
    CiphertextGroup sixOutputs,
    ConstCiphertext<DCRTPoly> carry) {
    if (sixOutputs.size() != 6) {
        OPENFHE_THROW("shared radix FBT must return b0,b1,b2,b3,G,P");
    }
    const auto& b0 = sixOutputs[0];
    const auto& b1 = sixOutputs[1];
    const auto& b2 = sixOutputs[2];
    const auto& b3 = sixOutputs[3];

    auto h1 = MultiplyReduce(carry, b0);
    auto b12 = MultiplyReduce(b1, b2);
    auto h2 = MultiplyReduce(h1, b1);
    auto h3 = MultiplyReduce(h1, b12);
    auto b23 = MultiplyReduce(b2, b3);
    auto h4 = MultiplyReduce(h2, b23);

    std::vector<Ciphertext<DCRTPoly>> output{
        LinearBit(b0, carry, h1),
        LinearBit(b1, h1, h2),
        LinearBit(b2, h2, h3),
        LinearBit(b3, h3, h4),
    };
    const uint32_t deepest = h4->GetLevel();
    for (auto& plane : output) {
        plane = Align(plane, deepest);
    }
    return output;
}

CiphertextGroup DenseRadixMultiplierZImpl::UnpackBooleanGroups(
    const std::vector<Ciphertext<DCRTPoly>>& planes) {
    if (planes.size() != 4) {
        OPENFHE_THROW("radix output requires four Boolean planes");
    }
    std::vector<Ciphertext<DCRTPoly>> result(2);
    for (uint32_t residue = 0; residue < 4; ++residue) {
        for (uint32_t group = 0; group < 2; ++group) {
            const uint32_t key = group * 4 + residue;
            const auto setupStart = std::chrono::steady_clock::now();
            outputPlans.emplace(
                key, BuildOutputPlan(planes[residue], group, residue));
            timing.setupExcluded +=
                Seconds(std::chrono::steady_clock::now() - setupStart);
        }
        auto terms = EvalBridgePlansRaw(
            planes[residue],
            {&outputPlans.at(residue), &outputPlans.at(4 + residue)});
        for (uint32_t group = 0; group < 2; ++group) {
            if (!result[group]) {
                result[group] = std::move(terms[group]);
            }
            else {
                z->EvalAddInPlace(result[group], terms[group]);
            }
        }
        outputPlans.erase(residue);
        outputPlans.erase(4 + residue);
    }
    for (auto& group : result) {
        z->ModReduceInPlace(group);
        group->SetZEncodingParams(
            ZEncodingParams(BModeFull, wordBits, radixBatch));
    }
    return CiphertextGroup(std::move(result));
}

CiphertextGroup DenseRadixMultiplierZImpl::EvalMultiply(
    CiphertextGroup left,
    CiphertextGroup right,
    uint32_t maximumB2BInputLevel) {
    RequirePair(left, "left multiplier input");
    RequirePair(right, "right multiplier input");
    if (left[0]->GetLevel() != right[0]->GetLevel() ||
        !left[0]->GetScalingFactorBFP().almostEqual(
            right[0]->GetScalingFactorBFP())) {
        OPENFHE_THROW("multiplier operands have different level/scale contracts");
    }
    timing = DenseRadixMultiplierTiming{};
    levels = DenseRadixMultiplierLevels{};
    levels.stableInput = left[0]->GetLevel();
    // Preserve the steady-state B2B boundary before the local operands are
    // aligned to the multiplier's level-16 circuit entry. The final B2B must
    // reproduce this boundary, not the scale of the aligned working copies.
    const auto stableInputScale = left[0]->GetScalingFactorBFP();

    if (rawBoundaryLevel > maximumB2BInputLevel) {
        OPENFHE_THROW("multiplier tail exceeds the native B2B entry boundary");
    }
    if (left[0]->GetLevel() > kCircuitInputLevel ||
        right[0]->GetLevel() > kCircuitInputLevel) {
        OPENFHE_THROW("B2B output is already below the Lattigo circuit input level");
    }
    left = CiphertextGroup({Align(left[0], kCircuitInputLevel),
                            Align(left[1], kCircuitInputLevel)});
    right = CiphertextGroup({Align(right[0], kCircuitInputLevel),
                             Align(right[1], kCircuitInputLevel)});

    PrepareInputPlans(left);
    const auto packStart = std::chrono::steady_clock::now();
    auto packedLeft = PackBooleanGroups(left);
    auto packedRight = PackBooleanGroups(right);
    inputPlans.clear();
    timing.inputPack = Seconds(std::chrono::steady_clock::now() - packStart);
    levels.packedRadix = packedLeft->GetLevel();
    if (levels.packedRadix != 17) {
        OPENFHE_THROW("Boolean/radix packing did not reproduce Lattigo level 10");
    }
    Observe("packed A", packedLeft);
    Observe("packed B", packedRight);

    double setupBefore = timing.setupExcluded;
    const auto convolutionStart = std::chrono::steady_clock::now();
    auto dftLeft = ForwardDFT(packedLeft);
    auto dftRight = ForwardDFT(packedRight);
    packedLeft.reset();
    packedRight.reset();
    forwardPlan.clear();
    auto frequencyProduct = MultiplyReduce(dftLeft, dftRight);
    dftLeft.reset();
    dftRight.reset();
    auto lazyDigits = InverseDFTMasked(frequencyProduct);
    frequencyProduct.reset();
    inversePlan.clear();
    timing.convolution =
        Seconds(std::chrono::steady_clock::now() - convolutionStart) -
        (timing.setupExcluded - setupBefore);
    levels.convolution = lazyDigits->GetLevel();
    if (levels.convolution != 20) {
        OPENFHE_THROW("DFT convolution did not reproduce Lattigo level 7");
    }
    Observe("convolution", lazyDigits);

    for (uint32_t pass = 0; pass < lazyCarryPasses; ++pass) {
        const auto start = std::chrono::steady_clock::now();
        lazyDigits = fhe->EvalRadix16LazyCarryFull(lazyDigits,
                                                   radixDigits,
                                                   radixBatch);
        timing.lazyCarry[pass] =
            Seconds(std::chrono::steady_clock::now() - start);
        levels.lazyCarry[pass] = lazyDigits->GetLevel();
        if (levels.lazyCarry[pass] != 21 + pass) {
            std::ostringstream message;
            message << "LazyCarry #" << (pass + 1) << " ended at level "
                    << levels.lazyCarry[pass] << ", expected " << (21 + pass);
            OPENFHE_THROW(message.str());
        }
        Observe("LazyCarry #" + std::to_string(pass + 1), lazyDigits);
    }

    const auto fbtStart = std::chrono::steady_clock::now();
    auto sixOutputs = fhe->EvalRadix31ToBooleanSixFullFirst(lazyDigits,
                                                            radixBatch);
    timing.sharedFBT = Seconds(std::chrono::steady_clock::now() - fbtStart);
    if (sixOutputs.size() != 6) {
        OPENFHE_THROW("shared radix FBT returned an unexpected output count");
    }
    for (uint32_t output = 1; output < 6; ++output) {
        if (sixOutputs[output]->GetLevel() != sixOutputs[0]->GetLevel() ||
            !sixOutputs[output]->GetScalingFactorBFP().almostEqual(
                sixOutputs[0]->GetScalingFactorBFP())) {
            OPENFHE_THROW("shared radix FBT outputs have inconsistent level/scale");
        }
    }
    lazyDigits.reset();
    levels.sharedFBTRaw = sixOutputs[0]->GetLevel();
    if (maximumB2BInputLevel < postFBTTailDepth) {
        OPENFHE_THROW("final B2B input runway is shorter than the Lattigo tail");
    }
    const uint32_t scheduledFBTLevel = kSharedFBTOutputLevel;
    if (levels.sharedFBTRaw > scheduledFBTLevel) {
        std::ostringstream message;
        message << "shared radix FBT output level " << levels.sharedFBTRaw
                << " already exceeds scheduled Lattigo tail input level "
                << scheduledFBTLevel;
        OPENFHE_THROW(message.str());
    }
    for (uint32_t output = 0; output < 6; ++output) {
        if (sixOutputs[output]->GetLevel() < scheduledFBTLevel) {
            sixOutputs[output] = Align(sixOutputs[output],
                                       scheduledFBTLevel);
        }
    }
    levels.sharedFBT = sixOutputs[0]->GetLevel();
    if (levels.sharedFBT != scheduledFBTLevel) {
        OPENFHE_THROW("shared radix FBT runway scheduling failed");
    }
    static const char* labels[6] = {"FBT b0", "FBT b1", "FBT b2",
                                     "FBT b3", "FBT G", "FBT P"};
    for (uint32_t output = 0; output < 6; ++output) {
        Observe(labels[output], sixOutputs[output]);
    }

    const auto prefixStart = std::chrono::steady_clock::now();
    auto carry = PrefixCarry(sixOutputs[5], sixOutputs[4]);
    timing.prefix = Seconds(std::chrono::steady_clock::now() - prefixStart);
    levels.prefix = carry->GetLevel();
    if (levels.prefix != scheduledFBTLevel + prefixLayers) {
        std::ostringstream message;
        message << prefixLayers << "-layer radix prefix ended at level "
                << levels.prefix << ", expected "
                << (scheduledFBTLevel + prefixLayers);
        OPENFHE_THROW(message.str());
    }
    Observe("radix carry", carry);
    sixOutputs[4].reset();
    sixOutputs[5].reset();

    const auto correctionStart = std::chrono::steady_clock::now();
    auto corrected = CorrectNibbles(sixOutputs, carry);
    sixOutputs = CiphertextGroup{};
    carry.reset();
    timing.correction =
        Seconds(std::chrono::steady_clock::now() - correctionStart);
    levels.correction = corrected[0]->GetLevel();
    if (levels.correction != scheduledFBTLevel + prefixLayers + 3) {
        std::ostringstream message;
        message << "depth-three nibble correction ended at level "
                << levels.correction << ", expected "
                << (scheduledFBTLevel + prefixLayers + 3);
        OPENFHE_THROW(message.str());
    }
    for (uint32_t plane = 0; plane < 4; ++plane) {
        Observe("corrected b" + std::to_string(plane), corrected[plane]);
    }

    setupBefore = timing.setupExcluded;
    const auto outputStart = std::chrono::steady_clock::now();
    auto raw = UnpackBooleanGroups(corrected);
    auto raw0 = raw[0];
    auto raw1 = raw[1];
    timing.outputPack = Seconds(std::chrono::steady_clock::now() - outputStart) -
                        (timing.setupExcluded - setupBefore);
    levels.rawBoolean = raw0->GetLevel();
    Observe("raw product group 0", raw0);
    Observe("raw product group 1", raw1);
    if (raw0->GetLevel() != rawBoundaryLevel ||
        raw1->GetLevel() != rawBoundaryLevel) {
        std::ostringstream message;
        message << "raw multiplier output level " << raw0->GetLevel()
                << " does not equal scheduled final B2B input level "
                << rawBoundaryLevel;
        OPENFHE_THROW(message.str());
    }

    const ZEncodingParams booleanParams(BModeFull, wordBits, radixBatch);
    raw0->SetZEncodingParams(booleanParams);
    raw1->SetZEncodingParams(booleanParams);
    const auto refreshStart = std::chrono::steady_clock::now();
    auto result = fhe->EvalBooleanToBooleanFull(
        CiphertextGroup({raw0, raw1}));
    raw0.reset();
    raw1.reset();
    timing.finalB2B = Seconds(std::chrono::steady_clock::now() - refreshStart);
    RequirePair(result, "final multiplier B2B output");
    result[0]->SetZEncodingParams(booleanParams);
    result[1]->SetZEncodingParams(booleanParams);
    levels.finalBoolean = result[0]->GetLevel();
    const bool levelMatches = levels.finalBoolean == levels.stableInput;
    const bool scale0Matches =
        result[0]->GetScalingFactorBFP().almostEqual(stableInputScale);
    const bool scale1Matches =
        result[1]->GetScalingFactorBFP().almostEqual(stableInputScale);
    if (!levelMatches || !scale0Matches || !scale1Matches) {
        std::ostringstream message;
        message << "final B2B boundary is not reusable: input level="
                << levels.stableInput << " output level="
                << levels.finalBoolean
                << " level_match=" << levelMatches
                << " scale0_match=" << scale0Matches
                << " scale1_match=" << scale1Matches
                << " input_log_scale="
                << std::log2(stableInputScale.convertToDouble())
                << " output0_log_scale="
                << std::log2(
                       result[0]->GetScalingFactorBFP().convertToDouble())
                << " output1_log_scale="
                << std::log2(
                       result[1]->GetScalingFactorBFP().convertToDouble());
        OPENFHE_THROW(message.str());
    }
    Observe("final product group 0", result[0]);
    Observe("final product group 1", result[1]);
    return result;
}

}  // namespace lbcrypto
