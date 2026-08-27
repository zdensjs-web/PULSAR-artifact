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
           lazyCarry[2] + lazyCarry[3] + sharedFBT + prefix + correction + outputPack +
           intermediateB2B + addCorrection;
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
    if (wordBits < 8 || wordBits > 2048 ||
        (wordBits & (wordBits - 1)) != 0 || wordBits % 4 != 0) {
        OPENFHE_THROW(
            "dense radix multiplier supports powers of two from 8 to 2048 bits");
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
    for (uint32_t distance = 1; distance < dimensions.radixSlice;
         distance <<= 1) {
        unique.insert(-static_cast<int32_t>(
            distance * dimensions.radixBatch));
    }
    unique.insert(static_cast<int32_t>(
        (dimensions.radixDigits - 1) * dimensions.radixBatch));
    unique.insert(static_cast<int32_t>(kSlots / 2));
    unique.insert(-static_cast<int32_t>(dimensions.radixBatch));
    unique.erase(0);
    return std::vector<int32_t>(unique.begin(), unique.end());
}

std::vector<int32_t>
DenseRadixMultiplierZImpl::ScaleDivisionRotationIndices(
    uint32_t wordBits,
    uint32_t shift,
    bool addMarker) {
    DenseRadixMultiplierZImpl dimensions(wordBits, LeveledZ{}, FHEZ{});
    auto base = RotationIndices(wordBits);
    std::set<int32_t> unique(base.begin(), base.end());
    if (addMarker) {
        for (uint32_t distance = 1; distance < wordBits; distance <<= 1) {
            unique.insert(-static_cast<int32_t>(
                distance * dimensions.booleanBatch));
        }
        const uint32_t correctionShift = shift + 1;
        unique.insert(NormalizeRotation(
            static_cast<int64_t>(correctionShift) *
                dimensions.booleanBatch,
            kSlots));
        if (correctionShift > 1) {
            unique.insert(NormalizeRotation(
                static_cast<int64_t>(correctionShift - 1) *
                    dimensions.booleanBatch,
                kSlots));
        }
    }
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
DenseRadixMultiplierZImpl::BuildPackedFullInverseDFTPlan(
    ConstCiphertext<DCRTPoly> input) const {
    const uint32_t baby = static_cast<uint32_t>(
        std::ceil(std::sqrt(static_cast<double>(radixSlice))));
    LinearPlan result(radixSlice);
    for (uint32_t offset = 0; offset < radixSlice; ++offset) {
        BigCVector diagonal(kSlots, BigComplex(BigFixedPoint::zero()));
        for (uint32_t row = 0; row < radixDigits; ++row) {
            const uint32_t lowColumn = (row + offset) % radixSlice;
            const double lowPhase = 2.0 * kPi *
                static_cast<double>(row) *
                static_cast<double>(lowColumn) /
                static_cast<double>(radixSlice);
            const double highPhase = 2.0 * kPi *
                static_cast<double>(row + radixDigits) *
                static_cast<double>(lowColumn) /
                static_cast<double>(radixSlice);
            const std::complex<double> coefficient =
                std::exp(std::complex<double>(0.0, lowPhase)) +
                std::complex<double>(0.0, 1.0) *
                std::exp(std::complex<double>(0.0, highPhase));
            const BigComplex value(
                BigFixedPoint::fromDouble(coefficient.real()),
                BigFixedPoint::fromDouble(coefficient.imag()));
            for (uint32_t word = 0; word < radixBatch; ++word) {
                diagonal[row * radixBatch + word] = value;
            }
        }
        result[offset] = EncodeDiagonal(
            std::move(diagonal), offset % baby, radixBatch, input);
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

DenseRadixMultiplierZImpl::LinearPlan
DenseRadixMultiplierZImpl::BuildQuotientOutputPlan(
    ConstCiphertext<DCRTPoly> input,
    uint32_t group,
    uint32_t residue,
    uint32_t sourceStart) const {
    if (group >= 2 || residue >= 4 || sourceStart >= 2 * wordBits) {
        OPENFHE_THROW("invalid quotient output plan");
    }
    const uint32_t logicalLength = wordBits;
    const uint32_t baby = static_cast<uint32_t>(
        std::ceil(std::sqrt(static_cast<double>(logicalLength))));
    LinearPlan result(logicalLength);
    const BigComplex one(BigFixedPoint::one());
    for (uint32_t logical = 0; logical < logicalLength; ++logical) {
        BigCVector diagonal(kSlots, BigComplex(BigFixedPoint::zero()));
        bool nonzero = false;
        for (uint32_t outputBit = 0; outputBit < wordBits; ++outputBit) {
            const uint32_t sourceBit = sourceStart + outputBit;
            if (sourceBit >= 2 * wordBits || sourceBit % 4 != residue) {
                continue;
            }
            const uint32_t digit = sourceBit / 4;
            for (uint32_t word = 0; word < booleanBatch; ++word) {
                const uint32_t from = digit * radixBatch +
                                      group * booleanBatch + word;
                const uint32_t to = outputBit * booleanBatch + word;
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

DenseRadixMultiplierZImpl::LinearPlan
DenseRadixMultiplierZImpl::BuildDenseShiftPlan(
    ConstCiphertext<DCRTPoly> input,
    uint32_t shift) const {
    if (shift >= wordBits) {
        OPENFHE_THROW("dense right shift is outside the word");
    }
    const uint32_t logicalLength = wordBits;
    const uint32_t baby = static_cast<uint32_t>(
        std::ceil(std::sqrt(static_cast<double>(logicalLength))));
    LinearPlan result(logicalLength);
    const BigComplex one(BigFixedPoint::one());
    for (uint32_t logical = 0; logical < logicalLength; ++logical) {
        BigCVector diagonal(kSlots, BigComplex(BigFixedPoint::zero()));
        bool nonzero = false;
        for (uint32_t outputBit = 0; outputBit + shift < wordBits;
             ++outputBit) {
            for (uint32_t word = 0; word < booleanBatch; ++word) {
                const uint32_t from =
                    (outputBit + shift) * booleanBatch + word;
                const uint32_t to = outputBit * booleanBatch + word;
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
DenseRadixMultiplierZImpl::EvalStreamingBridgePlansRaw(
    ConstCiphertext<DCRTPoly> input,
    uint32_t outputCount,
    const DiagonalFactory& factory) {
    if (!input || outputCount == 0 || !factory) {
        OPENFHE_THROW("invalid streaming bridge transform");
    }
    const uint32_t logicalLength = wordBits;
    const uint32_t stride = booleanBatch;
    const uint32_t baby = static_cast<uint32_t>(
        std::ceil(std::sqrt(static_cast<double>(logicalLength))));
    const uint32_t giant = (logicalLength + baby - 1) / baby;
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

    std::vector<Ciphertext<DCRTPoly>> outputs(outputCount);
    for (uint32_t output = 0; output < outputCount; ++output) {
        Ciphertext<DCRTPoly> result;
        for (uint32_t outer = 0; outer < giant; ++outer) {
            std::vector<Plaintext> encodedDiagonals(baby);
            const auto setupStart = std::chrono::steady_clock::now();
            Ciphertext<DCRTPoly> inner;
#pragma omp parallel for schedule(dynamic) \
    num_threads(OpenFHEParallelControls.GetThreadLimit(baby))
            for (uint32_t babyIndex = 0; babyIndex < baby; ++babyIndex) {
                const uint32_t logical = outer * baby + babyIndex;
                if (logical >= logicalLength) {
                    continue;
                }
                auto diagonal = factory(logical, output);
                if (!diagonal.empty()) {
                    encodedDiagonals[babyIndex] = EncodeDiagonal(
                        std::move(diagonal), outer * baby, stride, input);
                }
            }
            timing.setupExcluded += Seconds(
                std::chrono::steady_clock::now() - setupStart);
            for (uint32_t babyIndex = 0; babyIndex < baby; ++babyIndex) {
                auto& encoded = encodedDiagonals[babyIndex];
                if (!encoded) {
                    continue;
                }
                auto term = z->EvalMult(
                    babyRotations[babyIndex], encoded);
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
            OPENFHE_THROW("streaming bridge transform produced no terms");
        }
        result->SetZEncodingParams(input->GetZEncodingParams());
        outputs[output] = std::move(result);
    }
    return outputs;
}

// Large-word transforms generate encoded diagonals on demand.
Ciphertext<DCRTPoly>
DenseRadixMultiplierZImpl::EvalStreamingLinearPlanRaw(
    ConstCiphertext<DCRTPoly> input,
    uint32_t logicalLength,
    uint32_t stride,
    const DiagonalFactory& factory) {
    if (!input || logicalLength == 0 || !factory) {
        OPENFHE_THROW("invalid streaming linear transform");
    }
    const uint32_t baby = static_cast<uint32_t>(
        std::ceil(std::sqrt(static_cast<double>(logicalLength))));
    const uint32_t giant = (logicalLength + baby - 1) / baby;
    const auto cc = input->GetCryptoContext();
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
    Ciphertext<DCRTPoly> result;
    for (uint32_t babyIndex = 0; babyIndex < baby; ++babyIndex) {
        std::vector<Plaintext> encodedDiagonals(giant);
        const auto setupStart = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic) \
    num_threads(OpenFHEParallelControls.GetThreadLimit(giant))
        for (uint32_t outer = 0; outer < giant; ++outer) {
            const uint32_t logical = outer * baby + babyIndex;
            if (logical >= logicalLength) {
                continue;
            }
            auto diagonal = factory(logical, 0);
            if (!diagonal.empty()) {
                encodedDiagonals[outer] = EncodeDiagonal(
                    std::move(diagonal), babyIndex, stride, input);
            }
        }
        timing.setupExcluded += Seconds(
            std::chrono::steady_clock::now() - setupStart);
        Ciphertext<DCRTPoly> inner;
        for (uint32_t outer = 0; outer < giant; ++outer) {
            auto& encoded = encodedDiagonals[outer];
            if (!encoded) {
                continue;
            }
            auto term = z->EvalMult(giantRotations[outer], encoded);
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
        OPENFHE_THROW("streaming linear transform produced no terms");
    }
    result->SetZEncodingParams(input->GetZEncodingParams());
    return result;
}

Ciphertext<DCRTPoly>
DenseRadixMultiplierZImpl::EvalStreamingInputPlanRaw(
    ConstCiphertext<DCRTPoly> input,
    uint32_t group) {
    if (group >= 2) {
        OPENFHE_THROW("Boolean input group must be zero or one");
    }
    auto outputs = EvalStreamingBridgePlansRaw(
        input, 1,
        [&](uint32_t logical, uint32_t) {
            BigCVector diagonal(kSlots, BigComplex(BigFixedPoint::zero()));
            bool nonzero = false;
            for (uint32_t digit = 0; digit < radixDigits; ++digit) {
                for (uint32_t residue = 0; residue < 4; ++residue) {
                    const uint32_t fromBase =
                        (4 * digit + residue) * booleanBatch;
                    const uint32_t toBase =
                        digit * radixBatch + group * booleanBatch;
                    const uint32_t delta =
                        (kSlots + fromBase - toBase) & (kSlots - 1);
                    if (delta != logical * booleanBatch) {
                        continue;
                    }
                    const BigComplex weight(BigFixedPoint::fromDouble(
                        static_cast<double>(uint32_t{1} << residue)));
                    for (uint32_t word = 0; word < booleanBatch; ++word) {
                        diagonal[toBase + word] += weight;
                    }
                    nonzero = true;
                }
            }
            return nonzero ? diagonal : BigCVector{};
        });
    return std::move(outputs[0]);
}

std::vector<Ciphertext<DCRTPoly>>
DenseRadixMultiplierZImpl::EvalStreamingOutputPlansRaw(
    ConstCiphertext<DCRTPoly> input,
    uint32_t residue,
    uint32_t sourceStart) {
    if (residue >= 4 || sourceStart >= 2 * wordBits) {
        OPENFHE_THROW("invalid streaming Boolean output map");
    }
    return EvalStreamingBridgePlansRaw(
        input, 2,
        [&](uint32_t logical, uint32_t group) {
            BigCVector diagonal(kSlots, BigComplex(BigFixedPoint::zero()));
            bool nonzero = false;
            for (uint32_t outputBit = 0; outputBit < wordBits; ++outputBit) {
                const uint32_t sourceBit = sourceStart + outputBit;
                if (sourceBit >= 2 * wordBits ||
                    sourceBit % 4 != residue) {
                    continue;
                }
                const uint32_t digit = sourceBit / 4;
                const uint32_t fromBase =
                    digit * radixBatch + group * booleanBatch;
                const uint32_t toBase = outputBit * booleanBatch;
                const uint32_t delta =
                    (kSlots + fromBase - toBase) & (kSlots - 1);
                if (delta != logical * booleanBatch) {
                    continue;
                }
                for (uint32_t word = 0; word < booleanBatch; ++word) {
                    diagonal[toBase + word] +=
                        BigComplex(BigFixedPoint::one());
                }
                nonzero = true;
            }
            return nonzero ? diagonal : BigCVector{};
        });
}

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::EvalStreamingDFTRaw(
    ConstCiphertext<DCRTPoly> input,
    bool inverseMasked,
    bool packedFull) {
    return EvalStreamingLinearPlanRaw(
        input, radixSlice, radixBatch,
        [&](uint32_t offset, uint32_t) {
            BigCVector diagonal(kSlots, BigComplex(BigFixedPoint::zero()));
            const uint32_t rowLimit =
                (inverseMasked || packedFull) ? radixDigits : radixSlice;
            const double normalization =
                (!inverseMasked && !packedFull) ?
                1.0 / std::sqrt(static_cast<double>(radixSlice)) : 1.0;
            const double sign = inverseMasked || packedFull ? 1.0 : -1.0;
            for (uint32_t row = 0; row < rowLimit; ++row) {
                const uint32_t column = (row + offset) % radixSlice;
                const double phase = sign * 2.0 * kPi *
                    static_cast<double>(row) *
                    static_cast<double>(column) /
                    static_cast<double>(radixSlice);
                std::complex<double> coefficient =
                    normalization *
                    std::exp(std::complex<double>(0.0, phase));
                if (packedFull) {
                    const double highPhase = 2.0 * kPi *
                        static_cast<double>(row + radixDigits) *
                        static_cast<double>(column) /
                        static_cast<double>(radixSlice);
                    coefficient += std::complex<double>(0.0, 1.0) *
                        std::exp(std::complex<double>(0.0, highPhase));
                }
                const BigComplex value(
                    BigFixedPoint::fromDouble(coefficient.real()),
                    BigFixedPoint::fromDouble(coefficient.imag()));
                for (uint32_t word = 0; word < radixBatch; ++word) {
                    diagonal[row * radixBatch + word] = value;
                }
            }
            return diagonal;
        });
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

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::MultiplyMaskReduce(
    ConstCiphertext<DCRTPoly> input,
    BigCVector mask) {
    auto plaintext = EncodeDiagonal(std::move(mask), 0, 1, input);
    auto product = z->EvalMult(input, plaintext);
    z->ModReduceInPlace(product);
    product->SetZEncodingParams(input->GetZEncodingParams());
    return product;
}

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::PackBooleanGroups(
    CiphertextGroup input) {
    RequirePair(input, "Boolean/radix pack input");
    if (wordBits >= 1024) {
        auto first = EvalStreamingInputPlanRaw(input[0], 0);
        auto second = EvalStreamingInputPlanRaw(input[1], 1);
        z->EvalAddInPlace(first, second);
        z->ModReduceInPlace(first);
        first->SetZEncodingParams(input[0]->GetZEncodingParams());
        return first;
    }
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
    if (wordBits >= 1024) {
        return;
    }
    for (uint32_t group = 0; group < 2; ++group) {
        const auto setupStart = std::chrono::steady_clock::now();
        inputPlans.emplace(group, BuildInputPlan(input[group], group));
        timing.setupExcluded +=
            Seconds(std::chrono::steady_clock::now() - setupStart);
    }
}

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::ForwardDFT(
    ConstCiphertext<DCRTPoly> input) {
    if (wordBits >= 1024) {
        auto result = EvalStreamingDFTRaw(input, false, false);
        z->ModReduceInPlace(result);
        result->SetZEncodingParams(input->GetZEncodingParams());
        return result;
    }
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
    if (wordBits >= 1024) {
        auto result = EvalStreamingDFTRaw(input, true, false);
        z->ModReduceInPlace(result);
        result->SetZEncodingParams(input->GetZEncodingParams());
        return result;
    }
    if (inversePlan.empty()) {
        const auto setupStart = std::chrono::steady_clock::now();
        inversePlan = BuildDFTPlan(input, true);
        timing.setupExcluded +=
            Seconds(std::chrono::steady_clock::now() - setupStart);
    }
    return EvalLinearPlan(input, inversePlan, radixSlice, radixBatch);
}

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::InverseDFTPackedFull(
    ConstCiphertext<DCRTPoly> input) {
    if (wordBits >= 1024) {
        auto result = EvalStreamingDFTRaw(input, false, true);
        z->ModReduceInPlace(result);
        result->SetZEncodingParams(input->GetZEncodingParams());
        return result;
    }
    if (packedFullInversePlan.empty()) {
        const auto setupStart = std::chrono::steady_clock::now();
        packedFullInversePlan = BuildPackedFullInverseDFTPlan(input);
        timing.setupExcluded +=
            Seconds(std::chrono::steady_clock::now() - setupStart);
    }
    return EvalLinearPlan(input, packedFullInversePlan,
                          radixSlice, radixBatch);
}

Plaintext DenseRadixMultiplierZImpl::EncodePublicFrequency(
    const std::vector<uint32_t>& multiplierDigits,
    ConstCiphertext<DCRTPoly> dftInput) const {
    if (!dftInput || multiplierDigits.size() != radixDigits) {
        OPENFHE_THROW("public radix multiplier has the wrong digit count");
    }
    for (const uint32_t digit : multiplierDigits) {
        if (digit >= 16) {
            OPENFHE_THROW("public radix multiplier digit is outside [0,16)");
        }
    }

    // This is the exact cleartext counterpart of ForwardDFT: the active
    // radix digits occupy rows [0,K), the K guard rows are zero, and both
    // operands carry the same 1/sqrt(2K) normalization. The values are
    // constant over the packed-word dimension, so one plaintext serves the
    // complete aggregate batch.
    BigCVector frequency(kSlots, BigComplex(BigFixedPoint::zero()));
    const double normalization =
        1.0 / std::sqrt(static_cast<double>(radixSlice));
    for (uint32_t row = 0; row < radixSlice; ++row) {
        std::complex<double> value(0.0, 0.0);
        for (uint32_t column = 0; column < radixDigits; ++column) {
            const double phase = -2.0 * kPi *
                static_cast<double>(row) * static_cast<double>(column) /
                static_cast<double>(radixSlice);
            value += static_cast<double>(multiplierDigits[column]) *
                     std::exp(std::complex<double>(0.0, phase));
        }
        value *= normalization;
        const BigComplex encoded(
            BigFixedPoint::fromDouble(value.real()),
            BigFixedPoint::fromDouble(value.imag()));
        for (uint32_t word = 0; word < radixBatch; ++word) {
            frequency[row * radixBatch + word] = encoded;
        }
    }

    const auto elementParams = dftInput->GetElements()[0].GetParams();
    const auto& towers = elementParams->GetParams();
    if (towers.size() < 2) {
        OPENFHE_THROW("public frequency product has no rescale prime");
    }
    const auto cryptoParams =
        std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(
            dftInput->GetCryptoParameters());
    const auto qLast = BigFixedPoint(
        towers.back()->GetModulus(), 0, false).scaleTo(128);
    const auto targetScale =
        cryptoParams->GetScalingFactorBFP(dftInput->GetLevel() + 1);
    const auto plaintextScale =
        targetScale * qLast / dftInput->GetScalingFactorBFP();
    return ZEncodingImpl::encodeC(
        CSlots(ZEncodingParams(CMode, frequency.size() * 2), frequency),
        elementParams, plaintextScale);
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

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::PrefixCarryWide(
    ConstCiphertext<DCRTPoly> propagate,
    ConstCiphertext<DCRTPoly> generate) {
    auto currentP = propagate->Clone();
    auto currentG = generate->Clone();
    for (uint32_t distance = 1; distance < radixSlice; distance <<= 1) {
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

std::vector<Ciphertext<DCRTPoly>>
DenseRadixMultiplierZImpl::CorrectNibblesWide(
    CiphertextGroup sixOutputs,
    ConstCiphertext<DCRTPoly> carry) {
    if (sixOutputs.size() != 6) {
        OPENFHE_THROW("wide shared FBT must return b0,b1,b2,b3,G,P");
    }
    const auto& b0 = sixOutputs[0];
    const auto& b1 = sixOutputs[1];
    const auto& b2 = sixOutputs[2];
    const auto& b3 = sixOutputs[3];
    const auto& localPropagate = sixOutputs[5];

    auto h1 = MultiplyReduce(carry, b0);
    auto h2 = MultiplyReduce(h1, b1);
    auto b01 = MultiplyReduce(b0, b1);
    auto cb2 = MultiplyReduce(carry, b2);
    auto h3 = MultiplyReduce(cb2, b01);
    auto h4 = MultiplyReduce(carry, localPropagate);

    const uint32_t deepest = std::max(h2->GetLevel(),
                                      h3->GetLevel());
    std::vector<Ciphertext<DCRTPoly>> output{
        LinearBit(b0, carry, h1),
        LinearBit(b1, h1, h2),
        LinearBit(b2, h2, h3),
        LinearBit(b3, h3, h4),
    };
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
        if (wordBits >= 1024) {
            auto terms = EvalStreamingOutputPlansRaw(
                planes[residue], residue, 0);
            for (uint32_t group = 0; group < 2; ++group) {
                if (!result[group]) {
                    result[group] = std::move(terms[group]);
                }
                else {
                    z->EvalAddInPlace(result[group], terms[group]);
                }
            }
            continue;
        }
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

CiphertextGroup DenseRadixMultiplierZImpl::ExtractQuotient(
    const std::vector<Ciphertext<DCRTPoly>>& planes,
    uint32_t sourceStart) {
    if (planes.size() != 4) {
        OPENFHE_THROW("quotient extraction requires four product planes");
    }
    std::vector<Ciphertext<DCRTPoly>> result(2);
    for (uint32_t residue = 0; residue < 4; ++residue) {
        if (wordBits >= 1024) {
            auto terms = EvalStreamingOutputPlansRaw(
                planes[residue], residue, sourceStart);
            for (uint32_t group = 0; group < 2; ++group) {
                if (!result[group]) {
                    result[group] = std::move(terms[group]);
                }
                else {
                    z->EvalAddInPlace(result[group], terms[group]);
                }
            }
            continue;
        }
        std::vector<LinearPlan> plans;
        plans.reserve(2);
        for (uint32_t group = 0; group < 2; ++group) {
            const auto setupStart = std::chrono::steady_clock::now();
            plans.push_back(BuildQuotientOutputPlan(
                planes[residue], group, residue, sourceStart));
            timing.setupExcluded +=
                Seconds(std::chrono::steady_clock::now() - setupStart);
        }
        auto terms = EvalBridgePlansRaw(
            planes[residue], {&plans[0], &plans[1]});
        for (uint32_t group = 0; group < 2; ++group) {
            if (!result[group]) {
                result[group] = std::move(terms[group]);
            }
            else {
                z->EvalAddInPlace(result[group], terms[group]);
            }
        }
    }
    const ZEncodingParams params(BModeFull, wordBits, radixBatch);
    for (auto& group : result) {
        z->ModReduceInPlace(group);
        group->SetZEncodingParams(params);
    }
    return CiphertextGroup(std::move(result));
}

CiphertextGroup DenseRadixMultiplierZImpl::DenseRightShift(
    CiphertextGroup input,
    uint32_t shift) {
    RequirePair(input, "dense right-shift input");
    if (shift >= wordBits) {
        OPENFHE_THROW("dense right shift is outside the word");
    }
    std::vector<Ciphertext<DCRTPoly>> output;
    output.reserve(2);
    for (uint32_t group = 0; group < 2; ++group) {
        const auto setupStart = std::chrono::steady_clock::now();
        auto plan = BuildDenseShiftPlan(input[group], shift);
        timing.setupExcluded +=
            Seconds(std::chrono::steady_clock::now() - setupStart);
        output.push_back(EvalBridgePlanRaw(input[group], plan));
        z->ModReduceInPlace(output.back());
        output.back()->SetZEncodingParams(
            ZEncodingParams(BModeFull, wordBits, radixBatch));
    }
    return CiphertextGroup(std::move(output));
}

Ciphertext<DCRTPoly> DenseRadixMultiplierZImpl::DenseAddAndShift(
    ConstCiphertext<DCRTPoly> left,
    ConstCiphertext<DCRTPoly> right,
    uint32_t shift) {
    if (shift == 0 || shift > wordBits) {
        OPENFHE_THROW("magic add correction has an invalid shift");
    }
    const uint32_t layers = static_cast<uint32_t>(std::log2(wordBits));
    if (left->GetLevel() != right->GetLevel()) {
        OPENFHE_THROW("magic add correction inputs have different levels");
    }
    const uint32_t correctionInputLevel = left->GetLevel();
    auto lhs = Align(left, correctionInputLevel);
    auto rhs = Align(right, correctionInputLevel);
    auto generate = MultiplyReduce(lhs, rhs);
    auto propagate = z->EvalSub(
        z->EvalAdd(Align(lhs, generate->GetLevel()),
                   Align(rhs, generate->GetLevel())),
        z->EvalAdd(generate, generate));

    BigCVector pMask(kSlots, BigComplex(BigFixedPoint::one()));
    BigCVector gMask(kSlots, BigComplex(BigFixedPoint::one()));
    for (uint32_t word = 0; word < booleanBatch; ++word) {
        pMask[word] = BigComplex(BigFixedPoint::zero());
    }
    propagate = MultiplyMaskReduce(propagate, std::move(pMask));
    generate = MultiplyMaskReduce(generate, std::move(gMask));

    for (uint32_t distance = 1; distance < wordBits; distance <<= 1) {
        auto rotatedP = Rotate(
            propagate, -static_cast<int32_t>(distance * booleanBatch));
        auto rotatedG = Rotate(
            generate, -static_cast<int32_t>(distance * booleanBatch));
        auto nextP = MultiplyReduce(propagate, rotatedP);
        auto propagatedG = MultiplyReduce(propagate, rotatedG);
        generate = AddAdjusted(generate, propagatedG);
        propagate = std::move(nextP);
    }
    if (generate->GetLevel() != correctionInputLevel + 2 + layers) {
        OPENFHE_THROW("magic add prefix consumed an unexpected depth");
    }
    auto base = z->EvalSub(
        z->EvalAdd(Align(lhs, generate->GetLevel()),
                   Align(rhs, generate->GetLevel())),
        z->EvalAdd(generate, generate));
    auto carryIn = Rotate(generate, -static_cast<int32_t>(booleanBatch));
    base = AddAdjusted(base, carryIn);

    auto shifted = Rotate(base, static_cast<int32_t>(shift * booleanBatch));
    auto carryEndpoint = shift == 1 ? generate->Clone() :
        Rotate(generate,
               static_cast<int32_t>((shift - 1) * booleanBatch));
    BigCVector shiftMask(kSlots, BigComplex(BigFixedPoint::zero()));
    BigCVector overflowMask(kSlots, BigComplex(BigFixedPoint::zero()));
    for (uint32_t bit = 0; bit < wordBits - shift; ++bit) {
        for (uint32_t word = 0; word < booleanBatch; ++word) {
            shiftMask[bit * booleanBatch + word] =
                BigComplex(BigFixedPoint::one());
        }
    }
    const uint32_t overflowBit = wordBits - shift;
    for (uint32_t word = 0; word < booleanBatch; ++word) {
        overflowMask[overflowBit * booleanBatch + word] =
            BigComplex(BigFixedPoint::one());
    }
    shifted = MultiplyMaskReduce(shifted, std::move(shiftMask));
    carryEndpoint = MultiplyMaskReduce(
        carryEndpoint, std::move(overflowMask));
    return AddAdjusted(shifted, carryEndpoint);
}

CiphertextGroup DenseRadixMultiplierZImpl::RefreshBooleanPair(
    CiphertextGroup raw,
    const BigFixedPoint& stableInputScale,
    const char* label) {
    RequirePair(raw, label);
    const ZEncodingParams params(BModeFull, wordBits, radixBatch);
    raw[0]->SetZEncodingParams(params);
    raw[1]->SetZEncodingParams(params);
    auto result = fhe->EvalBooleanToBooleanFull(raw);
    RequirePair(result, label);
    result[0]->SetZEncodingParams(params);
    result[1]->SetZEncodingParams(params);
    if (result[0]->GetLevel() != levels.stableInput ||
        !result[0]->GetScalingFactorBFP().almostEqual(stableInputScale) ||
        !result[1]->GetScalingFactorBFP().almostEqual(stableInputScale)) {
        OPENFHE_THROW(std::string(label) + " did not restore the steady-state boundary");
    }
    return result;
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
    const double packSetupBefore = timing.setupExcluded;
    const auto packStart = std::chrono::steady_clock::now();
    auto packedLeft = PackBooleanGroups(left);
    auto packedRight = PackBooleanGroups(right);
    inputPlans.clear();
    timing.inputPack = Seconds(std::chrono::steady_clock::now() - packStart) -
                       (timing.setupExcluded - packSetupBefore);
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

    return FinalizeProduct(std::move(lazyDigits), maximumB2BInputLevel,
                           stableInputScale);
}

DenseRadixFullProduct DenseRadixMultiplierZImpl::EvalFullMultiply(
    CiphertextGroup left,
    CiphertextGroup right,
    uint32_t maximumB2BInputLevel) {
    RequirePair(left, "left full-product input");
    RequirePair(right, "right full-product input");
    if (left[0]->GetLevel() != right[0]->GetLevel() ||
        !left[0]->GetScalingFactorBFP().almostEqual(
            right[0]->GetScalingFactorBFP())) {
        OPENFHE_THROW("full-product operands have different level/scale contracts");
    }
    timing = DenseRadixMultiplierTiming{};
    levels = DenseRadixMultiplierLevels{};
    levels.stableInput = left[0]->GetLevel();
    const auto stableInputScale = left[0]->GetScalingFactorBFP();
    if (rawBoundaryLevel > maximumB2BInputLevel) {
        OPENFHE_THROW("full-product tail exceeds the native B2B entry boundary");
    }
    if (left[0]->GetLevel() > kCircuitInputLevel ||
        right[0]->GetLevel() > kCircuitInputLevel) {
        OPENFHE_THROW("full-product input is below the circuit entry level");
    }
    left = CiphertextGroup({Align(left[0], kCircuitInputLevel),
                            Align(left[1], kCircuitInputLevel)});
    right = CiphertextGroup({Align(right[0], kCircuitInputLevel),
                             Align(right[1], kCircuitInputLevel)});

    PrepareInputPlans(left);
    const double packSetupBefore = timing.setupExcluded;
    const auto packStart = std::chrono::steady_clock::now();
    auto packedLeft = PackBooleanGroups(left);
    Observe("Boolean-to-radix A", packedLeft);
    inputPlans.clear();
    PrepareInputPlans(right);
    auto packedRight = PackBooleanGroups(right);
    Observe("Boolean-to-radix B", packedRight);
    inputPlans.clear();
    timing.inputPack = Seconds(std::chrono::steady_clock::now() - packStart) -
                       (timing.setupExcluded - packSetupBefore);
    levels.packedRadix = packedLeft->GetLevel();
    if (levels.packedRadix != 17 || packedRight->GetLevel() != 17) {
        OPENFHE_THROW("full-product Boolean/radix packing did not end at level 17");
    }

    const double setupBefore = timing.setupExcluded;
    const auto convolutionStart = std::chrono::steady_clock::now();
    auto dftLeft = ForwardDFT(packedLeft);
    Observe("forward DFT A", dftLeft);
    auto dftRight = ForwardDFT(packedRight);
    Observe("forward DFT B", dftRight);
    packedLeft.reset();
    packedRight.reset();
    forwardPlan.clear();
    auto frequencyProduct = MultiplyReduce(dftLeft, dftRight);
    Observe("pointwise ciphertext product", frequencyProduct);
    dftLeft.reset();
    dftRight.reset();
    auto lazyDigits = InverseDFTPackedFull(frequencyProduct);
    Observe("packed inverse DFT", lazyDigits);
    frequencyProduct.reset();
    packedFullInversePlan.clear();
    timing.convolution =
        Seconds(std::chrono::steady_clock::now() - convolutionStart) -
        (timing.setupExcluded - setupBefore);
    levels.convolution = lazyDigits->GetLevel();
    if (levels.convolution != 20) {
        OPENFHE_THROW("full ciphertext product convolution did not end at level 20");
    }
    return FinalizeFullProduct(std::move(lazyDigits), maximumB2BInputLevel,
                               stableInputScale);
}

CiphertextGroup DenseRadixMultiplierZImpl::EvalScaleMultiply(
    CiphertextGroup input,
    const std::vector<uint32_t>& multiplierDigits,
    uint32_t maximumB2BInputLevel) {
    RequirePair(input, "scale multiplier input");
    if (multiplierDigits.size() != radixDigits) {
        OPENFHE_THROW("public multiplier must contain exactly word_bits/4 digits");
    }
    for (const uint32_t digit : multiplierDigits) {
        if (digit >= 16) {
            OPENFHE_THROW("public multiplier radix digit is outside [0,16)");
        }
    }

    timing = DenseRadixMultiplierTiming{};
    levels = DenseRadixMultiplierLevels{};
    levels.stableInput = input[0]->GetLevel();
    const auto stableInputScale = input[0]->GetScalingFactorBFP();
    if (rawBoundaryLevel > maximumB2BInputLevel) {
        OPENFHE_THROW("scale multiplier tail exceeds the native B2B entry boundary");
    }
    if (input[0]->GetLevel() > kCircuitInputLevel) {
        OPENFHE_THROW("B2B output is already below the Lattigo circuit input level");
    }
    input = CiphertextGroup({Align(input[0], kCircuitInputLevel),
                             Align(input[1], kCircuitInputLevel)});

    PrepareInputPlans(input);
    const double packSetupBefore = timing.setupExcluded;
    const auto packStart = std::chrono::steady_clock::now();
    auto packed = PackBooleanGroups(input);
    inputPlans.clear();
    timing.inputPack = Seconds(std::chrono::steady_clock::now() - packStart) -
                       (timing.setupExcluded - packSetupBefore);
    levels.packedRadix = packed->GetLevel();
    if (levels.packedRadix != 17) {
        OPENFHE_THROW("Boolean/radix packing did not reproduce Lattigo level 10");
    }
    Observe("packed input", packed);

    double setupBefore = timing.setupExcluded;
    const auto convolutionStart = std::chrono::steady_clock::now();
    auto dftInput = ForwardDFT(packed);
    packed.reset();
    forwardPlan.clear();
    const auto publicSetupStart = std::chrono::steady_clock::now();
    auto publicFrequency =
        EncodePublicFrequency(multiplierDigits, dftInput);
    timing.setupExcluded += Seconds(
        std::chrono::steady_clock::now() - publicSetupStart);
    auto frequencyProduct = z->EvalMult(dftInput, publicFrequency);
    z->ModReduceInPlace(frequencyProduct);
    frequencyProduct->SetZEncodingParams(dftInput->GetZEncodingParams());
    dftInput.reset();
    publicFrequency.reset();
    auto lazyDigits = InverseDFTMasked(frequencyProduct);
    frequencyProduct.reset();
    inversePlan.clear();
    timing.convolution =
        Seconds(std::chrono::steady_clock::now() - convolutionStart) -
        (timing.setupExcluded - setupBefore);
    levels.convolution = lazyDigits->GetLevel();
    if (levels.convolution != 20) {
        OPENFHE_THROW(
            "public DFT convolution did not reproduce Lattigo level 7");
    }
    Observe("convolution", lazyDigits);

    return FinalizeProduct(std::move(lazyDigits), maximumB2BInputLevel,
                           stableInputScale);
}

DenseRadixFullProduct DenseRadixMultiplierZImpl::EvalFullScaleMultiply(
    CiphertextGroup input,
    const std::vector<uint32_t>& multiplierDigits,
    uint32_t maximumB2BInputLevel) {
    RequirePair(input, "full public multiplier input");
    if (multiplierDigits.size() != radixDigits) {
        OPENFHE_THROW("full public multiplier must contain word_bits/4 digits");
    }
    for (const uint32_t digit : multiplierDigits) {
        if (digit >= 16) {
            OPENFHE_THROW("full public multiplier digit is outside [0,16)");
        }
    }
    timing = DenseRadixMultiplierTiming{};
    levels = DenseRadixMultiplierLevels{};
    levels.stableInput = input[0]->GetLevel();
    const auto stableInputScale = input[0]->GetScalingFactorBFP();
    if (rawBoundaryLevel > maximumB2BInputLevel) {
        OPENFHE_THROW("full public-product tail exceeds the native B2B boundary");
    }
    if (input[0]->GetLevel() > kCircuitInputLevel) {
        OPENFHE_THROW("full public-product input is below the circuit entry level");
    }
    input = CiphertextGroup({Align(input[0], kCircuitInputLevel),
                             Align(input[1], kCircuitInputLevel)});
    PrepareInputPlans(input);
    const double packSetupBefore = timing.setupExcluded;
    const auto packStart = std::chrono::steady_clock::now();
    auto packed = PackBooleanGroups(input);
    inputPlans.clear();
    timing.inputPack = Seconds(std::chrono::steady_clock::now() - packStart) -
                       (timing.setupExcluded - packSetupBefore);
    levels.packedRadix = packed->GetLevel();
    if (levels.packedRadix != 17) {
        OPENFHE_THROW("full public-product Boolean/radix packing did not end at level 17");
    }
    const double setupBefore = timing.setupExcluded;
    const auto convolutionStart = std::chrono::steady_clock::now();
    auto dftInput = ForwardDFT(packed);
    packed.reset();
    forwardPlan.clear();
    const auto publicSetupStart = std::chrono::steady_clock::now();
    auto publicFrequency = EncodePublicFrequency(multiplierDigits, dftInput);
    timing.setupExcluded += Seconds(
        std::chrono::steady_clock::now() - publicSetupStart);
    auto frequencyProduct = z->EvalMult(dftInput, publicFrequency);
    z->ModReduceInPlace(frequencyProduct);
    frequencyProduct->SetZEncodingParams(dftInput->GetZEncodingParams());
    dftInput.reset();
    publicFrequency.reset();
    auto lazyDigits = InverseDFTPackedFull(frequencyProduct);
    frequencyProduct.reset();
    packedFullInversePlan.clear();
    timing.convolution =
        Seconds(std::chrono::steady_clock::now() - convolutionStart) -
        (timing.setupExcluded - setupBefore);
    levels.convolution = lazyDigits->GetLevel();
    if (levels.convolution != 20) {
        OPENFHE_THROW("full public-product convolution did not end at level 20");
    }
    Observe("full public product", lazyDigits);
    return FinalizeFullProduct(std::move(lazyDigits), maximumB2BInputLevel,
                               stableInputScale);
}

CiphertextGroup DenseRadixMultiplierZImpl::EvalScaleDivide(
    CiphertextGroup input,
    const std::vector<uint32_t>& magicDigits,
    uint32_t shift,
    bool addMarker,
    uint32_t maximumB2BInputLevel) {
    RequirePair(input, "scale divider input");
    if (shift >= wordBits) {
        OPENFHE_THROW("scale divider shift is outside the word");
    }
    timing = DenseRadixMultiplierTiming{};
    levels = DenseRadixMultiplierLevels{};
    levels.stableInput = input[0]->GetLevel();
    const auto stableInputScale = input[0]->GetScalingFactorBFP();
    if (magicDigits.empty()) {
        const auto shiftStart = std::chrono::steady_clock::now();
        auto raw = DenseRightShift(input, shift);
        timing.outputPack = Seconds(
            std::chrono::steady_clock::now() - shiftStart);
        levels.rawBoolean = raw[0]->GetLevel();
        const auto refreshStart = std::chrono::steady_clock::now();
        auto result = RefreshBooleanPair(raw, stableInputScale,
                                         "power-of-two scale_div");
        timing.finalB2B = Seconds(
            std::chrono::steady_clock::now() - refreshStart);
        levels.finalBoolean = result[0]->GetLevel();
        return result;
    }
    if (magicDigits.size() != radixDigits) {
        OPENFHE_THROW("scale divider magic must contain word_bits/4 digits");
    }
    if (27 > maximumB2BInputLevel) {
        OPENFHE_THROW("scale divider tail exceeds native B2B entry boundary");
    }
    if (input[0]->GetLevel() > kCircuitInputLevel) {
        OPENFHE_THROW("B2B output is below the scale_div circuit entry");
    }
    const auto stableInput = input;
    input = CiphertextGroup({Align(input[0], kCircuitInputLevel),
                             Align(input[1], kCircuitInputLevel)});

    PrepareInputPlans(input);
    const double packSetupBefore = timing.setupExcluded;
    const auto packStart = std::chrono::steady_clock::now();
    auto packed = PackBooleanGroups(input);
    inputPlans.clear();
    timing.inputPack = Seconds(std::chrono::steady_clock::now() - packStart) -
                       (timing.setupExcluded - packSetupBefore);
    levels.packedRadix = packed->GetLevel();
    if (levels.packedRadix != 17) {
        OPENFHE_THROW("scale_div Boolean/radix pack did not end at level 17");
    }
    Observe("packed input", packed);

    const double setupBefore = timing.setupExcluded;
    const auto convolutionStart = std::chrono::steady_clock::now();
    auto dftInput = ForwardDFT(packed);
    packed.reset();
    forwardPlan.clear();
    const auto publicSetupStart = std::chrono::steady_clock::now();
    auto publicFrequency = EncodePublicFrequency(magicDigits, dftInput);
    timing.setupExcluded += Seconds(
        std::chrono::steady_clock::now() - publicSetupStart);
    auto frequencyProduct = z->EvalMult(dftInput, publicFrequency);
    z->ModReduceInPlace(frequencyProduct);
    frequencyProduct->SetZEncodingParams(dftInput->GetZEncodingParams());
    dftInput.reset();
    publicFrequency.reset();
    auto lazyDigits = InverseDFTPackedFull(frequencyProduct);
    frequencyProduct.reset();
    packedFullInversePlan.clear();
    timing.convolution =
        Seconds(std::chrono::steady_clock::now() - convolutionStart) -
        (timing.setupExcluded - setupBefore);
    levels.convolution = lazyDigits->GetLevel();
    if (levels.convolution != 20) {
        OPENFHE_THROW("full-product DFT convolution did not end at level 20");
    }
    Observe("full convolution", lazyDigits);

    const uint32_t sourceStart = addMarker ? wordBits : wordBits + shift;
    return FinalizeDivisionProduct(
        std::move(lazyDigits), stableInput, sourceStart, shift,
        addMarker, maximumB2BInputLevel, stableInputScale);
}

CiphertextGroup DenseRadixMultiplierZImpl::FinalizeProduct(
    Ciphertext<DCRTPoly> lazyDigits,
    uint32_t maximumB2BInputLevel,
    const BigFixedPoint& stableInputScale) {
    if (!lazyDigits || lazyDigits->GetLevel() != 20) {
        OPENFHE_THROW("radix product finalization requires level-20 lazy digits");
    }

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

    const double setupBefore = timing.setupExcluded;
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

DenseRadixFullProduct DenseRadixMultiplierZImpl::FinalizeFullProduct(
    Ciphertext<DCRTPoly> lazyDigits,
    uint32_t maximumB2BInputLevel,
    const BigFixedPoint& stableInputScale) {
    if (!lazyDigits || lazyDigits->GetLevel() != 20) {
        OPENFHE_THROW("full-product finalization requires level-20 lazy digits");
    }
    const uint32_t widePrefixLayers = prefixLayers + 1;
    for (uint32_t pass = 0; pass < lazyCarryPasses; ++pass) {
        const auto start = std::chrono::steady_clock::now();
        lazyDigits = fhe->EvalRadix16PackedFullLazyCarry(
            lazyDigits, radixDigits, radixBatch);
        timing.lazyCarry[pass] =
            Seconds(std::chrono::steady_clock::now() - start);
        levels.lazyCarry[pass] = lazyDigits->GetLevel();
        if (levels.lazyCarry[pass] != 21 + pass) {
            OPENFHE_THROW("full-product LazyCarry level mismatch");
        }
        Observe("full LazyCarry #" + std::to_string(pass + 1), lazyDigits);
    }

    const auto fbtStart = std::chrono::steady_clock::now();
    auto sixOutputs = fhe->EvalRadix31PackedFullToBooleanSixFirst(
        lazyDigits, radixDigits, radixBatch);
    timing.sharedFBT = Seconds(std::chrono::steady_clock::now() - fbtStart);
    lazyDigits.reset();
    if (sixOutputs.size() != 6) {
        OPENFHE_THROW("full-product FBT returned the wrong output count");
    }
    levels.sharedFBTRaw = sixOutputs[5]->GetLevel();
    for (uint32_t output = 0; output < sixOutputs.size(); ++output) {
        if (sixOutputs[output]->GetLevel() > kSharedFBTOutputLevel) {
            std::ostringstream message;
            message << "full-product FBT output " << output
                    << " is at level " << sixOutputs[output]->GetLevel()
                    << ", beyond scheduled level "
                    << kSharedFBTOutputLevel;
            OPENFHE_THROW(message.str());
        }
        if (sixOutputs[output]->GetLevel() < kSharedFBTOutputLevel) {
            sixOutputs[output] = Align(sixOutputs[output],
                                       kSharedFBTOutputLevel);
        }
    }
    levels.sharedFBT = kSharedFBTOutputLevel;
    Observe("full-product FBT", sixOutputs[5]);

    const auto prefixStart = std::chrono::steady_clock::now();
    auto carry = PrefixCarryWide(sixOutputs[5], sixOutputs[4]);
    timing.prefix = Seconds(std::chrono::steady_clock::now() - prefixStart);
    levels.prefix = carry->GetLevel();
    if (levels.prefix != kSharedFBTOutputLevel + widePrefixLayers) {
        OPENFHE_THROW("full-product prefix level mismatch");
    }
    Observe("full-product radix carry", carry);

    const auto correctionStart = std::chrono::steady_clock::now();
    auto corrected = CorrectNibblesWide(sixOutputs, carry);
    sixOutputs = CiphertextGroup{};
    carry.reset();
    timing.correction =
        Seconds(std::chrono::steady_clock::now() - correctionStart);
    levels.correction = corrected[0]->GetLevel();
    if (levels.correction !=
        kSharedFBTOutputLevel + widePrefixLayers + 2) {
        OPENFHE_THROW("full-product nibble correction level mismatch");
    }
    Observe("full-product corrected digits", corrected[0]);

    const double setupBefore = timing.setupExcluded;
    const auto extractStart = std::chrono::steady_clock::now();
    auto lowRaw = ExtractQuotient(corrected, 0);
    auto highRaw = ExtractQuotient(corrected, wordBits);
    corrected.clear();
    timing.outputPack = Seconds(std::chrono::steady_clock::now() - extractStart) -
                        (timing.setupExcluded - setupBefore);
    const uint32_t expectedRawLevel =
        kSharedFBTOutputLevel + widePrefixLayers + 3;
    if (lowRaw[0]->GetLevel() != expectedRawLevel ||
        lowRaw[1]->GetLevel() != expectedRawLevel ||
        highRaw[0]->GetLevel() != expectedRawLevel ||
        highRaw[1]->GetLevel() != expectedRawLevel ||
        expectedRawLevel > maximumB2BInputLevel) {
        OPENFHE_THROW("full-product halves did not reach the B2B boundary");
    }
    levels.rawBoolean = expectedRawLevel;
    Observe("full-product raw low group 0", lowRaw[0]);
    Observe("full-product raw high group 0", highRaw[0]);

    const auto refreshStart = std::chrono::steady_clock::now();
    auto low = RefreshBooleanPair(lowRaw, stableInputScale,
                                  "full-product low-half B2B");
    lowRaw = CiphertextGroup{};
    Observe("full-product refreshed low group 0", low[0]);
    auto high = RefreshBooleanPair(highRaw, stableInputScale,
                                   "full-product high-half B2B");
    highRaw = CiphertextGroup{};
    Observe("full-product refreshed high group 0", high[0]);
    timing.finalB2B = Seconds(
        std::chrono::steady_clock::now() - refreshStart);
    levels.finalBoolean = low[0]->GetLevel();
    Observe("full-product low group 1", low[1]);
    Observe("full-product high group 1", high[1]);
    return DenseRadixFullProduct{std::move(low), std::move(high)};
}

CiphertextGroup DenseRadixMultiplierZImpl::FinalizeDivisionProduct(
    Ciphertext<DCRTPoly> lazyDigits,
    CiphertextGroup stableInput,
    uint32_t sourceStart,
    uint32_t shift,
    bool addMarker,
    uint32_t maximumB2BInputLevel,
    const BigFixedPoint& stableInputScale) {
    if (!lazyDigits || lazyDigits->GetLevel() != 20) {
        OPENFHE_THROW("scale_div finalization requires level-20 full product");
    }
    const uint32_t widePrefixLayers = prefixLayers + 1;
    for (uint32_t pass = 0; pass < lazyCarryPasses; ++pass) {
        const auto start = std::chrono::steady_clock::now();
        lazyDigits = fhe->EvalRadix16PackedFullLazyCarry(
            lazyDigits, radixDigits, radixBatch);
        timing.lazyCarry[pass] =
            Seconds(std::chrono::steady_clock::now() - start);
        levels.lazyCarry[pass] = lazyDigits->GetLevel();
        if (levels.lazyCarry[pass] != 21 + pass) {
            OPENFHE_THROW("packed full-product LazyCarry level mismatch");
        }
        Observe("LazyCarry #" + std::to_string(pass + 1), lazyDigits);
    }

    const auto fbtStart = std::chrono::steady_clock::now();
    auto sixOutputs = fhe->EvalRadix31PackedFullToBooleanSixFirst(
        lazyDigits, radixDigits, radixBatch);
    timing.sharedFBT = Seconds(std::chrono::steady_clock::now() - fbtStart);
    lazyDigits.reset();
    if (sixOutputs.size() != 6) {
        OPENFHE_THROW("packed full-product FBT returned the wrong output count");
    }
    levels.sharedFBTRaw = sixOutputs[5]->GetLevel();
    const uint32_t scheduledFBTLevel = kSharedFBTOutputLevel;
    for (uint32_t output = 0; output < sixOutputs.size(); ++output) {
        if (sixOutputs[output]->GetLevel() > scheduledFBTLevel) {
            OPENFHE_THROW("packed FBT output exceeds the scheduled runway");
        }
        if (sixOutputs[output]->GetLevel() < scheduledFBTLevel) {
            sixOutputs[output] = Align(sixOutputs[output],
                                       scheduledFBTLevel);
        }
    }
    levels.sharedFBT = scheduledFBTLevel;
    static const char* labels[6] = {"FBT b0", "FBT b1", "FBT b2",
                                     "FBT b3", "FBT G", "FBT P"};
    for (uint32_t output = 0; output < 6; ++output) {
        Observe(labels[output], sixOutputs[output]);
    }

    const auto prefixStart = std::chrono::steady_clock::now();
    auto carry = PrefixCarryWide(sixOutputs[5], sixOutputs[4]);
    timing.prefix = Seconds(std::chrono::steady_clock::now() - prefixStart);
    levels.prefix = carry->GetLevel();
    if (levels.prefix != scheduledFBTLevel + widePrefixLayers) {
        OPENFHE_THROW("full-product prefix did not consume log2(2K) levels");
    }
    Observe("full radix carry", carry);

    const auto correctionStart = std::chrono::steady_clock::now();
    auto corrected = CorrectNibblesWide(sixOutputs, carry);
    sixOutputs = CiphertextGroup{};
    carry.reset();
    timing.correction =
        Seconds(std::chrono::steady_clock::now() - correctionStart);
    levels.correction = corrected[0]->GetLevel();
    if (levels.correction != scheduledFBTLevel + widePrefixLayers + 2) {
        OPENFHE_THROW("wide nibble correction did not consume two levels");
    }
    for (uint32_t plane = 0; plane < 4; ++plane) {
        Observe("corrected b" + std::to_string(plane), corrected[plane]);
    }

    const double setupBefore = timing.setupExcluded;
    const auto extractStart = std::chrono::steady_clock::now();
    auto raw = ExtractQuotient(corrected, sourceStart);
    corrected.clear();
    timing.outputPack = Seconds(std::chrono::steady_clock::now() - extractStart) -
                        (timing.setupExcluded - setupBefore);
    levels.rawBoolean = raw[0]->GetLevel();
    const uint32_t expectedRawLevel =
        scheduledFBTLevel + widePrefixLayers + 3;
    if (levels.rawBoolean != expectedRawLevel ||
        raw[1]->GetLevel() != expectedRawLevel ||
        levels.rawBoolean > maximumB2BInputLevel) {
        OPENFHE_THROW("raw quotient did not reach its scheduled B2B boundary");
    }
    Observe(addMarker ? "raw q0 group 0" : "raw quotient group 0", raw[0]);
    Observe(addMarker ? "raw q0 group 1" : "raw quotient group 1", raw[1]);

    if (addMarker) {
        const auto refreshStart = std::chrono::steady_clock::now();
        auto q0 = RefreshBooleanPair(raw, stableInputScale,
                                     "intermediate q0 B2B");
        timing.intermediateB2B = Seconds(
            std::chrono::steady_clock::now() - refreshStart);
        levels.intermediateBoolean = q0[0]->GetLevel();
        const uint32_t correctionDepth =
            static_cast<uint32_t>(std::log2(wordBits)) + 3;
        const uint32_t correctionEntry =
            maximumB2BInputLevel - correctionDepth;
        stableInput = CiphertextGroup({
            Align(stableInput[0], correctionEntry),
            Align(stableInput[1], correctionEntry),
        });
        q0 = CiphertextGroup({Align(q0[0], correctionEntry),
                              Align(q0[1], correctionEntry)});
        const auto correctionStart2 = std::chrono::steady_clock::now();
        raw = CiphertextGroup({
            DenseAddAndShift(stableInput[0], q0[0], shift + 1),
            DenseAddAndShift(stableInput[1], q0[1], shift + 1),
        });
        timing.addCorrection = Seconds(
            std::chrono::steady_clock::now() - correctionStart2);
        levels.rawBoolean = raw[0]->GetLevel();
        if (levels.rawBoolean > maximumB2BInputLevel ||
            raw[1]->GetLevel() != levels.rawBoolean) {
            OPENFHE_THROW("add-marker correction exceeds the B2B boundary");
        }
        Observe("corrected quotient group 0", raw[0]);
        Observe("corrected quotient group 1", raw[1]);
    }

    const auto finalStart = std::chrono::steady_clock::now();
    auto result = RefreshBooleanPair(raw, stableInputScale,
                                     "final scale_div B2B");
    timing.finalB2B = Seconds(
        std::chrono::steady_clock::now() - finalStart);
    levels.finalBoolean = result[0]->GetLevel();
    Observe("final quotient group 0", result[0]);
    Observe("final quotient group 1", result[1]);
    return result;
}

}  // namespace lbcrypto
