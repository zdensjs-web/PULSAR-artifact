#include "scheme/ckksrns/z-dense-boolean.h"

#include "encoding/z-encoding.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace lbcrypto {

void DenseBooleanZImpl::RequireCompatible(ConstCiphertext<DCRTPoly> left,
                                          ConstCiphertext<DCRTPoly> right,
                                          const char* operation) {
    if (!left || !right) {
        OPENFHE_THROW(std::string(operation) + ": nil ciphertext");
    }
    if (left->GetLevel() != right->GetLevel()) {
        std::ostringstream message;
        message << operation << ": level mismatch " << left->GetLevel() << "/" << right->GetLevel();
        OPENFHE_THROW(message.str());
    }
    if (!left->GetScalingFactorBFP().almostEqual(right->GetScalingFactorBFP())) {
        OPENFHE_THROW(std::string(operation) + ": BigFixedPoint scale mismatch");
    }
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::Add(ConstCiphertext<DCRTPoly> left,
                                            ConstCiphertext<DCRTPoly> right) {
    RequireCompatible(left, right, "dense Boolean add");
    return z->EvalAdd(left, right);
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::Sub(ConstCiphertext<DCRTPoly> left,
                                            ConstCiphertext<DCRTPoly> right) {
    RequireCompatible(left, right, "dense Boolean sub");
    return z->EvalSub(left, right);
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::Double(ConstCiphertext<DCRTPoly> input) {
    return Add(input, input);
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::MultiplyAndReduce(ConstCiphertext<DCRTPoly> left,
                                                          ConstCiphertext<DCRTPoly> right) {
    RequireCompatible(left, right, "dense Boolean multiply");
    auto product = z->EvalMult(left, right);
    z->ModReduceInPlace(product);
    return product;
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::Align(ConstCiphertext<DCRTPoly> input, uint32_t level) {
    return z->AdjustCiphertextToLevel(input, level);
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::OneMinus(
    ConstCiphertext<DCRTPoly> input) {
    if (!input) {
        OPENFHE_THROW("dense Boolean complement received a nil ciphertext");
    }
    auto output = input->Clone();
    z->EvalMultInPlaceInC(
        output, BigComplex(BigFixedPoint::fromDouble(-1.0)),
        BigFixedPoint::one());
    z->EvalAddInPlaceInC(output, BigComplex(BigFixedPoint::one()));
    return output;
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::AdjustToLevel(ConstCiphertext<DCRTPoly> input,
                                                      uint32_t level) {
    if (!input) {
        OPENFHE_THROW("dense Boolean level adjustment received a nil ciphertext");
    }
    return Align(input, level);
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::Rotate(ConstCiphertext<DCRTPoly> input, int32_t offset) {
    auto rotated = input->GetCryptoContext()->EvalRotate(input, offset);
    rotated->SetScalingFactorBFP(input->GetScalingFactorBFP());
    rotated->SetZEncodingParams(input->GetZEncodingParams());
    return rotated;
}

Plaintext DenseBooleanZImpl::GetMaskPlaintext(ConstCiphertext<DCRTPoly> input,
                                              uint32_t wordBits,
                                              uint32_t wordsPerCiphertext,
                                              MaskKind kind,
                                              uint32_t amount) {
    const auto level = input->GetLevel();
    const MaskKey key{level, wordBits, wordsPerCiphertext, kind, amount};
    const auto cached = maskCache.find(key);
    if (cached != maskCache.end()) {
        return cached->second;
    }

    const auto slotCount = input->GetCryptoContext()->GetRingDimension() / 2;
    if (static_cast<uint64_t>(wordBits) * wordsPerCiphertext != slotCount) {
        OPENFHE_THROW("dense Boolean mask layout does not fill all complex slots");
    }

    const auto half = BigFixedPoint::fromDouble(0.5);
    BigComplex activeValue(BigFixedPoint::one());
    if (kind == MaskKind::HalfOnes || kind == MaskKind::HalfClearWordHeads) {
        activeValue = BigComplex(half);
    }
    else if (kind == MaskKind::QuarterOnes) {
        activeValue = BigComplex(BigFixedPoint::fromDouble(0.25));
    }
    else if (kind == MaskKind::ImaginaryOnes) {
        activeValue = BigComplex(BigFixedPoint::zero(), BigFixedPoint::one());
    }

    std::vector<BigComplex> values(slotCount, activeValue);
    if (kind == MaskKind::ClearWordHeads || kind == MaskKind::HalfClearWordHeads) {
        for (uint32_t word = 0; word < wordsPerCiphertext; ++word) {
            values[word] = BigComplex(BigFixedPoint::zero());
        }
    }
    else if (kind == MaskKind::WordHeads) {
        std::fill(values.begin(), values.end(),
                  BigComplex(BigFixedPoint::zero()));
        for (uint32_t word = 0; word < wordsPerCiphertext; ++word) {
            values[word] = activeValue;
        }
    }
    else if (kind == MaskKind::WordTails) {
        std::fill(values.begin(), values.end(),
                  BigComplex(BigFixedPoint::zero()));
        const size_t tailOffset =
            static_cast<size_t>(wordBits - 1) * wordsPerCiphertext;
        for (uint32_t word = 0; word < wordsPerCiphertext; ++word) {
            values[tailOffset + word] = activeValue;
        }
    }
    else if (kind == MaskKind::LogicalRightShift) {
        std::fill(values.begin(), values.end(),
                  BigComplex(BigFixedPoint::zero()));
        if (amount < wordBits) {
            for (uint32_t bit = 0; bit < wordBits - amount; ++bit) {
                const size_t offset =
                    static_cast<size_t>(bit) * wordsPerCiphertext;
                for (uint32_t word = 0; word < wordsPerCiphertext; ++word) {
                    values[offset + word] = activeValue;
                }
            }
        }
    }

    const auto elementParams = input->GetElements()[0].GetParams();
    const auto& towers = elementParams->GetParams();
    if (towers.size() < 2) {
        OPENFHE_THROW("dense Boolean mask multiplication has no rescale prime");
    }
    const auto qLast = BigFixedPoint(towers.back()->GetModulus(), 0, false).scaleTo(128);
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(input->GetCryptoParameters());
    const auto targetScale = cryptoParams->GetScalingFactorBFP(level + 1);
    const auto plaintextScale = targetScale * qLast / input->GetScalingFactorBFP();
    const ZEncodingParams denseParams(BModeFull, 256, 256);
    auto plaintext = ZEncodingImpl::encodeC(CSlots(denseParams, values), elementParams, plaintextScale);
    maskCache.emplace(key, plaintext);
    return plaintext;
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::MultiplyMaskAndReduce(ConstCiphertext<DCRTPoly> input,
                                                              uint32_t wordBits,
                                                              uint32_t wordsPerCiphertext,
                                                              MaskKind kind,
                                                              uint32_t amount) {
    auto product = z->EvalMult(
        input,
        GetMaskPlaintext(input, wordBits, wordsPerCiphertext, kind, amount));
    z->ModReduceInPlace(product);
    return product;
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalKoggeStoneAdd(ConstCiphertext<DCRTPoly> left,
                                                          ConstCiphertext<DCRTPoly> right,
                                                          uint32_t wordBits,
                                                          uint32_t wordsPerCiphertext) {
    RequireCompatible(left, right, "Kogge-Stone input");
    if (wordBits == 0 || (wordBits & (wordBits - 1)) != 0) {
        OPENFHE_THROW("Kogge-Stone word width must be a nonzero power of two");
    }

    const uint32_t inputLevel = left->GetLevel();
    const uint32_t requiredDepth = KoggeStoneDepth(wordBits);
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(left->GetCryptoParameters());
    if (inputLevel + requiredDepth > cryptoParams->GetMultiplicativeDepth()) {
        std::ostringstream message;
        message << "Kogge-Stone needs " << requiredDepth << " levels from input level " << inputLevel
                << ", but multiplicative depth is " << cryptoParams->GetMultiplicativeDepth();
        OPENFHE_THROW(message.str());
    }

    auto generate = MultiplyAndReduce(left, right);
    auto leftAtGenerate = Align(left, generate->GetLevel());
    auto rightAtGenerate = Align(right, generate->GetLevel());
    auto propagate = Sub(Add(leftAtGenerate, rightAtGenerate), Double(generate));

    propagate = MultiplyMaskAndReduce(propagate, wordBits, wordsPerCiphertext, MaskKind::ClearWordHeads);
    generate = MultiplyMaskAndReduce(generate, wordBits, wordsPerCiphertext, MaskKind::Ones);

    for (uint32_t distance = 1; distance < wordBits; distance <<= 1) {
        const int32_t rotation = -static_cast<int32_t>(distance * wordsPerCiphertext);
        auto rotatedP = Rotate(propagate, rotation);
        auto rotatedG = Rotate(generate, rotation);
        auto nextP = MultiplyAndReduce(propagate, rotatedP);
        auto propagatedG = MultiplyAndReduce(propagate, rotatedG);
        auto generateAtNext = Align(generate, propagatedG->GetLevel());
        auto nextG = Add(generateAtNext, propagatedG);
        propagate = std::move(nextP);
        generate = std::move(nextG);
    }

    auto leftAtFinal = Align(left, generate->GetLevel());
    auto rightAtFinal = Align(right, generate->GetLevel());
    auto baseXor = Sub(Add(leftAtFinal, rightAtFinal), Double(generate));
    auto carryIn = Rotate(generate, -static_cast<int32_t>(wordsPerCiphertext));
    baseXor = MultiplyMaskAndReduce(baseXor, wordBits, wordsPerCiphertext, MaskKind::Ones);
    carryIn = MultiplyMaskAndReduce(carryIn, wordBits, wordsPerCiphertext, MaskKind::ClearWordHeads);
    return Add(baseXor, carryIn);
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalPulsarAdd(ConstCiphertext<DCRTPoly> left,
                                                     ConstCiphertext<DCRTPoly> right,
                                                     uint32_t wordBits,
                                                     uint32_t wordsPerCiphertext) {
    RequireCompatible(left, right, "PULSAR addition input");
    if (wordBits == 0 || (wordBits & (wordBits - 1)) != 0) {
        OPENFHE_THROW("PULSAR word width must be a nonzero power of two");
    }

    const uint32_t inputLevel = left->GetLevel();
    const uint32_t requiredDepth = KoggeStoneDepth(wordBits);
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(left->GetCryptoParameters());
    if (inputLevel + requiredDepth > cryptoParams->GetMultiplicativeDepth()) {
        std::ostringstream message;
        message << "PULSAR addition needs " << requiredDepth << " levels from input level " << inputLevel
                << ", but multiplicative depth is " << cryptoParams->GetMultiplicativeDepth();
        OPENFHE_THROW(message.str());
    }

    auto prefix = EvalPulsarAddBegin(left, right, wordBits, wordsPerCiphertext);
    while (prefix.nextDistance < wordBits) {
        EvalPulsarPrefixAdvance(prefix, wordBits, wordsPerCiphertext);
    }
    return EvalPulsarAddFinish(prefix, wordBits, wordsPerCiphertext);
}

DenseBooleanZImpl::WordResult DenseBooleanZImpl::EvalPulsarAddWithCarry(
    ConstCiphertext<DCRTPoly> left,
    ConstCiphertext<DCRTPoly> right,
    uint32_t wordBits,
    uint32_t wordsPerCiphertext) {
    auto prefix = EvalPulsarAddBegin(
        left, right, wordBits, wordsPerCiphertext);
    while (prefix.nextDistance < wordBits) {
        EvalPulsarPrefixAdvance(prefix, wordBits, wordsPerCiphertext);
    }
    auto value = EvalPulsarAddFinish(
        prefix, wordBits, wordsPerCiphertext);
    auto conjugate = z->EvalConjugateInC(prefix.packed);
    auto doubleCarry = Sub(prefix.packed, conjugate);
    z->EvalMultInPlaceInC(
        doubleCarry,
        BigComplex(BigFixedPoint::zero(),
                   BigFixedPoint::fromDouble(-1.0)),
        BigFixedPoint::one());
    auto carry = ScaleByHalf(doubleCarry);
    carry = MultiplyMaskAndReduce(
        carry, wordBits, wordsPerCiphertext, MaskKind::WordTails);
    for (uint32_t distance = 1; distance < wordBits; distance <<= 1) {
        carry = Add(carry, Rotate(
            carry, static_cast<int32_t>(distance * wordsPerCiphertext)));
    }
    return WordResult{std::move(value), std::move(carry)};
}

DenseBooleanZImpl::WordResult DenseBooleanZImpl::EvalPulsarAddWithCarryIn(
    ConstCiphertext<DCRTPoly> left,
    ConstCiphertext<DCRTPoly> right,
    ConstCiphertext<DCRTPoly> carryIn,
    uint32_t wordBits,
    uint32_t wordsPerCiphertext) {
    RequireCompatible(left, right, "PULSAR carry-input addition operands");
    RequireCompatible(left, carryIn, "PULSAR carry-input addition carry");
    auto carryHead = MultiplyMaskAndReduce(
        carryIn, wordBits, wordsPerCiphertext, MaskKind::WordHeads);
    const uint32_t entryLevel = carryHead->GetLevel();
    auto adjustedLeft = Align(left, entryLevel);
    auto adjustedRight = Align(right, entryLevel);
    auto adjustedCarry = Align(carryIn, entryLevel);
    auto prefix = EvalPulsarAddBegin(
        adjustedLeft, adjustedRight, wordBits, wordsPerCiphertext);
    while (prefix.nextDistance < wordBits) {
        EvalPulsarPrefixAdvance(prefix, wordBits, wordsPerCiphertext);
    }

    auto conjugate = z->EvalConjugateInC(prefix.packed);
    auto intervalPropagate = Add(prefix.packed, conjugate);
    auto doubleGenerate = Sub(prefix.packed, conjugate);
    z->EvalMultInPlaceInC(
        doubleGenerate,
        BigComplex(BigFixedPoint::zero(),
                   BigFixedPoint::fromDouble(-1.0)),
        BigFixedPoint::one());
    auto propagatedCarry = MultiplyAndReduce(
        intervalPropagate,
        Align(adjustedCarry, intervalPropagate->GetLevel()));
    auto doubleCarryOut = Add(
        Align(doubleGenerate, propagatedCarry->GetLevel()),
        Double(propagatedCarry));
    auto shiftedCarry = Rotate(
        doubleCarryOut, -static_cast<int32_t>(wordsPerCiphertext));
    auto wordCarryIn = MultiplyMaskAndReduce(
        shiftedCarry, wordBits, wordsPerCiphertext,
        MaskKind::HalfClearWordHeads);
    wordCarryIn = Add(
        wordCarryIn, Align(carryHead, wordCarryIn->GetLevel()));

    auto localSum = Align(prefix.localSum, wordCarryIn->GetLevel());
    auto outgoing = Align(doubleCarryOut, wordCarryIn->GetLevel());
    auto value = Sub(Add(localSum, wordCarryIn), outgoing);

    auto carry = ScaleByHalf(doubleCarryOut);
    carry = MultiplyMaskAndReduce(
        carry, wordBits, wordsPerCiphertext, MaskKind::WordTails);
    for (uint32_t distance = 1; distance < wordBits; distance <<= 1) {
        carry = Add(carry, Rotate(
            carry, static_cast<int32_t>(distance * wordsPerCiphertext)));
    }
    return WordResult{std::move(value), std::move(carry)};
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalPulsarSub(
    ConstCiphertext<DCRTPoly> left,
    ConstCiphertext<DCRTPoly> right,
    uint32_t wordBits,
    uint32_t wordsPerCiphertext) {
    RequireCompatible(left, right, "PULSAR subtraction input");
    const uint32_t inputLevel = left->GetLevel();
    const uint32_t requiredDepth = KoggeStoneDepth(wordBits);
    const auto cryptoParams =
        std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(
            left->GetCryptoParameters());
    if (inputLevel + requiredDepth >
        cryptoParams->GetMultiplicativeDepth()) {
        OPENFHE_THROW("PULSAR subtraction has insufficient levels");
    }

    auto prefix = EvalPulsarBorrowBegin(
        left, right, wordBits, wordsPerCiphertext);
    while (prefix.nextDistance < wordBits) {
        EvalPulsarPrefixAdvance(prefix, wordBits, wordsPerCiphertext);
    }
    return EvalPulsarSubFinish(prefix, wordBits, wordsPerCiphertext);
}

DenseBooleanZImpl::WordResult DenseBooleanZImpl::EvalPulsarSubWithBorrow(
    ConstCiphertext<DCRTPoly> left,
    ConstCiphertext<DCRTPoly> right,
    uint32_t wordBits,
    uint32_t wordsPerCiphertext) {
    auto prefix = EvalPulsarBorrowBegin(
        left, right, wordBits, wordsPerCiphertext);
    while (prefix.nextDistance < wordBits) {
        EvalPulsarPrefixAdvance(prefix, wordBits, wordsPerCiphertext);
    }
    auto value = EvalPulsarSubFinish(
        prefix, wordBits, wordsPerCiphertext);
    auto conjugate = z->EvalConjugateInC(prefix.packed);
    auto doubleBorrow = Sub(prefix.packed, conjugate);
    z->EvalMultInPlaceInC(
        doubleBorrow,
        BigComplex(BigFixedPoint::zero(),
                   BigFixedPoint::fromDouble(-1.0)),
        BigFixedPoint::one());
    auto borrow = ScaleByHalf(doubleBorrow);
    borrow = MultiplyMaskAndReduce(
        borrow, wordBits, wordsPerCiphertext, MaskKind::WordTails);
    for (uint32_t distance = 1; distance < wordBits; distance <<= 1) {
        borrow = Add(borrow, Rotate(
            borrow, static_cast<int32_t>(distance * wordsPerCiphertext)));
    }
    return WordResult{std::move(value), std::move(borrow)};
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalPulsarGreaterEqual(
    ConstCiphertext<DCRTPoly> left,
    ConstCiphertext<DCRTPoly> right,
    uint32_t wordBits,
    uint32_t wordsPerCiphertext) {
    RequireCompatible(left, right, "PULSAR comparison input");
    const uint32_t inputLevel = left->GetLevel();
    const uint32_t requiredDepth = KoggeStoneDepth(wordBits);
    const auto cryptoParams =
        std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(
            left->GetCryptoParameters());
    if (inputLevel + requiredDepth >
        cryptoParams->GetMultiplicativeDepth()) {
        OPENFHE_THROW("PULSAR comparison has insufficient levels");
    }

    auto prefix = EvalPulsarBorrowBegin(
        left, right, wordBits, wordsPerCiphertext);
    while (prefix.nextDistance < wordBits) {
        EvalPulsarPrefixAdvance(prefix, wordBits, wordsPerCiphertext);
    }
    return EvalPulsarGreaterEqualFinish(
        prefix, wordBits, wordsPerCiphertext);
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalPulsarEqual(
    ConstCiphertext<DCRTPoly> left,
    ConstCiphertext<DCRTPoly> right,
    uint32_t wordBits,
    uint32_t wordsPerCiphertext) {
    RequireCompatible(left, right, "PULSAR equality input");
    if (wordBits == 0 || (wordBits & (wordBits - 1)) != 0) {
        OPENFHE_THROW("PULSAR word width must be a nonzero power of two");
    }

    const uint32_t inputLevel = left->GetLevel();
    const uint32_t requiredDepth = KoggeStoneDepth(wordBits);
    const auto cryptoParams =
        std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(
            left->GetCryptoParameters());
    if (inputLevel + requiredDepth >
        cryptoParams->GetMultiplicativeDepth()) {
        OPENFHE_THROW("PULSAR equality has insufficient levels");
    }

    auto product = MultiplyAndReduce(left, right);
    auto leftAtProduct = Align(left, product->GetLevel());
    auto rightAtProduct = Align(right, product->GetLevel());
    auto mismatch = Sub(
        Add(leftAtProduct, rightAtProduct), Double(product));
    auto localPropagate = OneMinus(mismatch);
    auto halfPropagate = MultiplyMaskAndReduce(
        localPropagate, wordBits, wordsPerCiphertext,
        MaskKind::HalfClearWordHeads);
    auto imaginaryMismatch = MultiplyMaskAndReduce(
        mismatch, wordBits, wordsPerCiphertext,
        MaskKind::ImaginaryOnes);

    PrefixState prefix;
    prefix.packed = Add(halfPropagate, imaginaryMismatch);
    prefix.localPropagate = std::move(localPropagate);
    prefix.nextDistance = 1;
    while (prefix.nextDistance < wordBits) {
        EvalPulsarPrefixAdvance(prefix, wordBits, wordsPerCiphertext);
    }

    auto conjugate = z->EvalConjugateInC(prefix.packed);
    auto doubleMismatch = Sub(prefix.packed, conjugate);
    z->EvalMultInPlaceInC(
        doubleMismatch,
        BigComplex(BigFixedPoint::zero(),
                   BigFixedPoint::fromDouble(-1.0)),
        BigFixedPoint::one());
    auto equal = OneMinus(ScaleByHalf(doubleMismatch));
    equal = MultiplyMaskAndReduce(
        equal, wordBits, wordsPerCiphertext, MaskKind::WordTails);
    for (uint32_t distance = 1; distance < wordBits; distance <<= 1) {
        equal = Add(
            equal,
            Rotate(equal,
                   static_cast<int32_t>(distance * wordsPerCiphertext)));
    }
    return equal;
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalBooleanAnd(
    ConstCiphertext<DCRTPoly> left,
    ConstCiphertext<DCRTPoly> right) {
    const uint32_t level = std::max(left->GetLevel(), right->GetLevel());
    return MultiplyAndReduce(Align(left, level), Align(right, level));
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalBooleanXor(
    ConstCiphertext<DCRTPoly> left,
    ConstCiphertext<DCRTPoly> right) {
    const uint32_t level = std::max(left->GetLevel(), right->GetLevel());
    auto alignedLeft = Align(left, level);
    auto alignedRight = Align(right, level);
    auto product = MultiplyAndReduce(alignedLeft, alignedRight);
    alignedLeft = Align(alignedLeft, product->GetLevel());
    alignedRight = Align(alignedRight, product->GetLevel());
    return Sub(Add(alignedLeft, alignedRight), Double(product));
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalBooleanNot(
    ConstCiphertext<DCRTPoly> input) {
    return OneMinus(input);
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalBooleanChoose(
    ConstCiphertext<DCRTPoly> condition,
    ConstCiphertext<DCRTPoly> trueValue,
    ConstCiphertext<DCRTPoly> falseValue) {
    const uint32_t level = std::max(
        {condition->GetLevel(), trueValue->GetLevel(),
         falseValue->GetLevel()});
    auto alignedCondition = Align(condition, level);
    auto alignedTrue = Align(trueValue, level);
    auto alignedFalse = Align(falseValue, level);
    auto selectedDifference = MultiplyAndReduce(
        alignedCondition, Sub(alignedTrue, alignedFalse));
    return Add(Align(alignedFalse, selectedDifference->GetLevel()),
               selectedDifference);
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalBooleanMajority(
    ConstCiphertext<DCRTPoly> first,
    ConstCiphertext<DCRTPoly> second,
    ConstCiphertext<DCRTPoly> third) {
    const uint32_t level = std::max(
        {first->GetLevel(), second->GetLevel(), third->GetLevel()});
    auto alignedFirst = Align(first, level);
    auto alignedSecond = Align(second, level);
    auto firstAndSecond = MultiplyAndReduce(alignedFirst, alignedSecond);
    alignedFirst = Align(alignedFirst, firstAndSecond->GetLevel());
    alignedSecond = Align(alignedSecond, firstAndSecond->GetLevel());
    auto firstXorSecond = Sub(
        Add(alignedFirst, alignedSecond), Double(firstAndSecond));
    auto alignedThird = Align(third, firstXorSecond->GetLevel());
    auto thirdAndDifference = MultiplyAndReduce(
        alignedThird, firstXorSecond);
    return Add(Align(firstAndSecond, thirdAndDifference->GetLevel()),
               thirdAndDifference);
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalBooleanSelect(
    ConstCiphertext<DCRTPoly> falseValue,
    ConstCiphertext<DCRTPoly> trueValue,
    ConstCiphertext<DCRTPoly> condition) {
    RequireCompatible(falseValue, trueValue, "Boolean selection values");
    RequireCompatible(falseValue, condition, "Boolean selection condition");
    auto difference = Sub(trueValue, falseValue);
    auto selectedDifference = MultiplyAndReduce(difference, condition);
    auto base = Align(falseValue, selectedDifference->GetLevel());
    return Add(base, selectedDifference);
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalRotateWords(
    ConstCiphertext<DCRTPoly> input,
    int32_t offset) {
    if (!input) {
        OPENFHE_THROW("word rotation received a nil ciphertext");
    }
    return Rotate(input, offset);
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalRotateBitsRight(
    ConstCiphertext<DCRTPoly> input,
    uint32_t amount,
    uint32_t wordBits,
    uint32_t wordsPerCiphertext) {
    if (!input || wordBits == 0 || amount >= wordBits) {
        OPENFHE_THROW("invalid right-rotation arguments");
    }
    if (amount == 0) {
        return input->Clone();
    }
    return Rotate(
        input, static_cast<int32_t>(amount * wordsPerCiphertext));
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalShiftBitsRight(
    ConstCiphertext<DCRTPoly> input,
    uint32_t amount,
    uint32_t wordBits,
    uint32_t wordsPerCiphertext) {
    if (!input || wordBits == 0 || amount == 0 || amount >= wordBits) {
        OPENFHE_THROW("invalid logical-right-shift arguments");
    }
    auto rotated = Rotate(
        input, static_cast<int32_t>(amount * wordsPerCiphertext));
    return MultiplyMaskAndReduce(
        rotated, wordBits, wordsPerCiphertext,
        MaskKind::LogicalRightShift, amount);
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalBitClean(
    ConstCiphertext<DCRTPoly> input) {
    if (!input) {
        OPENFHE_THROW("Bit Clean received a nil ciphertext");
    }
    auto square = MultiplyAndReduce(input, input);
    auto inputAtSquare = Align(input, square->GetLevel());
    auto cube = MultiplyAndReduce(square, inputAtSquare);
    auto squareAtOutput = Align(square, cube->GetLevel());
    auto threeSquare = Add(Double(squareAtOutput), squareAtOutput);
    auto twoCube = Double(cube);
    return Sub(threeSquare, twoCube);
}

DenseBooleanZImpl::PrefixState DenseBooleanZImpl::EvalPulsarAddBegin(
    ConstCiphertext<DCRTPoly> left,
    ConstCiphertext<DCRTPoly> right,
    uint32_t wordBits,
    uint32_t wordsPerCiphertext) {
    RequireCompatible(left, right, "PULSAR addition input");
    if (wordBits == 0 || (wordBits & (wordBits - 1)) != 0) {
        OPENFHE_THROW("PULSAR word width must be a nonzero power of two");
    }

    auto localGenerate = MultiplyAndReduce(left, right);
    auto leftAtGenerate = Align(left, localGenerate->GetLevel());
    auto rightAtGenerate = Align(right, localGenerate->GetLevel());
    auto localSum = Add(leftAtGenerate, rightAtGenerate);
    auto localPropagate = Sub(localSum, Double(localGenerate));
    auto halfPropagate = MultiplyMaskAndReduce(
        localPropagate, wordBits, wordsPerCiphertext, MaskKind::HalfClearWordHeads);
    auto imaginaryGenerate = MultiplyMaskAndReduce(
        localGenerate, wordBits, wordsPerCiphertext, MaskKind::ImaginaryOnes);

    PrefixState prefix;
    prefix.packed = Add(halfPropagate, imaginaryGenerate);
    prefix.localPropagate = std::move(localPropagate);
    prefix.localSum = std::move(localSum);
    prefix.nextDistance = 1;
    return prefix;
}

DenseBooleanZImpl::PrefixState DenseBooleanZImpl::EvalPulsarBorrowBegin(
    ConstCiphertext<DCRTPoly> left,
    ConstCiphertext<DCRTPoly> right,
    uint32_t wordBits,
    uint32_t wordsPerCiphertext) {
    RequireCompatible(left, right, "PULSAR borrow input");
    if (wordBits == 0 || (wordBits & (wordBits - 1)) != 0) {
        OPENFHE_THROW("PULSAR word width must be a nonzero power of two");
    }

    auto product = MultiplyAndReduce(left, right);
    auto leftAtProduct = Align(left, product->GetLevel());
    auto rightAtProduct = Align(right, product->GetLevel());
    auto inputSum = Add(leftAtProduct, rightAtProduct);
    auto localDifference = Sub(leftAtProduct, rightAtProduct);
    auto localPropagate = OneMinus(Sub(inputSum, Double(product)));
    auto localGenerate = Sub(rightAtProduct, product);
    auto halfPropagate = MultiplyMaskAndReduce(
        localPropagate, wordBits, wordsPerCiphertext,
        MaskKind::HalfClearWordHeads);
    auto imaginaryGenerate = MultiplyMaskAndReduce(
        localGenerate, wordBits, wordsPerCiphertext,
        MaskKind::ImaginaryOnes);

    PrefixState prefix;
    prefix.packed = Add(halfPropagate, imaginaryGenerate);
    prefix.localPropagate = std::move(localPropagate);
    prefix.localSum = std::move(localDifference);
    prefix.nextDistance = 1;
    return prefix;
}

void DenseBooleanZImpl::EvalPulsarPrefixAdvance(PrefixState& prefix,
                                                uint32_t wordBits,
                                                uint32_t wordsPerCiphertext) {
    if (!prefix.packed) {
        OPENFHE_THROW("PULSAR prefix advance received an incomplete state");
    }
    if (prefix.nextDistance >= wordBits) {
        OPENFHE_THROW("PULSAR prefix advance exceeds the word width");
    }

    const int32_t rotation =
        -static_cast<int32_t>(prefix.nextDistance * wordsPerCiphertext);
    auto rotatedState = Rotate(prefix.packed, rotation);
    auto conjugate = z->EvalConjugateInC(prefix.packed);
    auto currentPropagate = Add(prefix.packed, conjugate);
    auto currentImaginaryGenerate = MultiplyMaskAndReduce(
        Sub(prefix.packed, conjugate), wordBits, wordsPerCiphertext,
        MaskKind::HalfOnes);
    auto nextState = MultiplyAndReduce(currentPropagate, rotatedState);
    prefix.packed = Add(nextState, currentImaginaryGenerate);
    prefix.nextDistance <<= 1;
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalPulsarAddFinish(
    const PrefixState& prefix,
    uint32_t wordBits,
    uint32_t wordsPerCiphertext) {
    if (!prefix.packed || prefix.nextDistance != wordBits) {
        OPENFHE_THROW("PULSAR addition finish requires a complete prefix state");
    }

    auto conjugate = z->EvalConjugateInC(prefix.packed);
    auto doubleGenerate = Sub(prefix.packed, conjugate);
    z->EvalMultInPlaceInC(
        doubleGenerate,
        BigComplex(BigFixedPoint::zero(), BigFixedPoint::fromDouble(-1.0)),
        BigFixedPoint::one());
    auto doubleCarryIn = Rotate(
        doubleGenerate, -static_cast<int32_t>(wordsPerCiphertext));
    auto carryIn = MultiplyMaskAndReduce(
        doubleCarryIn, wordBits, wordsPerCiphertext,
        MaskKind::HalfClearWordHeads);
    if (prefix.localSum) {
        auto localSumAtOutput = Align(prefix.localSum, carryIn->GetLevel());
        auto doubleGenerateAtOutput = Align(doubleGenerate, carryIn->GetLevel());
        return Sub(Add(localSumAtOutput, carryIn), doubleGenerateAtOutput);
    }
    if (!prefix.localPropagate) {
        OPENFHE_THROW("refreshed prefix state is missing the local propagate value");
    }
    auto localAtCarry = Align(prefix.localPropagate, carryIn->GetLevel());
    auto product = MultiplyAndReduce(localAtCarry, carryIn);
    auto localAtOutput = Align(localAtCarry, product->GetLevel());
    auto carryAtOutput = Align(carryIn, product->GetLevel());
    return Sub(Add(localAtOutput, carryAtOutput), Double(product));
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalPulsarSubFinish(
    const PrefixState& prefix,
    uint32_t wordBits,
    uint32_t wordsPerCiphertext) {
    if (!prefix.packed || prefix.nextDistance != wordBits) {
        OPENFHE_THROW(
            "PULSAR subtraction finish requires a complete prefix state");
    }

    auto conjugate = z->EvalConjugateInC(prefix.packed);
    auto doubleBorrowOut = Sub(prefix.packed, conjugate);
    z->EvalMultInPlaceInC(
        doubleBorrowOut,
        BigComplex(BigFixedPoint::zero(),
                   BigFixedPoint::fromDouble(-1.0)),
        BigFixedPoint::one());
    auto doubleBorrowIn = Rotate(
        doubleBorrowOut, -static_cast<int32_t>(wordsPerCiphertext));
    auto borrowIn = MultiplyMaskAndReduce(
        doubleBorrowIn, wordBits, wordsPerCiphertext,
        MaskKind::HalfClearWordHeads);

    if (prefix.localSum) {
        auto difference = Align(prefix.localSum, borrowIn->GetLevel());
        auto outgoing = Align(doubleBorrowOut, borrowIn->GetLevel());
        return Add(Sub(difference, borrowIn), outgoing);
    }
    if (!prefix.localPropagate) {
        OPENFHE_THROW(
            "refreshed borrow state is missing the local propagate value");
    }
    auto localXor = OneMinus(prefix.localPropagate);
    localXor = Align(localXor, borrowIn->GetLevel());
    auto product = MultiplyAndReduce(localXor, borrowIn);
    auto xorAtOutput = Align(localXor, product->GetLevel());
    auto borrowAtOutput = Align(borrowIn, product->GetLevel());
    return Sub(Add(xorAtOutput, borrowAtOutput), Double(product));
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalPulsarGreaterEqualFinish(
    const PrefixState& prefix,
    uint32_t wordBits,
    uint32_t wordsPerCiphertext) {
    if (!prefix.packed || prefix.nextDistance != wordBits) {
        OPENFHE_THROW(
            "PULSAR comparison finish requires a complete prefix state");
    }

    auto conjugate = z->EvalConjugateInC(prefix.packed);
    auto doubleBorrow = Sub(prefix.packed, conjugate);
    z->EvalMultInPlaceInC(
        doubleBorrow,
        BigComplex(BigFixedPoint::zero(),
                   BigFixedPoint::fromDouble(-1.0)),
        BigFixedPoint::one());
    auto borrow = ScaleByHalf(doubleBorrow);
    auto greaterEqual = OneMinus(borrow);
    greaterEqual = MultiplyMaskAndReduce(
        greaterEqual, wordBits, wordsPerCiphertext,
        MaskKind::WordTails);
    for (uint32_t distance = 1; distance < wordBits; distance <<= 1) {
        greaterEqual = Add(
            greaterEqual,
            Rotate(greaterEqual,
                   static_cast<int32_t>(distance * wordsPerCiphertext)));
    }
    return greaterEqual;
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::EvalPrefixBootEncode(
    const PrefixState& prefix,
    uint32_t wordBits,
    uint32_t wordsPerCiphertext) {
    if (!prefix.packed || !prefix.localPropagate) {
        OPENFHE_THROW("Prefix Boot encoding received an incomplete state");
    }

    auto conjugate = z->EvalConjugateInC(prefix.packed);
    auto doublePropagate = Double(Add(prefix.packed, conjugate));
    auto doubleGenerate = Sub(prefix.packed, conjugate);
    z->EvalMultInPlaceInC(
        doubleGenerate,
        BigComplex(BigFixedPoint::zero(), BigFixedPoint::fromDouble(-1.0)),
        BigFixedPoint::one());
    auto local = Align(prefix.localPropagate, prefix.packed->GetLevel());
    auto joint = Add(Sub(doubleGenerate, doublePropagate), local);
    z->EvalAddInPlaceInC(joint, BigComplex(BigFixedPoint::one()));

    // u=2(G-P)+p is in {-1,...,3}. Store the equivalent residue u+1 in
    // {0,...,4}; the functional refresh performs the division by eight by
    // changing scaling metadata, as in the radix functional bootstraps.
    (void)wordBits;
    (void)wordsPerCiphertext;
    return joint;
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::PackRealChannels(
    ConstCiphertext<DCRTPoly> first,
    ConstCiphertext<DCRTPoly> second) {
    RequireCompatible(first, second, "real-channel packing");
    auto imaginarySecond = second->Clone();
    z->EvalMultInPlaceInC(
        imaginarySecond,
        BigComplex(BigFixedPoint::zero(), BigFixedPoint::one()),
        BigFixedPoint::one());
    return Add(first, imaginarySecond);
}

std::array<Ciphertext<DCRTPoly>, 2> DenseBooleanZImpl::SplitRealChannels(
    ConstCiphertext<DCRTPoly> packed) {
    if (!packed) {
        OPENFHE_THROW("real-channel split received a nil ciphertext");
    }
    auto conjugate = z->EvalConjugateInC(packed);
    auto first = Add(packed, conjugate);
    auto second = Sub(packed, conjugate);
    z->EvalMultInPlaceInC(
        second,
        BigComplex(BigFixedPoint::zero(), BigFixedPoint::fromDouble(-1.0)),
        BigFixedPoint::one());
    return {first, second};
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::ScaleByTwo(
    ConstCiphertext<DCRTPoly> input) {
    if (!input) {
        OPENFHE_THROW("scale-by-two received a nil ciphertext");
    }
    return Double(input);
}

Ciphertext<DCRTPoly> DenseBooleanZImpl::ScaleByHalf(
    ConstCiphertext<DCRTPoly> input) {
    if (!input) {
        OPENFHE_THROW("scale-by-half received a nil ciphertext");
    }
    auto output = input->Clone();
    z->EvalMultInPlaceInC(
        output,
        BigComplex(BigFixedPoint::fromDouble(0.5)),
        BigFixedPoint::fromDouble(2.0));
    return output;
}

}  // namespace lbcrypto
