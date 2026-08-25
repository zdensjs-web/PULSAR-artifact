#ifndef SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_DENSE_BOOLEAN_H_
#define SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_DENSE_BOOLEAN_H_

#include "scheme/ckksrns/z-leveledshe.h"

#include <map>
#include <array>
#include <tuple>
#include <utility>

namespace lbcrypto {

// Evaluates guard-free, bit-major Boolean circuits while preserving the
// BigFixedPoint metadata required by the FHE-SIMD-ALU bootstrap code.
class DenseBooleanZImpl {
public:
    explicit DenseBooleanZImpl(LeveledZ leveled) : z(std::move(leveled)) {}

    struct PrefixState {
        Ciphertext<DCRTPoly> packed;
        Ciphertext<DCRTPoly> localPropagate;
        Ciphertext<DCRTPoly> localSum;
        uint32_t nextDistance = 1;
    };

    struct WordResult {
        Ciphertext<DCRTPoly> value;
        Ciphertext<DCRTPoly> flag;
    };

    Ciphertext<DCRTPoly> EvalKoggeStoneAdd(ConstCiphertext<DCRTPoly> left,
                                           ConstCiphertext<DCRTPoly> right,
                                           uint32_t wordBits,
                                           uint32_t wordsPerCiphertext);

    Ciphertext<DCRTPoly> EvalPulsarAdd(ConstCiphertext<DCRTPoly> left,
                                      ConstCiphertext<DCRTPoly> right,
                                      uint32_t wordBits,
                                       uint32_t wordsPerCiphertext);

    WordResult EvalPulsarAddWithCarry(
        ConstCiphertext<DCRTPoly> left,
        ConstCiphertext<DCRTPoly> right,
        uint32_t wordBits,
        uint32_t wordsPerCiphertext);

    WordResult EvalPulsarAddWithCarryIn(
        ConstCiphertext<DCRTPoly> left,
        ConstCiphertext<DCRTPoly> right,
        ConstCiphertext<DCRTPoly> carryIn,
        uint32_t wordBits,
        uint32_t wordsPerCiphertext);

    Ciphertext<DCRTPoly> EvalPulsarSub(ConstCiphertext<DCRTPoly> left,
                                      ConstCiphertext<DCRTPoly> right,
                                      uint32_t wordBits,
                                      uint32_t wordsPerCiphertext);

    WordResult EvalPulsarSubWithBorrow(
        ConstCiphertext<DCRTPoly> left,
        ConstCiphertext<DCRTPoly> right,
        uint32_t wordBits,
        uint32_t wordsPerCiphertext);

    Ciphertext<DCRTPoly> EvalPulsarGreaterEqual(
        ConstCiphertext<DCRTPoly> left,
        ConstCiphertext<DCRTPoly> right,
        uint32_t wordBits,
        uint32_t wordsPerCiphertext);

    Ciphertext<DCRTPoly> EvalPulsarEqual(
        ConstCiphertext<DCRTPoly> left,
        ConstCiphertext<DCRTPoly> right,
        uint32_t wordBits,
        uint32_t wordsPerCiphertext);

    Ciphertext<DCRTPoly> EvalBooleanAnd(ConstCiphertext<DCRTPoly> left,
                                       ConstCiphertext<DCRTPoly> right);

    Ciphertext<DCRTPoly> EvalBooleanXor(ConstCiphertext<DCRTPoly> left,
                                       ConstCiphertext<DCRTPoly> right);

    Ciphertext<DCRTPoly> EvalBooleanNot(ConstCiphertext<DCRTPoly> input);

    Ciphertext<DCRTPoly> EvalBooleanChoose(
        ConstCiphertext<DCRTPoly> condition,
        ConstCiphertext<DCRTPoly> trueValue,
        ConstCiphertext<DCRTPoly> falseValue);

    Ciphertext<DCRTPoly> EvalBooleanMajority(
        ConstCiphertext<DCRTPoly> first,
        ConstCiphertext<DCRTPoly> second,
        ConstCiphertext<DCRTPoly> third);

    Ciphertext<DCRTPoly> EvalBooleanSelect(
        ConstCiphertext<DCRTPoly> falseValue,
        ConstCiphertext<DCRTPoly> trueValue,
        ConstCiphertext<DCRTPoly> condition);

    Ciphertext<DCRTPoly> EvalRotateWords(
        ConstCiphertext<DCRTPoly> input,
        int32_t offset);

    Ciphertext<DCRTPoly> EvalRotateBitsRight(
        ConstCiphertext<DCRTPoly> input,
        uint32_t amount,
        uint32_t wordBits,
        uint32_t wordsPerCiphertext);

    Ciphertext<DCRTPoly> EvalShiftBitsRight(
        ConstCiphertext<DCRTPoly> input,
        uint32_t amount,
        uint32_t wordBits,
        uint32_t wordsPerCiphertext);

    Ciphertext<DCRTPoly> EvalBitClean(ConstCiphertext<DCRTPoly> input);

    PrefixState EvalPulsarAddBegin(ConstCiphertext<DCRTPoly> left,
                                   ConstCiphertext<DCRTPoly> right,
                                   uint32_t wordBits,
                                   uint32_t wordsPerCiphertext);

    PrefixState EvalPulsarBorrowBegin(ConstCiphertext<DCRTPoly> left,
                                      ConstCiphertext<DCRTPoly> right,
                                      uint32_t wordBits,
                                      uint32_t wordsPerCiphertext);

    void EvalPulsarPrefixAdvance(PrefixState& prefix,
                                 uint32_t wordBits,
                                 uint32_t wordsPerCiphertext);

    Ciphertext<DCRTPoly> EvalPulsarAddFinish(const PrefixState& prefix,
                                             uint32_t wordBits,
                                             uint32_t wordsPerCiphertext);

    Ciphertext<DCRTPoly> EvalPulsarSubFinish(const PrefixState& prefix,
                                             uint32_t wordBits,
                                             uint32_t wordsPerCiphertext);

    Ciphertext<DCRTPoly> EvalPulsarGreaterEqualFinish(
        const PrefixState& prefix,
        uint32_t wordBits,
        uint32_t wordsPerCiphertext);

    Ciphertext<DCRTPoly> EvalPrefixBootEncode(const PrefixState& prefix,
                                              uint32_t wordBits,
                                              uint32_t wordsPerCiphertext);

    Ciphertext<DCRTPoly> PackRealChannels(ConstCiphertext<DCRTPoly> first,
                                          ConstCiphertext<DCRTPoly> second);

    std::array<Ciphertext<DCRTPoly>, 2> SplitRealChannels(
        ConstCiphertext<DCRTPoly> packed);

    Ciphertext<DCRTPoly> ScaleByTwo(ConstCiphertext<DCRTPoly> input);

    Ciphertext<DCRTPoly> ScaleByHalf(ConstCiphertext<DCRTPoly> input);

    Ciphertext<DCRTPoly> AdjustToLevel(ConstCiphertext<DCRTPoly> input,
                                       uint32_t level);

    static constexpr uint32_t KoggeStoneDepth(uint32_t wordBits) {
        uint32_t layers = 0;
        for (uint32_t distance = 1; distance < wordBits; distance <<= 1) {
            ++layers;
        }
        return layers + 3;
    }

private:
    enum class MaskKind : uint32_t {
        Ones,
        ClearWordHeads,
        HalfOnes,
        HalfClearWordHeads,
        QuarterOnes,
        ImaginaryOnes,
        WordHeads,
        WordTails,
        LogicalRightShift,
    };

    using MaskKey =
        std::tuple<uint32_t, uint32_t, uint32_t, MaskKind, uint32_t>;

    Ciphertext<DCRTPoly> Add(ConstCiphertext<DCRTPoly> left, ConstCiphertext<DCRTPoly> right);
    Ciphertext<DCRTPoly> Sub(ConstCiphertext<DCRTPoly> left, ConstCiphertext<DCRTPoly> right);
    Ciphertext<DCRTPoly> Double(ConstCiphertext<DCRTPoly> input);
    Ciphertext<DCRTPoly> MultiplyAndReduce(ConstCiphertext<DCRTPoly> left,
                                           ConstCiphertext<DCRTPoly> right);
    Ciphertext<DCRTPoly> MultiplyMaskAndReduce(ConstCiphertext<DCRTPoly> input,
                                               uint32_t wordBits,
                                               uint32_t wordsPerCiphertext,
                                               MaskKind kind,
                                               uint32_t amount = 0);
    Ciphertext<DCRTPoly> Rotate(ConstCiphertext<DCRTPoly> input, int32_t offset);
    Ciphertext<DCRTPoly> Align(ConstCiphertext<DCRTPoly> input, uint32_t level);
    Ciphertext<DCRTPoly> OneMinus(ConstCiphertext<DCRTPoly> input);
    Plaintext GetMaskPlaintext(ConstCiphertext<DCRTPoly> input,
                               uint32_t wordBits,
                               uint32_t wordsPerCiphertext,
                               MaskKind kind,
                               uint32_t amount = 0);
    static void RequireCompatible(ConstCiphertext<DCRTPoly> left,
                                  ConstCiphertext<DCRTPoly> right,
                                  const char* operation);

    LeveledZ z;
    std::map<MaskKey, Plaintext> maskCache;
};

using DenseBooleanZ = std::shared_ptr<DenseBooleanZImpl>;

}  // namespace lbcrypto

#endif  // SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_DENSE_BOOLEAN_H_
