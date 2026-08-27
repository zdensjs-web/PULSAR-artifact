#ifndef SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_DENSE_RADIX_MULTIPLIER_H_
#define SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_DENSE_RADIX_MULTIPLIER_H_

#include "scheme/ckksrns/z-fhe.h"
#include "scheme/ckksrns/z-leveledshe.h"

#include <chrono>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace lbcrypto {

struct DenseRadixMultiplierTiming {
    double inputPack = 0.0;
    double convolution = 0.0;
    double lazyCarry[4] = {0.0, 0.0, 0.0, 0.0};
    double sharedFBT = 0.0;
    double prefix = 0.0;
    double correction = 0.0;
    double outputPack = 0.0;
    double intermediateB2B = 0.0;
    double addCorrection = 0.0;
    double finalB2B = 0.0;
    double setupExcluded = 0.0;

    double CircuitSeconds() const;
    double OnlineSeconds() const;
};

struct DenseRadixMultiplierLevels {
    uint32_t stableInput = 0;
    uint32_t packedRadix = 0;
    uint32_t convolution = 0;
    uint32_t lazyCarry[4] = {0, 0, 0, 0};
    uint32_t sharedFBTRaw = 0;
    uint32_t sharedFBT = 0;
    uint32_t prefix = 0;
    uint32_t correction = 0;
    uint32_t rawBoolean = 0;
    uint32_t intermediateBoolean = 0;
    uint32_t finalBoolean = 0;
};

struct DenseRadixFullProduct {
    CiphertextGroup low;
    CiphertextGroup high;
};

// Exact OpenFHE port of the verified Lattigo Boolean/radix-16 multiplier.
// The public entry point accepts and returns two dense, guard-free Boolean
// ciphertexts. The word width determines every layout dimension; every
// supported configuration occupies all 32768 complex slots.
class DenseRadixMultiplierZImpl {
public:
    DenseRadixMultiplierZImpl(uint32_t wordBits,
                              LeveledZ leveled,
                              FHEZ functional);

    CiphertextGroup EvalMultiply(CiphertextGroup left, CiphertextGroup right,
                                 uint32_t maximumB2BInputLevel);

    DenseRadixFullProduct EvalFullMultiply(
        CiphertextGroup left,
        CiphertextGroup right,
        uint32_t maximumB2BInputLevel);

    // Public-constant counterpart of EvalMultiply. The scalar is supplied as
    // little-endian radix-16 digits so the core remains independent of any
    // application-side large-integer type. There are no scalar-specific fast
    // paths: every value follows the same PolyCMult-compatible DFT pipeline.
    CiphertextGroup EvalScaleMultiply(
        CiphertextGroup input,
        const std::vector<uint32_t>& multiplierDigits,
        uint32_t maximumB2BInputLevel);

    DenseRadixFullProduct EvalFullScaleMultiply(
        CiphertextGroup input,
        const std::vector<uint32_t>& multiplierDigits,
        uint32_t maximumB2BInputLevel);

    // Exact unsigned public division descriptor used by libdivide and by the
    // validated Lattigo full-product implementation. An empty magic vector
    // denotes the power-of-two descriptor; otherwise floor(X/d) is recovered
    // from the high half of X*magic, with the optional add-marker correction.
    CiphertextGroup EvalScaleDivide(
        CiphertextGroup input,
        const std::vector<uint32_t>& magicDigits,
        uint32_t shift,
        bool addMarker,
        uint32_t maximumB2BInputLevel);

    using StageObserver = std::function<void(
        const std::string&, ConstCiphertext<DCRTPoly>)>;

    void SetStageObserver(StageObserver observer) {
        stageObserver = std::move(observer);
    }

    static void DecodeLazyCarryReference(std::vector<int64_t>& digits);

    const DenseRadixMultiplierTiming& GetTiming() const {
        return timing;
    }

    const DenseRadixMultiplierLevels& GetLevels() const {
        return levels;
    }

    // Extra rotations used by the two sparse linear maps, radix DFTs and
    // radix carry prefix. FHEZ bootstrap rotations are generated separately.
    static std::vector<int32_t> RotationIndices(uint32_t wordBits);
    static std::vector<int32_t> ScaleDivisionRotationIndices(
        uint32_t wordBits,
        uint32_t shift,
        bool addMarker);

    uint32_t WordBits() const { return wordBits; }
    uint32_t NativeBatch() const { return booleanBatch; }
    uint32_t AggregateBatch() const { return radixBatch; }
    uint32_t RadixDigits() const { return radixDigits; }
    uint32_t LazyCarryPasses() const { return lazyCarryPasses; }
    uint32_t PrefixLayers() const { return prefixLayers; }
    uint32_t RawBoundaryLevel() const { return rawBoundaryLevel; }

private:
    static constexpr uint32_t kSlots = 32768;
    // Lattigo reports remaining levels while OpenFHE reports consumed levels.
    // The multiplier starts at level 16. The full-product FBT reaches level
    // 18 because its delimiter consumes one level before the common tail.
    static constexpr uint32_t kCircuitInputLevel = 16;
    static constexpr uint32_t kSharedFBTOutputLevel = 18;

    // Lattigo's lintrans.LinearTransformation stores plaintext diagonals
    // encoded once at one fixed level and scale. Keep the same contract here:
    // a plan never retains BigCVector sources and never populates a runtime
    // (scale, modulus) cache.
    using LinearPlan = std::vector<Plaintext>;
    using DiagonalFactory =
        std::function<BigCVector(uint32_t logical, uint32_t output)>;

    Ciphertext<DCRTPoly> EvalLinearPlan(ConstCiphertext<DCRTPoly> input,
                                        LinearPlan& plan,
                                        uint32_t logicalLength,
                                        uint32_t stride);
    Ciphertext<DCRTPoly> EvalLinearPlanRaw(
        ConstCiphertext<DCRTPoly> input,
        LinearPlan& plan,
        uint32_t logicalLength,
        uint32_t stride);
    Ciphertext<DCRTPoly> EvalLinearPlanFromRotations(
        ConstCiphertext<DCRTPoly> input,
        LinearPlan& plan,
        uint32_t logicalLength,
        uint32_t stride,
        const std::vector<Ciphertext<DCRTPoly>>& giantRotations);
    Ciphertext<DCRTPoly> EvalBridgePlanRaw(
        ConstCiphertext<DCRTPoly> input,
        LinearPlan& plan);
    std::vector<Ciphertext<DCRTPoly>> EvalBridgePlansRaw(
        ConstCiphertext<DCRTPoly> input,
        const std::vector<LinearPlan*>& plans);
    Ciphertext<DCRTPoly> EvalBridgePlanFromRotations(
        ConstCiphertext<DCRTPoly> input,
        LinearPlan& plan,
        const std::vector<Ciphertext<DCRTPoly>>& babyRotations);
    std::vector<Ciphertext<DCRTPoly>> EvalStreamingBridgePlansRaw(
        ConstCiphertext<DCRTPoly> input,
        uint32_t outputCount,
        const DiagonalFactory& factory);
    Ciphertext<DCRTPoly> EvalStreamingLinearPlanRaw(
        ConstCiphertext<DCRTPoly> input,
        uint32_t logicalLength,
        uint32_t stride,
        const DiagonalFactory& factory);
    Ciphertext<DCRTPoly> EvalStreamingInputPlanRaw(
        ConstCiphertext<DCRTPoly> input,
        uint32_t group);
    std::vector<Ciphertext<DCRTPoly>> EvalStreamingOutputPlansRaw(
        ConstCiphertext<DCRTPoly> input,
        uint32_t residue,
        uint32_t sourceStart);
    Ciphertext<DCRTPoly> EvalStreamingDFTRaw(
        ConstCiphertext<DCRTPoly> input,
        bool inverseMasked,
        bool packedFull);
    void PrepareInputPlans(CiphertextGroup input);
    Ciphertext<DCRTPoly> PackBooleanGroups(CiphertextGroup input);
    Ciphertext<DCRTPoly> ForwardDFT(ConstCiphertext<DCRTPoly> input);
    Ciphertext<DCRTPoly> InverseDFTMasked(ConstCiphertext<DCRTPoly> input);
    Ciphertext<DCRTPoly> InverseDFTPackedFull(ConstCiphertext<DCRTPoly> input);
    Plaintext EncodePublicFrequency(
        const std::vector<uint32_t>& multiplierDigits,
        ConstCiphertext<DCRTPoly> dftInput) const;
    Ciphertext<DCRTPoly> MultiplyReduce(ConstCiphertext<DCRTPoly> left,
                                        ConstCiphertext<DCRTPoly> right);
    Ciphertext<DCRTPoly> MultiplyMaskReduce(
        ConstCiphertext<DCRTPoly> input,
        BigCVector mask);
    Ciphertext<DCRTPoly> Rotate(ConstCiphertext<DCRTPoly> input, int32_t offset);
    Ciphertext<DCRTPoly> Align(ConstCiphertext<DCRTPoly> input, uint32_t level);
    Ciphertext<DCRTPoly> AddAdjusted(ConstCiphertext<DCRTPoly> left,
                                     ConstCiphertext<DCRTPoly> right);
    Ciphertext<DCRTPoly> LinearBit(ConstCiphertext<DCRTPoly> bit,
                                   ConstCiphertext<DCRTPoly> carryIn,
                                   ConstCiphertext<DCRTPoly> carryOut);
    Ciphertext<DCRTPoly> PrefixCarry(ConstCiphertext<DCRTPoly> propagate,
                                     ConstCiphertext<DCRTPoly> generate);
    Ciphertext<DCRTPoly> PrefixCarryWide(ConstCiphertext<DCRTPoly> propagate,
                                         ConstCiphertext<DCRTPoly> generate);
    std::vector<Ciphertext<DCRTPoly>> CorrectNibbles(
        CiphertextGroup sixOutputs,
        ConstCiphertext<DCRTPoly> carry);
    std::vector<Ciphertext<DCRTPoly>> CorrectNibblesWide(
        CiphertextGroup sixOutputs,
        ConstCiphertext<DCRTPoly> carry);
    CiphertextGroup UnpackBooleanGroups(
        const std::vector<Ciphertext<DCRTPoly>>& planes);
    CiphertextGroup FinalizeProduct(
        Ciphertext<DCRTPoly> lazyDigits,
        uint32_t maximumB2BInputLevel,
        const BigFixedPoint& stableInputScale);
    DenseRadixFullProduct FinalizeFullProduct(
        Ciphertext<DCRTPoly> lazyDigits,
        uint32_t maximumB2BInputLevel,
        const BigFixedPoint& stableInputScale);
    CiphertextGroup FinalizeDivisionProduct(
        Ciphertext<DCRTPoly> lazyDigits,
        CiphertextGroup stableInput,
        uint32_t sourceStart,
        uint32_t shift,
        bool addMarker,
        uint32_t maximumB2BInputLevel,
        const BigFixedPoint& stableInputScale);
    CiphertextGroup ExtractQuotient(
        const std::vector<Ciphertext<DCRTPoly>>& planes,
        uint32_t sourceStart);
    CiphertextGroup DenseRightShift(CiphertextGroup input,
                                    uint32_t shift);
    Ciphertext<DCRTPoly> DenseAddAndShift(
        ConstCiphertext<DCRTPoly> left,
        ConstCiphertext<DCRTPoly> right,
        uint32_t shift);
    CiphertextGroup RefreshBooleanPair(
        CiphertextGroup raw,
        const BigFixedPoint& stableInputScale,
        const char* label);

    LinearPlan BuildInputPlan(ConstCiphertext<DCRTPoly> input,
                              uint32_t group) const;
    LinearPlan BuildDFTPlan(ConstCiphertext<DCRTPoly> input,
                            bool inverseMasked) const;
    LinearPlan BuildPackedFullInverseDFTPlan(
        ConstCiphertext<DCRTPoly> input) const;
    LinearPlan BuildOutputPlan(ConstCiphertext<DCRTPoly> input,
                               uint32_t group,
                               uint32_t residue) const;
    LinearPlan BuildQuotientOutputPlan(ConstCiphertext<DCRTPoly> input,
                                       uint32_t group,
                                       uint32_t residue,
                                       uint32_t sourceStart) const;
    LinearPlan BuildDenseShiftPlan(ConstCiphertext<DCRTPoly> input,
                                   uint32_t shift) const;
    static Plaintext EncodeDiagonal(BigCVector diagonal,
                                    uint32_t babyIndex,
                                    uint32_t stride,
                                    ConstCiphertext<DCRTPoly> input);
    static void RequirePair(CiphertextGroup pair, const char* label);
    static double Seconds(std::chrono::steady_clock::duration duration);
    void Observe(const std::string& label,
                 ConstCiphertext<DCRTPoly> ciphertext) const;

    LeveledZ z;
    FHEZ fhe;
    uint32_t wordBits = 0;
    uint32_t booleanBatch = 0;
    uint32_t radixDigits = 0;
    uint32_t radixSlice = 0;
    uint32_t radixBatch = 0;
    uint32_t lazyCarryPasses = 0;
    uint32_t prefixLayers = 0;
    uint32_t postFBTTailDepth = 0;
    uint32_t rawBoundaryLevel = 0;
    std::map<uint32_t, LinearPlan> inputPlans;
    LinearPlan forwardPlan;
    LinearPlan inversePlan;
    LinearPlan packedFullInversePlan;
    std::map<uint32_t, LinearPlan> outputPlans;
    DenseRadixMultiplierTiming timing;
    DenseRadixMultiplierLevels levels;
    StageObserver stageObserver;
};

using DenseRadixMultiplierZ = std::shared_ptr<DenseRadixMultiplierZImpl>;

}  // namespace lbcrypto

#endif  // SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_DENSE_RADIX_MULTIPLIER_H_
