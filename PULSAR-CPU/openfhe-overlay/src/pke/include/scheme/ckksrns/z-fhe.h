#ifndef SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_FHE_H_
#define SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_FHE_H_

#include "z-leveledshe.h"
#include "z-advancedshe.h"
#include "z-fhe-utils.h"
#include "z-fhe-constants.h"

namespace lbcrypto {

class ZBootstrapPlaintextCacheImpl {
public:
    ZBootstrapPlaintextCacheImpl()  = default;
    ~ZBootstrapPlaintextCacheImpl() = default;

    ZBootstrapPlaintextCacheImpl(const BigCVector& val) : m_value(val) {}

    Plaintext GetPlaintext(const BigFixedPoint& scalingFactor,
                           const std::shared_ptr<typename DCRTPoly::Params>& elementParams);

    void ClearPlaintextCache();

    BigCVector& getValue() {
        return m_value;
    }

private:
    BigCVector m_value;

    // scalingFactor, modulus
    using Key = std::tuple<BigFixedPoint, BigInteger>;
    std::map<Key, Plaintext> m_cache;
};

using ZBootstrapPlaintextCache = std::shared_ptr<ZBootstrapPlaintextCacheImpl>;

class ZBootstrapPrecom {
public:
    ZBootstrapPrecom() = default;

    virtual ~ZBootstrapPrecom() = default;

    ZBootstrapPrecom(const ZBootstrapPrecom& rhs) = default;

    ZBootstrapPrecom(ZBootstrapPrecom&& rhs) noexcept = default;

    // level budget for homomorphic encoding, number of layers to collapse in one level,
    // number of layers remaining to be collapsed in one level to have exactly the number
    // of levels specified in the level budget, the number of rotations in one level,
    // the baby step and giant step in the baby-step giant-step strategy, the number of
    // rotations in the remaining level, the baby step and giant step in the baby-step
    // giant-step strategy for the remaining level
    struct ckks_boot_params m_paramsEnc;

    // level budget for homomorphic decoding, number of layers to collapse in one level,
    // number of layers remaining to be collapsed in one level to have exactly the number
    // of levels specified in the level budget, the number of rotations in one level,
    // the baby step and giant step in the baby-step giant-step strategy, the number of
    // rotations in the remaining level, the baby step and giant step in the baby-step
    // giant-step strategy for the remaining level
    struct ckks_boot_params m_paramsDec;

    // number of complex slots for which the bootstrapping is performed
    // In Z we have two kinds of slots: complex slots and Z slots
    uint32_t m_cSlots;

    // indicates whether the linear transforms are for sparse packing
    bool m_isSparse;

    // Linear map U0; used in decoding
    std::vector<ZBootstrapPlaintextCache> m_U0Pre;

    // Conj(U0^T); used in encoding
    std::vector<ZBootstrapPlaintextCache> m_U0hatTPre;
    // Conj(U0^T); used in encoding; scaled N so do not need to normalize
    std::vector<ZBootstrapPlaintextCache> m_U0hatTPreScaledN;
    // Conj(U0^T); used in encoding; scaled N * K so do not need to normalize for Chebyshev
    std::vector<ZBootstrapPlaintextCache> m_U0hatTPreScaledNK;

    // coefficients corresponding to U0; used in decoding
    std::vector<std::vector<ZBootstrapPlaintextCache>> m_U0PreFFT;

    // coefficients corresponding to conj(U0^T); used in encoding
    std::vector<std::vector<ZBootstrapPlaintextCache>> m_U0hatTPreFFT;

    // coefficients corresponding to conj(U0^T); used in encoding; scaled N so do not need to normalize
    std::vector<std::vector<ZBootstrapPlaintextCache>> m_U0hatTPreFFTScaledN;
    // coefficients corresponding to conj(U0^T); used in encoding; scaled N * K so do not need to normalize
    std::vector<std::vector<ZBootstrapPlaintextCache>> m_U0hatTPreFFTScaledNK;

    //
    // coefficients for ZU and ZV = ZUInverse
    //
    // digonal, 1st digonal, ..., zN/2-1th diagonal, -1th diagonal, ..., -zN/2+1 th diagonal
    std::vector<ZBootstrapPlaintextCache> m_ZUPre;
    std::vector<ZBootstrapPlaintextCache> m_ZVPre;

    // coefficients for ZV with preprocessing for b0
    std::vector<ZBootstrapPlaintextCache> m_ZVSpecialB0Pre;

    // coefficients for ZU and ZV in A2Ae
    std::vector<ZBootstrapPlaintextCache> m_ZUSpecialA2AePre;
    std::vector<ZBootstrapPlaintextCache> m_ZVSpecialA2AePre;

    // Parameters for scaleZV, i.e., whether we scale down by zSlots during multiplying ZV
    // If larger than this threshold, we do not scale down during multiplying ZV
    // And a manual scaling down is needed (i.e. takes two levels for ZV LT)
    uint32_t m_zSlotsThresholdForScaling;

    // Pre-computation for fully packed case
    std::vector<ZBootstrapPlaintextCache> m_ZU0Pre;
    std::vector<ZBootstrapPlaintextCache> m_ZU1Pre;
    std::vector<ZBootstrapPlaintextCache> m_ZV0Pre;
    std::vector<ZBootstrapPlaintextCache> m_ZV1Pre;
    std::vector<ZBootstrapPlaintextCache> m_ZV0SpecialB0Pre;
    std::vector<ZBootstrapPlaintextCache> m_ZU0SpecialA2AePre;
    std::vector<ZBootstrapPlaintextCache> m_ZU1SpecialA2AePre;
    std::vector<ZBootstrapPlaintextCache> m_ZV0SpecialA2AePre;
    std::vector<ZBootstrapPlaintextCache> m_ZV1SpecialA2AePre;

    // Parameters for ArithToBoolean
    // Num bits per iteration
    uint32_t m_w;
    // lower bits below this threshold is not cleaned
    int32_t m_cutoff;
    // Interpolation order
    uint32_t m_lutIDOrder;
    uint32_t m_lutMSBOrder;
    // coefficients for LUTs
    std::vector<BigComplex> m_lutIDCoeffs;
    std::vector<BigComplex> m_lutMSBCoeffs;
};

class FHEZImpl {
public:
    FHEZImpl(LeveledZ z, AdvancedZ advZ) : z(z), advZ(advZ) {}

    void EvalBootstrapSetup(const CryptoContextImpl<DCRTPoly>& cc, uint32_t zN, uint32_t zSlots,
                            std::vector<uint32_t> levelBudget = {2, 2}, std::vector<uint32_t> dim1 = {0, 0},
                            uint32_t w = 4, int32_t arithToBooleanCutoff = -16, uint32_t lutOrder = 1);

    // Generates only C2S/S2C, conjugation and sparse-ModRaise keys.
    // Generic A2B/B2A rotations are not used by the radix multiplier.
    void EvalBootstrapKeyGenBooleanRadixOnly(
        const PrivateKey<DCRTPoly> privateKey,
        uint32_t zN,
        uint32_t zSlots);

    // Releases generic Z/A2B/B2A plans not referenced by the exact radix
    // multiplier, retaining only its SlotsToCoeffs/CoeffsToSlots FFT plans.
    void ReleaseUnusedBooleanRadixPrecom(uint32_t cSlots);

    void EvalBootstrapKeyGen(const PrivateKey<DCRTPoly> privateKey, uint32_t zN, uint32_t zSlots);

public:
    Ciphertext<DCRTPoly> EvalLinearTransform(std::vector<ZBootstrapPlaintextCache>& A,
                                             ConstCiphertext<DCRTPoly>& ct) const;

    Ciphertext<DCRTPoly> EvalZLinearTransform(std::vector<ZBootstrapPlaintextCache>& A,
                                              ConstCiphertext<DCRTPoly>& ct) const;

    Ciphertext<DCRTPoly> EvalCoeffsToSlots(const std::vector<std::vector<ZBootstrapPlaintextCache>>& A,
                                           ConstCiphertext<DCRTPoly>& ctxt) const;

    Ciphertext<DCRTPoly> EvalSlotsToCoeffs(const std::vector<std::vector<ZBootstrapPlaintextCache>>& A,
                                           ConstCiphertext<DCRTPoly>& ctxt) const;

public:
    Ciphertext<DCRTPoly> EvalTruncate(ConstCiphertext<DCRTPoly>& ct) const;

    Ciphertext<DCRTPoly> EvalModRaise(ConstCiphertext<DCRTPoly>& ct) const;

    void EvalPartialSumInPlace(Ciphertext<DCRTPoly>& ct) const;

    Ciphertext<DCRTPoly> EvalModRaisePartialSum(ConstCiphertext<DCRTPoly>& ct) const;

    enum Z2COption { Z2C_NORMAL, Z2C_SPECIAL_B0, Z2C_SPECIAL_A2AE };
    CiphertextGroup EvalZ2C(ConstCiphertext<DCRTPoly>& ct, Z2COption z2cOption) const;

    Ciphertext<DCRTPoly> EvalC2R(ConstCiphertext<DCRTPoly>& ct) const;

    enum R2CScalingOption { SCALE_N, SCALE_NK, SCALE_N_PRE, SCALE_NK_PRE };
    CiphertextGroup EvalR2C(ConstCiphertext<DCRTPoly>& ct, R2CScalingOption scalingOption) const;

    enum C2ZOption { C2Z_NORMAL, C2Z_SPECIAL_A2AE };
    Ciphertext<DCRTPoly> EvalC2Z(CiphertextGroup ct, C2ZOption c2zOption) const;

    Ciphertext<DCRTPoly> EvalZ2R(ConstCiphertext<DCRTPoly>& ct, Z2COption z2cOption) const;

    Ciphertext<DCRTPoly> EvalR2Z(ConstCiphertext<DCRTPoly>& ct, R2CScalingOption scalingOption,
                                 C2ZOption c2zOption) const;

    Ciphertext<DCRTPoly> EvalArithToArithHigh(ConstCiphertext<DCRTPoly>& ctxt) const;

    Ciphertext<DCRTPoly> EvalArithToArithNoise(ConstCiphertext<DCRTPoly>& ctxt) const;

    Ciphertext<DCRTPoly> EvalArithToArith(ConstCiphertext<DCRTPoly>& ctxt) const;

    Ciphertext<DCRTPoly> EvalArithToBooleanSparse(ConstCiphertext<DCRTPoly>& ctxt);

    CiphertextGroup EvalArithToBooleanFull(ConstCiphertext<DCRTPoly>& ctxt);

    CiphertextGroup EvalArithToBooleanBatched(CiphertextGroup ctxts);

    // Dispatcher
    CiphertextGroup EvalArithToBoolean(CiphertextGroup ctxts);

    Ciphertext<DCRTPoly> EvalBooleanToBooleanSparse(ConstCiphertext<DCRTPoly>& ctxt) const;

    CiphertextGroup EvalBooleanToBooleanFull(CiphertextGroup ctxt) const;

    void ClearBooleanBootstrapPlaintextCaches(uint32_t cSlots) const;

    // Maps the five Prefix Boot residues {0,...,4} to {-2,...,2}/8.
    CiphertextGroup EvalFiveStateIdentityFull(CiphertextGroup ctxt) const;

    // Maps each five-state Prefix Boot residue directly to Z=P/2+iG and p.
    CiphertextGroup EvalFiveStatePrefixFull(CiphertextGroup ctxt) const;

    // One full-ring radix state is transformed once and evaluated against ten
    // real LUTs sharing the exponential approximation and polynomial powers.
    // Output order is b0..b3,inc0..inc3,G,P for roots x in [0,30], where
    // inc=(x mod 16 + 1) mod 16. The incremented planes make final carry
    // correction a single selector level.
    CiphertextGroup EvalRadix31ToBooleanTenFullFirst(ConstCiphertext<DCRTPoly>& ctxt) const;


    // Selects between two Boolean ciphertexts in one multiplication level:
    // zero + choice * (one-zero).
    Ciphertext<DCRTPoly> EvalBooleanSelectBit(ConstCiphertext<DCRTPoly>& choice,
                                              ConstCiphertext<DCRTPoly>& zero,
                                              ConstCiphertext<DCRTPoly>& one) const;


    // Boolean-preserving one-level multiplication used by carry-prefix and
    // runway gates. Computes input^2 followed by one ModReduce.
    Ciphertext<DCRTPoly> EvalBooleanSquareBit(ConstCiphertext<DCRTPoly>& input) const;
    // Exact Lattigo radix-16 LazyCarry: periodic residue FBT followed by
    // one linear carry shift. The active half contains 64 radix digits and
    // the second half is the zero guard region.
    Ciphertext<DCRTPoly> EvalRadix16LazyCarryFull(
        ConstCiphertext<DCRTPoly>& ctxt,
        uint32_t activeDigits,
        uint32_t radixBatch) const;

    // Exact Lattigo shared FBT output order: b0,b1,b2,b3,G,P. Three
    // complex polynomials share one C2R/ModRaise/R2C/exp front end and one
    // encrypted power basis.
    CiphertextGroup EvalRadix31ToBooleanSixFullFirst(
        ConstCiphertext<DCRTPoly>& ctxt,
        uint32_t radixBatch) const;


    // Exact Lattigo full-product scale_div path. The low K digits are stored
    // in the real component and the high K digits in the imaginary component.
    Ciphertext<DCRTPoly> EvalRadix16PackedFullLazyCarry(
        ConstCiphertext<DCRTPoly>& ctxt,
        uint32_t halfDigits,
        uint32_t radixBatch) const;

    CiphertextGroup EvalRadix31PackedFullToBooleanSixFirst(
        ConstCiphertext<DCRTPoly>& ctxt,
        uint32_t halfDigits,
        uint32_t radixBatch) const;

    // Dispatcher
    CiphertextGroup EvalBooleanToBoolean(CiphertextGroup ctxt) const;

    Ciphertext<DCRTPoly> EvalBooleanToArith(CiphertextGroup ctxt) const;

private:
    ZBootstrapPrecom& GetBootPrecom(uint32_t slots) const {
        auto pair = m_bootPrecomMap.find(slots);
        if (pair != m_bootPrecomMap.end())
            return *(pair->second);
        OPENFHE_THROW("Precomputations for " + std::to_string(slots) + " slots not found.");
    }

    void ApplyDoubleAngleIterations(Ciphertext<DCRTPoly>& ciphertext, uint32_t numIter) const;

    Ciphertext<DCRTPoly> internalArithToBooleanIteration(ConstCiphertext<DCRTPoly>& ctxt,
                                                         const std::vector<BigComplex>& lutCoeffs) const;

    Ciphertext<DCRTPoly> internalBooleanToBooleanLTsSparse(ConstCiphertext<DCRTPoly>& ctxt) const;

    CiphertextGroup internalBooleanToBooleanLTsFull(CiphertextGroup ctxt) const;

    CiphertextGroup internalBooleanToBooleanCustomTwoLUTSparse(ConstCiphertext<DCRTPoly>& ctxt,
                                                               const std::vector<BigComplex>& lutCoeffs,
                                                               const std::vector<BigComplex>& lutCoeffs2) const;

    CiphertextGroup internalBooleanToBooleanCustomTwoLUTFull(CiphertextGroup ctxt,
                                                             const std::vector<BigComplex>& lutCoeffs,
                                                             const std::vector<BigComplex>& lutCoeffs2) const;

    //------------------------------------------------------------------------------
    // Precomputations for ZCoeffsToSlots and SlotsToZCoeffs
    //------------------------------------------------------------------------------
    std::vector<ZBootstrapPlaintextCache> EvalZLinearTransformPrecompute(const CryptoContextImpl<DCRTPoly>& cc,
                                                                         const BigCMatrix& A, uint32_t zSlots,
                                                                         BigFixedPoint scale) const;

    std::vector<ZBootstrapPlaintextCache> EvalZLinearTransformPrecompute(const CryptoContextImpl<DCRTPoly>& cc,
                                                                         const BigCMatrix& A, const BigCMatrix& B,
                                                                         uint32_t zSlots, BigFixedPoint scale) const;

    //------------------------------------------------------------------------------
    // Precomputations for CoeffsToSlots and SlotsToCoeffs
    //------------------------------------------------------------------------------

    std::vector<ZBootstrapPlaintextCache> EvalLinearTransformPrecompute(const CryptoContextImpl<DCRTPoly>& cc,
                                                                        const BigCMatrix& A, BigFixedPoint scale) const;

    std::vector<ZBootstrapPlaintextCache> EvalLinearTransformPrecompute(const CryptoContextImpl<DCRTPoly>& cc,
                                                                        const BigCMatrix& A, const BigCMatrix& B,
                                                                        uint32_t orientation,
                                                                        BigFixedPoint scale) const;

    std::vector<std::vector<ZBootstrapPlaintextCache>> EvalCoeffsToSlotsPrecompute(
        const CryptoContextImpl<DCRTPoly>& cc, const BigCVector& pows, const std::vector<uint32_t>& rotGroup,
        bool flag_i, std::vector<BigFixedPoint> scales) const;

    std::vector<std::vector<ZBootstrapPlaintextCache>> EvalSlotsToCoeffsPrecompute(
        const CryptoContextImpl<DCRTPoly>& cc, const BigCVector& pows, const std::vector<uint32_t>& rotGroup,
        bool flag_i) const;

    //------------------------------------------------------------------------------
    // Find Rotation Indices
    //------------------------------------------------------------------------------
    std::vector<int32_t> FindBootstrapRotationIndices(uint32_t zN, uint32_t zSlots, uint32_t M);

    // ATTN: The following 3 functions are helper methods to be called in FindBootstrapRotationIndices() only.
    // so they DO NOT remove possible duplicates and automorphisms corresponding to 0 and M/4.
    // These methods completely depend on FindBootstrapRotationIndices() to do that.
    std::vector<uint32_t> FindLinearTransformRotationIndices(uint32_t slots, uint32_t M);
    std::vector<uint32_t> FindCoeffsToSlotsRotationIndices(uint32_t slots, uint32_t M);
    std::vector<uint32_t> FindSlotsToCoeffsRotationIndices(uint32_t slots, uint32_t M);

    //------------------------------------------------------------------------------
    // Mask Plaintext for Arithmetic to Boolean Bootstrapping
    //------------------------------------------------------------------------------

    Plaintext getATBMaskSparsePacking(uint32_t iter, uint32_t w, uint32_t zN, uint32_t zSlots,
                                      const std::shared_ptr<typename DCRTPoly::Params>& elementParams,
                                      const BigFixedPoint& scalingFactor);

    Plaintext getATBMaskFullPacking(uint32_t iter, uint32_t w, uint32_t zN, uint32_t zSlots,
                                    const std::shared_ptr<typename DCRTPoly::Params>& elementParams,
                                    const BigFixedPoint& scalingFactor);

    Plaintext getATBSubtractMaskFullPacking(uint32_t iter, uint32_t w, uint32_t zN, uint32_t zSlots,
                                            const std::shared_ptr<typename DCRTPoly::Params>& elementParams,
                                            const BigFixedPoint& scalingFactor, BigFixedPoint scaleDown);

    Plaintext getATBMask(uint32_t iter, uint32_t w, uint32_t zN, uint32_t zSlots,
                         const std::shared_ptr<typename DCRTPoly::Params>& elementParams,
                         const BigFixedPoint& scalingFactor);

private:
    // iter, w, zN, zSlots, modulus, scalingFactor
    using MaskPlaintextKey = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, BigInteger, BigFixedPoint>;
    std::map<MaskPlaintextKey, Plaintext> m_atbMaskPtxtCache;

    // iter, w, zN, zSlots, modulus, scalingFactor, scaleDown
    using SubtractMaskPlaintextKey =
        std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, BigInteger, BigFixedPoint, BigFixedPoint>;
    std::map<SubtractMaskPlaintextKey, Plaintext> m_atbSubtractMaskPtxtCache;

private:
    // corresponds to probability of less than 2^{-128}
    static constexpr uint32_t K_SPARSE_ENCAPSULATED = 16;

    // number of double-angle iterations in CKKS bootstrapping. Must be static because it is used in a static function.
    // same value is used for both SPARSE and ENCAPSULATED_SPARSE
    static constexpr uint32_t R_SPARSE = 3;

private:
    LeveledZ z;
    AdvancedZ advZ;

    // key tuple is dim1, levelBudgetEnc, levelBudgetDec
    // key tuple is cSlots, zSlots,
    std::map<uint32_t, std::shared_ptr<ZBootstrapPrecom>> m_bootPrecomMap;
};

using FHEZ = std::shared_ptr<FHEZImpl>;

}  // namespace lbcrypto

#endif  //SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_FHE_H_
