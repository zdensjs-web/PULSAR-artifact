#ifndef SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_LEVELEDSHE_H_
#define SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_LEVELEDSHE_H_

#include "math/hal/bigfixedpoint.h"
#include "cryptocontext.h"

namespace lbcrypto {

class LeveledZImpl;
using LeveledZ = std::shared_ptr<LeveledZImpl>;

class CiphertextGroup {
public:
    CiphertextGroup() = default;
    CiphertextGroup(Ciphertext<DCRTPoly> ct) : parts({ct}) {}
    CiphertextGroup(std::vector<Ciphertext<DCRTPoly>> cts) : parts(cts) {}

    std::vector<Ciphertext<DCRTPoly>> getParts() const {
        return parts;
    }

    Ciphertext<DCRTPoly>& operator[](size_t idx) {
        return parts[idx];
    }

    size_t size() const {
        return parts.size();
    }

    operator Ciphertext<DCRTPoly>() const {
        if (parts.size() != 1) {
            OPENFHE_THROW("Cannot convert CiphertextGroup with multiple parts to single Ciphertext");
        }
        return parts[0];
    }

    using MapFunc = std::function<Ciphertext<DCRTPoly>(ConstCiphertext<DCRTPoly>&)>;
    CiphertextGroup map(MapFunc func) const {
        std::vector<Ciphertext<DCRTPoly>> result;
        for (const auto& part : parts) {
            result.push_back(func(part));
        }
        return CiphertextGroup(result);
    }

    using MapWideFunc = std::function<CiphertextGroup(ConstCiphertext<DCRTPoly>&)>;
    CiphertextGroup mapWide(MapWideFunc func) const {
        std::vector<Ciphertext<DCRTPoly>> result;
        for (const auto& part : parts) {
            auto newParts = func(part);
            for (auto& newPart : newParts.getParts()) {
                result.push_back(newPart);
            }
        }
        return CiphertextGroup(result);
    }

    CiphertextGroup append(const CiphertextGroup& other) const {
        std::vector<Ciphertext<DCRTPoly>> result = parts;
        for (const auto& part : other.getParts()) {
            result.push_back(part);
        }
        return CiphertextGroup(result);
    }

private:
    std::vector<Ciphertext<DCRTPoly>> parts;
};

class UserZImpl;
class FHEZImpl;
class AdvancedZImpl;
class DenseRadixMultiplierZImpl;
class DenseBooleanZImpl;
class LeveledZImpl {
    friend class UserZImpl;
    friend class FHEZImpl;
    friend class AdvancedZImpl;
    friend class DenseRadixMultiplierZImpl;
    friend class DenseBooleanZImpl;

private:
    //
    // Generic methods involving Ciphertext and Plaintext
    //
    Ciphertext<DCRTPoly> EvalAdd(ConstCiphertext<DCRTPoly> ct, Plaintext ptxt);
    Ciphertext<DCRTPoly> EvalSub(ConstCiphertext<DCRTPoly> ct, Plaintext ptxt);
    Ciphertext<DCRTPoly> EvalAdd(ConstCiphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2);
    Ciphertext<DCRTPoly> EvalSub(ConstCiphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2);
    Ciphertext<DCRTPoly> EvalNegate(ConstCiphertext<DCRTPoly> ct1);
    // They do not mod reduce
    Ciphertext<DCRTPoly> EvalMult(ConstCiphertext<DCRTPoly> ct, Plaintext ptxt);
    Ciphertext<DCRTPoly> EvalMult(ConstCiphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2);
    Ciphertext<DCRTPoly> EvalSquare(ConstCiphertext<DCRTPoly> ct1);
    void EvalAddInPlace(Ciphertext<DCRTPoly> ct, Plaintext ptxt);
    void EvalAddInPlace(Ciphertext<DCRTPoly> ct, ConstCiphertext<DCRTPoly> ct2);
    void EvalSubInPlace(Ciphertext<DCRTPoly> ct, Plaintext ptxt);
    void EvalSubInPlace(Ciphertext<DCRTPoly> ct, ConstCiphertext<DCRTPoly> ct2);
    void EvalNegateInPlace(Ciphertext<DCRTPoly> ct);
    void EvalMultInPlace(Ciphertext<DCRTPoly> ct, Plaintext ptxt);
    // Buggy
    //void EvalSquareInPlace(Ciphertext<DCRTPoly> ct);

    // Used internally
    Ciphertext<DCRTPoly> EvalMultScalar(ConstCiphertext<DCRTPoly> ct, BigInteger scalar);
    void EvalMultScalarInPlace(Ciphertext<DCRTPoly> ct, BigInteger scalar);

    //
    // Level management
    //
private:
    // Should not allow user to call level reduce
    void LevelReduceInPlace(Ciphertext<DCRTPoly>& ciphertext, size_t levels = 1);

public:
    void ModReduceInPlace(Ciphertext<DCRTPoly>& ciphertext, size_t levels = 1);

    //
    // Cross level adjustment
    //
    Ciphertext<DCRTPoly> AdjustCiphertextToLevel(ConstCiphertext<DCRTPoly> ciphertext, size_t level);

private:
    Ciphertext<DCRTPoly> AdjustCiphertext(ConstCiphertext<DCRTPoly> ct, ConstCiphertext<DCRTPoly> ctTarget);

    // Automatic adjustment family
    Ciphertext<DCRTPoly> EvalMultWithAdjust(ConstCiphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2);
    Ciphertext<DCRTPoly> EvalAddWithAdjust(ConstCiphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2);
    Ciphertext<DCRTPoly> EvalSubWithAdjust(ConstCiphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2);

    void EvalAddWithAdjustInPlace(Ciphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2);
    void EvalSubWithAdjustInPlace(Ciphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2);

    //
    // Operations in C
    //
    // Mainly used by bootstrapping/AdvancedSHE

    void EvalAddInPlaceInC(Ciphertext<DCRTPoly> ct, const BigComplex& ptxt);
    void EvalMultInPlaceInC(Ciphertext<DCRTPoly> ct, const BigComplex& ptxt,
                            BigFixedPoint scalingFactor = BigFixedPoint::zero());

    Ciphertext<DCRTPoly> EvalAddInC(ConstCiphertext<DCRTPoly> ct, const BigComplex& ptxt);

    Ciphertext<DCRTPoly> EvalMultInC(ConstCiphertext<DCRTPoly> ct, const BigComplex& ptxt,
                                     BigFixedPoint scalingFactor = BigFixedPoint::zero());

    Ciphertext<DCRTPoly> EvalConjugateInC(ConstCiphertext<DCRTPoly> ct);

private:
    // value, scalingFactor, modulus
    using BCInCPlaintextKey = std::tuple<BigComplex, BigFixedPoint, BigInteger>;
    struct BCInCPlaintextKeyCompare {
        bool operator()(const BCInCPlaintextKey& a, const BCInCPlaintextKey& b) const;
    };

    std::map<BCInCPlaintextKey, Plaintext, BCInCPlaintextKeyCompare> m_bcInCPlaintextCache;

    Plaintext GetBCInCPlaintext(const BigComplex& value, const BigFixedPoint& scalingFactor,
                                const std::shared_ptr<typename DCRTPoly::Params>& elementParams);

    // zN, scaling factor, modulus
    using tPlaintextKey = std::tuple<BigInteger, BigFixedPoint, BigInteger>;
    struct tPlaintextKeyCompare {
        bool operator()(const tPlaintextKey& a, const tPlaintextKey& b) const;
    };

    std::map<tPlaintextKey, Plaintext, tPlaintextKeyCompare> m_tPlaintextCache;
    std::map<tPlaintextKey, Plaintext, tPlaintextKeyCompare> m_tInvPlaintextCache;

    Plaintext GetTPlaintext(uint32_t zN, const BigFixedPoint& scalingFactor,
                            const std::shared_ptr<typename DCRTPoly::Params>& elementParams);
    Plaintext GetTInvPlaintext(uint32_t zN, const BigFixedPoint& scalingFactor,
                               const std::shared_ptr<typename DCRTPoly::Params>& elementParams);
};

using LeveledZ = std::shared_ptr<LeveledZImpl>;

}  // namespace lbcrypto

#endif  //SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_LEVELEDSHE_H_