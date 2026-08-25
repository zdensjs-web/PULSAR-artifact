#ifndef SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_USER_H_
#define SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_USER_H_

#include "scheme/ckksrns/z-leveledshe.h"

namespace lbcrypto {

// User facing API
// LeveledSHE API are too internal...
class UserZImpl {
public:
    UserZImpl(LeveledZ z) : z(z) {}

    //
    // Operations in Z
    //
    // ct-ct versions.
    Ciphertext<DCRTPoly> EvalAddInZ(ConstCiphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2);
    Ciphertext<DCRTPoly> EvalSubInZ(ConstCiphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2);
    Ciphertext<DCRTPoly> EvalNegateInZ(ConstCiphertext<DCRTPoly> ct1);

    // Here is short cut multiplication, ct1 * ct2 where one is in binary encoding
    Ciphertext<DCRTPoly> EvalMultShortInZ(ConstCiphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2);
    // Here is full multiplication, ct1 * ct2 * t
    Ciphertext<DCRTPoly> EvalMultFullInZ(ConstCiphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2);

    // ct-pt versions. Here ptxt will be ZEncoded
    Ciphertext<DCRTPoly> EvalAddPtInZ(ConstCiphertext<DCRTPoly> ct, BigInteger ptxt);
    Ciphertext<DCRTPoly> EvalSubPtInZ(ConstCiphertext<DCRTPoly> ct, BigInteger ptxt);
    Ciphertext<DCRTPoly> EvalMultPtInZ(ConstCiphertext<DCRTPoly> ct, BigInteger ptxt);

    Ciphertext<DCRTPoly> EvalAddPtVecInZ(ConstCiphertext<DCRTPoly> ct, std::vector<BigInteger> ptxt);
    Ciphertext<DCRTPoly> EvalSubPtVecInZ(ConstCiphertext<DCRTPoly> ct, std::vector<BigInteger> ptxt);
    Ciphertext<DCRTPoly> EvalMultPtVecInZ(ConstCiphertext<DCRTPoly> ct, std::vector<BigInteger> ptxt);

    // Conversion between [m]_t / t and [m]_t
    Ciphertext<DCRTPoly> EvalMultTInZ(ConstCiphertext<DCRTPoly> ct);
    Ciphertext<DCRTPoly> EvalMultTInvInZ(ConstCiphertext<DCRTPoly> ct);

    //
    // Boolean Mode Operations
    //

    // ct-ct version
    CiphertextGroup EvalBooleanAND(CiphertextGroup ct1, CiphertextGroup ct2);
    CiphertextGroup EvalBooleanOR(CiphertextGroup ct1, CiphertextGroup ct2);
    CiphertextGroup EvalBooleanXOR(CiphertextGroup ct1, CiphertextGroup ct2);
    CiphertextGroup EvalBooleanNOT(CiphertextGroup ct1);

    // ct-pt version
    CiphertextGroup EvalBooleanANDPt(CiphertextGroup ct1, BigInteger ptxt);
    CiphertextGroup EvalBooleanORPt(CiphertextGroup ct1, BigInteger ptxt);
    CiphertextGroup EvalBooleanXORPt(CiphertextGroup ct1, BigInteger ptxt);

    CiphertextGroup EvalBooleanANDPtVec(CiphertextGroup ct1, std::vector<BigInteger> ptxt);
    CiphertextGroup EvalBooleanORPtVec(CiphertextGroup ct1, std::vector<BigInteger> ptxt);
    CiphertextGroup EvalBooleanXORPtVec(CiphertextGroup ct1, std::vector<BigInteger> ptxt);

    // Here left/right is defined in a big-endian manner
    CiphertextGroup EvalBooleanShiftLeft(CiphertextGroup ct1, uint64_t offset, bool rotate = false);
    CiphertextGroup EvalBooleanShiftRight(CiphertextGroup ct1, uint64_t offset);
    // Wrappers for rotate only
    CiphertextGroup EvalBooleanRotateLeft(CiphertextGroup ct1, uint64_t offset);
    CiphertextGroup EvalBooleanRotateRight(CiphertextGroup ct1, uint64_t offset);

    CiphertextGroup EvalSignExtract(CiphertextGroup ct1);

    void Setup(const PrivateKey<DCRTPoly> privateKey, uint32_t zN, uint32_t zSlots, bool enableSignExtract,
               // This is zSlot shifts
               std::vector<int32_t> zSlotRotates,
               // The following are bit shifts
               std::vector<uint32_t> leftShifts, std::vector<uint32_t> rightShifts, std::vector<uint32_t> leftRotates,
               std::vector<uint32_t> rightRotates);

private:
    LeveledZ z;
};

using UserZ = std::shared_ptr<UserZImpl>;

#define VERIFY_ARITH_ENCODING(ct)                          \
    do {                                                   \
        auto zEncParams = ct->GetZEncodingParams();        \
        if (zEncParams.getEncodingType() != ZMode) {       \
            OPENFHE_THROW("Ciphertext not in arith mode"); \
        }                                                  \
    } while (0)

#define VERIFY_BOOL_ENCODING(ctGroup)                                                                   \
    do {                                                                                                \
        auto zEncParams = ctGroup[0]->GetZEncodingParams();                                             \
        if (zEncParams.getEncodingType() != BModeFull && zEncParams.getEncodingType() != BModeSparse) { \
            OPENFHE_THROW("Ciphertext not in boolean mode");                                            \
        }                                                                                               \
    } while (0)

}  // namespace lbcrypto

#endif  //SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_USER_H_