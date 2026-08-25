#include "scheme/ckksrns/z-user-advanced.h"

namespace lbcrypto {

void UserZImpl::Setup(const PrivateKey<DCRTPoly> privateKey, uint32_t zN, uint32_t zSlots, bool enableSignExtract,
                      std::vector<int32_t> zSlotRotates, std::vector<uint32_t> leftShifts,
                      std::vector<uint32_t> rightShifts, std::vector<uint32_t> leftRotates,
                      std::vector<uint32_t> rightRotates) {
    auto cc       = privateKey->GetCryptoContext();
    auto ringDim  = cc->GetRingDimension();
    auto algo     = cc->GetScheme();
    auto cSlots   = (zN / 2) * (zSlots);
    auto isSparse = (cSlots < ringDim / 2);

    std::vector<int32_t> indices;
    if (enableSignExtract) {
        if (!isSparse) {
            // Full packing
            indices.push_back(static_cast<int32_t>(zN / 2 - 1));
        }
        else {
            auto offset1 = (zN / 2) * (zSlots) + (zN / 2 - 1);
            indices.push_back(offset1);
        }
    }

    // zSlot rotates
    for (auto zsr : zSlotRotates) {
        int32_t index = (zN / 2) * zsr;
        indices.push_back(index);
    }

    // Left shift
    for (auto ls : leftShifts) {
        int32_t index = -static_cast<int32_t>(ls);
        if (ls >= zN) {
            OPENFHE_THROW("Left shift amount exceeds zN");
        }
        if (ls < zN / 2) {
            if (!isSparse) {
                indices.push_back(-index);
                auto offset2 = -index + static_cast<int32_t>(zN / 2);
                indices.push_back(offset2);
            }
            else {
                indices.push_back(-index);
                int32_t offset2 = (zN / 2) * (zSlots - 1) + index;
                indices.push_back(-offset2);
            }
        }
        else {
            if (!isSparse) {
                auto offset3 = -index + static_cast<int32_t>(zN / 2);
                indices.push_back(offset3);
            }
            else {
                int32_t offset3 = (zN / 2) * (zSlots - 1) + index;
                indices.push_back(-offset3);
            }
        }
    }

    // Right shift
    for (auto rs : rightShifts) {
        if (rs >= zN) {
            OPENFHE_THROW("Right shift amount exceeds zN");
        }
        int32_t index = static_cast<int32_t>(rs);
        if (rs < zN / 2) {
            if (!isSparse) {
                indices.push_back(index);
                int32_t offset2 = -index + static_cast<int32_t>(zN / 2);
                indices.push_back(-offset2);
            }
            else {
                indices.push_back(index);
                int32_t offset2 = (zN / 2) * (zSlots - 1) + index;
                indices.push_back(offset2);
            }
        }
        else {
            if (!isSparse) {
                int32_t offset3 = index - static_cast<int32_t>(zN / 2);
                indices.push_back(offset3);
            }
            else {
                int32_t offset3 = (zN / 2) * (zSlots - 1) + index;
                indices.push_back(offset3);
            }
        }
    }

    // Left rotate/right rotate
    for (auto rr : rightRotates) {
        leftRotates.push_back(zN - rr);
    }
    for (auto lr : leftRotates) {
        int32_t index = -static_cast<int32_t>(lr);
        if (lr >= zN) {
            OPENFHE_THROW("Left shift amount exceeds zN");
        }
        if (lr < zN / 2) {
            if (!isSparse) {
                indices.push_back(-index);
                auto offset2 = -index + static_cast<int32_t>(zN / 2);
                indices.push_back(offset2);
            }
            else {
                indices.push_back(-index);
                int32_t offset2 = (zN / 2) * (zSlots - 1) + index;
                indices.push_back(-offset2);
                int32_t offset3 = -(zN / 2) * (zSlots - 1) + index;
                indices.push_back(-offset3);
            }
        }
        else {
            if (!isSparse) {
                auto offset1 = -index + static_cast<int32_t>(zN / 2);
                indices.push_back(offset1);
                int32_t offset2 = -index + static_cast<int64_t>(zN);
                indices.push_back(offset2);
            }
            else {
                int32_t offset1 = (zN / 2) * (zSlots - 1) + index;
                indices.push_back(-offset1);
                int32_t offset2 = (zN / 2) * (zSlots + 1) - index;
                indices.push_back(offset2);
                int64_t offset3 = (zN)-index;
                indices.push_back(offset3);
            }
        }
    }

    // Deduplicate indices
    std::set<int32_t> indexSet(indices.begin(), indices.end());
    indices.assign(indexSet.begin(), indexSet.end());

    auto evalKeys = algo->EvalAtIndexKeyGen(privateKey, indices);
    cc->InsertEvalAutomorphismKey(evalKeys, privateKey->GetKeyTag());
}

CiphertextGroup UserZAdvancedImpl::EvalLessThan(Ciphertext<DCRTPoly> ct1, Ciphertext<DCRTPoly> ct2) {
    VERIFY_ARITH_ENCODING(ct1);
    VERIFY_ARITH_ENCODING(ct2);
    // Compute ct1 - ct2
    auto ctDiff     = userZ->EvalSubInZ(ct1, ct2);
    auto ctDiffBool = fheZ->EvalArithToBoolean(ctDiff);

    // Sign extract
    auto sign = userZ->EvalSignExtract(ctDiffBool);
    return sign;
}

CiphertextGroup UserZAdvancedImpl::EvalRotateInZ(CiphertextGroup ct, int32_t index) {
    auto zEncodingparams = ct[0]->GetZEncodingParams();
    auto zN              = zEncodingparams.getZN();
    auto cc              = ct[0]->GetCryptoContext();
    if (zEncodingparams.isZMode() || zEncodingparams.isBModeFull()) {
        return ct.map(
            [&](ConstCiphertext<DCRTPoly>& c) -> Ciphertext<DCRTPoly> { return cc->EvalRotate(c, index * (zN / 2)); });
    }
    else if (zEncodingparams.isBModeSparse()) {
        // One way is mask and rotate, but it requires extra evalMult.
        // We directly prohibit this now... ask rotation in ZMode instead.
        OPENFHE_THROW("Rotation in boolean sparse mode is not supported in this implementation");
    }
    else {
        OPENFHE_THROW("Unsupported encoding type for rotation");
    }
}

//CiphertextGroup UserZAdvancedImpl::EvalEqualTo(Ciphertext<DCRTPoly> ct1, Ciphertext<DCRTPoly> ct2) {
//    // TODO: optimize it using reduce operation....
//    VERIFY_ARITH_ENCODING(ct1);
//    VERIFY_ARITH_ENCODING(ct2);
//    //// Compute ct1 - ct2
//    //auto ctDiff = userZ->EvalSubInZ(ct1, ct2);
//    //// Compute ct2 - ct1
//    //auto ctDiff2    = userZ->EvalSubInZ(ct2, ct1);
//
//    //auto ctDiffBool = fheZ->EvalArithToBoolean(CiphertextGroup({ctDiff, ctDiff2}));
//
//    //// Sign extract
//    //auto sign = userZ->EvalSignExtract(ctDiffBool.append(ctDiff2Bool));
//
//    //// NOT sign
//    //auto signNeg = userZ->EvalBooleanNOT(sign);
//    //return signNeg;
//}

}  // namespace lbcrypto