#include "scheme/ckksrns/z-leveledshe.h"
#include "scheme/ckksrns/z-user.h"
#include "encoding/z-encoding.h"

namespace lbcrypto {

CiphertextGroup UserZImpl::EvalBooleanAND(CiphertextGroup ct1, CiphertextGroup ct2) {
    VERIFY_BOOL_ENCODING(ct1);
    VERIFY_BOOL_ENCODING(ct2);
    if (ct1.size() != ct2.size()) {
        OPENFHE_THROW("CiphertextGroup size mismatch");
    }
    std::vector<Ciphertext<DCRTPoly>> resVec;
    for (size_t i = 0; i != ct1.size(); ++i) {
        auto res = z->EvalMultWithAdjust(ct1[i], ct2[i]);
        z->ModReduceInPlace(res);
        resVec.push_back(res);
    }
    return resVec;
}

CiphertextGroup UserZImpl::EvalBooleanOR(CiphertextGroup ct1, CiphertextGroup ct2) {
    VERIFY_BOOL_ENCODING(ct1);
    VERIFY_BOOL_ENCODING(ct2);
    if (ct1.size() != ct2.size()) {
        OPENFHE_THROW("CiphertextGroup size mismatch");
    }
    std::vector<Ciphertext<DCRTPoly>> resVec;
    for (size_t i = 0; i != ct1.size(); ++i) {
        auto addRes = z->EvalAddWithAdjust(ct1[i], ct2[i]);
        auto res    = z->EvalMultWithAdjust(ct1[i], ct2[i]);
        z->ModReduceInPlace(res);
        auto orRes = z->EvalSubWithAdjust(addRes, res);
        resVec.push_back(orRes);
    }
    return resVec;
}

CiphertextGroup UserZImpl::EvalBooleanXOR(CiphertextGroup ct1, CiphertextGroup ct2) {
    VERIFY_BOOL_ENCODING(ct1);
    VERIFY_BOOL_ENCODING(ct2);
    if (ct1.size() != ct2.size()) {
        OPENFHE_THROW("CiphertextGroup size mismatch");
    }
    std::vector<Ciphertext<DCRTPoly>> resVec;
    for (size_t i = 0; i != ct1.size(); ++i) {
        auto addRes = z->EvalAddWithAdjust(ct1[i], ct2[i]);
        auto res    = z->EvalMultWithAdjust(ct1[i], ct2[i]);
        res         = z->EvalMultScalar(res, 2);
        z->ModReduceInPlace(res);
        auto orRes = z->EvalSubWithAdjust(addRes, res);
        resVec.push_back(orRes);
    }
    return resVec;
}

CiphertextGroup UserZImpl::EvalBooleanNOT(CiphertextGroup ct1) {
    VERIFY_BOOL_ENCODING(ct1);
    std::vector<Ciphertext<DCRTPoly>> resVec;
    for (size_t i = 0; i != ct1.size(); ++i) {
        auto res = z->EvalNegate(ct1[i]);
        z->EvalAddInPlaceInC(res, BigFixedPoint::one());
        resVec.push_back(res);
    }
    return resVec;
}

CiphertextGroup UserZImpl::EvalBooleanANDPt(CiphertextGroup ct, BigInteger ptxt) {
    VERIFY_BOOL_ENCODING(ct);
    std::vector<Ciphertext<DCRTPoly>> resVec;
    // Encode ptxt in Z encoding
    auto zN        = ct[0]->GetZEncodingParams().getZN();
    auto zSlots    = ct[0]->GetZEncodingParams().getZSlots();
    auto elemParam = ct[0]->GetElements()[0].GetParams();
    auto sf        = ct[0]->GetScalingFactorBFP();

    std::vector<ZEncoding> ptxtEncoded;
    if (ct[0]->GetZEncodingParams().isBModeFull()) {
        if (ct.size() != 2) {
            OPENFHE_THROW("CiphertextGroup size mismatch for full boolean encoding");
        }
        ptxtEncoded = ZEncodingImpl::encodeBooleanFullSingle(ptxt, zN, elemParam, sf);
    }
    else if (ct[0]->GetZEncodingParams().isBModeSparse()) {
        // Have to replicate, as the usual zSlot = 1 to full zSlot wont work here
        std::vector<BigInteger> ptxts(zSlots, ptxt);
        ptxtEncoded.push_back(ZEncodingImpl::encodeBooleanSparse(ptxts, zN, zSlots, elemParam, sf));
    }
    else {
        OPENFHE_THROW("Invalid boolean encoding type");
    }
    for (size_t i = 0; i != ct.size(); ++i) {
        auto res = z->EvalMult(ct[i], ptxtEncoded[i]);
        z->ModReduceInPlace(res);
        resVec.push_back(res);
    }
    return resVec;
}

CiphertextGroup UserZImpl::EvalBooleanORPt(CiphertextGroup ct, BigInteger ptxt) {
    VERIFY_BOOL_ENCODING(ct);
    std::vector<Ciphertext<DCRTPoly>> resVec;
    // Encode ptxt in Z encoding
    auto zN        = ct[0]->GetZEncodingParams().getZN();
    auto zSlots    = ct[0]->GetZEncodingParams().getZSlots();
    auto elemParam = ct[0]->GetElements()[0].GetParams();
    auto sf        = ct[0]->GetScalingFactorBFP();

    std::vector<ZEncoding> ptxtEncoded;
    if (ct[0]->GetZEncodingParams().isBModeFull()) {
        if (ct.size() != 2) {
            OPENFHE_THROW("CiphertextGroup size mismatch for full boolean encoding");
        }
        ptxtEncoded = ZEncodingImpl::encodeBooleanFullSingle(ptxt, zN, elemParam, sf);
    }
    else if (ct[0]->GetZEncodingParams().isBModeSparse()) {
        // Have to replicate, as the usual zSlot = 1 to full zSlot wont work here
        std::vector<BigInteger> ptxts(zSlots, ptxt);
        ptxtEncoded.push_back(ZEncodingImpl::encodeBooleanSparse(ptxts, zN, zSlots, elemParam, sf));
    }
    else {
        OPENFHE_THROW("Invalid boolean encoding type");
    }
    for (size_t i = 0; i != ct.size(); ++i) {
        auto addRes = z->EvalAdd(ct[i], ptxtEncoded[i]);
        auto res    = z->EvalMult(ct[i], ptxtEncoded[i]);
        z->ModReduceInPlace(res);
        auto orRes = z->EvalSubWithAdjust(addRes, res);
        resVec.push_back(orRes);
    }
    return resVec;
}

CiphertextGroup UserZImpl::EvalBooleanXORPt(CiphertextGroup ct, BigInteger ptxt) {
    VERIFY_BOOL_ENCODING(ct);
    std::vector<Ciphertext<DCRTPoly>> resVec;
    // Encode ptxt in Z encoding
    auto zN        = ct[0]->GetZEncodingParams().getZN();
    auto zSlots    = ct[0]->GetZEncodingParams().getZSlots();
    auto elemParam = ct[0]->GetElements()[0].GetParams();
    auto sf        = ct[0]->GetScalingFactorBFP();

    std::vector<ZEncoding> ptxtEncoded;
    if (ct[0]->GetZEncodingParams().isBModeFull()) {
        if (ct.size() != 2) {
            OPENFHE_THROW("CiphertextGroup size mismatch for full boolean encoding");
        }
        ptxtEncoded = ZEncodingImpl::encodeBooleanFullSingle(ptxt, zN, elemParam, sf);
    }
    else if (ct[0]->GetZEncodingParams().isBModeSparse()) {
        // Have to replicate, as the usual zSlot = 1 to full zSlot wont work here
        std::vector<BigInteger> ptxts(zSlots, ptxt);
        ptxtEncoded.push_back(ZEncodingImpl::encodeBooleanSparse(ptxts, zN, zSlots, elemParam, sf));
    }
    else {
        OPENFHE_THROW("Invalid boolean encoding type");
    }
    for (size_t i = 0; i != ct.size(); ++i) {
        auto addRes = z->EvalAdd(ct[i], ptxtEncoded[i]);
        auto res    = z->EvalMult(ct[i], ptxtEncoded[i]);
        res         = z->EvalMultScalar(res, 2);
        z->ModReduceInPlace(res);
        auto xorRes = z->EvalSubWithAdjust(addRes, res);
        resVec.push_back(xorRes);
    }
    return resVec;
}

CiphertextGroup UserZImpl::EvalBooleanANDPtVec(CiphertextGroup ct, std::vector<BigInteger> ptxt) {
    VERIFY_BOOL_ENCODING(ct);
    std::vector<Ciphertext<DCRTPoly>> resVec;
    // Encode ptxt in Z encoding
    auto zN     = ct[0]->GetZEncodingParams().getZN();
    auto zSlots = ct[0]->GetZEncodingParams().getZSlots();
    if (ptxt.size() != zSlots) {
        OPENFHE_THROW("Plaintext size not matching zSlots in EvalBooleanANDPtVec");
    }
    auto elemParam = ct[0]->GetElements()[0].GetParams();
    auto sf        = ct[0]->GetScalingFactorBFP();

    std::vector<ZEncoding> ptxtEncoded;
    if (ct[0]->GetZEncodingParams().isBModeFull()) {
        if (ct.size() != 2) {
            OPENFHE_THROW("CiphertextGroup size mismatch for full boolean encoding");
        }
        ptxtEncoded = ZEncodingImpl::encodeBooleanFull(ptxt, zN, zSlots, elemParam, sf);
    }
    else if (ct[0]->GetZEncodingParams().isBModeSparse()) {
        ptxtEncoded.push_back(ZEncodingImpl::encodeBooleanSparse(ptxt, zN, zSlots, elemParam, sf));
    }
    else {
        OPENFHE_THROW("Invalid boolean encoding type");
    }
    for (size_t i = 0; i != ct.size(); ++i) {
        auto res = z->EvalMult(ct[i], ptxtEncoded[i]);
        z->ModReduceInPlace(res);
        resVec.push_back(res);
    }
    return resVec;
}

CiphertextGroup UserZImpl::EvalBooleanORPtVec(CiphertextGroup ct, std::vector<BigInteger> ptxt) {
    VERIFY_BOOL_ENCODING(ct);
    std::vector<Ciphertext<DCRTPoly>> resVec;
    // Encode ptxt in Z encoding
    auto zN     = ct[0]->GetZEncodingParams().getZN();
    auto zSlots = ct[0]->GetZEncodingParams().getZSlots();
    if (ptxt.size() != zSlots) {
        OPENFHE_THROW("Plaintext size not matching zSlots in EvalBooleanORPtVec");
    }
    auto elemParam = ct[0]->GetElements()[0].GetParams();
    auto sf        = ct[0]->GetScalingFactorBFP();

    std::vector<ZEncoding> ptxtEncoded;
    if (ct[0]->GetZEncodingParams().isBModeFull()) {
        if (ct.size() != 2) {
            OPENFHE_THROW("CiphertextGroup size mismatch for full boolean encoding");
        }
        ptxtEncoded = ZEncodingImpl::encodeBooleanFull(ptxt, zN, zSlots, elemParam, sf);
    }
    else if (ct[0]->GetZEncodingParams().isBModeSparse()) {
        ptxtEncoded.push_back(ZEncodingImpl::encodeBooleanSparse(ptxt, zN, zSlots, elemParam, sf));
    }
    else {
        OPENFHE_THROW("Invalid boolean encoding type");
    }
    for (size_t i = 0; i != ct.size(); ++i) {
        auto addRes = z->EvalAdd(ct[i], ptxtEncoded[i]);
        auto res    = z->EvalMult(ct[i], ptxtEncoded[i]);
        z->ModReduceInPlace(res);
        auto orRes = z->EvalSubWithAdjust(addRes, res);
        resVec.push_back(orRes);
    }
    return resVec;
}

CiphertextGroup UserZImpl::EvalBooleanXORPtVec(CiphertextGroup ct, std::vector<BigInteger> ptxt) {
    VERIFY_BOOL_ENCODING(ct);
    std::vector<Ciphertext<DCRTPoly>> resVec;
    // Encode ptxt in Z encoding
    auto zN     = ct[0]->GetZEncodingParams().getZN();
    auto zSlots = ct[0]->GetZEncodingParams().getZSlots();
    if (ptxt.size() != zSlots) {
        OPENFHE_THROW("Plaintext size not matching zSlots in EvalBooleanXORPtVec");
    }
    auto elemParam = ct[0]->GetElements()[0].GetParams();
    auto sf        = ct[0]->GetScalingFactorBFP();

    std::vector<ZEncoding> ptxtEncoded;
    if (ct[0]->GetZEncodingParams().isBModeFull()) {
        if (ct.size() != 2) {
            OPENFHE_THROW("CiphertextGroup size mismatch for full boolean encoding");
        }
        ptxtEncoded = ZEncodingImpl::encodeBooleanFull(ptxt, zN, zSlots, elemParam, sf);
    }
    else if (ct[0]->GetZEncodingParams().isBModeSparse()) {
        ptxtEncoded.push_back(ZEncodingImpl::encodeBooleanSparse(ptxt, zN, zSlots, elemParam, sf));
    }
    else {
        OPENFHE_THROW("Invalid boolean encoding type");
    }
    for (size_t i = 0; i != ct.size(); ++i) {
        auto addRes = z->EvalAdd(ct[i], ptxtEncoded[i]);
        auto res    = z->EvalMult(ct[i], ptxtEncoded[i]);
        res         = z->EvalMultScalar(res, 2);
        z->ModReduceInPlace(res);
        auto xorRes = z->EvalSubWithAdjust(addRes, res);
        resVec.push_back(xorRes);
    }
    return resVec;
}

CiphertextGroup UserZImpl::EvalBooleanShiftLeft(CiphertextGroup ct, uint64_t offset, bool rotate) {
    VERIFY_BOOL_ENCODING(ct);
    auto zEncodeParams = ct[0]->GetZEncodingParams();
    auto zN            = zEncodeParams.getZN();
    auto zSlots        = zEncodeParams.getZSlots();
    auto elemParam     = ct[0]->GetElements()[0].GetParams();
    auto sf            = ct[0]->GetScalingFactorBFP();
    auto cc            = ct[0]->GetCryptoContext();
    if (offset >= zN) {
        OPENFHE_THROW("Shift offset exceeds zN");
    }
    std::vector<Ciphertext<DCRTPoly>> resVec;
    if (offset < zN / 2) {
        if (zEncodeParams.isBModeFull()) {
            if (ct.size() != 2) {
                OPENFHE_THROW("CiphertextGroup size mismatch for full boolean encoding");
            }
            auto mask1 = (BigInteger(1) << (zN / 2 - offset)) - BigInteger(1);
            // Extra << (zN / 2) to shift to high part for encodeBooleanFullSingle
            auto mask2        = ((BigInteger(1) << (zN / 2)) - (BigInteger(1) << (zN / 2 - offset))) << (zN / 2);
            auto mask         = mask1 | mask2;
            auto maskEncoded  = ZEncodingImpl::encodeBooleanFullSingle(mask, zN, elemParam, sf);
            auto mask1Encoded = maskEncoded[0];
            auto mask2Encoded = maskEncoded[1];

            auto low0  = z->EvalMult(ct[0], mask1Encoded);
            auto high0 = z->EvalMult(ct[0], mask2Encoded);
            auto low1  = z->EvalMult(ct[1], mask1Encoded);
            auto high1 = z->EvalMult(ct[1], mask2Encoded);
            z->ModReduceInPlace(low0);
            z->ModReduceInPlace(high0);
            z->ModReduceInPlace(low1);
            z->ModReduceInPlace(high1);
            low0            = cc->EvalRotate(low0, -offset);
            low1            = cc->EvalRotate(low1, -offset);
            int64_t offset2 = -static_cast<int64_t>(offset) + static_cast<int64_t>(zN / 2);
            high0           = cc->EvalRotate(high0, offset2);
            high1           = cc->EvalRotate(high1, offset2);
            if (rotate) {
                z->EvalAddInPlace(low0, high1);
            }
            z->EvalAddInPlace(low1, high0);
            resVec.push_back(low0);
            resVec.push_back(low1);
        }
        else {
            auto mask1Low  = (BigInteger(1) << (zN / 2 - offset)) - BigInteger(1);
            auto mask1High = mask1Low << (zN / 2);
            auto mask2Low  = ((BigInteger(1) << (zN / 2)) - (BigInteger(1) << (zN / 2 - offset)));
            auto mask2High = mask2Low << (zN / 2);
            auto mask1     = mask1Low | mask1High;
            std::vector<BigInteger> mask1Replicated(zSlots, mask1);
            std::vector<BigInteger> mask2LowReplicated(zSlots, mask2Low);
            auto mask1Encoded = ZEncodingImpl::encodeBooleanSparse(mask1Replicated, zN, zSlots, elemParam, sf);
            auto mask2LowReplicatedEncoded =
                ZEncodingImpl::encodeBooleanSparse(mask2LowReplicated, zN, zSlots, elemParam, sf);
            auto part1 = z->EvalMult(ct[0], mask1Encoded);
            auto part2 = z->EvalMult(ct[0], mask2LowReplicatedEncoded);
            z->ModReduceInPlace(part1);
            z->ModReduceInPlace(part2);
            part1           = cc->EvalRotate(part1, -offset);
            int64_t offset2 = (zN / 2) * (zSlots - 1) + static_cast<int64_t>(offset);
            part2           = cc->EvalRotate(part2, -offset2);
            z->EvalAddInPlace(part1, part2);
            if (rotate) {
                // For rotate
                std::vector<BigInteger> mask2HighReplicated(zSlots, mask2High);
                auto mask2HighReplicatedEncoded =
                    ZEncodingImpl::encodeBooleanSparse(mask2HighReplicated, zN, zSlots, elemParam, sf);
                auto part3 = z->EvalMult(ct[1], mask2HighReplicatedEncoded);
                z->ModReduceInPlace(part3);
                int64_t offset3 = -(zN / 2) * (zSlots - 1) + static_cast<int64_t>(offset);
                part3           = cc->EvalRotate(part3, -offset3);
                z->EvalAddInPlace(part1, part3);
            }
            resVec.push_back(part1);
        }
    }
    else {
        if (zEncodeParams.isBModeFull()) {
            if (ct.size() != 2) {
                OPENFHE_THROW("CiphertextGroup size mismatch for full boolean encoding");
            }
            offset -= zN / 2;  // Adjust offset
            auto mask1 = (BigInteger(1) << (zN / 2 - offset)) - BigInteger(1);
            // Extra << (zN / 2) to shift to high part for encodeBooleanFullSingle
            auto mask2        = ((BigInteger(1) << (zN / 2)) - (BigInteger(1) << (zN / 2 - offset))) << (zN / 2);
            auto mask         = mask1 | mask2;
            auto maskEncoded  = ZEncodingImpl::encodeBooleanFullSingle(mask, zN, elemParam, sf);
            auto mask1Encoded = maskEncoded[0];
            auto mask2Encoded = maskEncoded[1];

            auto low0 = z->EvalMult(ct[0], mask1Encoded);
            z->ModReduceInPlace(low0);
            low0 = cc->EvalRotate(low0, -offset);

            // Reverse the order
            if (rotate) {
                auto high0 = z->EvalMult(ct[0], mask2Encoded);
                auto low1  = z->EvalMult(ct[1], mask1Encoded);
                auto high1 = z->EvalMult(ct[1], mask2Encoded);
                z->ModReduceInPlace(high0);
                z->ModReduceInPlace(low1);
                z->ModReduceInPlace(high1);
                low1            = cc->EvalRotate(low1, -offset);
                int64_t offset2 = -static_cast<int64_t>(offset) + static_cast<int64_t>(zN / 2);
                high0           = cc->EvalRotate(high0, offset2);
                high1           = cc->EvalRotate(high1, offset2);
                z->EvalAddInPlace(low1, high0);
                z->EvalAddInPlace(low0, high1);
                resVec.push_back(low1);
                resVec.push_back(low0);
            }
            else {
                auto zero = low0->Clone();
                for (size_t i = 0; i != zero->GetElements().size(); ++i) {
                    zero->GetElements()[i].SetValuesToZero();
                }
                resVec.push_back(zero);
                resVec.push_back(low0);
            }
        }
        else {
            offset -= zN / 2;  // Adjust offset
            auto mask1Low = (BigInteger(1) << (zN / 2 - offset)) - BigInteger(1);
            std::vector<BigInteger> mask1LowReplicated(zSlots, mask1Low);
            auto mask1LowEncoded = ZEncodingImpl::encodeBooleanSparse(mask1LowReplicated, zN, zSlots, elemParam, sf);
            auto part1           = z->EvalMult(ct[0], mask1LowEncoded);
            z->ModReduceInPlace(part1);
            auto offset1 = (zN / 2) * zSlots + offset;
            part1        = cc->EvalRotate(part1, -offset1);

            if (rotate) {
                auto mask1High = mask1Low << (zN / 2);
                auto mask2Low  = ((BigInteger(1) << (zN / 2)) - (BigInteger(1) << (zN / 2 - offset)));
                auto mask2High = mask2Low << (zN / 2);
                std::vector<BigInteger> mask1HighReplicated(zSlots, mask1High);
                auto mask1HighReplicatedEncoded =
                    ZEncodingImpl::encodeBooleanSparse(mask1HighReplicated, zN, zSlots, elemParam, sf);
                std::vector<BigInteger> mask2LowReplicated(zSlots, mask2Low);
                auto mask2LowReplicatedEncoded =
                    ZEncodingImpl::encodeBooleanSparse(mask2LowReplicated, zN, zSlots, elemParam, sf);
                std::vector<BigInteger> mask2HighReplicated(zSlots, mask2High);
                auto mask2HighReplicatedEncoded =
                    ZEncodingImpl::encodeBooleanSparse(mask2HighReplicated, zN, zSlots, elemParam, sf);

                auto part2 = z->EvalMult(ct[0], mask1HighReplicatedEncoded);
                z->ModReduceInPlace(part2);
                auto part3 = z->EvalMult(ct[0], mask2LowReplicatedEncoded);
                z->ModReduceInPlace(part3);
                auto part4 = z->EvalMult(ct[0], mask2HighReplicatedEncoded);
                z->ModReduceInPlace(part4);

                int64_t offset2 = (zN / 2) * (zSlots) - static_cast<int64_t>(offset);
                // Note: here we use positive offset...
                part2 = cc->EvalRotate(part2, offset2);
                z->EvalAddInPlace(part1, part2);

                // Positive offset
                int64_t offset3 = (zN / 2) - static_cast<int64_t>(offset);
                part3           = cc->EvalRotate(part3, offset3);
                z->EvalAddInPlace(part1, part3);

                int64_t offset4 = (zN / 2) - static_cast<int64_t>(offset);
                part4           = cc->EvalRotate(part4, offset4);
                z->EvalAddInPlace(part1, part4);
            }
            resVec.push_back(part1);
        }
    }
    return resVec;
}

CiphertextGroup UserZImpl::EvalBooleanShiftRight(CiphertextGroup ct, uint64_t offset) {
    VERIFY_BOOL_ENCODING(ct);
    auto zEncodeParams = ct[0]->GetZEncodingParams();
    auto zN            = zEncodeParams.getZN();
    auto zSlots        = zEncodeParams.getZSlots();
    auto elemParam     = ct[0]->GetElements()[0].GetParams();
    auto sf            = ct[0]->GetScalingFactorBFP();
    auto cc            = ct[0]->GetCryptoContext();
    if (offset >= zN) {
        OPENFHE_THROW("Shift offset exceeds zN");
    }
    std::vector<Ciphertext<DCRTPoly>> resVec;
    if (offset < zN / 2) {
        if (zEncodeParams.isBModeFull()) {
            if (ct.size() != 2) {
                OPENFHE_THROW("CiphertextGroup size mismatch for full boolean encoding");
            }
            auto mask1 = (BigInteger(1) << (offset)) - BigInteger(1);
            // Extra << (zN / 2) to shift to high part for encodeBooleanFullSingle
            auto mask2        = ((BigInteger(1) << (zN / 2)) - (BigInteger(1) << (offset))) << (zN / 2);
            auto mask         = mask1 | mask2;
            auto maskEncoded  = ZEncodingImpl::encodeBooleanFullSingle(mask, zN, elemParam, sf);
            auto mask1Encoded = maskEncoded[0];
            auto mask2Encoded = maskEncoded[1];

            auto high0 = z->EvalMult(ct[0], mask2Encoded);
            auto low1  = z->EvalMult(ct[1], mask1Encoded);
            auto high1 = z->EvalMult(ct[1], mask2Encoded);
            z->ModReduceInPlace(high0);
            z->ModReduceInPlace(low1);
            z->ModReduceInPlace(high1);
            high0           = cc->EvalRotate(high0, offset);
            int64_t offset2 = -static_cast<int64_t>(offset) + static_cast<int64_t>(zN / 2);
            low1            = cc->EvalRotate(low1, -offset2);
            high1           = cc->EvalRotate(high1, offset);
            z->EvalAddInPlace(high0, low1);
            resVec.push_back(high0);
            resVec.push_back(high1);
        }
        else {
            auto mask1Low  = (BigInteger(1) << (offset)) - BigInteger(1);
            auto mask1High = mask1Low << (zN / 2);
            auto mask2Low  = ((BigInteger(1) << (zN / 2)) - (BigInteger(1) << (offset)));
            auto mask2High = mask2Low << (zN / 2);
            auto mask2     = mask2Low | mask2High;
            std::vector<BigInteger> mask1HighReplicated(zSlots, mask1High);
            std::vector<BigInteger> mask2Replicated(zSlots, mask2);
            auto mask1HighReplicatedEncoded =
                ZEncodingImpl::encodeBooleanSparse(mask1HighReplicated, zN, zSlots, elemParam, sf);
            auto mask2ReplicatedEncoded =
                ZEncodingImpl::encodeBooleanSparse(mask2Replicated, zN, zSlots, elemParam, sf);
            auto part1 = z->EvalMult(ct[0], mask1HighReplicatedEncoded);
            auto part2 = z->EvalMult(ct[0], mask2ReplicatedEncoded);
            z->ModReduceInPlace(part1);
            z->ModReduceInPlace(part2);
            part1           = cc->EvalRotate(part1, offset);
            int64_t offset2 = (zN / 2) * (zSlots - 1) + static_cast<int64_t>(offset);
            part2           = cc->EvalRotate(part2, offset2);
            z->EvalAddInPlace(part1, part2);
            resVec.push_back(part1);
        }
    }
    else {
        if (zEncodeParams.isBModeFull()) {
            if (ct.size() != 2) {
                OPENFHE_THROW("CiphertextGroup size mismatch for full boolean encoding");
            }
            offset -= zN / 2;  // Adjust offset
            // Extra << (zN / 2) to shift to high part for encodeBooleanFullSingle
            auto mask2        = ((BigInteger(1) << (zN / 2)) - (BigInteger(1) << (offset))) << (zN / 2);
            auto maskEncoded  = ZEncodingImpl::encodeBooleanFullSingle(mask2, zN, elemParam, sf);
            auto mask2Encoded = maskEncoded[1];

            auto high1 = z->EvalMult(ct[1], mask2Encoded);
            z->ModReduceInPlace(high1);
            high1 = cc->EvalRotate(high1, offset);

            auto zero = high1->Clone();
            for (size_t i = 0; i != zero->GetElements().size(); ++i) {
                zero->GetElements()[i].SetValuesToZero();
            }
            resVec.push_back(high1);
            resVec.push_back(zero);
        }
        else {
            offset -= zN / 2;  // Adjust offset
            auto mask2High = ((BigInteger(1) << (zN / 2)) - (BigInteger(1) << (offset))) << (zN / 2);
            std::vector<BigInteger> mask2HighReplicated(zSlots, mask2High);
            auto mask2HighEncoded = ZEncodingImpl::encodeBooleanSparse(mask2HighReplicated, zN, zSlots, elemParam, sf);
            auto part1            = z->EvalMult(ct[0], mask2HighEncoded);
            z->ModReduceInPlace(part1);
            auto offset1 = (zN / 2) * zSlots + offset;
            part1        = cc->EvalRotate(part1, offset1);
            resVec.push_back(part1);
        }
    }
    return resVec;
}

CiphertextGroup UserZImpl::EvalBooleanRotateLeft(CiphertextGroup ct, uint64_t offset) {
    return EvalBooleanShiftLeft(ct, offset, true);
}

CiphertextGroup UserZImpl::EvalBooleanRotateRight(CiphertextGroup ct, uint64_t offset) {
    auto zN = ct[0]->GetZEncodingParams().getZN();
    offset  = (zN - (offset % zN)) % zN;
    return EvalBooleanShiftLeft(ct, offset, true);
}

CiphertextGroup UserZImpl::EvalSignExtract(CiphertextGroup ct) {
    VERIFY_BOOL_ENCODING(ct);
    auto zEncodeParams = ct[0]->GetZEncodingParams();
    auto zN            = zEncodeParams.getZN();
    return EvalBooleanShiftRight(ct, zN - 1);
}

}  // namespace lbcrypto