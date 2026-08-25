#include "scheme/ckksrns/z-leveledshe.h"
#include "encoding/z-encoding.h"
namespace lbcrypto {

//=============================================================================
// Generic methods
//=============================================================================

void LeveledZImpl::EvalAddInPlace(Ciphertext<DCRTPoly> ct, Plaintext ptxt) {
    if (ct->GetElements()[0].GetParams()->GetParams().size() !=
        ptxt->GetElement<DCRTPoly>().GetParams()->GetParams().size()) {
        OPENFHE_THROW("Ciphertext and Plaintext size mismatch in EvalAdd");
    }
    if (ptxt->GetElement<DCRTPoly>().GetFormat() != Format::EVALUATION) {
        OPENFHE_THROW("Plaintext must be in EVALUATION format in EvalMultDCRTPoly");
    }
    ZEncoding zEnc = std::dynamic_pointer_cast<ZEncodingImpl>(ptxt);
    if (!zEnc->GetScalingFactorBFP().almostEqual(ct->GetScalingFactorBFP())) {
        OPENFHE_THROW("Ciphertext and Plaintext scaling factor mismatch in EvalAdd");
    }
    auto zEncDCRTPoly = zEnc->GetElement<DCRTPoly>();

    auto& b = ct->GetElements()[0];
    b += zEncDCRTPoly;
}

Ciphertext<DCRTPoly> LeveledZImpl::EvalAdd(ConstCiphertext<DCRTPoly> ct, Plaintext ptxt) {
    auto ctNew = ct->Clone();
    EvalAddInPlace(ctNew, ptxt);
    return ctNew;
}

void LeveledZImpl::EvalAddInPlace(Ciphertext<DCRTPoly> ct, ConstCiphertext<DCRTPoly> ct2) {
    if (ct->GetElements()[0].GetParams()->GetParams().size() != ct2->GetElements()[0].GetParams()->GetParams().size()) {
        OPENFHE_THROW("Ciphertext size mismatch in EvalAddInplace");
    }
    if (!ct->GetScalingFactorBFP().almostEqual(ct2->GetScalingFactorBFP())) {
        OPENFHE_THROW("Ciphertext scaling factor mismatch in EvalAddInplace");
    }
    auto& cv1  = ct->GetElements();
    auto& cv2  = ct2->GetElements();
    uint32_t n = cv1.size();
    for (uint32_t i = 0; i < n; ++i)
        cv1[i] += cv2[i];
}

Ciphertext<DCRTPoly> LeveledZImpl::EvalAdd(ConstCiphertext<DCRTPoly> ct, ConstCiphertext<DCRTPoly> ct2) {
    auto ctNew = ct->Clone();
    EvalAddInPlace(ctNew, ct2);
    return ctNew;
}

void LeveledZImpl::EvalSubInPlace(Ciphertext<DCRTPoly> ct, Plaintext ptxt) {
    if (ct->GetElements()[0].GetParams()->GetParams().size() !=
        ptxt->GetElement<DCRTPoly>().GetParams()->GetParams().size()) {
        OPENFHE_THROW("Ciphertext and Plaintext size mismatch in EvalAdd");
    }
    if (ptxt->GetElement<DCRTPoly>().GetFormat() != Format::EVALUATION) {
        OPENFHE_THROW("Plaintext must be in EVALUATION format in EvalMultDCRTPoly");
    }
    ZEncoding zEnc = std::dynamic_pointer_cast<ZEncodingImpl>(ptxt);
    if (!zEnc->GetScalingFactorBFP().almostEqual(ct->GetScalingFactorBFP())) {
        OPENFHE_THROW("Ciphertext and Plaintext scaling factor mismatch in EvalAdd");
    }
    auto zEncDCRTPoly = zEnc->GetElement<DCRTPoly>();

    auto& b = ct->GetElements()[0];
    b -= zEncDCRTPoly;
}

Ciphertext<DCRTPoly> LeveledZImpl::EvalSub(ConstCiphertext<DCRTPoly> ct, Plaintext ptxt) {
    auto ctNew = ct->Clone();
    EvalSubInPlace(ctNew, ptxt);
    return ctNew;
}

void LeveledZImpl::EvalSubInPlace(Ciphertext<DCRTPoly> ct, ConstCiphertext<DCRTPoly> ct2) {
    if (ct->GetElements()[0].GetParams()->GetParams().size() != ct2->GetElements()[0].GetParams()->GetParams().size()) {
        OPENFHE_THROW("Ciphertext size mismatch in EvalAddInplace");
    }
    if (!ct->GetScalingFactorBFP().almostEqual(ct2->GetScalingFactorBFP())) {
        OPENFHE_THROW("Ciphertext scaling factor mismatch in EvalAddInplace");
    }
    auto& cv1  = ct->GetElements();
    auto& cv2  = ct2->GetElements();
    uint32_t n = cv1.size();
    for (uint32_t i = 0; i < n; ++i)
        cv1[i] -= cv2[i];
}

Ciphertext<DCRTPoly> LeveledZImpl::EvalSub(ConstCiphertext<DCRTPoly> ct, ConstCiphertext<DCRTPoly> ct2) {
    auto ctNew = ct->Clone();
    EvalSubInPlace(ctNew, ct2);
    return ctNew;
}

void LeveledZImpl::EvalNegateInPlace(Ciphertext<DCRTPoly> ct) {
    auto& cv1  = ct->GetElements();
    uint32_t n = cv1.size();
    for (uint32_t i = 0; i < n; ++i)
        cv1[i] = -cv1[i];
}

Ciphertext<DCRTPoly> LeveledZImpl::EvalNegate(ConstCiphertext<DCRTPoly> ct1) {
    auto ctNew = ct1->Clone();
    EvalNegateInPlace(ctNew);
    return ctNew;
}

void LeveledZImpl::EvalMultInPlace(Ciphertext<DCRTPoly> ct, Plaintext ptxt) {
    ZEncoding zEnc    = std::dynamic_pointer_cast<ZEncodingImpl>(ptxt);
    auto zEncDCRTPoly = zEnc->GetElement<DCRTPoly>();
    if (ct->GetElements()[0].GetParams()->GetParams().size() !=
        ptxt->GetElement<DCRTPoly>().GetParams()->GetParams().size()) {
        OPENFHE_THROW("Ciphertext and Plaintext size mismatch in EvalMult");
    }
    if (zEncDCRTPoly.GetFormat() != Format::EVALUATION) {
        OPENFHE_THROW("Plaintext must be in EVALUATION format in EvalMult");
    }
    auto ctSFBFP   = ct->GetScalingFactorBFP();
    auto ptxtSFBFP = zEnc->GetScalingFactorBFP();

    for (auto& a : ct->GetElements()) {
        a *= zEncDCRTPoly;
    }
    ct->SetScalingFactorBFP(ctSFBFP * ptxtSFBFP);
}

Ciphertext<DCRTPoly> LeveledZImpl::EvalMult(ConstCiphertext<DCRTPoly> ct, Plaintext ptxt) {
    auto ctNew = ct->Clone();
    EvalMultInPlace(ctNew, ptxt);
    return ctNew;
}

Ciphertext<DCRTPoly> LeveledZImpl::EvalMult(ConstCiphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2) {
    auto sfBFP1 = ct1->GetScalingFactorBFP();
    auto sfBFP2 = ct2->GetScalingFactorBFP();
    auto cc     = ct1->GetCryptoContext();
    // We use cc here for automatic relinearization
    // But we do not rely on automatic rescaling
    // TODO: remove use of cc
    auto ctNew = cc->EvalMult(ct1, ct2);
    ctNew->SetScalingFactorBFP(sfBFP1 * sfBFP2);
    return ctNew;
}

// Seems buggy.
// After inplace mult, scaling factor is not set correctly.
//void LeveledZImpl::EvalSquareInPlace(Ciphertext<DCRTPoly> ct) {
//    auto sfBFP = ct->GetScalingFactorBFP();
//    auto cc    = ct->GetCryptoContext();
//    // We use cc here for automatic relinearization
//    // But we do not rely on automatic rescaling
//    // TODO: remove use of cc
//    ct->SetScalingFactorBFP(sfBFP * sfBFP);
//    cc->EvalSquareInPlace(ct);
//    std::cout << "Scaling factor before squaring: " << std::log2(sfBFP.convertToDouble()) << std::endl;
//    ct->SetScalingFactorBFP(sfBFP * sfBFP);
//    std::cout << "Scaling factor after squaring: " << std::log2(ct->GetScalingFactorBFP().convertToDouble())
//              << std::endl;
//}
//
Ciphertext<DCRTPoly> LeveledZImpl::EvalSquare(ConstCiphertext<DCRTPoly> ct1) {
    return EvalMult(ct1, ct1);
}

Ciphertext<DCRTPoly> LeveledZImpl::EvalMultWithAdjust(ConstCiphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2) {
    // GetLevel() at qL is 0...
    // GetLevel() at q0 is L...
    if (ct1->GetLevel() < ct2->GetLevel()) {
        auto ct1Adjusted = AdjustCiphertext(ct1, ct2);
        return EvalMult(ct1Adjusted, ct2);
    }
    else if (ct1->GetLevel() > ct2->GetLevel()) {
        auto ct2Adjusted = AdjustCiphertext(ct2, ct1);
        return EvalMult(ct1, ct2Adjusted);
    }
    else {
        if (!ct1->GetScalingFactorBFP().almostEqual(ct2->GetScalingFactorBFP())) {
            OPENFHE_THROW("Scaling factors are not equal in gEvalMultWithAdjust");
        }
        return EvalMult(ct1, ct2);
    }
}

Ciphertext<DCRTPoly> LeveledZImpl::EvalAddWithAdjust(ConstCiphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2) {
    // GetLevel() at qL is 0...
    // GetLevel() at q0 is L...
    if (ct1->GetLevel() < ct2->GetLevel()) {
        auto ct1Adjusted = AdjustCiphertext(ct1, ct2);
        return EvalAdd(ct1Adjusted, ct2);
    }
    else if (ct1->GetLevel() > ct2->GetLevel()) {
        auto ct2Adjusted = AdjustCiphertext(ct2, ct1);
        return EvalAdd(ct1, ct2Adjusted);
    }
    else {
        if (!ct1->GetScalingFactorBFP().almostEqual(ct2->GetScalingFactorBFP())) {
            OPENFHE_THROW("Scaling factors are not equal in gEvalAddtWithAdjust");
        }
        return EvalAdd(ct1, ct2);
    }
}

void LeveledZImpl::EvalAddWithAdjustInPlace(Ciphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2) {
    if (ct2->GetLevel() < ct1->GetLevel()) {
        auto ct2Adjusted = AdjustCiphertext(ct2, ct1);
        EvalAddInPlace(ct1, ct2Adjusted);
    }
    else if (ct1->GetLevel() < ct2->GetLevel()) {
        auto ct1Adjusted = AdjustCiphertext(ct1, ct2);
        ct1->SetElements(ct1Adjusted->GetElements());
        ct1->SetLevel(ct1Adjusted->GetLevel());
        ct1->SetScalingFactorBFP(ct1Adjusted->GetScalingFactorBFP());
        EvalAddInPlace(ct1, ct2);
    }
    else {
        if (!ct1->GetScalingFactorBFP().almostEqual(ct2->GetScalingFactorBFP())) {
            OPENFHE_THROW("Scaling factors are not equal in gEvalAddWithAdjust");
        }
        EvalAddInPlace(ct1, ct2);
    }
}

Ciphertext<DCRTPoly> LeveledZImpl::EvalSubWithAdjust(ConstCiphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2) {
    // GetLevel() at qL is 0...
    // GetLevel() at q0 is L...
    if (ct1->GetLevel() < ct2->GetLevel()) {
        auto ct1Adjusted = AdjustCiphertext(ct1, ct2);
        return EvalSub(ct1Adjusted, ct2);
    }
    else if (ct1->GetLevel() > ct2->GetLevel()) {
        auto ct2Adjusted = AdjustCiphertext(ct2, ct1);
        return EvalSub(ct1, ct2Adjusted);
    }
    else {
        if (!ct1->GetScalingFactorBFP().almostEqual(ct2->GetScalingFactorBFP())) {
            OPENFHE_THROW("Scaling factors are not equal in gEvalSubWithAdjust");
        }
        return EvalSub(ct1, ct2);
    }
}

void LeveledZImpl::EvalSubWithAdjustInPlace(Ciphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2) {
    if (ct2->GetLevel() < ct1->GetLevel()) {
        auto ct2Adjusted = AdjustCiphertext(ct2, ct1);
        EvalSubInPlace(ct1, ct2Adjusted);
    }
    else if (ct1->GetLevel() < ct2->GetLevel()) {
        auto ct1Adjusted = AdjustCiphertext(ct1, ct2);
        ct1->SetElements(ct1Adjusted->GetElements());
        ct1->SetLevel(ct1Adjusted->GetLevel());
        ct1->SetScalingFactorBFP(ct1Adjusted->GetScalingFactorBFP());
        EvalSubInPlace(ct1, ct2);
    }
    else {
        if (!ct1->GetScalingFactorBFP().almostEqual(ct2->GetScalingFactorBFP())) {
            OPENFHE_THROW("Scaling factors are not equal in gEvalSubWithAdjustInPlace");
        }
        EvalSubInPlace(ct1, ct2);
    }
}

void LeveledZImpl::EvalMultScalarInPlace(Ciphertext<DCRTPoly> ct, BigInteger scalar) {
    auto elemParams = ct->GetElements()[0].GetParams();
    auto n          = elemParams->GetRingDimension();
    if constexpr (HEXL_MUL_ENABLE) {
        for (auto& a : ct->GetElements()) {
            auto& mVectors = a.GetAllElements();
            auto t         = a.GetNumOfElements();
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(8))
            for (size_t i = 0; i < t; ++i) {
                auto qi             = mVectors[i].GetModulus();
                uint64_t qiUInt     = qi.template ConvertToInt<uint64_t>();
                uint64_t* op1       = reinterpret_cast<uint64_t*>(&mVectors[i][0]);
                uint64_t scalarInqi = scalar.Mod(qi).template ConvertToInt<uint64_t>();
                std::vector<uint64_t> op2Vec(n, scalarInqi);
                intel::hexl::EltwiseMultMod(op1, op1, op2Vec.data(), n, qiUInt, 1);
            }
        }
    }
    else {
        for (auto& a : ct->GetElements()) {
            auto& mVectors = a.GetAllElements();
            auto t         = a.GetNumOfElements();
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(8))
            for (size_t i = 0; i < t; ++i) {
                auto qi         = mVectors[i].GetModulus();
                auto mui        = mVectors[i].GetRootOfUnity();
                auto scalarInqi = scalar.Mod(qi);
                for (size_t j = 0; j < n; ++j) {
                    mVectors[i][j] = scalarInqi.ModMulFast(mVectors[i][j], qi, mui);
                }
            }
        }
    }
}

Ciphertext<DCRTPoly> LeveledZImpl::EvalMultScalar(ConstCiphertext<DCRTPoly> ct, BigInteger scalar) {
    auto ctNew = ct->Clone();
    EvalMultScalarInPlace(ctNew, scalar);
    return ctNew;
}

void LeveledZImpl::LevelReduceInPlace(Ciphertext<DCRTPoly>& ciphertext, size_t levels) {
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(ciphertext->GetCryptoParameters());

    auto& cv = ciphertext->GetElements();

    ciphertext->SetLevel(ciphertext->GetLevel() + levels);
    for (size_t i = 0; i < levels; ++i) {
        for (auto& dcrtpoly : cv)
            dcrtpoly.DropLastElement();
    }
}

void LeveledZImpl::ModReduceInPlace(Ciphertext<DCRTPoly>& ciphertext, size_t levels) {
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(ciphertext->GetCryptoParameters());

    auto& cv = ciphertext->GetElements();

    size_t sizeQ  = cryptoParams->GetElementParams()->GetParams().size();
    size_t sizeQl = cv[0].GetNumOfElements();
    size_t diffQl = sizeQ - sizeQl;

    ciphertext->SetLevel(ciphertext->GetLevel() + levels);

    for (size_t i = 0; i < levels; ++i) {
        for (auto& dcrtpoly : cv)
            dcrtpoly.DropLastElementAndScale(cryptoParams->GetQlQlInvModqlDivqlModq(diffQl + i),
                                             cryptoParams->GetqlInvModq(diffQl + i));
        // We manually track scaling factor
        auto ql      = cryptoParams->GetElementParams()->GetParams()[sizeQl - 1 - i]->GetModulus();
        auto qlBigFP = BigFixedPoint(ql, 0, false).scaleTo(128);
        ciphertext->SetScalingFactorBFP(ciphertext->GetScalingFactorBFP() / qlBigFP);

        // Old code from OpenFHE
        // double modReduceFactor = cryptoParams->GetModReduceFactor(sizeQl - 1 - i);
        // ciphertext->SetScalingFactor(ciphertext->GetScalingFactor() / modReduceFactor);
    }
}

Ciphertext<DCRTPoly> LeveledZImpl::AdjustCiphertextToLevel(ConstCiphertext<DCRTPoly> ct, size_t level) {
    if (ct->GetLevel() == level) {
        return ct->Clone();
    }
    else if (ct->GetLevel() > level) {
        OPENFHE_THROW("Can not adjust ct to smaller level (i.e. more moduli)");
    }

    auto ctBFP        = ct->GetScalingFactorBFP();
    auto cc           = ct->GetCryptoContext();
    auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto sfTarget     = cryptoParams->GetScalingFactorBFP(level);

    auto ctBFPLog2       = std::log2(ctBFP.convertToDouble());
    auto sfTargetBFPLog2 = std::log2(sfTarget.convertToDouble());

    // The case of Noise Deg = 2 is not handled now.
    // TODO: Should track noise degree...
    // FIXME: actually very bad thing can happen here...
    // The selection of moduli chain can lead to large scaling factors
    // Use this as a safeguard for now
    if (ctBFPLog2 > 100 || sfTargetBFPLog2 > 100) {
        std::cout << "ctBFPLog2: " << ctBFPLog2 << ", ctTargetBFPLog2: " << sfTargetBFPLog2 << std::endl;
        OPENFHE_THROW("Can not Adjust Ciphertext with large scaling factor");
    }

    auto sizeQl       = ct->GetElements()[0].GetNumOfElements();
    auto sizeQlTarget = cryptoParams->GetMultiplicativeDepth() + 1 - level;

    auto qlTargetPlusOne    = cryptoParams->GetElementParams()->GetParams()[sizeQlTarget]->GetModulus();
    auto qlTargetPlusOneBFP = BigFixedPoint(qlTargetPlusOne, 0, false).scaleTo(128);

    auto ctNew = ct->Clone();
    LevelReduceInPlace(ctNew, sizeQl - sizeQlTarget - 1);

    auto adjustFactorBFP = sfTarget * qlTargetPlusOneBFP / ctBFP;
    auto adjustFactor    = adjustFactorBFP.getRoundedBigInteger();
    EvalMultScalarInPlace(ctNew, adjustFactor);
    ModReduceInPlace(ctNew);
    ctNew->SetScalingFactorBFP(sfTarget);
    return ctNew;
}

Ciphertext<DCRTPoly> LeveledZImpl::AdjustCiphertext(ConstCiphertext<DCRTPoly> ct, ConstCiphertext<DCRTPoly> ctTarget) {
    auto ctBFP       = ct->GetScalingFactorBFP();
    auto ctTargetBFP = ctTarget->GetScalingFactorBFP();

    auto ctBFPLog2       = std::log2(ctBFP.convertToDouble());
    auto ctTargetBFPLog2 = std::log2(ctTargetBFP.convertToDouble());
    // The case of Noise Deg = 2 is not handled now.
    // TODO: Should track noise degree...
    // FIXME: actually very bad thing can happen here...
    // The selection of moduli chain can lead to large scaling factors
    // Use this as a safeguard for now
    if (ctBFPLog2 > 100 || ctTargetBFPLog2 > 100) {
        std::cout << "ctBFPLog2: " << ctBFPLog2 << ", ctTargetBFPLog2: " << ctTargetBFPLog2 << std::endl;
        OPENFHE_THROW("Can not Adjust Ciphertext with large scaling factor");
    }

    auto sizeQl       = ct->GetElements()[0].GetNumOfElements();
    auto sizeQlTarget = ctTarget->GetElements()[0].GetNumOfElements();
    if (sizeQl == sizeQlTarget) {
        if (!ctBFP.almostEqual(ctTargetBFP)) {
            OPENFHE_THROW("Can not Adjust Ciphertext");
        }
        return ct->Clone();
    }
    if (sizeQl < sizeQlTarget) {
        OPENFHE_THROW("Can not Adjust Ciphertext to larger size");
    }
    auto ctNew = ct->Clone();

    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(ct->GetCryptoParameters());
    auto qlTargetPlusOne    = cryptoParams->GetElementParams()->GetParams()[sizeQlTarget]->GetModulus();
    auto qlTargetPlusOneBFP = BigFixedPoint(qlTargetPlusOne, 0, false).scaleTo(128);

    LevelReduceInPlace(ctNew, sizeQl - sizeQlTarget - 1);

    auto adjustFactorBFP = ctTargetBFP * qlTargetPlusOneBFP / ctBFP;
    auto adjustFactor    = adjustFactorBFP.getRoundedBigInteger();
    EvalMultScalarInPlace(ctNew, adjustFactor);
    ModReduceInPlace(ctNew);
    ctNew->SetScalingFactorBFP(ctTargetBFP);
    return ctNew;
}

//
// Plaintext Caches
//

Plaintext LeveledZImpl::GetBCInCPlaintext(const BigComplex& value, const BigFixedPoint& scalingFactor,
                                          const std::shared_ptr<typename DCRTPoly::Params>& elementParams) {
    auto q = elementParams->GetModulus();

    auto key = std::make_tuple(value, scalingFactor, q);
    if (m_bcInCPlaintextCache.find(key) != m_bcInCPlaintextCache.end()) {
        return m_bcInCPlaintextCache[key];
    }

    Plaintext ptxt = ZEncodingImpl::encodeC(value, elementParams, scalingFactor);
    // Store in cache
    // TODO: is this thread safe?
    m_bcInCPlaintextCache[key] = ptxt;
    return ptxt;
}

Plaintext LeveledZImpl::GetTPlaintext(uint32_t zN, const BigFixedPoint& scalingFactor,
                                      const std::shared_ptr<typename DCRTPoly::Params>& elementParams) {
    auto q   = elementParams->GetModulus();
    auto key = std::make_tuple(zN, scalingFactor, q);
    if (m_tPlaintextCache.find(key) != m_tPlaintextCache.end()) {
        return m_tPlaintextCache[key];
    }

    Plaintext tPtxt        = ZEncodingImpl::encodeTInZ(zN, elementParams, scalingFactor);
    m_tPlaintextCache[key] = tPtxt;
    return tPtxt;
}

Plaintext LeveledZImpl::GetTInvPlaintext(uint32_t zN, const BigFixedPoint& scalingFactor,
                                         const std::shared_ptr<typename DCRTPoly::Params>& elementParams) {
    auto q   = elementParams->GetModulus();
    auto key = std::make_tuple(zN, scalingFactor, q);
    if (m_tInvPlaintextCache.find(key) != m_tInvPlaintextCache.end()) {
        return m_tInvPlaintextCache[key];
    }

    Plaintext tPtxt           = ZEncodingImpl::encodeTInvInZ(zN, elementParams, scalingFactor);
    m_tInvPlaintextCache[key] = tPtxt;
    return tPtxt;
}

//
// Helpers
//

bool LeveledZImpl::BCInCPlaintextKeyCompare::operator()(const BCInCPlaintextKey& a, const BCInCPlaintextKey& b) const {
    if (std::get<0>(a).getReal().almostEqual(std::get<0>(b).getReal())) {
        if (std::get<0>(a).getImag().almostEqual(std::get<0>(b).getImag())) {
            if (std::get<1>(a).almostEqual(std::get<1>(b))) {
                return std::get<2>(a) < std::get<2>(b);
            }
            return std::get<1>(a) < std::get<1>(b);
        }
        return std::get<0>(a).getImag() < std::get<0>(b).getImag();
    }
    return std::get<0>(a).getReal() < std::get<0>(b).getReal();
}

bool LeveledZImpl::tPlaintextKeyCompare::operator()(const tPlaintextKey& a, const tPlaintextKey& b) const {
    if (std::get<0>(a) == std::get<0>(b)) {
        if (std::get<1>(a).almostEqual(std::get<1>(b))) {
            return std::get<2>(a) < std::get<2>(b);
        }
        else {
            return std::get<1>(a) < std::get<1>(b);
        }
    }
    return std::get<0>(a) < std::get<0>(b);
}

}  // namespace lbcrypto