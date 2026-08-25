#include "scheme/ckksrns/z-leveledshe.h"
#include "scheme/ckksrns/z-user.h"
#include "encoding/z-encoding.h"

namespace lbcrypto {

//
// Operations in Z
//

Ciphertext<DCRTPoly> UserZImpl::EvalAddInZ(ConstCiphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2) {
    VERIFY_ARITH_ENCODING(ct1);
    VERIFY_ARITH_ENCODING(ct2);
    return z->EvalAddWithAdjust(ct1, ct2);
}
Ciphertext<DCRTPoly> UserZImpl::EvalSubInZ(ConstCiphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2) {
    VERIFY_ARITH_ENCODING(ct1);
    VERIFY_ARITH_ENCODING(ct2);
    return z->EvalSubWithAdjust(ct1, ct2);
}
Ciphertext<DCRTPoly> UserZImpl::EvalNegateInZ(ConstCiphertext<DCRTPoly> ct1) {
    VERIFY_ARITH_ENCODING(ct1);
    return z->EvalNegate(ct1);
}

Ciphertext<DCRTPoly> UserZImpl::EvalAddPtInZ(ConstCiphertext<DCRTPoly> ct, BigInteger ptxt) {
    VERIFY_ARITH_ENCODING(ct);
    // Encode ptxt in Z encoding
    auto zN        = ct->GetZEncodingParams().getZN();
    auto elemParam = ct->GetElements()[0].GetParams();
    auto sf        = ct->GetScalingFactorBFP();
    auto plaintext = ZEncodingImpl::encodeArithSingle(ptxt, zN, elemParam, sf);
    return z->EvalAdd(ct, plaintext);
}

Ciphertext<DCRTPoly> UserZImpl::EvalSubPtInZ(ConstCiphertext<DCRTPoly> ct, BigInteger ptxt) {
    VERIFY_ARITH_ENCODING(ct);
    // Encode ptxt in Z encoding
    auto zN        = ct->GetZEncodingParams().getZN();
    auto elemParam = ct->GetElements()[0].GetParams();
    auto sf        = ct->GetScalingFactorBFP();
    auto plaintext = ZEncodingImpl::encodeArithSingle(ptxt, zN, elemParam, sf);
    return z->EvalSub(ct, plaintext);
}

Ciphertext<DCRTPoly> UserZImpl::EvalMultPtInZ(ConstCiphertext<DCRTPoly> ct, BigInteger ptxt) {
    VERIFY_ARITH_ENCODING(ct);
    // Encode ptxt in Z encoding
    auto zN        = ct->GetZEncodingParams().getZN();
    auto elemParam = ct->GetElements()[0].GetParams();
    auto sf        = ct->GetScalingFactorBFP();
    // encodeBinary here is crucial: making it become MultShort
    auto plaintext = ZEncodingImpl::encodeZ({ZPolynomial::encodeBinary(zN, ptxt)}, zN, 1, elemParam, sf);
    // This is actually MultShort
    auto res = z->EvalMult(ct, plaintext);
    z->ModReduceInPlace(res);
    return res;
}

Ciphertext<DCRTPoly> UserZImpl::EvalMultShortInZ(ConstCiphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2) {
    VERIFY_ARITH_ENCODING(ct1);
    VERIFY_ARITH_ENCODING(ct2);
    auto ct = z->EvalMultWithAdjust(ct1, ct2);
    z->ModReduceInPlace(ct);
    return ct;
}

Ciphertext<DCRTPoly> UserZImpl::EvalMultTInZ(ConstCiphertext<DCRTPoly> ct) {
    VERIFY_ARITH_ENCODING(ct);
    // Multiply by tPtxt
    auto zN         = ct->GetZEncodingParams().getZN();
    auto elemParam  = ct->GetElements()[0].GetParams();
    auto sf         = ct->GetScalingFactorBFP();
    Plaintext tPtxt = z->GetTPlaintext(zN, sf, elemParam);

    auto ct2 = z->EvalMult(ct, tPtxt);
    z->ModReduceInPlace(ct2);
    return ct2;
}

Ciphertext<DCRTPoly> UserZImpl::EvalMultTInvInZ(ConstCiphertext<DCRTPoly> ct) {
    VERIFY_ARITH_ENCODING(ct);
    // Multiply by tPtxt
    auto zN            = ct->GetZEncodingParams().getZN();
    auto elemParam     = ct->GetElements()[0].GetParams();
    auto sf            = ct->GetScalingFactorBFP();
    Plaintext tInvPtxt = z->GetTInvPlaintext(zN, sf, elemParam);

    auto ct2 = z->EvalMult(ct, tInvPtxt);
    z->ModReduceInPlace(ct2);
    return ct2;
}

Ciphertext<DCRTPoly> UserZImpl::EvalMultFullInZ(ConstCiphertext<DCRTPoly> ct1, ConstCiphertext<DCRTPoly> ct2) {
    VERIFY_ARITH_ENCODING(ct1);
    VERIFY_ARITH_ENCODING(ct2);
    auto ct = z->EvalMultWithAdjust(ct1, ct2);
    z->ModReduceInPlace(ct);

    // Multiply by tPtxt
    auto zN         = ct->GetZEncodingParams().getZN();
    auto elemParam  = ct->GetElements()[0].GetParams();
    auto sf         = ct->GetScalingFactorBFP();
    Plaintext tPtxt = z->GetTPlaintext(zN, sf, elemParam);
    ct              = z->EvalMult(ct, tPtxt);
    z->ModReduceInPlace(ct);
    return ct;
}

Ciphertext<DCRTPoly> UserZImpl::EvalAddPtVecInZ(ConstCiphertext<DCRTPoly> ct, std::vector<BigInteger> ptxt) {
    VERIFY_ARITH_ENCODING(ct);
    auto zSlots = ct->GetZEncodingParams().getZSlots();
    if (ptxt.size() != zSlots) {
        OPENFHE_THROW("Plaintext size not matching zSlots in EvalAddInZ");
    }
    // Encode ptxt in Z encoding
    auto zN        = ct->GetZEncodingParams().getZN();
    auto elemParam = ct->GetElements()[0].GetParams();
    auto sf        = ct->GetScalingFactorBFP();
    auto plaintext = ZEncodingImpl::encodeArith(ptxt, zN, zSlots, elemParam, sf);
    return z->EvalAdd(ct, plaintext);
}

Ciphertext<DCRTPoly> UserZImpl::EvalSubPtVecInZ(ConstCiphertext<DCRTPoly> ct, std::vector<BigInteger> ptxt) {
    VERIFY_ARITH_ENCODING(ct);
    auto zSlots = ct->GetZEncodingParams().getZSlots();
    if (ptxt.size() != zSlots) {
        OPENFHE_THROW("Plaintext size not matching zSlots in EvalSubInZ");
    }
    // Encode ptxt in Z encoding
    auto zN        = ct->GetZEncodingParams().getZN();
    auto elemParam = ct->GetElements()[0].GetParams();
    auto sf        = ct->GetScalingFactorBFP();
    auto plaintext = ZEncodingImpl::encodeArith(ptxt, zN, zSlots, elemParam, sf);
    return z->EvalSub(ct, plaintext);
}

Ciphertext<DCRTPoly> UserZImpl::EvalMultPtVecInZ(ConstCiphertext<DCRTPoly> ct, std::vector<BigInteger> ptxt) {
    VERIFY_ARITH_ENCODING(ct);
    auto zSlots = ct->GetZEncodingParams().getZSlots();
    if (ptxt.size() != zSlots) {
        OPENFHE_THROW("Plaintext size not matching zSlots in EvalMultInZ");
    }
    // Encode ptxt in Z encoding
    auto zN        = ct->GetZEncodingParams().getZN();
    auto elemParam = ct->GetElements()[0].GetParams();
    auto sf        = ct->GetScalingFactorBFP();
    // encodeBinary here is crucial: making it become MultShort
    std::vector<ZPolynomial> zPolys(ptxt.size(), ZPolynomial(zN));
    for (size_t i = 0; i != ptxt.size(); ++i) {
        zPolys[i] = ZPolynomial::encodeBinary(zN, ptxt[i]);
    }
    auto plaintext = ZEncodingImpl::encodeZ(zPolys, zN, zSlots, elemParam, sf);
    // This is actually MultShort
    auto res = z->EvalMult(ct, plaintext);
    z->ModReduceInPlace(res);
    return res;
}

//
// Operations in C
//

void LeveledZImpl::EvalAddInPlaceInC(Ciphertext<DCRTPoly> ct, const BigComplex& ptxt) {
    auto elemParams = ct->GetElements()[0].GetParams();
    auto ptxtEnc    = GetBCInCPlaintext(ptxt, ct->GetScalingFactorBFP(), elemParams);
    EvalAddInPlace(ct, ptxtEnc);
}

Ciphertext<DCRTPoly> LeveledZImpl::EvalAddInC(ConstCiphertext<DCRTPoly> ct, const BigComplex& ptxt) {
    auto elemParams = ct->GetElements()[0].GetParams();
    auto ptxtEnc    = GetBCInCPlaintext(ptxt, ct->GetScalingFactorBFP(), elemParams);
    return EvalAdd(ct, ptxtEnc);
}

// We do not automatically rescale...these are used internally
void LeveledZImpl::EvalMultInPlaceInC(Ciphertext<DCRTPoly> ct, const BigComplex& ptxt, BigFixedPoint scalingFactor) {
    auto elemParams = ct->GetElements()[0].GetParams();
    if (scalingFactor.equalZero()) {
        scalingFactor = ct->GetScalingFactorBFP();
    }
    auto ptxtEnc = GetBCInCPlaintext(ptxt, scalingFactor, elemParams);
    EvalMultInPlace(ct, ptxtEnc);
}

Ciphertext<DCRTPoly> LeveledZImpl::EvalMultInC(ConstCiphertext<DCRTPoly> ct, const BigComplex& ptxt,
                                               BigFixedPoint scalingFactor) {
    auto elemParams = ct->GetElements()[0].GetParams();
    if (scalingFactor.equalZero()) {
        scalingFactor = ct->GetScalingFactorBFP();
    }
    auto ptxtEnc = GetBCInCPlaintext(ptxt, scalingFactor, elemParams);
    return EvalMult(ct, ptxtEnc);
}

Ciphertext<DCRTPoly> LeveledZImpl::EvalConjugateInC(ConstCiphertext<DCRTPoly> ct) {
    uint32_t N = ct->GetElements()[0].GetRingDimension();
    std::vector<uint32_t> vec(N);
    PrecomputeAutoMap(N, 2 * N - 1, &vec);

    auto result = ct->Clone();

    auto cc         = ct->GetCryptoContext();
    auto algo       = ct->GetCryptoContext()->GetScheme();
    auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(ct->GetKeyTag());
    algo->KeySwitchInPlace(result, evalKeyMap.at(2 * N - 1));

    auto& rcv = result->GetElements();
    rcv[0]    = rcv[0].AutomorphismTransform(2 * N - 1, vec);
    rcv[1]    = rcv[1].AutomorphismTransform(2 * N - 1, vec);
    return result;
}

}  // namespace lbcrypto