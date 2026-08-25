#include "scheme/ckksrns/z-fhe.h"

namespace lbcrypto {

static BigFixedPoint getLinearTransformPlaintextScale(ConstCiphertext<DCRTPoly>& ct);

//------------------------------------------------------------------------------
// EVALUATION: ZCoeffsToSlots and SlotsToZCoeffs
//------------------------------------------------------------------------------

Ciphertext<DCRTPoly> FHEZImpl::EvalZLinearTransform(std::vector<ZBootstrapPlaintextCache>& A,
                                                    ConstCiphertext<DCRTPoly>& ct) const {
    // Computing the baby-step bStep and the giant-step gStep.
    const uint32_t zNDiv2 = (A.size() + 1) / 2;
    // FUNNY that bStep = g...
    const uint32_t bStep = std::ceil(std::sqrt(zNDiv2));
    const uint32_t gStep = std::ceil(static_cast<double>(zNDiv2) / bStep);

    auto cc     = ct->GetCryptoContext();
    auto digits = cc->EvalFastRotationPrecompute(ct);

    // hoisted automorphisms
    std::vector<Ciphertext<DCRTPoly>> fastRotation(bStep - 1);
    std::vector<Ciphertext<DCRTPoly>> fastRotationNeg(bStep - 1);
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(bStep - 1))
    for (uint32_t j = 1; j < bStep; ++j)
        fastRotation[j - 1] = cc->EvalFastRotationExt(ct, j, digits, true);
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(bStep - 1))
    for (uint32_t j = 1; j < bStep; ++j)
        fastRotationNeg[j - 1] = cc->EvalFastRotationExt(ct, -static_cast<int32_t>(j), digits, true);

    auto elementParams = fastRotation[0]->GetElements()[0].GetParams();
    auto sfBFP         = getLinearTransformPlaintextScale(ct);

    // Now initialize cache if not cached
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(A.size()))
    for (size_t i = 0; i < A.size(); ++i) {
        A[i]->GetPlaintext(sfBFP, elementParams);
    }

    const uint32_t M = cc->GetCyclotomicOrder();
    const uint32_t N = cc->GetRingDimension();
    std::vector<uint32_t> map(N);
    Ciphertext<DCRTPoly> result;
    DCRTPoly first;

    // Used for neg loop initialization of inner
    Ciphertext<DCRTPoly> starter;
    for (uint32_t j = 0; j < gStep; ++j) {
        auto inner = z->EvalMult(cc->KeySwitchExt(ct, true), A[bStep * j]->GetPlaintext(sfBFP, elementParams));
        for (uint32_t i = 1; i < bStep; ++i) {
            if (bStep * j + i < zNDiv2)
                z->EvalAddInPlace(
                    inner, z->EvalMult(fastRotation[i - 1], A[bStep * j + i]->GetPlaintext(sfBFP, elementParams)));
        }

        if (j == 0) {
            // We only want to inherit metadata
            starter = inner->Clone();
            for (auto& cv : starter->GetElements())
                cv.SetValuesToZero();

            first         = cc->KeySwitchDownFirstElement(inner);
            auto elements = inner->GetElements();
            elements[0].SetValuesToZero();
            inner->SetElements(std::move(elements));
            result = std::move(inner);
        }
        else {
            inner = cc->KeySwitchDown(inner);
            // Find the automorphism index that corresponds to rotation index index.
            uint32_t autoIndex = FindAutomorphismIndex2nComplex(bStep * j, M);
            PrecomputeAutoMap(N, autoIndex, &map);
            first += inner->GetElements()[0].AutomorphismTransform(autoIndex, map);

            auto&& innerDigits = cc->EvalFastRotationPrecompute(inner);
            z->EvalAddInPlace(result, cc->EvalFastRotationExt(inner, bStep * j, innerDigits, false));
        }
    }
    // Neg
    for (uint32_t j = 0; j < gStep; ++j) {
        Ciphertext<DCRTPoly> inner;
        if (j == 0) {
            // Initialize inner to zero ciphertext
            inner = starter;
        }
        else {
            inner =
                z->EvalMult(cc->KeySwitchExt(ct, true), A[zNDiv2 - 1 + bStep * j]->GetPlaintext(sfBFP, elementParams));
        }
        for (uint32_t i = 1; i < bStep; ++i) {
            if (bStep * j + i < zNDiv2)
                z->EvalAddInPlace(inner,
                                  z->EvalMult(fastRotationNeg[i - 1],
                                              A[zNDiv2 - 1 + bStep * j + i]->GetPlaintext(sfBFP, elementParams)));
        }

        if (j == 0) {
            first += cc->KeySwitchDownFirstElement(inner);
            auto elements = inner->GetElements();
            elements[0].SetValuesToZero();
            inner->SetElements(std::move(elements));
            z->EvalAddInPlace(result, inner);
        }
        else {
            inner = cc->KeySwitchDown(inner);
            // Find the automorphism index that corresponds to rotation index index.
            uint32_t autoIndex = FindAutomorphismIndex2nComplex(-bStep * static_cast<int32_t>(j), M);
            PrecomputeAutoMap(N, autoIndex, &map);
            first += inner->GetElements()[0].AutomorphismTransform(autoIndex, map);

            auto&& innerDigits = cc->EvalFastRotationPrecompute(inner);
            z->EvalAddInPlace(result,
                              cc->EvalFastRotationExt(inner, -bStep * static_cast<int32_t>(j), innerDigits, false));
        }
    }
    result = cc->KeySwitchDown(result);
    result->GetElements()[0] += first;
    return result;
}

//------------------------------------------------------------------------------
// EVALUATION: CoeffsToSlots and SlotsToCoeffs; possibly fused with truncate
//------------------------------------------------------------------------------

static BigFixedPoint getLinearTransformPlaintextScale(ConstCiphertext<DCRTPoly>& ct) {
    auto sfBFP = ct->GetScalingFactorBFP();
    // When at l = 1, we fuse EvalTruncate for C2R
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(ct->GetCryptoParameters());
    auto mulDepth           = cryptoParams->GetMultiplicativeDepth();
    auto l                  = mulDepth - ct->GetLevel();
    if (l == 1) {
        const auto& paramsQ = cryptoParams->GetElementParams()->GetParams();
        auto q0   = BigInteger(paramsQ[0]->GetModulus());
        auto q1   = BigInteger(paramsQ[1]->GetModulus());
        auto qBFP = BigFixedPoint(q0 * q1, 0, false).scaleTo(128);
        return qBFP / sfBFP;
    }

    // Choose the plaintext scale so that the ModReduce following this
    // transform returns the ciphertext to the scale assigned to the next
    // level. This is necessary when q0 and the scaling moduli have different
    // sizes: a ModRaised ciphertext still carries the q0 scale.
    const auto level = ct->GetLevel();
    if (level < mulDepth) {
        const auto& paramsQ = cryptoParams->GetElementParams()->GetParams();
        const auto qLast = paramsQ[paramsQ.size() - 1 - level]->GetModulus();
        const auto qLastBFP = BigFixedPoint(qLast, 0, false).scaleTo(128);
        const auto targetScale = cryptoParams->GetScalingFactorBFP(level + 1);
        return targetScale * qLastBFP / sfBFP;
    }
    return sfBFP;
}

Ciphertext<DCRTPoly> FHEZImpl::EvalLinearTransform(std::vector<ZBootstrapPlaintextCache>& A,
                                                   ConstCiphertext<DCRTPoly>& ct) const {
    // Computing the baby-step bStep and the giant-step gStep.
    const uint32_t slots = A.size();
    const auto& p        = GetBootPrecom(slots);
    const uint32_t bStep = (p.m_paramsEnc.g == 0) ? std::ceil(std::sqrt(slots)) : p.m_paramsEnc.g;
    const uint32_t gStep = std::ceil(static_cast<double>(slots) / bStep);

    auto cc     = ct->GetCryptoContext();
    auto digits = cc->EvalFastRotationPrecompute(ct);

    // hoisted automorphisms
    std::vector<Ciphertext<DCRTPoly>> fastRotation(bStep - 1);
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(bStep - 1))
    for (uint32_t j = 1; j < bStep; ++j)
        fastRotation[j - 1] = cc->EvalFastRotationExt(ct, j, digits, true);

    auto elementParams = fastRotation[0]->GetElements()[0].GetParams();
    auto sfBFP         = getLinearTransformPlaintextScale(ct);

    // Now initialize cache if not cached
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(A.size()))
    for (size_t i = 0; i < A.size(); ++i) {
        A[i]->GetPlaintext(sfBFP, elementParams);
    }

    const uint32_t M = cc->GetCyclotomicOrder();
    const uint32_t N = cc->GetRingDimension();
    std::vector<uint32_t> map(N);
    Ciphertext<DCRTPoly> result;
    DCRTPoly first;
    for (uint32_t j = 0; j < gStep; ++j) {
        auto inner = z->EvalMult(cc->KeySwitchExt(ct, true), A[bStep * j]->GetPlaintext(sfBFP, elementParams));
        for (uint32_t i = 1; i < bStep; ++i) {
            if (bStep * j + i < slots)
                z->EvalAddInPlace(
                    inner, z->EvalMult(fastRotation[i - 1], A[bStep * j + i]->GetPlaintext(sfBFP, elementParams)));
        }

        if (j == 0) {
            first         = cc->KeySwitchDownFirstElement(inner);
            auto elements = inner->GetElements();
            elements[0].SetValuesToZero();
            inner->SetElements(std::move(elements));
            result = std::move(inner);
        }
        else {
            inner = cc->KeySwitchDown(inner);
            // Find the automorphism index that corresponds to rotation index index.
            uint32_t autoIndex = FindAutomorphismIndex2nComplex(bStep * j, M);
            PrecomputeAutoMap(N, autoIndex, &map);
            first += inner->GetElements()[0].AutomorphismTransform(autoIndex, map);

            auto&& innerDigits = cc->EvalFastRotationPrecompute(inner);
            z->EvalAddInPlace(result, cc->EvalFastRotationExt(inner, bStep * j, innerDigits, false));
        }
    }
    result = cc->KeySwitchDown(result);
    result->GetElements()[0] += first;
    return result;
}

Ciphertext<DCRTPoly> FHEZImpl::EvalCoeffsToSlots(const std::vector<std::vector<ZBootstrapPlaintextCache>>& A,
                                                 ConstCiphertext<DCRTPoly>& ctxt) const {
    //const uint32_t slots = ctxt->GetSlots();

    auto cSlots   = ctxt->GetZEncodingParams().getCSlots();
    const auto& p = GetBootPrecom(cSlots).m_paramsEnc;

    // precompute the inner and outer rotations
    std::vector<std::vector<int32_t>> rot_out(p.lvlb, std::vector<int32_t>(p.b + p.bRem));
    std::vector<std::vector<int32_t>> rot_in(p.lvlb, std::vector<int32_t>(p.numRotations + 1));

    int32_t stop    = -1;
    int32_t flagRem = 0;
    if (p.remCollapse != 0) {
        stop    = 0;
        flagRem = 1;

        // remainder corresponds to index 0 in encoding and to last index in decoding
        rot_in[0].resize(p.numRotationsRem + 1);
    }

    auto cc = ctxt->GetCryptoContext();

    const uint32_t M4 = cc->GetCyclotomicOrder() / 4;

    int32_t offset = static_cast<int32_t>((p.numRotations + 1) / 2) - 1;
    for (int32_t s = p.lvlb - 1; s > stop; --s) {
        int32_t scale = (1 << ((s - flagRem) * p.layersCollapse + p.remCollapse));
        for (uint32_t i = 0; i < p.b; ++i)
            rot_out[s][i] = ReduceRotation(scale * p.g * i, M4);
        for (uint32_t j = 0; j < p.g; ++j)
            rot_in[s][j] = ReduceRotation(scale * (j - offset), cSlots);
    }

    if (flagRem == 1) {
        offset = static_cast<int32_t>((p.numRotationsRem + 1) / 2) - 1;
        for (uint32_t i = 0; i < p.bRem; ++i)
            rot_out[stop][i] = ReduceRotation(p.gRem * i, M4);
        for (uint32_t j = 0; j < p.gRem; ++j)
            rot_in[stop][j] = ReduceRotation(j - offset, cSlots);
    }

    auto result = ctxt->Clone();

    uint32_t N = cc->GetRingDimension();
    std::vector<uint32_t> map(N);

    auto algo               = cc->GetScheme();
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());

    // hoisted automorphisms
    const int32_t smax = -1 + p.lvlb;
    for (int32_t s = smax; s > stop; --s) {
        if (s != smax)
            z->ModReduceInPlace(result);

        // computes the NTTs for each CRT limb (for the hoisted automorphisms used later on)
        auto digits = cc->EvalFastRotationPrecompute(result);
        std::vector<Ciphertext<DCRTPoly>> fastRotation(p.g);
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(p.g))
        for (uint32_t j = 0; j < p.g; ++j)
            fastRotation[j] = (rot_in[s][j] != 0) ? cc->EvalFastRotationExt(result, rot_in[s][j], digits, true) :
                                                    cc->KeySwitchExt(result, true);

        auto elementParams = fastRotation[0]->GetElements()[0].GetParams();
        // Here we are never possible to fuse truncate
        auto sfBFP = getLinearTransformPlaintextScale(fastRotation[0]);

        // Now initialize cache if not cached
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(A[s].size()))
        for (size_t i = 0; i < A[s].size(); ++i) {
            A[s][i]->GetPlaintext(sfBFP, elementParams);
        }

        Ciphertext<DCRTPoly> outer;
        DCRTPoly first;
        for (uint32_t i = 0; i < p.b; ++i) {
            // for the first iteration with j=0:
            uint32_t G = p.g * i;
            auto inner = z->EvalMult(fastRotation[0], A[s][G]->GetPlaintext(sfBFP, elementParams));
            // continue the loop
            for (uint32_t j = 1; j < p.g; ++j) {
                if ((G + j) != p.numRotations)
                    z->EvalAddInPlace(inner,
                                      z->EvalMult(fastRotation[j], A[s][G + j]->GetPlaintext(sfBFP, elementParams)));
            }

            if (i == 0) {
                first = cc->KeySwitchDownFirstElement(inner);
                outer = std::move(inner);
                outer->GetElements()[0].SetValuesToZero();
            }
            else {
                if (rot_out[s][i] != 0) {
                    inner = cc->KeySwitchDown(inner);
                    // Find the automorphism index that corresponds to rotation index index.
                    uint32_t autoIndex = FindAutomorphismIndex2nComplex(rot_out[s][i], cc->GetCyclotomicOrder());
                    PrecomputeAutoMap(N, autoIndex, &map);
                    first += inner->GetElements()[0].AutomorphismTransform(autoIndex, map);
                    auto&& innerDigits = cc->EvalFastRotationPrecompute(inner);
                    z->EvalAddInPlace(outer, cc->EvalFastRotationExt(inner, rot_out[s][i], innerDigits, false));
                }
                else {
                    first += cc->KeySwitchDownFirstElement(inner);
                    auto& elements = inner->GetElements();
                    elements[0].SetValuesToZero();
                    z->EvalAddInPlace(outer, inner);
                }
            }
        }
        result = cc->KeySwitchDown(outer);
        result->GetElements()[0] += first;
    }

    if (flagRem == 1) {
        z->ModReduceInPlace(result);

        // computes the NTTs for each CRT limb (for the hoisted automorphisms used later on)
        auto digits = cc->EvalFastRotationPrecompute(result);
        std::vector<Ciphertext<DCRTPoly>> fastRotationRem(p.gRem);
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(p.gRem))
        for (uint32_t j = 0; j < p.gRem; ++j)
            fastRotationRem[j] = (rot_in[stop][j] != 0) ?
                                     cc->EvalFastRotationExt(result, rot_in[stop][j], digits, true) :
                                     cc->KeySwitchExt(result, true);

        auto elementParams = fastRotationRem[0]->GetElements()[0].GetParams();
        auto sfBFP         = getLinearTransformPlaintextScale(fastRotationRem[0]);

        // Now initialize cache if not cached
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(A[stop].size()))
        for (size_t i = 0; i < A[stop].size(); ++i) {
            A[stop][i]->GetPlaintext(sfBFP, elementParams);
        }

        Ciphertext<DCRTPoly> outer;
        DCRTPoly first;
        for (uint32_t i = 0; i < p.bRem; ++i) {
            // for the first iteration with j=0:
            int32_t GRem = p.gRem * i;
            auto inner   = z->EvalMult(fastRotationRem[0], A[stop][GRem]->GetPlaintext(sfBFP, elementParams));
            // continue the loop
            for (uint32_t j = 1; j < p.gRem; ++j) {
                if ((GRem + j) != p.numRotationsRem)
                    z->EvalAddInPlace(
                        inner, z->EvalMult(fastRotationRem[j], A[stop][GRem + j]->GetPlaintext(sfBFP, elementParams)));
            }

            if (i == 0) {
                first = cc->KeySwitchDownFirstElement(inner);
                outer = std::move(inner);
                outer->GetElements()[0].SetValuesToZero();
            }
            else {
                if (rot_out[stop][i] != 0) {
                    inner = cc->KeySwitchDown(inner);
                    // Find the automorphism index that corresponds to rotation index index.
                    uint32_t autoIndex = FindAutomorphismIndex2nComplex(rot_out[stop][i], cc->GetCyclotomicOrder());
                    PrecomputeAutoMap(N, autoIndex, &map);
                    first += inner->GetElements()[0].AutomorphismTransform(autoIndex, map);
                    auto&& innerDigits = cc->EvalFastRotationPrecompute(inner);
                    z->EvalAddInPlace(outer, cc->EvalFastRotationExt(inner, rot_out[stop][i], innerDigits, false));
                }
                else {
                    first += cc->KeySwitchDownFirstElement(inner);
                    auto elements = inner->GetElements();
                    elements[0].SetValuesToZero();
                    inner->SetElements(std::move(elements));
                    z->EvalAddInPlace(outer, inner);
                }
            }
        }
        result = cc->KeySwitchDown(outer);
        result->GetElements()[0] += first;
    }
    return result;
}

Ciphertext<DCRTPoly> FHEZImpl::EvalSlotsToCoeffs(const std::vector<std::vector<ZBootstrapPlaintextCache>>& A,
                                                 ConstCiphertext<DCRTPoly>& ctxt) const {
    auto cSlots   = ctxt->GetZEncodingParams().getCSlots();
    const auto& p = GetBootPrecom(cSlots).m_paramsDec;

    // precompute the inner and outer rotations
    std::vector<std::vector<int32_t>> rot_out(p.lvlb, std::vector<int32_t>(p.b + p.bRem));
    std::vector<std::vector<int32_t>> rot_in(p.lvlb, std::vector<int32_t>(p.numRotations + 1));
    const int32_t flagRem = (p.remCollapse == 0) ? 0 : 1;
    if (flagRem == 1) {
        // remainder corresponds to index 0 in encoding and to last index in decoding
        rot_in[p.lvlb - 1].resize(p.numRotationsRem + 1);
    }

    auto cc = ctxt->GetCryptoContext();

    const uint32_t M4    = cc->GetCyclotomicOrder() / 4;
    const int32_t smax   = p.lvlb - flagRem;
    const int32_t offset = static_cast<int32_t>((p.numRotations + 1) / 2) - 1;
    for (int32_t s = 0; s < smax; ++s) {
        const int32_t scale = 1 << (s * p.layersCollapse);
        for (uint32_t j = 0; j < p.g; ++j)
            rot_in[s][j] = ReduceRotation((j - offset) * scale, M4);
        for (uint32_t i = 0; i < p.b; ++i)
            rot_out[s][i] = ReduceRotation((p.g * i) * scale, M4);
    }

    if (flagRem == 1) {
        const int32_t scaleRem  = 1 << (smax * p.layersCollapse);
        const int32_t offsetRem = static_cast<int32_t>((p.numRotationsRem + 1) / 2) - 1;
        for (uint32_t j = 0; j < p.gRem; ++j)
            rot_in[smax][j] = ReduceRotation((j - offsetRem) * scaleRem, M4);
        for (uint32_t i = 0; i < p.bRem; ++i)
            rot_out[smax][i] = ReduceRotation((p.gRem * i) * scaleRem, M4);
    }

    //  No need for Encrypted Bit Reverse
    auto result = ctxt->Clone();

    uint32_t N = cc->GetRingDimension();
    std::vector<uint32_t> map(N);

    auto algo               = cc->GetScheme();
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());

    // hoisted automorphisms
    for (int32_t s = 0; s < smax; ++s) {
        if (s != 0)
            z->ModReduceInPlace(result);

        // computes the NTTs for each CRT limb (for the hoisted automorphisms used later on)
        auto digits = cc->EvalFastRotationPrecompute(result);
        std::vector<Ciphertext<DCRTPoly>> fastRotation(p.g);
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(p.g))
        for (uint32_t j = 0; j < p.g; ++j)
            fastRotation[j] = (rot_in[s][j] != 0) ? cc->EvalFastRotationExt(result, rot_in[s][j], digits, true) :
                                                    cc->KeySwitchExt(result, true);

        auto elementParams = fastRotation[0]->GetElements()[0].GetParams();
        auto sfBFP         = getLinearTransformPlaintextScale(fastRotation[0]);

        // Now initialize cache if not cached
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(A[s].size()))
        for (size_t i = 0; i < A[s].size(); ++i) {
            A[s][i]->GetPlaintext(sfBFP, elementParams);
        }

        Ciphertext<DCRTPoly> outer;
        DCRTPoly first;
        for (uint32_t i = 0; i < p.b; ++i) {
            // for the first iteration with j=0:
            uint32_t G = i * p.g;
            auto inner = z->EvalMult(fastRotation[0], A[s][G]->GetPlaintext(sfBFP, elementParams));
            // continue the loop
            for (uint32_t j = 1; j < p.g; ++j) {
                if ((G + j) != p.numRotations)
                    z->EvalAddInPlace(inner,
                                      z->EvalMult(fastRotation[j], A[s][G + j]->GetPlaintext(sfBFP, elementParams)));
            }

            if (i == 0) {
                first         = cc->KeySwitchDownFirstElement(inner);
                auto elements = inner->GetElements();
                elements[0].SetValuesToZero();
                inner->SetElements(std::move(elements));
                outer = std::move(inner);
            }
            else {
                if (rot_out[s][i] != 0) {
                    inner = cc->KeySwitchDown(inner);
                    // Find the automorphism index that corresponds to rotation index index.
                    auto autoIndex = FindAutomorphismIndex2nComplex(rot_out[s][i], cc->GetCyclotomicOrder());
                    PrecomputeAutoMap(N, autoIndex, &map);
                    first += inner->GetElements()[0].AutomorphismTransform(autoIndex, map);
                    auto&& innerDigits = cc->EvalFastRotationPrecompute(inner);
                    z->EvalAddInPlace(outer, cc->EvalFastRotationExt(inner, rot_out[s][i], innerDigits, false));
                }
                else {
                    first += cc->KeySwitchDownFirstElement(inner);
                    auto elements = inner->GetElements();
                    elements[0].SetValuesToZero();
                    inner->SetElements(std::move(elements));
                    z->EvalAddInPlace(outer, inner);
                }
            }
        }
        result = cc->KeySwitchDown(outer);
        result->GetElements()[0] += first;
    }

    if (flagRem == 1) {
        z->ModReduceInPlace(result);

        // computes the NTTs for each CRT limb (for the hoisted automorphisms used later on)
        auto digits = cc->EvalFastRotationPrecompute(result);
        std::vector<Ciphertext<DCRTPoly>> fastRotationRem(p.gRem);
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(p.gRem))
        for (uint32_t j = 0; j < p.gRem; ++j)
            fastRotationRem[j] = (rot_in[smax][j] != 0) ?
                                     cc->EvalFastRotationExt(result, rot_in[smax][j], digits, true) :
                                     cc->KeySwitchExt(result, true);

        auto elementParams = fastRotationRem[0]->GetElements()[0].GetParams();
        auto sfBFP         = getLinearTransformPlaintextScale(fastRotationRem[0]);

        // Now initialize cache if not cached
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(A[smax].size()))
        for (size_t i = 0; i < A[smax].size(); ++i) {
            A[smax][i]->GetPlaintext(sfBFP, elementParams);
        }

        Ciphertext<DCRTPoly> outer;
        DCRTPoly first;
        for (uint32_t i = 0; i < p.bRem; ++i) {
            // for the first iteration with j=0:
            uint32_t GRem = i * p.gRem;
            auto inner    = z->EvalMult(fastRotationRem[0], A[smax][GRem]->GetPlaintext(sfBFP, elementParams));
            // continue the loop
            for (uint32_t j = 1; j < p.gRem; ++j) {
                if ((GRem + j) != p.numRotationsRem)
                    z->EvalAddInPlace(
                        inner, z->EvalMult(fastRotationRem[j], A[smax][GRem + j]->GetPlaintext(sfBFP, elementParams)));
            }

            if (i == 0) {
                first         = cc->KeySwitchDownFirstElement(inner);
                auto elements = inner->GetElements();
                elements[0].SetValuesToZero();
                inner->SetElements(std::move(elements));
                outer = std::move(inner);
            }
            else {
                if (rot_out[smax][i] != 0) {
                    inner = cc->KeySwitchDown(inner);
                    // Find the automorphism index that corresponds to rotation index index.
                    auto autoIndex = FindAutomorphismIndex2nComplex(rot_out[smax][i], cc->GetCyclotomicOrder());
                    PrecomputeAutoMap(N, autoIndex, &map);
                    first += inner->GetElements()[0].AutomorphismTransform(autoIndex, map);
                    auto innerDigits = cc->EvalFastRotationPrecompute(inner);
                    z->EvalAddInPlace(outer, cc->EvalFastRotationExt(inner, rot_out[smax][i], innerDigits, false));
                }
                else {
                    first += cc->KeySwitchDownFirstElement(inner);
                    auto elements = inner->GetElements();
                    elements[0].SetValuesToZero();
                    inner->SetElements(std::move(elements));
                    z->EvalAddInPlace(outer, inner);
                }
            }
        }
        result = cc->KeySwitchDown(outer);
        result->GetElements()[0] += first;
    }
    return result;
}

}  // namespace lbcrypto
