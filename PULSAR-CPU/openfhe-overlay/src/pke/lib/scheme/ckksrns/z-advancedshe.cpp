#include "cryptocontext.h"
#include "scheme/ckksrns/ckksrns-utils.h"
#include "scheme/ckksrns/z-leveledshe.h"
#include "scheme/ckksrns/z-advancedshe.h"

namespace lbcrypto {

static uint32_t Degree(const std::vector<BigComplex>& coefficients) {
    uint32_t i = coefficients.size();
    if (i == 0)
        OPENFHE_THROW("Coefficients vector can not be empty");
    while (i > 0) {
        if (!coefficients[--i].equalZero())
            break;
    }
    return i;
}

//===================================================================================
// Chebyshev related
//===================================================================================

/* f and g are vectors of Chebyshev interpolation coefficients of the two polynomials.
We assume their dominant coefficient is not zero. LongDivisionChebyshev returns the
vector of Chebyshev interpolation coefficients for the quotient and remainder of the
division f/g. longDiv is a struct that contains the vectors of coefficients for the
quotient and rest. We assume that the zero-th coefficient is c0, not c0/2 and returns
the same format.*/
static std::shared_ptr<longDiv<BigComplex>> LongDivisionChebyshev(const std::vector<BigComplex>& f,
                                                                  const std::vector<BigComplex>& g) {
    auto n = Degree(f);
    if (n != f.size() - 1)
        OPENFHE_THROW("The dominant coefficient of the divident is zero");
    auto k = Degree(g);
    if (k != g.size() - 1)
        OPENFHE_THROW("The dominant coefficient of the divisor is zero");
    if (n < k)
        return std::make_shared<longDiv<BigComplex>>(std::vector<BigComplex>(1), f);

    auto res = std::make_shared<longDiv<BigComplex>>();

    auto& q = res->q;
    q.resize(n - k + 1);

    auto& r = res->r;
    r       = f;

    std::vector<BigComplex> d;
    d.reserve(g.size() + n);

    while (n > k) {
        d.clear();
        d.resize(n + 1);

        q[n - k] = BigFixedPoint::two() * r.back();
        if (IsNotEqualOne(g[k].convertToComplex()))
            q[n - k] /= g.back();

        if (k == n - k) {
            d.front() = BigFixedPoint::two() * g[n - k];
            for (uint32_t i = 1; i < 2 * k + 1; ++i)
                d[i] = g[std::abs(static_cast<int32_t>(n - k - i))];
        }
        else {
            if (k > (n - k)) {
                d.front() = BigFixedPoint::two() * g[n - k];
                for (uint32_t i = 1; i < k - (n - k) + 1; ++i) {
                    d[i] = g[std::abs(static_cast<int32_t>(n - k - i))] + g[static_cast<size_t>(n - k + i)];
                }
                for (uint32_t i = k - (n - k) + 1; i < n + 1; ++i) {
                    d[i] = g[std::abs(static_cast<int32_t>(i - n + k))];
                }
            }
            else {
                d[n - k] = g.front();
                for (uint32_t i = n - 2 * k; i < n + 1; ++i) {
                    if (i != n - k) {
                        d[i] = g[std::abs(int32_t(i - n + k))];
                    }
                }
            }
        }

        if (IsNotEqualOne(r.back().convertToComplex())) {
            // d *= f[n]
            std::transform(d.begin(), d.end(), d.begin(),
                           std::bind(std::multiplies<BigComplex>(), std::placeholders::_1, r.back()));
        }
        if (IsNotEqualOne(g.back().convertToComplex())) {
            // d /= g[k]
            std::transform(d.begin(), d.end(), d.begin(),
                           std::bind(std::divides<BigComplex>(), std::placeholders::_1, g.back()));
        }
        // f-=d
        std::transform(r.begin(), r.end(), d.begin(), r.begin(), std::minus<BigComplex>());
        if (r.size() > 1) {
            n = Degree(r);
            r.resize(n + 1);
        }
    }

    if (n == k) {
        d = g;

        q.front() = r.back();
        if (IsNotEqualOne(g.back().convertToComplex())) {
            q.front() /= g.back();  // q[0] /= g[k]
        }
        if (IsNotEqualOne(r.back().convertToComplex())) {
            // d *= f[n]
            std::transform(d.begin(), d.end(), d.begin(),
                           std::bind(std::multiplies<BigComplex>(), std::placeholders::_1, r.back()));
        }
        if (IsNotEqualOne(g.back().convertToComplex())) {
            // d /= g[k]
            std::transform(d.begin(), d.end(), d.begin(),
                           std::bind(std::divides<BigComplex>(), std::placeholders::_1, g.back()));
        }
        // f-=d
        std::transform(r.begin(), r.end(), d.begin(), r.begin(), std::minus<BigComplex>());
        if (r.size() > 1) {
            n = Degree(r);
            r.resize(n + 1);
        }
    }
    q.front() *= BigFixedPoint::two();  // Because we want to have [c0] in the last spot, not [c0/2]
    return res;
}

std::shared_ptr<seriesPowers<DCRTPoly>> AdvancedZImpl::internalEvalChebyPolysPS(ConstCiphertext<DCRTPoly>& x,
                                                                                uint32_t degree) {
    auto degs  = ComputeDegreesPS(degree);
    uint32_t k = degs[0];
    uint32_t m = degs[1];

    auto cc = x->GetCryptoContext();
    std::vector<Ciphertext<DCRTPoly>> T(k);
    // no linear transformation is needed if a = -1, b = 1
    // T_1(y) = y
    T[0] = x->Clone();

    // Computes Chebyshev polynomials up to degree k
    // for y: T_1(y) = y, T_2(y), ... , T_k(y)
    // uses binary tree multiplication
    for (uint32_t i = 2; i <= k; ++i) {
        if (i & 0x1) {  // if i is odd
            // compute T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
            T[i - 1] = z->EvalMultWithAdjust(T[i / 2 - 1], T[i / 2]);
            z->EvalAddInPlace(T[i - 1], T[i - 1]);
            z->ModReduceInPlace(T[i - 1]);
            // TODO: maybe we can hoist T0Adjust
            z->EvalSubWithAdjustInPlace(T[i - 1], T[0]);
        }
        else {
            // compute T_{2i}(y) = 2*T_i(y)^2 - 1
            T[i - 1] = z->EvalMult(T[i / 2 - 1], T[i / 2 - 1]);
            z->EvalAddInPlace(T[i - 1], T[i - 1]);
            z->ModReduceInPlace(T[i - 1]);
            auto one = BigFixedPoint::one();
            z->EvalAddInPlaceInC(T[i - 1], -one);
        }
    }

    // Adjust them to have same depth and scaling factor
    for (uint32_t i = 1; i < k; ++i) {
        if (T[i - 1]->GetLevel() != T[k - 1]->GetLevel()) {
            T[i - 1] = z->AdjustCiphertext(T[i - 1], T[k - 1]);
        }
        else {
            if (!T[i - 1]->GetScalingFactorBFP().almostEqual(T[k - 1]->GetScalingFactorBFP())) {
                OPENFHE_THROW("Scaling factors are not equal!");
            }
        }
    }

    std::vector<Ciphertext<DCRTPoly>> T2(m);
    // T2[0] is used as a placeholder
    T2[0] = T.back();

    // computes T_{k(2*m - 1)}(y)
    auto T2km1 = T.back();

    for (uint32_t i = 1; i < m; ++i) {
        // Compute the Chebyshev polynomials T_k(y), T_{2k}(y), T_{4k}(y), ... , T_{2^{m-1}k}(y)
        T2[i] = z->EvalMult(T2[i - 1], T2[i - 1]);
        z->EvalAddInPlace(T2[i], T2[i]);
        z->ModReduceInPlace(T2[i]);
        auto one = BigFixedPoint::one();
        z->EvalAddInPlaceInC(T2[i], -one);

        // compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
        T2km1 = z->EvalMultWithAdjust(T2km1, T2[i]);
        z->EvalAddInPlace(T2km1, T2km1);
        z->ModReduceInPlace(T2km1);
        z->EvalSubWithAdjustInPlace(T2km1, T2[0]);
    }

    return std::make_shared<seriesPowers<DCRTPoly>>(std::move(T), std::move(T2), std::move(T2km1), k, m);
}

Ciphertext<DCRTPoly> AdvancedZImpl::EvalPartialLinearWSum(const std::vector<Ciphertext<DCRTPoly>>& ciphertexts,
                                                          const std::vector<BigComplex>& constants, uint32_t limit) {
    if (0 == limit)
        limit = ciphertexts.size();

    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(ciphertexts[0]->GetCryptoParameters());

    auto cc = ciphertexts[0]->GetCryptoContext();

    std::vector<Ciphertext<DCRTPoly>> cts(limit);
    cts[0] = ciphertexts[0]->Clone();
    // Check to see if input ciphertexts are of same level
    // and adjust if needed to the max level among them
    uint32_t maxLevel = cts[0]->GetLevel();
    uint32_t maxIdx   = 0;
    for (uint32_t i = 1; i < limit; ++i) {
        cts[i] = ciphertexts[i]->Clone();
        if (cts[i]->GetLevel() > maxLevel) {
            maxLevel = cts[i]->GetLevel();
            maxIdx   = i;
        }
    }

    auto algo = cc->GetScheme();
    auto& ctm = cts[maxIdx];
    for (uint32_t i = 0; i < maxIdx; ++i)
        cts[i] = z->AdjustCiphertext(cts[i], ctm);
    for (uint32_t i = maxIdx + 1; i < limit; ++i)
        cts[i] = z->AdjustCiphertext(cts[i], ctm);

    z->EvalMultInPlaceInC(cts[0], constants[1]);
    for (uint32_t i = 1; i < limit; ++i) {
        z->EvalMultInPlaceInC(cts[i], constants[i + 1]);
        z->EvalAddInPlace(cts[0], cts[i]);
    }
    z->ModReduceInPlace(cts[0]);
    return cts[0];
}

Ciphertext<DCRTPoly> AdvancedZImpl::InnerEvalChebyshevPS(ConstCiphertext<DCRTPoly>& x,
                                                         const std::vector<BigComplex>& coefficients, uint32_t k,
                                                         uint32_t m, const std::vector<Ciphertext<DCRTPoly>>& T,
                                                         const std::vector<Ciphertext<DCRTPoly>>& T2) {
    // Compute k*2^{m-1}-k because we use it a lot
    uint32_t k2m2k = k * (1 << (m - 1)) - k;

    // Divide coefficients by T^{k*2^{m-1}}
    std::vector<BigComplex> Tkm(k2m2k + k + 1);
    Tkm.back() = BigFixedPoint::one();
    auto divqr = LongDivisionChebyshev(coefficients, Tkm);

    // Subtract x^{k(2^{m-1} - 1)} from r
    auto& r2 = divqr->r;
    if (uint32_t n = Degree(r2); static_cast<int32_t>(k2m2k - n) <= 0) {
        r2.resize(n + 1);
        r2[k2m2k] -= BigFixedPoint::one();
    }
    else {
        r2.resize(k2m2k + 1);
        r2.back() = -BigFixedPoint::one();
    }

    auto divcs = LongDivisionChebyshev(r2, divqr->q);
    auto cc    = x->GetCryptoContext();

    Ciphertext<DCRTPoly> cu, qu, su;

    {
        // Evaluate q and s2 at u.
        // If their degrees are larger than k, then recursively apply the Paterson-Stockmeyer algorithm.
        if (Degree(divqr->q) > k) {
            qu = InnerEvalChebyshevPS(x, divqr->q, k, m - 1, T, T2);
        }
        else {
            // dq = k from construction
            // perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients

            // the highest order coefficient will always be a power of two up to 2^{m-1} because q is "monic" but the Chebyshev rule adds a factor of 2
            // we don't need to increase the depth by multiplying the highest order coefficient, but instead checking and summing, since we work with m <= 4.
            qu                   = T[k - 1]->Clone();
            const uint32_t limit = std::log2(divqr->q.back().getReal().convertToDouble());
            for (uint32_t i = 0; i < limit; ++i)
                z->EvalAddInPlace(qu, qu);

            // adds the free term (at x^0)
            z->EvalAddInPlaceInC(qu, divqr->q.front() / BigFixedPoint::two());
            // The number of levels of qu is the same as the number of levels of T[k-1] + 1.
            // Will only get here when m = 2, so the number of levels of qu and T2[m-1] will be the same.

            divqr->q.resize(k);
            if (uint32_t n = Degree(divqr->q); n > 0)
                z->EvalAddWithAdjustInPlace(qu, EvalPartialLinearWSum(T, divqr->q, n));
        }
    }

    {
        // Add x^{k(2^{m-1} - 1)} to s
        auto& s2 = divcs->r;
        s2.resize(k2m2k + 1);
        s2.back() = BigFixedPoint::one();

        if (Degree(s2) > k) {
            su = InnerEvalChebyshevPS(x, s2, k, m - 1, T, T2);
        }
        else {
            // the highest order coefficient will always be 1 because s2 is monic.
            su = T[k - 1]->Clone();

            // ds = k from construction
            // perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
            s2.resize(k);
            if (uint32_t n = Degree(s2); n > 0)
                z->EvalAddWithAdjustInPlace(su, EvalPartialLinearWSum(T, s2, n));

            // adds the free term (at x^0)
            z->EvalAddInPlaceInC(su, s2.front() / BigFixedPoint::two());

            // The number of levels of su is the same as the number of levels of T[k-1] or T[k-1] + 1. Need to reduce it to T2[m-1] + 1.
            // New Code: Maybe automatic adjustment
            //gLevelReduceInPlace(su);
        }
    }

    if (uint32_t n = Degree(divcs->q); n >= 1) {
        if (n == 1) {
            if (IsNotEqualOne(divcs->q[1].convertToComplex())) {
                cu = z->EvalMultInC(T.front(), divcs->q[1]);
                z->ModReduceInPlace(cu);
            }
            else {
                cu = T.front()->Clone();
            }
        }
        else {
            cu = EvalPartialLinearWSum(T, divcs->q, n);
        }

        // adds the free term (at x^0)
        z->EvalAddInPlaceInC(cu, divcs->q.front() / BigFixedPoint::two());

        // Need to reduce levels up to the level of T2[m-1].
        // New code: Need adjust below
        // gLevelReduceInPlace(cu, (T2[m - 1]->GetLevel() - cu->GetLevel()));
    }

    cu = cu ? z->EvalAddWithAdjust(T2[m - 1], cu) : z->EvalAddInC(T2[m - 1], divcs->q.front() / BigFixedPoint::two());

    auto result = z->EvalMultWithAdjust(cu, qu);
    z->ModReduceInPlace(result);
    z->EvalAddWithAdjustInPlace(result, su);
    return result;
}

Ciphertext<DCRTPoly> AdvancedZImpl::internalEvalChebyshevSeriesPSWithPrecomp(
    const std::shared_ptr<seriesPowers<DCRTPoly>>& ctxtPolys, const std::vector<BigComplex>& coefficients) {
    auto& T     = ctxtPolys->powersRe;
    auto& T2    = ctxtPolys->powers2Re;
    auto& T2km1 = ctxtPolys->power2km1Re;
    auto k      = ctxtPolys->k;
    auto m      = ctxtPolys->m;

    // Compute k*2^{m-1}-k because we use it a lot
    uint32_t k2m2k = k * (1 << (m - 1)) - k;

    // Add T^{k(2^m - 1)}(y) to the polynomial that has to be evaluated
    auto f2 = coefficients;
    f2.resize(Degree(f2) + 1);
    f2.resize(2 * k2m2k + k + 1);
    f2.back() = BigFixedPoint::one();

    return z->EvalSubWithAdjust(InnerEvalChebyshevPS(T[0], f2, k, m, T, T2), T2km1);
}

Ciphertext<DCRTPoly> AdvancedZImpl::EvalChebyshevSeriesPS(ConstCiphertext<DCRTPoly>& x,
                                                          const std::vector<BigComplex>& coeffs) {
    return internalEvalChebyshevSeriesPSWithPrecomp(internalEvalChebyPolysPS(x, Degree(coeffs)), coeffs);
}

//===================================================================================
// PolyPS related
//===================================================================================

std::shared_ptr<seriesPowers<DCRTPoly>> AdvancedZImpl::internalEvalPowersPS(ConstCiphertext<DCRTPoly>& x,
                                                                            uint32_t degree) {
    auto degs  = ComputeDegreesPS(degree);
    uint32_t k = degs[0];
    uint32_t m = degs[1];

    std::vector<Ciphertext<DCRTPoly>> powers(k);
    powers[0] = x->Clone();

    // computes all powers up to k for x
    uint32_t powerOf2 = 2;
    uint32_t rem      = 0;
    for (uint32_t i = 2; i <= k; ++i) {
        if (rem == 0) {
            powers[i - 1] = z->EvalMult(powers[(powerOf2 >> 1) - 1], powers[(powerOf2 >> 1) - 1]);
        }
        else {
            powers[i - 1] = z->EvalMultWithAdjust(powers[powerOf2 - 1], powers[rem - 1]);
        }

        if (++rem == powerOf2) {
            powerOf2 <<= 1;
            rem = 0;
        }
        z->ModReduceInPlace(powers[i - 1]);
    }

    // Adjust them to have same depth and scaling factor
    for (uint32_t i = 1; i < k; ++i) {
        if (powers[i - 1]->GetLevel() != powers[k - 1]->GetLevel()) {
            powers[i - 1] = z->AdjustCiphertext(powers[i - 1], powers[k - 1]);
        }
        else {
            assert(powers[i - 1]->GetScalingFactorBFP().almostEqual(powers[k - 1]->GetScalingFactorBFP()) &&
                   "Scaling factors are not equal!");
        }
    }

    // computes powers of form k*2^i for x and the product of the powers in power2, that yield x^{k(2*m - 1)}
    std::vector<Ciphertext<DCRTPoly>> powers2(m);
    powers2[0] = powers.back();

    auto power2km1 = powers.back();

    for (uint32_t i = 1; i < m; ++i) {
        powers2[i] = z->EvalMult(powers2[i - 1], powers2[i - 1]);
        z->ModReduceInPlace(powers2[i]);
        power2km1 = z->EvalMultWithAdjust(powers2[i], power2km1);
        z->ModReduceInPlace(power2km1);
    }

    return std::make_shared<seriesPowers<DCRTPoly>>(std::move(powers), std::move(powers2), std::move(power2km1), k, m);
}

static std::shared_ptr<longDiv<BigComplex>> LongDivisionPoly(const std::vector<BigComplex>& f,
                                                             const std::vector<BigComplex>& g) {
    auto n = Degree(f);
    if (n != f.size() - 1)
        OPENFHE_THROW("The dominant coefficient of the divident is zero");
    auto k = Degree(g);
    if (k != g.size() - 1)
        OPENFHE_THROW("The dominant coefficient of the divisor is zero");
    if (n < k)
        return std::make_shared<longDiv<BigComplex>>(std::vector<BigComplex>(1), f);

    auto res = std::make_shared<longDiv<BigComplex>>();

    auto& q = res->q;
    q.resize(n - k + 1);

    auto& r = res->r;
    r       = f;

    std::vector<BigComplex> d;
    d.reserve(g.size() + n);

    while (n >= k) {
        // d is g padded with zeros before up to n
        d.clear();
        d.resize(n - k);
        d.insert(d.end(), g.begin(), g.end());

        q[n - k] = r.back();
        if (IsNotEqualOne(g[k].convertToComplex()))
            q[n - k] /= g.back();

        // d *= q[n - k]
        std::transform(d.begin(), d.end(), d.begin(),
                       std::bind(std::multiplies<BigComplex>(), std::placeholders::_1, q[n - k]));
        // f-=d
        std::transform(r.begin(), r.end(), d.begin(), r.begin(), std::minus<BigComplex>());
        if (r.size() > 1) {
            n = Degree(r);
            r.resize(n + 1);
        }
    }
    return res;
}

Ciphertext<DCRTPoly> AdvancedZImpl::InnerEvalPolyPS(ConstCiphertext<DCRTPoly>& x,
                                                    const std::vector<BigComplex>& coefficients, uint32_t k, uint32_t m,
                                                    const std::vector<Ciphertext<DCRTPoly>>& powers,
                                                    const std::vector<Ciphertext<DCRTPoly>>& powers2) {
    // Compute k*2^m because we use it often
    uint32_t k2m2k = k * (1 << (m - 1)) - k;

    // Divide coefficients by x^{k*2^{m-1}}
    std::vector<BigComplex> xkm(k2m2k + k + 1);
    xkm.back() = BigFixedPoint::one();
    auto divqr = LongDivisionPoly(coefficients, xkm);

    // Subtract x^{k(2^{m-1} - 1)} from r
    auto& r2 = divqr->r;
    if (auto n = Degree(r2); static_cast<int32_t>(k2m2k - n) <= 0) {
        r2.resize(n + 1);
        r2[k2m2k] -= BigFixedPoint::one();
    }
    else {
        r2.resize(k2m2k + 1);
        r2.back() = -BigFixedPoint::one();
    }

    auto divcs = LongDivisionPoly(r2, divqr->q);
    auto cc    = x->GetCryptoContext();

    Ciphertext<DCRTPoly> cu, qu, su;

#pragma omp task shared(qu)
    {
        // Evaluate q and s2 at u.
        // If their degrees are larger than k, then recursively apply the Paterson-Stockmeyer algorithm.

        if (Degree(divqr->q) > k) {
            qu = InnerEvalPolyPS(x, divqr->q, k, m - 1, powers, powers2);
        }
        else {
            qu = z->EvalAddInC(powers[k - 1], divqr->q.front());
            divqr->q.resize(k);
            if (uint32_t n = Degree(divqr->q); n > 0)
                z->EvalAddWithAdjustInPlace(qu, EvalPartialLinearWSum(powers, divqr->q, n));
        }
    }

#pragma omp task shared(su)
    {
        // Add x^{k(2^{m-1} - 1)} to s
        auto& s2 = divcs->r;
        s2.resize(k2m2k + 1);
        s2.back() = BigFixedPoint::one();

        if (Degree(s2) > k) {
            su = InnerEvalPolyPS(x, s2, k, m - 1, powers, powers2);
        }
        else {
            su = z->EvalAddInC(powers[k - 1], s2.front());
            s2.resize(k);
            if (uint32_t n = Degree(s2); n > 0)
                z->EvalAddWithAdjustInPlace(su, EvalPartialLinearWSum(powers, s2, n));
        }
    }

    if (uint32_t n = Degree(divcs->q); n == 0) {
        cu = z->EvalAddInC(powers2[m - 1], divcs->q.front());
    }
    else if (n == 1) {
        if (IsNotEqualOne(divcs->q[1].convertToComplex())) {
            cu = z->EvalMultInC(powers.front(), divcs->q[1]);
            z->ModReduceInPlace(cu);
            cu = z->EvalAddWithAdjust(cu, powers2[m - 1]);
        }
        else {
            cu = z->EvalAddWithAdjust(powers2[m - 1], powers.front());
        }
        z->EvalAddInPlaceInC(cu, divcs->q.front());
    }
    else {
        cu = z->EvalAddWithAdjust(powers2[m - 1], EvalPartialLinearWSum(powers, divcs->q, n));
        z->EvalAddInPlaceInC(cu, divcs->q.front());
    }

#pragma omp taskwait

    auto result = z->EvalMultWithAdjust(cu, qu);
    z->ModReduceInPlace(result);
    z->EvalAddWithAdjustInPlace(result, su);
    return result;
}

Ciphertext<DCRTPoly> AdvancedZImpl::internalEvalPolyPSWithPrecomp(
    const std::shared_ptr<seriesPowers<DCRTPoly>>& ctxtPowers, const std::vector<BigComplex>& coefficients) {
    auto& powers    = ctxtPowers->powersRe;
    auto& powers2   = ctxtPowers->powers2Re;
    auto& power2km1 = ctxtPowers->power2km1Re;
    auto k          = ctxtPowers->k;
    auto m          = ctxtPowers->m;

    // Compute k*2^{m-1}-k because we use it a lot
    uint32_t k2m2k = k * (1 << (m - 1)) - k;

    // Add T^{k(2^m - 1)}(y) to the polynomial that has to be evaluated
    auto f2 = coefficients;
    f2.resize(Degree(f2) + 1);
    f2.resize(2 * k2m2k + k + 1);
    f2.back() = BigFixedPoint::one();

    Ciphertext<DCRTPoly> result;
#pragma omp parallel num_threads(OpenFHEParallelControls.GetThreadLimit(6 * m + 2))
    {
#pragma omp single
        result = z->EvalSubWithAdjust(InnerEvalPolyPS(powers[0], f2, k, m, powers, powers2), power2km1);
    }
    return result;
}

std::shared_ptr<seriesPowers<DCRTPoly>> AdvancedZImpl::EvalPowers(ConstCiphertext<DCRTPoly>& ciphertext,
                                                                  const std::vector<BigComplex>& coefficients) {
    return internalEvalPowersPS(ciphertext, Degree(coefficients));
}

std::shared_ptr<seriesPowers<DCRTPoly>>
AdvancedZImpl::EvalPowersLattigoDepth6(
    ConstCiphertext<DCRTPoly>& x,
    const std::vector<BigComplex>& coefficients) {
    if (!x || Degree(coefficients) != 61) {
        OPENFHE_THROW(
            "the Lattigo depth-six power basis requires a degree-61 polynomial");
    }

    // Lattigo evaluates this Hermite polynomial with logDegree=6 and a
    // depth-six split. OpenFHE's generic selector chooses k=9,m=3 (depth 7)
    // for degree 61. The exact same polynomial fits k*(2^m-1)=62 with
    // k=2,m=5, which restores the six-level Lattigo schedule.
    constexpr uint32_t k = 2;
    constexpr uint32_t m = 5;

    std::vector<Ciphertext<DCRTPoly>> powers(k);
    powers[0] = x->Clone();
    powers[1] = z->EvalMult(powers[0], powers[0]);
    z->ModReduceInPlace(powers[1]);
    powers[0] = z->AdjustCiphertext(powers[0], powers[1]);

    std::vector<Ciphertext<DCRTPoly>> powers2(m);
    powers2[0] = powers.back();
    auto power2km1 = powers.back();
    for (uint32_t i = 1; i < m; ++i) {
        powers2[i] = z->EvalMult(powers2[i - 1], powers2[i - 1]);
        z->ModReduceInPlace(powers2[i]);
        power2km1 = z->EvalMultWithAdjust(powers2[i], power2km1);
        z->ModReduceInPlace(power2km1);
    }

    return std::make_shared<seriesPowers<DCRTPoly>>(
        std::move(powers), std::move(powers2), std::move(power2km1), k, m);
}

Ciphertext<DCRTPoly> AdvancedZImpl::EvalPolyWithPrecomp(std::shared_ptr<seriesPowers<DCRTPoly>> ctxtPowers,
                                                        const std::vector<BigComplex>& coeffs) {
    return internalEvalPolyPSWithPrecomp(ctxtPowers, coeffs);
}

}  // namespace lbcrypto