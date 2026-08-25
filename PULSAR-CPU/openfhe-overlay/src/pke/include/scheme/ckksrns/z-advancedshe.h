#ifndef SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_ADVANCEDSHE_H_
#define SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_ADVANCEDSHE_H_

#include "math/hal/bigfixedpoint.h"
#include "cryptocontext.h"
#include "z-leveledshe.h"

namespace lbcrypto {

class AdvancedZImpl {
public:
    AdvancedZImpl(LeveledZ z) : z(z) {}

    Ciphertext<DCRTPoly> EvalChebyshevSeriesPS(ConstCiphertext<DCRTPoly>& x, const std::vector<BigComplex>& coeffs);

    std::shared_ptr<seriesPowers<DCRTPoly>> EvalPowers(ConstCiphertext<DCRTPoly>& ciphertext,
                                                       const std::vector<BigComplex>& coefficients);

    // Degree-61 Paterson-Stockmeyer power basis with the Lattigo
    // depth-six schedule (k=2,m=5).
    std::shared_ptr<seriesPowers<DCRTPoly>> EvalPowersLattigoDepth6(
        ConstCiphertext<DCRTPoly>& ciphertext,
        const std::vector<BigComplex>& coefficients);

    Ciphertext<DCRTPoly> EvalPolyWithPrecomp(std::shared_ptr<seriesPowers<DCRTPoly>> ctxtPowers,
                                             const std::vector<BigComplex>& coeffs);

private:
    // WSum related
    Ciphertext<DCRTPoly> EvalPartialLinearWSum(const std::vector<Ciphertext<DCRTPoly>>& ciphertexts,
                                               const std::vector<BigComplex>& constants, uint32_t limit = 0);

    // ChebyshevPS related
    std::shared_ptr<seriesPowers<DCRTPoly>> internalEvalChebyPolysPS(ConstCiphertext<DCRTPoly>& x, uint32_t degree);

    Ciphertext<DCRTPoly> InnerEvalChebyshevPS(ConstCiphertext<DCRTPoly>& x, const std::vector<BigComplex>& coefficients,
                                              uint32_t k, uint32_t m, const std::vector<Ciphertext<DCRTPoly>>& T,
                                              const std::vector<Ciphertext<DCRTPoly>>& T2);

    Ciphertext<DCRTPoly> internalEvalChebyshevSeriesPSWithPrecomp(
        const std::shared_ptr<seriesPowers<DCRTPoly>>& ctxtPolys, const std::vector<BigComplex>& coefficients);

    // PS related
    std::shared_ptr<seriesPowers<DCRTPoly>> internalEvalPowersPS(ConstCiphertext<DCRTPoly>& x, uint32_t degree);

    Ciphertext<DCRTPoly> InnerEvalPolyPS(ConstCiphertext<DCRTPoly>& x, const std::vector<BigComplex>& coefficients,
                                         uint32_t k, uint32_t m, const std::vector<Ciphertext<DCRTPoly>>& powers,
                                         const std::vector<Ciphertext<DCRTPoly>>& powers2);

    Ciphertext<DCRTPoly> internalEvalPolyPSWithPrecomp(const std::shared_ptr<seriesPowers<DCRTPoly>>& ctxtPowers,
                                                       const std::vector<BigComplex>& coefficients);

private:
    LeveledZ z;
};

using AdvancedZ = std::shared_ptr<AdvancedZImpl>;

}  // namespace lbcrypto

#endif  //SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_ADVANCEDSHE_H_