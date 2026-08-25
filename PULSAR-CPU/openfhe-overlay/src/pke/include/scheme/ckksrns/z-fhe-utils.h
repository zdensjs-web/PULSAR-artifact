#ifndef SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_UTILS_H_
#define SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_UTILS_H_

#include "math/hal/bigfixedpoint.h"
#include "ckksrns-utils.h"

namespace lbcrypto {

/**
 * Extracts shifted diagonal of matrix A.
 *
 * @param &A square linear map.
 * @param index the index by which the diagonal shifted.
 *
 * @return the vector corresponding to the shifted diagonal
 */
BigCVector ExtractShiftedDiagonal(const BigCMatrix& A, int i);

/**
 * Computes the coefficients for the FFT encoding for CoeffEncodingCollapse such that every
 * iteration occupies one level.
 *
 * @param pows vector of roots of unity powers.
 * @param rotGroup rotation group indices to appropriately choose the elements of pows to compute iFFT.
 * @param flag_i flag that is 0 when we compute the coefficients for conj(U_0^T) and is 1 for conj(i*U_0^T).
 */
BigCMatrix CoeffEncodingOneLevel(const BigCVector& pows, const std::vector<uint32_t>& rotGroup, bool flag_i);

/**
 * Computes the coefficients for the FFT decoding for CoeffDecodingCollapse such that every
 * iteration occupies one level.
 *
 * @param pows vector of roots of unity powers.
 * @param rotGroup rotation group indices to appropriately choose the elements of pows to compute iFFT.
 * @param flag_i flag that is 0 when we compute the coefficients for U_0 and is 1 for i*U_0.
 */
BigCMatrix CoeffDecodingOneLevel(const BigCVector& pows, const std::vector<uint32_t>& rotGroup, bool flag_i);

/**
 * Computes the coefficients for the given level budget for the FFT encoding. Needed in
 * EvalLTFFTPrecomputeEncoding.
 *
 * @param pows vector of roots of unity powers.
 * @param rotGroup rotation group indices to appropriately choose the elements of pows to compute iFFT.
 * @param levelBudget the user specified budget for levels.
 * @param flag_i flag that is 0 when we compute the coefficients for conj(U_0^T) and is 1 for conj(i*U_0^T).
 */
std::vector<std::vector<std::vector<BigComplex>>> CoeffEncodingCollapse(const BigCVector& pows,
                                                                        const std::vector<uint32_t>& rotGroup,
                                                                        uint32_t levelBudget, bool flag_i);

/**
 * Computes the coefficients for the given level budget for the FFT decoding. Needed in
 * EvalLTFFTPrecomputeDecoding.
 *
 * @param pows vector of roots of unity powers.
 * @param rotGroup rotation group indices to appropriately choose the elements of pows to compute FFT.
 * @param levelBudget the user specified budget for levels.
 * @param flag_i flag that is 0 when we compute the coefficients for U_0 and is 1 for i*U_0.
 */
std::vector<std::vector<std::vector<BigComplex>>> CoeffDecodingCollapse(const BigCVector& pows,
                                                                        const std::vector<uint32_t>& rotGroup,
                                                                        uint32_t levelBudget, bool flag_i);

}  // namespace lbcrypto

#endif  //SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_UTILS_H_