#include "scheme/ckksrns/z-fhe-utils.h"

namespace lbcrypto {

namespace {  // this namespace should stay unnamed

/**
 * Computes parameters to ensure the encoding and decoding computations take exactly the
 * specified number of levels. More specifically, it returns a vector than contains
 * layers (the number of layers to collapse in one level), rows (how many such levels),
 * rem (the number of layers remaining to be collapsed in one level)
 *
 * @param logSlots the base 2 logarithm of the number of slots.
 * @param budget the allocated level budget for the computation.
 */
std::vector<uint32_t> SelectLayers(uint32_t logSlots, uint32_t budget = 4) {
    uint32_t layers = std::ceil(static_cast<double>(logSlots) / budget);
    uint32_t rows   = logSlots / layers;
    uint32_t rem    = logSlots % layers;
    uint32_t dim    = rows + (rem != 0);

    // the above choice ensures dim <= budget
    if (dim < budget) {
        layers -= 1;
        rows = logSlots / layers;
        rem  = logSlots - rows * layers;
        dim  = rows + (rem != 0);

        // the above choice endures dim >=budget
        if (dim > budget) {
            while (dim != budget) {
                --rows;
                rem = logSlots - rows * layers;
                dim = rows + (rem != 0);
            }
        }
    }
    return {layers, rows, rem};
}

}  // namespace

BigCVector ExtractShiftedDiagonal(const BigCMatrix& A, int index) {
    uint32_t cols = A[0].size();
    uint32_t rows = A.size();
    BigCVector result(cols);
    for (uint32_t k = 0; k < cols; ++k)
        result[k] = A[k % rows][(k + index) % cols];
    return result;
}

BigCMatrix CoeffEncodingOneLevel(const BigCVector& pows, const std::vector<uint32_t>& rotGroup, bool flag_i) {
    //static const BigComplex I(BigFixedPoint::zero(), BigFixedPoint::one());
    //static const BigComplex neg_exp_M_PI = std::exp(-M_PI / 2 * I);
    static const BigComplex neg_exp_M_PI(BigFixedPoint::zero(), -BigFixedPoint::one());

    const uint32_t dim   = pows.size() - 1;
    const uint32_t slots = rotGroup.size();

    // Each outer iteration from the FFT algorithm can be written a weighted sum of
    // three terms: the input shifted right by a power of two, the unshifted input,
    // and the input shifted left by a power of two. For each outer iteration
    // (log2(size) in total), the matrix coeff stores the coefficients in the
    // following order: the coefficients associated to the input shifted right,
    // the coefficients for the non-shifted input and the coefficients associated
    // to the input shifted left.
    const auto log2slots = static_cast<uint32_t>(std::log2(slots));
    BigCMatrix coeff(3 * log2slots, BigCVector(slots));

    for (uint32_t m = slots; m > 1; m >>= 1) {
        uint32_t s   = std::log2(m) - 1;
        auto& coeff0 = coeff[s];
        auto& coeff1 = coeff[(s += log2slots)];
        auto& coeff2 = coeff[(s += log2slots)];

        const BigComplex b = flag_i && (m == 2) ? neg_exp_M_PI : BigFixedPoint::one();
        for (uint32_t k = 0; k < slots; k += m) {
            const uint32_t lenq  = m << 2;
            const uint32_t lenh  = m >> 1;
            const uint32_t klenh = k + lenh;
            std::fill(coeff2.begin() + k, coeff2.begin() + klenh, b);  // shifted left
            std::fill(coeff1.begin() + k, coeff1.begin() + klenh, b);  // not shifted
            for (uint32_t j = 0, jklenh = klenh; j < lenh; ++j, ++jklenh) {
                const auto w   = b * pows[(lenq - (rotGroup[j] % lenq)) * (dim / lenq)];
                coeff1[jklenh] = -w;  // not shifted
                coeff0[jklenh] = w;   // shifted right
            }
        }
    }
    return coeff;
}

BigCMatrix CoeffDecodingOneLevel(const BigCVector& pows, const std::vector<uint32_t>& rotGroup, bool flag_i) {
    //constexpr std::complex<double> I(0.0, 1.0);
    //static const std::complex<double> pos_exp_M_PI = std::exp(M_PI / 2 * I);
    static const BigComplex pos_exp_M_PI(BigFixedPoint::zero(), BigFixedPoint::one());

    const uint32_t dim   = pows.size() - 1;
    const uint32_t slots = rotGroup.size();

    // Each outer iteration from the FFT algorithm can be written a weighted sum of
    // three terms: the input shifted right by a power of two, the unshifted input,
    // and the input shifted left by a power of two. For each outer iteration
    // (log2(size) in total), the matrix coeff stores the coefficients in the
    // following order: the coefficients associated to the input shifted right,
    // the coefficients for the non-shifted input and the coefficients associated
    // to the input shifted left.
    const auto log2slots = static_cast<uint32_t>(std::log2(slots));
    BigCMatrix coeff(3 * log2slots, BigCVector(slots));

    for (uint32_t m = 2; m <= slots; m <<= 1) {
        uint32_t s   = std::log2(m) - 1;
        auto& coeff0 = coeff[s];
        auto& coeff1 = coeff[(s += log2slots)];
        auto& coeff2 = coeff[(s += log2slots)];

        const BigComplex b = flag_i && (m == 2) ? pos_exp_M_PI : BigFixedPoint::one();
        for (uint32_t k = 0; k < slots; k += m) {
            const uint32_t lenq  = m << 2;
            const uint32_t lenh  = m >> 1;
            const uint32_t klenh = k + lenh;
            std::fill(coeff0.begin() + klenh, coeff0.begin() + klenh + lenh, b);  // shifted right
            std::fill(coeff1.begin() + k, coeff1.begin() + klenh, b);             // not shifted
            for (uint32_t j = 0, jk = k; j < lenh; ++j, ++jk) {
                const auto w      = b * pows[(rotGroup[j] % lenq) * (dim / lenq)];
                coeff2[jk]        = w;   // shifted left
                coeff1[jk + lenh] = -w;  // not shifted
            }
        }
    }
    return coeff;
}

std::vector<std::vector<std::vector<BigComplex>>> CoeffEncodingCollapse(const BigCVector& pows,
                                                                        const std::vector<uint32_t>& rotGroup,
                                                                        uint32_t levelBudget, bool flag_i) {
    const uint32_t slots = rotGroup.size();
    if (!slots)
        OPENFHE_THROW("rotGroup can not be empty");
    if (!levelBudget)
        OPENFHE_THROW("levelBudget can not be 0");

    const uint32_t log2slots = static_cast<uint32_t>(std::log2(slots));
    // Need to compute how many layers are collapsed in each of the level from the budget.
    // If there is no exact division between the maximum number of possible levels (log(slots)) and the
    // level budget, the last level will contain the remaining layers collapsed.
    const std::vector<uint32_t> dims = SelectLayers(log2slots, levelBudget);
    const uint32_t layersCollapse    = dims[0];
    const uint32_t remCollapse       = dims[2];

    const uint32_t dimCollapse = levelBudget;
    const uint32_t flagRem     = (remCollapse == 0) ? 0 : 1;

    const uint32_t numRotations    = (1U << (layersCollapse + 1)) - 1;
    const uint32_t numRotationsRem = (1U << (remCollapse + 1)) - 1;

    // Computing the coefficients for encoding for the given level budget
    BigCMatrix coeff1 = CoeffEncodingOneLevel(pows, rotGroup, flag_i);

    // Coeff stores the coefficients for the given budget of levels
    std::vector<std::vector<std::vector<BigComplex>>> coeff(
        dimCollapse, std::vector<std::vector<BigComplex>>(numRotations, std::vector<BigComplex>(slots)));
    if (flagRem) {
        // this one corresponds to the first index in encoding (same applies to the last index in decoding too)
        coeff[0] = std::vector<std::vector<BigComplex>>(numRotationsRem, std::vector<BigComplex>(slots));
    }

    if (layersCollapse) {  // this condition is necessary for the code executed before the inner loop
        std::vector<std::vector<BigComplex>> zeros(numRotations, std::vector<BigComplex>(slots, BigFixedPoint::zero()));
        for (int32_t s = dimCollapse - 1; s >= static_cast<int32_t>(flagRem); --s) {
            // top is an index, so it can't be negative. let's check that
            if (log2slots < (dimCollapse - 1 - s) * layersCollapse + 1)
                OPENFHE_THROW("top can not be negative");
            uint32_t top = log2slots - (dimCollapse - 1 - s) * layersCollapse - 1;

            coeff[s][0] = coeff1[top];
            coeff[s][1] = coeff1[top + log2slots];
            coeff[s][2] = coeff1[top + 2 * log2slots];
            for (uint32_t l = 1; l < layersCollapse; ++l) {
                auto temp = coeff[s];
                coeff[s]  = zeros;

                for (uint32_t u = 0; u < (1U << (l + 1)) - 1; ++u) {
                    for (uint32_t k = 0; k < slots; ++k) {
                        coeff[s][2 * u][k] +=
                            coeff1[top - l][k] * temp[u][ReduceRotation(k - (1U << (top - l)), slots)];
                        coeff[s][2 * u + 1][k] += coeff1[top - l + log2slots][k] * temp[u][k];
                        coeff[s][2 * u + 2][k] +=
                            coeff1[top - l + 2 * log2slots][k] * temp[u][ReduceRotation(k + (1U << (top - l)), slots)];
                    }
                }
            }
        }
    }

    if (flagRem && remCollapse) {
        std::vector<std::vector<BigComplex>> zeros(numRotationsRem,
                                                   std::vector<BigComplex>(slots, BigFixedPoint::zero()));
        uint32_t s = 0;
        // top is an index, so it can't be negative. let's check that
        if (log2slots < (dimCollapse - 1 - s) * layersCollapse - 1)
            OPENFHE_THROW("top can not be negative");
        uint32_t top = log2slots - (dimCollapse - 1 - s) * layersCollapse - 1;

        coeff[s][0] = coeff1[top];
        coeff[s][1] = coeff1[top + log2slots];
        coeff[s][2] = coeff1[top + 2 * log2slots];
        for (uint32_t l = 1; l < remCollapse; ++l) {
            auto temp = coeff[s];
            coeff[s]  = zeros;

            for (uint32_t u = 0; u < (1U << (l + 1)) - 1; ++u) {
                for (uint32_t k = 0; k < slots; ++k) {
                    coeff[s][2 * u][k] += coeff1[top - l][k] * temp[u][ReduceRotation(k - (1U << (top - l)), slots)];
                    coeff[s][2 * u + 1][k] += coeff1[top - l + log2slots][k] * temp[u][k];
                    coeff[s][2 * u + 2][k] +=
                        coeff1[top - l + 2 * log2slots][k] * temp[u][ReduceRotation(k + (1U << (top - l)), slots)];
                }
            }
        }
    }

    return coeff;
}

std::vector<std::vector<std::vector<BigComplex>>> CoeffDecodingCollapse(const BigCVector& pows,
                                                                        const std::vector<uint32_t>& rotGroup,
                                                                        uint32_t levelBudget, bool flag_i) {
    const uint32_t slots = rotGroup.size();
    if (!slots)
        OPENFHE_THROW("rotGroup can not be empty");
    if (!levelBudget)
        OPENFHE_THROW("levelBudget can not be 0");

    const uint32_t log2slots = static_cast<uint32_t>(std::log2(slots));

    // Need to compute how many layers are collapsed in each of the level from the budget.
    // If there is no exact division between the maximum number of possible levels (log(slots)) and the
    // level budget, the last level will contain the remaining layers collapsed.
    std::vector<uint32_t> dims    = SelectLayers(log2slots, levelBudget);
    const uint32_t layersCollapse = dims[0];
    const uint32_t rowsCollapse   = dims[1];
    const uint32_t remCollapse    = dims[2];

    const uint32_t dimCollapse = levelBudget;
    const uint32_t flagRem     = (remCollapse == 0) ? 0 : 1;

    uint32_t numRotations    = (1U << (layersCollapse + 1)) - 1;
    uint32_t numRotationsRem = (1U << (remCollapse + 1)) - 1;

    // Computing the coefficients for decoding for the given level budget
    BigCMatrix coeff1 = CoeffDecodingOneLevel(pows, rotGroup, flag_i);

    // Coeff stores the coefficients for the given budget of levels
    std::vector<std::vector<std::vector<BigComplex>>> coeff(
        dimCollapse, std::vector<std::vector<BigComplex>>(numRotations, std::vector<BigComplex>(slots)));
    if (flagRem) {
        // this one corresponds to the last index in decoding (same applies to the first index in encoding too)
        coeff[dimCollapse - 1] = std::vector<std::vector<BigComplex>>(numRotationsRem, std::vector<BigComplex>(slots));
    }

    if (layersCollapse) {  // this condition is necessary for the code executed before the inner loop
        std::vector<std::vector<BigComplex>> zeros(numRotations, std::vector<BigComplex>(slots, BigFixedPoint::zero()));
        for (uint32_t s = 0; s < rowsCollapse; ++s) {
            coeff[s][0] = coeff1[s * layersCollapse];
            coeff[s][1] = coeff1[log2slots + s * layersCollapse];
            coeff[s][2] = coeff1[2 * log2slots + s * layersCollapse];

            for (uint32_t l = 1; l < layersCollapse; ++l) {
                auto temp = coeff[s];
                coeff[s]  = zeros;
                for (uint32_t t = 0; t < 3; ++t) {
                    uint32_t shift = (t == 0) ? 0 : ((t == 1) ? (1U << l) : (1U << (l + 1)));
                    for (uint32_t u = 0; u < (1U << (l + 1)) - 1; ++u) {
                        for (uint32_t k = 0; k < slots; ++k) {
                            coeff[s][u + shift][k] += coeff1[s * layersCollapse + l + t * log2slots][k] * temp[u][k];
                        }
                    }
                }
            }
        }
    }

    if (flagRem && remCollapse) {  // check if (remCollapse > 0). it is necessary for the code executed before the loop
        const uint32_t s = rowsCollapse;

        coeff[s][0] = coeff1[s * layersCollapse];
        coeff[s][1] = coeff1[log2slots + s * layersCollapse];
        coeff[s][2] = coeff1[2 * log2slots + s * layersCollapse];

        std::vector<std::vector<BigComplex>> zeros(numRotationsRem,
                                                   std::vector<BigComplex>(slots, BigFixedPoint::zero()));
        for (uint32_t l = 1; l < remCollapse; ++l) {
            auto temp = coeff[s];
            coeff[s]  = zeros;
            for (uint32_t t = 0; t < 3; ++t) {
                uint32_t shift = (t == 0) ? 0 : ((t == 1) ? (1U << l) : (1U << (l + 1)));
                for (uint32_t u = 0; u < (1U << (l + 1)) - 1; ++u) {
                    for (uint32_t k = 0; k < slots; ++k) {
                        coeff[s][u + shift][k] += coeff1[s * layersCollapse + l + t * log2slots][k] * temp[u][k];
                    }
                }
            }
        }
    }

    return coeff;
}

}  // namespace lbcrypto