#include "math/z-constants.h"

namespace lbcrypto {

std::unordered_map<uint32_t, ZLinearTransform::PrecomputedValues> ZLinearTransform::precomputedValues;

void ZLinearTransform::Initialize(uint32_t zN) {
#pragma omp critical
    if (precomputedValues.find(zN) == precomputedValues.end()) {
        precomputedValues.insert({zN, PrecomputedValues(zN)});
    }
}

ZLinearTransform::PrecomputedValues::PrecomputedValues(uint32_t zN) {
    m_zN    = zN;
    m_ZU    = GetZU(zN);
    m_ZUInv = GetZUInverse(zN);
}

BigCMatrix ZLinearTransform::PrecomputedValues::GetZU(uint32_t zN) {
    // Vandermond matrix
    BigCMatrix zu(zN / 2, std::vector<BigComplex>(zN));
    for (size_t i = 0; i != zN / 2; ++i) {
        zu[i][0] = BigFixedPoint::one();
        for (size_t j = 1; j != zN; ++j) {
            zu[i][j] = zu[i][j - 1] * Z_ROOTS_MAP.at(zN)[i];
        }
    }
    return zu;
}

BigCMatrix ZLinearTransform::PrecomputedValues::GetZUInverse(uint32_t zN) {
    std::vector<std::vector<BigComplex>> zUInv(zN, std::vector<BigComplex>(zN / 2));
    // Build the inverse Vandermond matrix
    for (size_t j = 0; j != zN / 2; ++j) {
        auto xi = Z_ROOTS_MAP.at(zN)[j];
        std::vector<BigComplex> xiPowers;
        auto one   = BigFixedPoint::one();
        auto zNbig = BigFixedPoint::positive(zN);
        xiPowers.push_back(one);
        for (size_t p = 1; p != zN; ++p) {
            xiPowers.push_back(xiPowers[p - 1] * xi);
        }
        // build power map
        for (size_t i = 0; i != zN; ++i) {
            // d = zN * xi^(zN-1) - 1
            auto d = (zNbig * xiPowers[zN - 1]) - one;
            if (i == 0) {
                zUInv[i][j] = (xiPowers[zN - 1] - one) / d;
            }
            else {
                zUInv[i][j] = xiPowers[zN - 1 - i] / d;
            }
        }
    }
    return zUInv;
}

const BigCMatrix& ZLinearTransform::GetZU(uint32_t zN) {
    auto it = precomputedValues.find(zN);
    if (it == precomputedValues.end()) {
        OPENFHE_THROW("ZLinearTransform::Initialize not called for " + std::to_string(zN));
    }
    return it->second.m_ZU;
}

const BigCMatrix& ZLinearTransform::GetZUInverse(uint32_t zN) {
    auto it = precomputedValues.find(zN);
    if (it == precomputedValues.end()) {
        OPENFHE_THROW("ZLinearTransform::Initialize not called for " + std::to_string(zN));
    }
    return it->second.m_ZUInv;
}

BigCVector ZLinearTransform::MultZU(uint32_t zN, const BigFPVector& input) {
    auto& U = GetZU(zN);
    assert(input.size() == U[0].size() && "Input size does not match expected size for U");

    // This is the vandermond matrix
    std::vector<BigComplex> result;

    size_t halfSize = input.size() / 2;
    for (size_t i = 0; i < halfSize; ++i) {
        BigComplex sum;
        for (size_t j = 0; j < input.size(); ++j) {
            sum = sum + U[i][j] * input[j];
        }
        result.push_back(sum);
    }
    return result;
}

BigFPVector ZLinearTransform::MultZUInverse(uint32_t zN, const BigCVector& input) {
    auto& UInv = GetZUInverse(zN);
    std::vector<BigFixedPoint> result;
    assert(input.size() == UInv[0].size() && "Input size does not match expected size for multiplyByZUInverse");

    for (size_t i = 0; i != zN; ++i) {
        BigComplex sum;
        for (size_t j = 0; j != zN / 2; ++j) {
            sum = sum + UInv[i][j] * input[j];
        }
        // z + conj(z) = 2*real(z)
        auto two = BigFixedPoint::positive(2);
        result.push_back(two * sum.getReal());
    }
    return result;
}

}  // namespace lbcrypto