//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2023, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

/*
  This file contains the discrete fourier transform implementation.
 */

#include "math/dftransform-bigcomplex.h"
#include "math/nbtheory.h"

#include "utils/inttypes.h"
#include "utils/parallel.h"

#include <complex>
#include <vector>

namespace lbcrypto {

std::unordered_map<uint32_t, DiscreteFourierTransformBigComplex::PrecomputedValues>
    DiscreteFourierTransformBigComplex::precomputedValues;

DiscreteFourierTransformBigComplex::PrecomputedValues::PrecomputedValues(uint32_t m, uint32_t nh) {
    m_M  = m;
    m_Nh = nh;

    m_rotGroup.resize(m_Nh);
    uint32_t fivePows = 1;
    for (size_t i = 0; i < m_Nh; ++i) {
        m_rotGroup[i] = fivePows;
        fivePows *= 5;
        fivePows %= m_M;
    }

    m_ksiPows.resize(m_M + 1);
    m_ksiPows[0] = BigComplex(BigFixedPoint::one(), BigFixedPoint::zero());
    m_ksiPows[1] = R_ROOT_MAP.at(m);
    for (size_t j = 2; j < m_M; ++j) {
        m_ksiPows[j] = m_ksiPows[j - 1] * m_ksiPows[1];
    }
    m_ksiPows[m_M] = m_ksiPows[0];
}

void DiscreteFourierTransformBigComplex::Initialize(uint32_t m, uint32_t nh) {
#pragma omp critical
    // add a PrecomputedValues object to the map of precomputedValues only if it doesn't already exist for the given cyclotomic order
    if (precomputedValues.find(m) == precomputedValues.end()) {
        precomputedValues.insert({m, PrecomputedValues(m, nh)});
    }
}

void DiscreteFourierTransformBigComplex::FFTSpecialInv(BigCVector& vals, uint32_t cyclOrder) {
    // check if the precomputed table exists for the given cyclotomic order
    const auto it = precomputedValues.find(cyclOrder);
    if (it == precomputedValues.end()) {
        std::string errMsg("DiscreteFourierTransform::Initialize() must be called for cyclOrder = ");
        errMsg += std::to_string(cyclOrder);
        OPENFHE_THROW(errMsg);
    }

    const uint32_t valsSize = vals.size();
    for (size_t len = valsSize; len >= 1; len >>= 1) {
        for (size_t i = 0; i < valsSize; i += len) {
            size_t lenh = len >> 1;
            size_t lenq = len << 2;
            size_t gap  = it->second.m_M / lenq;
            for (size_t j = 0; j < lenh; ++j) {
                size_t idx   = (lenq - (it->second.m_rotGroup[j] % lenq)) * gap;
                BigComplex u = vals[i + j] + vals[i + j + lenh];
                BigComplex v = vals[i + j] - vals[i + j + lenh];
                v *= it->second.m_ksiPows[idx];
                vals[i + j]        = u;
                vals[i + j + lenh] = v;
            }
        }
    }
    BitReverse(vals);

    for (size_t i = 0; i < valsSize; ++i) {
        vals[i] /= BigFixedPoint::positive(valsSize);
    }
}

void DiscreteFourierTransformBigComplex::FFTSpecial(BigCVector& vals, uint32_t cyclOrder) {
    // check if the precomputed table exists for the given cyclotomic order
    const auto it = precomputedValues.find(cyclOrder);
    if (it == precomputedValues.end()) {
        std::string errMsg("DiscreteFourierTransform::Initialize() must be called for cyclOrder = ");
        errMsg += std::to_string(cyclOrder);
        OPENFHE_THROW(errMsg);
    }
    const PrecomputedValues& prepValues = it->second;

    BitReverse(vals);
    uint32_t size = vals.size();
    for (size_t len = 2; len <= size; len <<= 1) {
        size_t lenh = len >> 1;
        size_t lenq = len << 2;
        size_t gap  = prepValues.m_M / lenq;
        for (size_t i = 0; i < size; i += len) {
            for (size_t j = 0; j < lenh; ++j) {
                int64_t idx  = ((prepValues.m_rotGroup[j] % lenq)) * gap;
                BigComplex u = vals[i + j];
                BigComplex v = vals[i + j + lenh];
                v *= prepValues.m_ksiPows[idx];
                vals[i + j]        = u + v;
                vals[i + j + lenh] = u - v;
            }
        }
    }
}

void DiscreteFourierTransformBigComplex::BitReverse(BigCVector& vals) {
    uint32_t size = vals.size();
    for (size_t i = 1, j = 0; i < size; ++i) {
        size_t bit = size >> 1;
        for (; j >= bit; bit >>= 1) {
            j -= bit;
        }
        j += bit;
        if (i < j) {
            std::swap(vals[i], vals[j]);
        }
    }
}

}  // namespace lbcrypto
