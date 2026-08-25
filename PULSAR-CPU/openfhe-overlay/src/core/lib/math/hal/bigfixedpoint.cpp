#include "math/hal/bigfixedpoint.h"

namespace lbcrypto {

//=============================================================================
// BigFixedPoint Implementation
//=============================================================================

BigFixedPoint operator+(const BigFixedPoint& a, const BigFixedPoint& b) {
    if (a.getLog2Scale() == b.getLog2Scale()) {
        auto aValue = a.getValue();
        auto bValue = b.getValue();
        auto aNeg   = a.getNeg();
        auto bNeg   = b.getNeg();
        if (aNeg == bNeg) {
            return BigFixedPoint(aValue + bValue, a.getLog2Scale(), aNeg);
        }
        else {
            if (aValue > bValue) {
                return BigFixedPoint(aValue - bValue, a.getLog2Scale(), aNeg);
            }
            else if (bValue > aValue) {
                return BigFixedPoint(bValue - aValue, a.getLog2Scale(), bNeg);
            }
            else {
                // equal values, return zero
                return BigFixedPoint(BigInteger(0), a.getLog2Scale(), false);
            }
        }
    }
    else if (a.getLog2Scale() > b.getLog2Scale()) {
        int diff           = a.getLog2Scale() - b.getLog2Scale();
        BigInteger scaledB = b.getValue() << diff;
        BigFixedPoint bigScaledB(scaledB, a.getLog2Scale(), b.getNeg());
        return a + bigScaledB;
    }
    else {
        int diff           = b.getLog2Scale() - a.getLog2Scale();
        BigInteger scaledA = a.getValue() << diff;
        BigFixedPoint bigScaledA(scaledA, b.getLog2Scale(), a.getNeg());
        return bigScaledA + b;
    }
}

BigFixedPoint operator-(const BigFixedPoint& a, const BigFixedPoint& b) {
    BigFixedPoint negB = b;
    negB               = BigFixedPoint(negB.getValue(), negB.getLog2Scale(), !negB.getNeg());
    return a + negB;
}

BigFixedPoint operator-(const BigFixedPoint& a) {
    return BigFixedPoint(a.getValue(), a.getLog2Scale(), !a.getNeg());
}

// make it scale preserving
BigFixedPoint operator*(const BigFixedPoint& a, const BigFixedPoint& b) {
    BigInteger prod = a.getValue() * b.getValue();
    int log2S       = a.getLog2Scale() + b.getLog2Scale();
    bool neg        = (a.getNeg() != b.getNeg());
    return BigFixedPoint(prod, log2S, neg).scaleTo(std::max(a.getLog2Scale(), b.getLog2Scale()));
}

BigFixedPoint operator/(const BigFixedPoint& a, const BigFixedPoint& b) {
    // Scale up the dividend to preserve precision
    BigInteger scaledA = a.getValue() << b.getLog2Scale();
    BigInteger quot    = scaledA / b.getValue();
    int log2S          = a.getLog2Scale();
    bool neg           = (a.getNeg() != b.getNeg());
    return BigFixedPoint(quot, log2S, neg);
}

bool operator<(const BigFixedPoint& a, const BigFixedPoint& b) {
    auto c = a - b;
    if (c.equalZero()) {
        return false;
    }
    return c.getNeg();
}

//=============================================================================
// BigComplex Implementation
//=============================================================================

BigComplex operator+(const BigComplex& a, const BigComplex& b) {
    return BigComplex(a.getReal() + b.getReal(), a.getImag() + b.getImag());
}
BigComplex operator-(const BigComplex& a, const BigComplex& b) {
    return BigComplex(a.getReal() - b.getReal(), a.getImag() - b.getImag());
}
BigComplex operator*(const BigComplex& a, const BigComplex& b) {
    BigFixedPoint rePart = (a.getReal() * b.getReal()) - (a.getImag() * b.getImag());
    BigFixedPoint imPart = (a.getReal() * b.getImag()) + (a.getImag() * b.getReal());
    return BigComplex(rePart, imPart);
}
BigComplex operator/(const BigComplex& a, const BigComplex& b) {
    BigFixedPoint denom  = (b.getReal() * b.getReal()) + (b.getImag() * b.getImag());
    BigFixedPoint rePart = ((a.getReal() * b.getReal()) + (a.getImag() * b.getImag())) / denom;
    BigFixedPoint imPart = ((a.getImag() * b.getReal()) - (a.getReal() * b.getImag())) / denom;
    return BigComplex(rePart, imPart);
}

BigComplex& BigComplex::operator+=(const BigComplex& a) {
    *this = *this + a;
    return *this;
}
BigComplex& BigComplex::operator-=(const BigComplex& a) {
    *this = *this - a;
    return *this;
}
BigComplex& BigComplex::operator*=(const BigComplex& a) {
    *this = *this * a;
    return *this;
}
BigComplex& BigComplex::operator/=(const BigComplex& a) {
    *this = *this / a;
    return *this;
}

BigComplex operator-(const BigComplex& a) {
    return BigComplex(-a.getReal(), -a.getImag());
}

BigCVector ToCVector(const BigFPVector& input) {
    BigCVector result;
    result.reserve(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        result.push_back(input[i]);
    }
    return result;
}

BigFPVector ToReal(const BigCVector& input) {
    BigFPVector result;
    result.reserve(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        result.push_back(input[i].getReal());
    }
    return result;
}

std::vector<std::complex<double>> ToStdComplexVector(const BigCVector& input) {
    std::vector<std::complex<double>> result;
    result.reserve(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        result.push_back(input[i].convertToComplex());
    }
    return result;
}

BigCVector FromStdComplexVector(const std::vector<std::complex<double>>& input) {
    BigCVector result;
    result.reserve(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        result.push_back(
            BigComplex(BigFixedPoint::fromDouble(input[i].real()), BigFixedPoint::fromDouble(input[i].imag())));
    }
    return result;
}

}  // namespace lbcrypto