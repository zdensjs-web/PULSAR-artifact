#include "math/z-polynomial.h"
#include "math/dftransform.h"
#include "math/dftransform-bigcomplex.h"

namespace lbcrypto {

CSlots ZPolynomial::toCSlots() const {
    return CSlots(ZEncodingParams(ZMode, zN, 1), ZLinearTransform::MultZU(zN, coefficients));
}

ZPolynomial CSlots::getZPolynomial(size_t slotIndex) const {
    if (!params.isZMode()) {
        OPENFHE_THROW("CSlots::getZPolynomial: not in ZMode");
    }
    if (slotIndex >= params.getZSlots()) {
        OPENFHE_THROW("CSlots::getZPolynomial: slotIndex out of range");
    }
    auto zN = params.getZN();
    std::vector<BigComplex> cSlotsForIndex(zN / 2, BigFixedPoint::zero());
    for (size_t i = 0; i != zN / 2; ++i) {
        cSlotsForIndex[i] = slots[(slotIndex * (zN / 2)) + i];
    }
    return ZPolynomial(ZLinearTransform::MultZUInverse(zN, cSlotsForIndex));
}

CSlots RPolynomial::toCSlots() const {
    auto m      = coefficients.size() * 2;
    auto cSlots = coefficients.size() / 2;
    std::vector<std::complex<double>> forwardDouble(cSlots);
    for (size_t i = 0; i != cSlots; ++i) {
        //forward[i] = BigComplex(coefficients[i], coefficients[i + cSlots]);
        forwardDouble[i] =
            std::complex<double>(coefficients[i].convertToDouble(), coefficients[i + cSlots].convertToDouble());
    }
    DiscreteFourierTransform::FFTSpecial(forwardDouble, m);

    BigCVector forward(cSlots);
    for (size_t i = 0; i != cSlots; ++i) {
        forward[i] = BigComplex(BigFixedPoint::fromDouble(forwardDouble[i].real()),
                                BigFixedPoint::fromDouble(forwardDouble[i].imag()));
    }

    // TOO SLOW...
    //DiscreteFourierTransformBigComplex::FFTSpecial(forward, m);
    return CSlots(params, forward);
}

RPolynomial CSlots::toRPolynomial() const {
    auto m             = slots.size() * 4;
    BigCVector inverse = slots;
    std::vector<std::complex<double>> inverseInDouble(inverse.size());
    for (size_t i = 0; i != inverse.size(); ++i) {
        inverseInDouble[i] =
            std::complex<double>(inverse[i].getReal().convertToDouble(), inverse[i].getImag().convertToDouble());
    }
#ifdef HIGH_PREC
    // ....DiscreteFourierTransformBigComplex is too slow....
    DiscreteFourierTransformBigComplex::FFTSpecialInv(inverse, m);

    std::vector<BigFixedPoint> rValues(2 * slots.size());
    for (size_t i = 0; i != inverse.size(); ++i) {
        for (size_t i = 0; i != inverse.size(); ++i) {
            rValues[i]                = inverse[i].getReal();
            rValues[i + slots.size()] = inverse[i].getImag();
        }
    }
    return RPolynomial(params, rValues);
#else
    DiscreteFourierTransform::FFTSpecialInv(inverseInDouble, m);
    std::vector<double> rValues(2 * slots.size());
    for (size_t i = 0; i != inverse.size(); ++i) {
        rValues[i]                = inverseInDouble[i].real();
        rValues[i + slots.size()] = inverseInDouble[i].imag();
    }
    std::vector<BigFixedPoint> rValuesBFP(2 * slots.size());
    for (size_t i = 0; i != rValuesBFP.size(); ++i) {
        rValuesBFP[i] = BigFixedPoint::fromDouble(rValues[i]);
    }
    return RPolynomial(params, rValuesBFP);
#endif
}

// TODO: deprecate into BMode...
std::pair<uint64_t, double> CSlots::getIntegerAndErrorAtBooleanMode(size_t slotIndex) const {
    if (!params.isZMode()) {
        OPENFHE_THROW("CSlots::getIntegerAtBooleanMode: not in ZMode");
    }
    if (slotIndex >= params.getZSlots()) {
        OPENFHE_THROW("CSlots::getIntegerAtBooleanMode: slotIndex out of range");
    }
    auto zN                = params.getZN();
    auto zSlots            = params.getZSlots();
    uint64_t reconstructed = 0;
    double log2MaxError    = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i != zN; ++i) {
        auto index = slotIndex * (zN / 2) + i;
        if (i >= zN / 2) {
            index += (zSlots - 1) * (zN / 2);
        }
        auto bit         = slots[index].getReal().convertToDouble();
        auto integerPart = std::round(bit);
        auto fracPart    = bit - integerPart;
        reconstructed += (static_cast<uint64_t>(integerPart) << (i));
        log2MaxError = std::max(log2MaxError, std::log2(std::abs(fracPart)));
    }
    return {reconstructed, log2MaxError};
}

uint64_t CSlots::getIntegerAtBooleanMode(size_t slotIndex) const {
    return getIntegerAndErrorAtBooleanMode(slotIndex).first;
}

double CSlots::getIntegerErrorAtBooleanMode(size_t slotIndex) const {
    return getIntegerAndErrorAtBooleanMode(slotIndex).second;
}

std::ostream& operator<<(std::ostream& os, const ZEncodingType& type) {
    switch (type) {
        case INVALID:
            os << "INVALID";
            break;
        case ZMode:
            os << "ZMode";
            break;
        case BModeSparse:
            os << "BModeSparse";
            break;
        case BModeFull:
            os << "BModeFull";
            break;
        case CMode:
            os << "CMode";
            break;
        default:
            os << "Unknown Encoding Type";
    }
    return os;
}

}  // namespace lbcrypto