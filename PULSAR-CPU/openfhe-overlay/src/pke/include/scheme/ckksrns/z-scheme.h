
#ifndef LBCRYPTO_CRYPTO_CKKSRNS_Z_SCHEME_H
#define LBCRYPTO_CRYPTO_CKKSRNS_Z_SCHEME_H

#include "schemerns/rns-scheme.h"

#include <memory>
#include <string>

/**
 * @namespace lbcrypto
 * The namespace of lbcrypto
 */
namespace lbcrypto {

class SchemeZ : public SchemeRNS {
public:
    SchemeZ() {
        // this->m_ParamsGen = std::make_shared<ParameterGenerationCKKSRNS>();
    }

    virtual ~SchemeZ() = default;

    bool operator==(const SchemeBase<DCRTPoly>& sch) const override {
        return (typeid(sch) == typeid(SchemeZ));
    }

    void Enable(PKESchemeFeature feature) override;
};

}  // namespace lbcrypto

#endif  // LBCRYPTO_CRYPTO_CKKSRNS_Z_SCHEME_H