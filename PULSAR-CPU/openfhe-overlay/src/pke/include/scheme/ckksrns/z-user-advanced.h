#ifndef SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_USER_ADVANCED_H_
#define SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_USER_ADVANCED_H_

#include "scheme/ckksrns/z-leveledshe.h"
#include "scheme/ckksrns/z-fhe.h"
#include "scheme/ckksrns/z-user.h"

namespace lbcrypto {

// User facing API
// LeveledSHE API are too internal...
class UserZAdvancedImpl {
public:
    UserZAdvancedImpl(LeveledZ z, FHEZ fheZ, UserZ userZ) : z(z), fheZ(fheZ), userZ(userZ) {}

    CiphertextGroup EvalLessThan(Ciphertext<DCRTPoly> ct1, Ciphertext<DCRTPoly> ct2);
    // CiphertextGroup EvalEqualTo(Ciphertext<DCRTPoly> ct1, Ciphertext<DCRTPoly> ct2);

    // This works for both ZMode and BMode
    CiphertextGroup EvalRotateInZ(CiphertextGroup ct, int32_t index);

private:
    LeveledZ z;
    FHEZ fheZ;
    UserZ userZ;
};

using UserZAdvanced = std::shared_ptr<UserZAdvancedImpl>;

}  // namespace lbcrypto

#endif  //SRC_PKE_INCLUDE_SCHEME_CKKSRNS_Z_USER_ADVANCED_H_