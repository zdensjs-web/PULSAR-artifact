#include "word_operator_runtime_driver.h"

int main(int argc, char** argv) {
    return kswordapp::Run(
        argc, argv, ksword::OperatorKind::ScalarRightShift);
}
