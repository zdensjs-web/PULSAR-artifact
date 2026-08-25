#include "comparison_runtime_driver.h"

int main(int argc, char** argv) {
    return kscomparisonapp::Run(
        argc, argv, ksgreaterthan::Relation::LessThan);
}
