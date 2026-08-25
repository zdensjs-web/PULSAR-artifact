#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
TESTS = ROOT / "tests"


def main() -> None:
    source_files = sorted(SRC.glob("*.cpp")) + sorted(SRC.glob("*.h"))
    if len(source_files) != 33:
        raise RuntimeError(f"expected 33 source files, found {len(source_files)}")

    missing_includes: list[str] = []
    for path in source_files:
        source = path.read_text(encoding="utf-8")
        for include in re.findall(r'^#include\s+"([^"]+)"', source, re.MULTILINE):
            if not (SRC / include).is_file():
                missing_includes.append(f"{path.name}: {include}")
    if missing_includes:
        raise RuntimeError("missing local includes: " + ", ".join(missing_includes))

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    target_count = cmake.count("pulsar_gpu_add_operator(")
    if target_count != 19:
        raise RuntimeError(f"expected 19 CMake targets, found {target_count}")

    tests = sorted(TESTS.glob("*_test.py"))
    if len(tests) != 7:
        raise RuntimeError(f"expected 7 plaintext tests, found {len(tests)}")

    print("PULSAR-GPU repository verification")
    print("==================================")
    print("production source files : 33")
    print("executable targets      : 19")
    print("plaintext test programs : 7")
    print("local include closure   : PASS")
    print("verdict                 : PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"verification failed: {error}", file=sys.stderr)
        raise
