#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
OVERLAY = ROOT / "openfhe-overlay"


def require_file(relative: str) -> Path:
    path = ROOT / relative
    if not path.is_file():
        raise RuntimeError(f"missing required file: {relative}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"empty required file: {relative}")
    return path


def require_text(relative: str, markers: tuple[str, ...]) -> None:
    text = require_file(relative).read_text(encoding="utf-8")
    for marker in markers:
        if marker not in text:
            raise RuntimeError(f"{relative} is missing marker: {marker}")


def main() -> None:
    required = (
        "README.md",
        "LICENSE",
        "NOTICE",
        "VERSION",
        "install_pulsar_openfhe.py",
        "run_pulsar_cpu_operator.sh",
        "run_pulsar_mul.sh",
        "run_pulsar_add_chain.sh",
        "run_pulsar_transfer.sh",
        "run_pulsar_auction.sh",
        "run_pulsar_sha256.sh",
        "tests/single_round_plain_test.py",
        "tests/multiplier_plain_test.py",
        "tests/sha256_plain_test.py",
        "openfhe-overlay/src/core/include/math/hal/bigfixedpoint.h",
        "openfhe-overlay/src/pke/include/encoding/z-encoding.h",
        "openfhe-overlay/src/pke/include/scheme/ckksrns/z-fhe.h",
        "openfhe-overlay/src/pke/include/scheme/ckksrns/z-dense-boolean.h",
        "openfhe-overlay/src/pke/include/scheme/ckksrns/z-dense-radix-multiplier.h",
        "openfhe-overlay/src/pke/examples/benchmark-pulsar-single-round.cpp",
        "openfhe-overlay/src/pke/examples/benchmark-pulsar-multiply.cpp",
        "openfhe-overlay/src/pke/examples/benchmark-pulsar-add-chain.cpp",
        "openfhe-overlay/src/pke/examples/pulsar-auction-benchmark.inc",
        "openfhe-overlay/src/pke/examples/pulsar-sha256-benchmark.inc",
    )
    for relative in required:
        require_file(relative)

    require_text(
        "openfhe-overlay/src/pke/include/scheme/ckksrns/z-dense-boolean.h",
        (
            "EvalPulsarAdd",
            "EvalPulsarGreaterEqual",
            "EvalPulsarEqual",
            "EvalBooleanXor",
            "EvalPrefixBootEncode",
        ),
    )
    require_text(
        "openfhe-overlay/src/pke/examples/benchmark-pulsar-add-chain.cpp",
        ('mode == "transfer"', 'mode == "auction"', 'mode == "sha256"'),
    )
    require_text(
        "openfhe-overlay/src/pke/examples/benchmark-pulsar-single-round.cpp",
        ("Operation::Add", "Operation::GreaterThan", "Operation::Equal", "Operation::Xor"),
    )

    single = require_file(
        "openfhe-overlay/src/pke/examples/benchmark-pulsar-single-round.cpp"
    ).read_text(encoding="utf-8").lower()
    runner = require_file("run_pulsar_cpu_operator.sh").read_text(
        encoding="utf-8"
    ).lower()
    if "operation::mixed" in single or "add_xor_gt" in single:
        raise RuntimeError("discarded MIXED benchmark is present")
    if "add|gt|eq|xor|mixed" in runner:
        raise RuntimeError("discarded MIXED runner is present")

    forbidden_names = ("rsa", "benchmark-applications.cpp")
    for path in ROOT.rglob("*"):
        if path.is_file() and any(name in path.name.lower() for name in forbidden_names):
            raise RuntimeError(f"excluded benchmark is present: {path.relative_to(ROOT)}")

    overlay_files = sum(1 for path in OVERLAY.rglob("*") if path.is_file())
    print("PULSAR-CPU repository audit")
    print("===========================")
    print(f"OpenFHE overlay files : {overlay_files}")
    print("operators             : ADD, GT, EQ, XOR, MUL")
    print("applications          : Transfer, Auction, SHA-256")
    print("chained scheduler     : ADD")
    print("excluded              : MIXED, RSA, baseline Vault")
    print("dependency            : OpenFHE 1.4.0 source tree")
    print("overall verdict       : PASS")


if __name__ == "__main__":
    main()
