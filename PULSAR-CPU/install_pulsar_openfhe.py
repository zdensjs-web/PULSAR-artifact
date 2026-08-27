#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


VERSION = "1.3.2"
SUPPORTED_OPENFHE = (1, 4, 0)
RUN_SCRIPTS = (
    "run_pulsar_cpu_operator.sh",
    "run_pulsar_add.sh",
    "run_pulsar_gt.sh",
    "run_pulsar_eq.sh",
    "run_pulsar_xor.sh",
    "run_pulsar_mixed.sh",
    "run_pulsar_mul.sh",
    "run_pulsar_add_chain.sh",
    "run_pulsar_transfer.sh",
    "run_pulsar_auction.sh",
    "run_pulsar_auction_16_256.sh",
    "run_pulsar_vault.sh",
    "run_pulsar_sha256.sh",
)


def read_openfhe_version(root: Path) -> tuple[int, int, int]:
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    values = []
    for name in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(
            rf"set\(OPENFHE_VERSION_{name}\s+([0-9]+)\)", cmake
        )
        if match is None:
            raise RuntimeError(f"cannot read OPENFHE_VERSION_{name}")
        values.append(int(match.group(1)))
    return tuple(values)  # type: ignore[return-value]


def copy_with_backup(source: Path, target: Path, backup_root: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.read_bytes() == source.read_bytes():
        return False
    if target.exists() and not target.is_file():
        raise RuntimeError(f"target is not a regular file: {target}")
    if target.is_file():
        relative = target.relative_to(backup_root.parent)
        backup = backup_root / relative
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            print(f"backup   : {backup}")
    shutil.copy2(source, target)
    print(f"install  : {target}")
    return True


def run_plaintext_checks(package: Path) -> None:
    checks = (
        package / "tests/single_round_plain_test.py",
        package / "tests/multiplier_plain_test.py",
        package / "tests/sha256_plain_test.py",
    )
    for check in checks:
        subprocess.run([sys.executable, str(check)], check=True)


def install(package: Path, root: Path, skip_checks: bool) -> None:
    root = root.expanduser().resolve()
    if not (root / "src/pke/CMakeLists.txt").is_file():
        raise RuntimeError(f"not an OpenFHE source tree: {root}")
    version = read_openfhe_version(root)
    if version != SUPPORTED_OPENFHE:
        expected = ".".join(map(str, SUPPORTED_OPENFHE))
        actual = ".".join(map(str, version))
        raise RuntimeError(
            f"PULSAR-CPU expects OpenFHE {expected}, found {actual}"
        )

    if not skip_checks:
        run_plaintext_checks(package)

    overlay = package / "openfhe-overlay"
    sources = sorted(path for path in overlay.rglob("*") if path.is_file())
    if not sources:
        raise RuntimeError("the OpenFHE overlay is empty")

    backup_root = root / f".pulsar-openfhe-backup-v{VERSION}"
    changed = 0
    for source in sources:
        target = root / source.relative_to(overlay)
        changed += int(copy_with_backup(source, target, backup_root))

    for name in RUN_SCRIPTS:
        source = package / name
        target = root / name
        changed += int(copy_with_backup(source, target, backup_root))
        target.chmod(target.stat().st_mode | 0o111)

    print(f"Installed PULSAR-CPU {VERSION}: {changed} changed files")
    build = root / "build"
    if (build / "CMakeCache.txt").is_file():
        print("Refreshing the existing CMake build graph...")
        subprocess.run(["cmake", "-S", str(root), "-B", str(build)], check=True)

    print("Next:")
    print(f"  cd {root}")
    if not (build / "CMakeCache.txt").is_file():
        print("  cmake -S . -B build -DBUILD_EXAMPLES=ON")
    print(
        "  cmake --build build --target "
        "benchmark-pulsar-single-round benchmark-pulsar-multiply "
        "benchmark-pulsar-add-chain benchmark-pulsar-vault -j\"$(nproc)\""
    )
    print("  ./run_pulsar_add.sh 128 12")
    print("  ./run_pulsar_mixed.sh 128 12 1")
    print("  ./run_pulsar_transfer.sh 128 0.01")
    print("  ./run_pulsar_auction.sh 128 0.01")
    print("  ./run_pulsar_vault.sh 256 3 3 12 1")
    print("  ./run_pulsar_sha256.sh 1 0.01 12")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install PULSAR-CPU into an OpenFHE 1.4.0 source tree"
    )
    parser.add_argument("openfhe_root", type=Path)
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="skip plaintext checks before installation",
    )
    args = parser.parse_args()
    install(Path(__file__).resolve().parent, args.openfhe_root, args.skip_checks)


if __name__ == "__main__":
    main()
