#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${PULSAR_BUILD_ROOT:-${ROOT_DIR}/.build}"
JOBS="${PULSAR_BUILD_JOBS:-$(nproc)}"

BASE_ARCHIVE="${ROOT_DIR}/vendor/fhe-simd-alu-08f1eb8.tar.gz"
HEXL_ARCHIVE="${ROOT_DIR}/vendor/hexl-1.2.6.tar.gz"
CPU_FEATURES_ARCHIVE="${ROOT_DIR}/vendor/cpu-features-32b49eb.tar.gz"
OPENFHE_SRC="${WORK_DIR}/openfhe"
OPENFHE_BUILD="${OPENFHE_SRC}/build"
HEXL_SRC="${WORK_DIR}/hexl-src"
HEXL_BUILD="${WORK_DIR}/hexl-build"
HEXL_INSTALL="${WORK_DIR}/hexl-install"
CPU_FEATURES_SRC="${WORK_DIR}/cpu-features-src"
CPU_FEATURES_BUILD="${WORK_DIR}/cpu-features-build"
CPU_FEATURES_INSTALL="${WORK_DIR}/cpu-features-install"

for command in cmake make gcc g++ python3 autoconf automake libtoolize tar sha256sum sed; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "missing required command: ${command}" >&2
        echo "install the Ubuntu packages listed in README.md and rerun this script" >&2
        exit 1
    fi
done

if [[ ! -f "${BASE_ARCHIVE}" || ! -f "${HEXL_ARCHIVE}" || ! -f "${CPU_FEATURES_ARCHIVE}" ]]; then
    echo "vendored source archives are missing under ${ROOT_DIR}/vendor" >&2
    exit 1
fi

(
    cd "${ROOT_DIR}/vendor"
    sha256sum --check SHA256SUMS
)

mkdir -p "${WORK_DIR}"

if [[ ! -f "${OPENFHE_SRC}/CMakeLists.txt" ]]; then
    mkdir -p "${OPENFHE_SRC}"
    tar -xzf "${BASE_ARCHIVE}" -C "${OPENFHE_SRC}"
fi

sed -i 's/\r$//' "${OPENFHE_SRC}/third-party/gperftools/autogen.sh"
sed -i '/^[[:space:]]*ACLOCAL_AMFLAGS[[:space:]]*=/d' \
    "${OPENFHE_SRC}/third-party/gperftools/Makefile.am"
chmod +x "${OPENFHE_SRC}/third-party/gperftools/autogen.sh"

if [[ ! -f "${HEXL_SRC}/CMakeLists.txt" ]]; then
    mkdir -p "${HEXL_SRC}"
    tar -xzf "${HEXL_ARCHIVE}" -C "${HEXL_SRC}"
fi

if [[ ! -f "${CPU_FEATURES_SRC}/CMakeLists.txt" ]]; then
    mkdir -p "${CPU_FEATURES_SRC}"
    tar -xzf "${CPU_FEATURES_ARCHIVE}" -C "${CPU_FEATURES_SRC}"
fi

python3 "${ROOT_DIR}/install_pulsar_openfhe.py" "${OPENFHE_SRC}"
sed -i 's/\r$//' "${OPENFHE_SRC}"/run_pulsar_*.sh

cmake -S "${CPU_FEATURES_SRC}" -B "${CPU_FEATURES_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_INSTALL_PREFIX="${CPU_FEATURES_INSTALL}" \
    -DBUILD_PIC=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=OFF
cmake --build "${CPU_FEATURES_BUILD}" --target install -j"${JOBS}"

cmake -S "${HEXL_SRC}" -B "${HEXL_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${HEXL_INSTALL}" \
    -DCMAKE_PREFIX_PATH="${CPU_FEATURES_INSTALL}" \
    -DHEXL_BENCHMARK=OFF \
    -DHEXL_COVERAGE=OFF \
    -DHEXL_DOCS=OFF \
    -DHEXL_EXPERIMENTAL=OFF \
    -DHEXL_SHARED_LIB=ON \
    -DHEXL_TESTING=OFF
cmake --build "${HEXL_BUILD}" --target install -j"${JOBS}"

cmake -S "${OPENFHE_SRC}" -B "${OPENFHE_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${HEXL_INSTALL};${CPU_FEATURES_INSTALL}" \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_UNITTESTS=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DGIT_SUBMOD_AUTO=OFF \
    -DWITH_OPENMP=ON \
    -DWITH_NTL=ON \
    -DMATHBACKEND=6 \
    -DWITH_TCM=ON \
    -DWITH_INTEL_HEXL=ON \
    -DINTEL_HEXL_PREBUILT=ON \
    -DINTEL_HEXL_HINT_DIR="${HEXL_INSTALL}"

cmake --build "${OPENFHE_BUILD}" --target tcm -j"${JOBS}"
cmake --build "${OPENFHE_BUILD}" --target \
    benchmark-pulsar-single-round \
    benchmark-pulsar-multiply \
    benchmark-pulsar-add-chain \
    benchmark-pulsar-vault \
    -j"${JOBS}"

echo
echo "PULSAR-CPU build completed."
echo "Source and run scripts: ${OPENFHE_SRC}"
echo "Example:"
echo "  cd ${OPENFHE_SRC}"
echo "  ./run_pulsar_add.sh 128 12"
