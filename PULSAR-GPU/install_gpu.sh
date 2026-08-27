#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="${PULSAR_GPU_BUILD_ROOT:-$ROOT_DIR/.build}"
PREFIX="${PULSAR_GPU_PREFIX:-$ROOT_DIR/.local}"
DEPS_ROOT="$BUILD_ROOT/deps"
OPENFHE_SOURCE="$DEPS_ROOT/openfhe-src"
OPENFHE_BUILD="$BUILD_ROOT/openfhe-build"
FIDESLIB_SOURCE="$ROOT_DIR/third_party/FIDESlib"
FIDESLIB_BUILD="$BUILD_ROOT/fideslib-build"
PULSAR_BUILD="$BUILD_ROOT/pulsar-build"
OPENFHE_PATCH="$ROOT_DIR/patches/openfhe-pulsar-v1.5.1.patch"

OPENFHE_REPOSITORY="https://github.com/openfheorg/openfhe-development.git"
OPENFHE_TAG="v1.5.1"
OPENFHE_COMMIT="1306d14f8c26bb6150d3e6ad54f28dfe1007689e"
MINIMUM_CMAKE_VERSION="3.25.2"

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

version_at_least() {
    local actual="$1"
    local required="$2"
    [[ "$(printf '%s\n%s\n' "$required" "$actual" | sort -V | head -n1)" == "$required" ]]
}

select_cmake() {
    local requested="${CMAKE_BIN:-}"
    local system_cmake=""
    local candidate=""
    local version=""

    if [[ -n "$requested" ]]; then
        candidate="$(command -v "$requested" 2>/dev/null || true)"
        [[ -n "$candidate" ]] || fail "CMAKE_BIN does not name an executable: $requested"
        version="$("$candidate" --version | awk 'NR==1 {print $3}')"
        version_at_least "$version" "$MINIMUM_CMAKE_VERSION" || \
            fail "CMake $MINIMUM_CMAKE_VERSION or newer is required; $candidate is $version"
        printf '%s\n' "$candidate"
        return
    fi

    system_cmake="$(command -v cmake 2>/dev/null || true)"
    for candidate in "$system_cmake" "$HOME/.local/bin/cmake" /opt/cmake-*/bin/cmake; do
        [[ -n "$candidate" && -x "$candidate" ]] || continue
        version="$("$candidate" --version | awk 'NR==1 {print $3}')"
        if version_at_least "$version" "$MINIMUM_CMAKE_VERSION"; then
            printf '%s\n' "$candidate"
            return
        fi
    done

    fail "CMake $MINIMUM_CMAKE_VERSION or newer was not found; set CMAKE_BIN to its path"
}

require_command git
require_command g++
require_command make
require_command nvidia-smi
require_command sort

CMAKE_COMMAND="$(select_cmake)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
[[ -x "$CUDACXX" ]] || fail "CUDA compiler not found: $CUDACXX"

if [[ -z "${JOBS:-}" ]]; then
    JOBS="$(nproc 2>/dev/null || printf '4')"
    (( JOBS > 8 )) && JOBS=8
fi

if [[ -z "${FIDESLIB_ARCH:-}" ]]; then
    FIDESLIB_ARCH="$(
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits |
        tr -d ' .' | sort -u | sed '/^$/d;s/$/-real/' | paste -sd';' -
    )"
fi
[[ -n "$FIDESLIB_ARCH" ]] || fail "could not determine the CUDA compute capability"

[[ -f "$FIDESLIB_SOURCE/CMakeLists.txt" ]] || fail "bundled FIDESlib source is missing"
[[ -f "$OPENFHE_PATCH" ]] || fail "OpenFHE patch is missing"
grep -q 'EvalTwistedProduct' "$FIDESLIB_SOURCE/api/CryptoContext.hpp" || \
    fail "bundled FIDESlib does not contain the PULSAR extensions"

mkdir -p "$DEPS_ROOT" "$PREFIX"

if [[ ! -d "$OPENFHE_SOURCE/.git" ]]; then
    [[ ! -e "$OPENFHE_SOURCE" ]] || \
        fail "$OPENFHE_SOURCE exists but is not a Git checkout; remove .build and retry"
    echo "[1/4] Cloning OpenFHE $OPENFHE_TAG"
    git clone --branch "$OPENFHE_TAG" --recursive "$OPENFHE_REPOSITORY" "$OPENFHE_SOURCE"
else
    echo "[1/4] Reusing the existing OpenFHE source"
    git -C "$OPENFHE_SOURCE" submodule update --init --recursive
fi

actual_openfhe_commit="$(git -C "$OPENFHE_SOURCE" rev-parse HEAD)"
[[ "$actual_openfhe_commit" == "$OPENFHE_COMMIT" ]] || \
    fail "unexpected OpenFHE commit: $actual_openfhe_commit"

patch_stamp="$OPENFHE_SOURCE/.pulsar-openfhe-patch-applied"
if [[ ! -f "$patch_stamp" ]]; then
    if git -C "$OPENFHE_SOURCE" apply --check "$OPENFHE_PATCH"; then
        git -C "$OPENFHE_SOURCE" apply "$OPENFHE_PATCH"
    elif ! git -C "$OPENFHE_SOURCE" apply --reverse --check "$OPENFHE_PATCH"; then
        fail "OpenFHE source is neither clean nor already patched"
    fi
    touch "$patch_stamp"
fi

echo "[2/4] Building patched OpenFHE"
"$CMAKE_COMMAND" -S "$OPENFHE_SOURCE" -B "$OPENFHE_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DBUILD_STATIC=ON \
    -DBUILD_SHARED=ON \
    -DBUILD_UNITTESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DBUILD_EXTRAS=OFF \
    -DWITH_OPENMP=ON \
    -DWITH_TCM=OFF \
    -DWITH_NTL=OFF \
    -DWITH_NATIVEOPT=OFF
"$CMAKE_COMMAND" --build "$OPENFHE_BUILD" --target install -j"$JOBS"

echo "[3/4] Building bundled PULSAR FIDESlib"
"$CMAKE_COMMAND" -S "$FIDESLIB_SOURCE" -B "$FIDESLIB_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_PATH="$CUDA_HOME" \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" \
    -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DFIDESLIB_ARCH="$FIDESLIB_ARCH" \
    -DFIDESLIB_INSTALL_OPENFHE=OFF \
    -DFIDESLIB_INSTALL_PREFIX="$PREFIX" \
    -DOPENFHE_INSTALL_PREFIX="$PREFIX" \
    -DFIDESLIB_COMPILE_TESTS=OFF \
    -DFIDESLIB_COMPILE_BENCHMARKS=OFF
"$CMAKE_COMMAND" --build "$FIDESLIB_BUILD" --target install -j"$JOBS"

echo "[4/4] Building PULSAR-GPU"
"$CMAKE_COMMAND" -S "$ROOT_DIR" -B "$PULSAR_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$PREFIX" \
    -Dfideslib_DIR="$PREFIX/share/fideslib/cmake"
"$CMAKE_COMMAND" --build "$PULSAR_BUILD" -j"$JOBS"
"$CMAKE_COMMAND" --install "$PULSAR_BUILD" --prefix "$PREFIX"

cat <<EOF

PULSAR-GPU installation completed.
  CUDA architectures: $FIDESLIB_ARCH
  Install prefix:     $PREFIX
  Build directory:    $BUILD_ROOT

Load the runtime environment with:
  source "$ROOT_DIR/env.sh"
EOF
