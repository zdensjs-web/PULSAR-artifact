# PULSAR-GPU

PULSAR-GPU is the GPU implementation of the PULSAR fixed-width integer operators. It evaluates Boolean-encoded encrypted integers with CKKS and uses [FIDESlib](https://github.com/CAPS-UMU/FIDESlib) as the CUDA backend. The implementation supports packed execution, complex-slot prefix computation, and multi-GPU execution.

This repository is a research artifact. The current release contains the production operator executables and plaintext regression tests. End-to-end application drivers are not included in this release.

## Operators

PULSAR-GPU supports power-of-two word widths from 8 to 256 bits. The default width is 256 bits.

| Group | Operation | Executable |
|---|---|---|
| Arithmetic | Addition | `pulsar_gpu_add` |
| Arithmetic | Subtraction | `pulsar_gpu_sub` |
| Arithmetic | Multiplication modulo `2^w` | `pulsar_gpu_mul` |
| Division | Public-scalar division | `pulsar_gpu_scalar_div` |
| Division | Ciphertext-ciphertext division | `pulsar_gpu_div` |
| Comparison | `GT`, `GE`, `LT`, `LE`, `EQ`, `NE` | `pulsar_gpu_gt`, `pulsar_gpu_ge`, `pulsar_gpu_lt`, `pulsar_gpu_le`, `pulsar_gpu_eq`, `pulsar_gpu_ne` |
| Bitwise | `XOR`, `AND`, `OR`, `NOT` | `pulsar_gpu_xor`, `pulsar_gpu_and`, `pulsar_gpu_or`, `pulsar_gpu_not` |
| Shift | Public left/right shift | `pulsar_gpu_scalar_shl`, `pulsar_gpu_scalar_shr` |
| Shift | Encrypted left/right shift | `pulsar_gpu_shl`, `pulsar_gpu_shr` |

Addition, subtraction, and comparison use a parallel-prefix circuit with complex packing. Bit positions are packed across CKKS slots to evaluate many integers in parallel. Arithmetic and comparison executables perform the final ciphertext refresh required by their standalone benchmark contract. `NOT` does not perform bootstrapping.

## Requirements

- Linux with an NVIDIA CUDA-capable GPU
- NVIDIA CUDA 12 or 13
- GCC 11 or later
- CMake 3.25.2 or later
- OpenMP development files
- [FIDESlib](https://github.com/CAPS-UMU/FIDESlib) and the FIDESlib-compatible patched OpenFHE build
- NCCL for multi-GPU execution when enabled in FIDESlib

The known-good environment uses the patched FIDESlib/OpenFHE tree used during PULSAR-GPU development. Compatibility with an unmodified upstream FIDESlib checkout is not guaranteed when the upstream API changes.

## Install FIDESlib

Follow the [official FIDESlib installation instructions](https://github.com/CAPS-UMU/FIDESlib). A typical user-local installation is:

```bash
git clone https://github.com/CAPS-UMU/FIDESlib.git
cmake -S FIDESlib -B FIDESlib/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DFIDESLIB_INSTALL_OPENFHE=ON \
  -DFIDESLIB_INSTALL_PREFIX="$HOME/.local" \
  -DOPENFHE_INSTALL_PREFIX="$HOME/.local"
cmake --build FIDESlib/build -j"$(nproc)"
cmake --build FIDESlib/build --target install -j"$(nproc)"
```

CUDA installation and GPU architecture selection depend on the host system. Consult the FIDESlib documentation before changing its CMake options.

## Build PULSAR-GPU

Clone the repository and point CMake to the FIDESlib installation prefix:

```bash
git clone <PULSAR-GPU repository URL>
cd PULSAR-GPU

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/.local"
cmake --build build -j"$(nproc)"
```

All executables are written to `build/bin`. To build one operator only:

```bash
cmake --build build --target pulsar_gpu_add -j"$(nproc)"
```

Optional installation:

```bash
cmake --install build --prefix "$HOME/.local"
```

## Run

Every executable accepts `--help`. The common options are:

- `--bits N`: word width, one of `8`, `16`, `32`, `64`, `128`, or `256`
- `--words N`: number of packed words per ciphertext
- `--gpus N`: use logical GPU indices `0` through `N-1`
- `--devices LIST`: select logical GPU indices explicitly

`CUDA_VISIBLE_DEVICES` selects the physical devices visible to the process. `--gpus` and `--devices` select logical indices within that visible set.

### Addition

```bash
CUDA_VISIBLE_DEVICES=0,1 \
./build/bin/pulsar_gpu_add --bits 256 --gpus 2
```

### Comparison

```bash
CUDA_VISIBLE_DEVICES=0 \
./build/bin/pulsar_gpu_gt --bits 128 --gpus 1
```

### Multiplication

The multiplier also accepts `--merge-batch N`, which controls the number of row merges performed between GPU synchronization points.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
./build/bin/pulsar_gpu_mul --bits 256 --gpus 2 --merge-batch 8
```

### Ciphertext-ciphertext division

The divider also accepts `--parallel-blocks N`.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
./build/bin/pulsar_gpu_div --bits 256 --gpus 2 --parallel-blocks 2
```

Use the executable's `--help` output as the authoritative description of supported options and defaults.

## Verification

The repository includes tests for the plaintext algebra, packing rules, and level schedules. These tests do not require a GPU build:

```bash
python3 verify_repository.py
python3 tests/subtractor_head_zero_plain_test.py
python3 tests/comparison_head_zero_plain_test.py
python3 tests/word_operator_plain_test.py
python3 tests/multiplier_plain_test.py
python3 tests/scalar_divider_plain_test.py
python3 tests/cipher_divider_plain_test.py
python3 tests/cipher_divider_level_schedule_test.py
```

Full ciphertext correctness and timing must be verified on a machine with the matching FIDESlib/OpenFHE build and NVIDIA GPUs.

## Repository Structure

```text
PULSAR-GPU/
|-- CMakeLists.txt
|-- README.md
|-- src/                  # Operator implementations and shared runtime code
|-- tests/                # Plaintext algebra and schedule regression tests
`-- verify_repository.py  # Static package verification
```

## Acknowledgments

PULSAR-GPU builds on [FIDESlib](https://github.com/CAPS-UMU/FIDESlib), a CUDA CKKS backend interoperable with [OpenFHE](https://github.com/openfheorg/openfhe-development).
