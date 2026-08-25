# PULSAR-CPU

PULSAR-CPU is the OpenFHE implementation of PULSAR's Boolean-encoded integer
operators, refresh scheduler, and CPU application benchmarks. The release
supports dense SIMD packing, complex-packed parallel prefix computation,
Boolean-to-Boolean bootstrapping, Prefix Boot, and Q33/43-bit-scale
evaluation.

PULSAR-CPU depends on an OpenFHE 1.4.0 source tree. The bootstrapping code used
by PULSAR originated in the FHE-SIMD-ALU OpenFHE prototype, but the required
extension is included under `openfhe-overlay/`. Users do **not** need to clone,
build, or link FHE-SIMD-ALU separately. See [NOTICE](NOTICE) for provenance.

## Included benchmarks

The release contains the following single-instruction benchmarks:

- `ADD`: unsigned addition modulo `2^w`.
- `GT`: strict unsigned comparison `A > B`.
- `EQ`: unsigned equality comparison `A == B`.
- `XOR`: bitwise exclusive OR.
- `MUL`: unsigned multiplication modulo `2^w`.

It also contains the currently validated PULSAR CPU workloads:

- **Chained ADD:** an arbitrary number of consecutive additions scheduled with
  Boolean-to-Boolean bootstrapping and Prefix Boot.
- **Transfer:** `moved = sender >= amount ? amount : 0`, followed by sender and
  receiver updates.
- **Auction:** a tree reduction that returns the largest encrypted bid.
- **SHA-256:** between 1 and 64 compression rounds over one padded 512-bit
  block. A 64-round execution performs the complete compression function.

MIXED, RSA, division, and the FHE-SIMD-ALU baseline applications are not part
of this release. A PULSAR CPU implementation of Vault was not present in the
validated source set and is therefore not included.

## Parameters and packing

The benchmarks use the parameter profile from the CPU evaluation:

| Parameter | Value |
|---|---:|
| Ring dimension | `N = 2^16` |
| Complex slots | `2^15` |
| Multiplicative depth | 32 (`Q` count 33) |
| Scaling modulus | 43 bits |
| First modulus | 52 bits |
| Hybrid key-switching digits | 6 |
| Security setting | `HEStd_128_classic` |
| Enforced bound | `log2(QP) <= 1747` |

Bits at the same bit position in different integers occupy adjacent CKKS
slots. Two ciphertext groups fill the available logical slots:

| Word width | Batch per ciphertext | Aggregate batch |
|---:|---:|---:|
| 16 | 2,048 | 4,096 |
| 32 | 1,024 | 2,048 |
| 64 | 512 | 1,024 |
| 128 | 256 | 512 |
| 256 | 128 | 256 |

## Repository structure

```text
PULSAR-CPU/
|-- openfhe-overlay/        OpenFHE and bootstrapping extensions
|   `-- src/
|       |-- core/          fixed-point and complex-transform support
|       `-- pke/           FHEZ, PULSAR operators, schedulers, benchmarks
|-- tests/                  plaintext and repository checks
|-- install_pulsar_openfhe.py
|-- run_pulsar_*.sh
|-- LICENSE
`-- NOTICE
```

The installer copies the overlay into an OpenFHE source tree. Existing files
are backed up under `.pulsar-openfhe-backup-v1.3.0/` before replacement.
OpenFHE uses recursive source discovery, so no handwritten CMake target patch
is required. CMake must be reconfigured after installation to discover the
new benchmark files.

## Requirements

- Linux x86-64
- OpenFHE 1.4.0 source code
- CMake 3.16 or newer and a C++17 compiler
- OpenMP
- NTL and GMP
- Intel HEXL and tcmalloc for the evaluated configuration
- Python 3.10 or newer for installation and plaintext checks
- Sufficient memory for `N=2^16` bootstrapping key generation

On Ubuntu 22.04, the basic build dependencies can be installed with:

```bash
sudo apt update
sudo apt install -y build-essential git cmake clang libomp-dev \
  libntl-dev libgmp-dev autoconf libtool python3
```

## Installation

Obtain the pinned OpenFHE source release and install the PULSAR extension:

```bash
git clone --branch v1.4.0 --depth 1 \
  https://github.com/openfheorg/openfhe-development.git \
  ~/openfhe-development

cd ~/PULSAR-CPU
python3 install_pulsar_openfhe.py ~/openfhe-development
```

Configure and build the three executables:

```bash
cd ~/openfhe-development
CC=clang CXX=clang++ cmake -S . -B build \
  -DBUILD_EXAMPLES=ON \
  -DWITH_OPENMP=ON \
  -DWITH_NTL=ON \
  -DWITH_TCM=ON \
  -DWITH_INTEL_HEXL=ON \
  -DINTEL_HEXL_HINT_DIR="$PWD/build/install" \
  -DMATHBACKEND=6

cmake --build build --target \
  benchmark-pulsar-single-round \
  benchmark-pulsar-multiply \
  benchmark-pulsar-add-chain \
  -j"$(nproc)"
```

The installer requires exactly OpenFHE 1.4.0 because the overlay replaces a
small number of OpenFHE internals needed for `FLEXIBLEMANUAL` scaling and FHEZ
metadata. Installing into another OpenFHE release is rejected instead of
silently producing an incompatible build.

## Single-instruction benchmarks

Each script accepts the word width and CPU thread count:

```bash
cd ~/openfhe-development
./run_pulsar_add.sh 128 12
./run_pulsar_gt.sh 128 12
./run_pulsar_eq.sh 128 12
./run_pulsar_xor.sh 128 12
./run_pulsar_mul.sh 128 12 1
```

Supported widths are `16`, `32`, `64`, `128`, and `256`. The generic Boolean
operator interface is:

```bash
./run_pulsar_cpu_operator.sh <add|gt|eq|xor> \
  <word_bits> <threads> [output_refresh]
```

`output_refresh` defaults to `1`. The default measurement excludes setup and
input refresh, evaluates the instruction once, and includes a shared output
Boolean-to-Boolean refresh. Set the argument to `0` only to inspect the raw
instruction circuit. Multiplication has a separate interface:

```bash
./run_pulsar_mul.sh <word_bits> <threads> [repeats]
```

## Scheduler and applications

The chained-addition interface exposes the word width, number of additions,
and initial/final refresh controls:

```bash
OMP_NUM_THREADS=12 ./run_pulsar_add_chain.sh \
  <word_bits> <rounds> <initial_refresh:0|1> <final_refresh:0|1>
```

Application commands are:

```bash
OMP_NUM_THREADS=12 ./run_pulsar_transfer.sh 128 0.01
OMP_NUM_THREADS=12 ./run_pulsar_auction.sh 128 0.01
./run_pulsar_auction_16_256.sh 0.01
./run_pulsar_sha256.sh 1 0.01 12
./run_pulsar_sha256.sh 64 0.01 12
```

The threshold argument controls profiling-based Bit Clean scheduling. Setup,
key generation, encryption, input refresh, decryption, and correctness audits
are excluded from reported circuit latency. Online Bit Clean and refresh
operations selected during execution are included.

## Correctness checks

Run checks that do not require an OpenFHE build:

```bash
python3 verify_repository.py
python3 tests/single_round_plain_test.py
python3 tests/multiplier_plain_test.py
python3 tests/sha256_plain_test.py
```

Encrypted benchmarks decrypt and audit every output group after the timed
region. Successful runs finish with `OVERALL status=PASS`.

## License and attribution

PULSAR-CPU is distributed under the BSD 2-Clause license. OpenFHE-derived
files retain their original notices. Bootstrapping provenance and the boundary
between OpenFHE, FHE-SIMD-ALU-derived support, and PULSAR-specific code are
summarized in [NOTICE](NOTICE).
