# PULSAR-CPU

PULSAR-CPU is the OpenFHE implementation of PULSAR's Boolean-encoded integer
operators, refresh scheduler, and CPU application benchmarks. The release
supports dense SIMD packing, complex-packed parallel prefix computation,
Boolean-to-Boolean bootstrapping, Prefix Boot, and Q33/43-bit-scale
evaluation.

PULSAR-CPU bundles the pinned FHE-SIMD-ALU OpenFHE baseline required by the
implementation, together with Intel HEXL 1.2.6, Google cpu_features, and the
other source dependencies that are not installed as Ubuntu packages.
Installation does not clone source code from GitHub. See [NOTICE](NOTICE) for
provenance.

## Included benchmarks

The release contains the following instruction benchmarks:

- `ADD`: unsigned addition modulo `2^w`.
- `GT`: strict unsigned comparison `A > B`.
- `EQ`: unsigned equality comparison `A == B`.
- `XOR`: bitwise exclusive OR.
- `MIXED`: consecutive `ADD`, `XOR`, and `GT` instructions over the same
  Boolean representation.
- `MUL`: unsigned multiplication modulo `2^w`.

It also contains the currently validated PULSAR CPU workloads:

- **Chained ADD:** an arbitrary number of consecutive additions scheduled with
  Boolean-to-Boolean bootstrapping and Prefix Boot.
- **Transfer:** `moved = sender >= amount ? amount : 0`, followed by sender and
  receiver updates.
- **Auction:** a tree reduction that returns the largest encrypted bid.
- **Vault:** multiplies each encrypted deposit by a public exchange rate and
  divides the low `w`-bit product by a public rate scale.
- **SHA-256:** between 1 and 64 compression rounds over one padded 512-bit
  block. A 64-round execution performs the complete compression function.

RSA, encrypted division, and the FHE-SIMD-ALU baseline applications are not
part of this release.

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
|-- tests/                  plaintext correctness checks
|-- vendor/                 pinned dependency sources and checksums
|-- install_pulsar_openfhe.py
|-- setup_cpu.sh            source extraction, configuration, and build
|-- run_pulsar_*.sh
|-- LICENSE
`-- NOTICE
```

`setup_cpu.sh` extracts the pinned baseline into `.build/openfhe`, builds the
vendored Intel HEXL source, applies `openfhe-overlay/`, builds tcmalloc, and
then builds the PULSAR executables. The installer backs up replaced baseline
files under `.pulsar-openfhe-backup-v1.3.2/`.

## Requirements

The complete build has been validated on Ubuntu 22.04 with GCC 11.4 and
CMake 3.22.1.

- Linux x86-64
- CMake 3.16 or newer and a C++17 compiler
- OpenMP
- NTL and GMP
- Autoconf, Automake, and Libtool for the bundled tcmalloc build
- Python 3.10 or newer for installation and plaintext checks
- Sufficient memory for `N=2^16` bootstrapping key generation

On a fresh Ubuntu 22.04 server, install the system dependencies with:

```bash
sudo apt update
sudo apt install -y build-essential cmake libntl-dev libgmp-dev \
  autoconf automake libtool python3
```

## Installation

From the `PULSAR-CPU` directory, run the self-contained setup script:

```bash
cd PULSAR-artifact/PULSAR-CPU
bash setup_cpu.sh
```

The script uses all available CPU cores by default. To limit compilation
parallelism, set `PULSAR_BUILD_JOBS`, for example:

```bash
PULSAR_BUILD_JOBS=4 bash setup_cpu.sh
```

The resulting source tree and run scripts are placed in
`PULSAR-CPU/.build/openfhe`. The evaluated build configuration enables Intel
HEXL, NTL backend 6, OpenMP, and tcmalloc. The vendored archives are pinned to
FHE-SIMD-ALU commit `08f1eb87434e7be072cba889270a8400bbffc08e`, Intel HEXL
1.2.6, and Google cpu_features commit
`32b49eb5e7809052a28422cfde2f2745fbb0eb76`. Their checksums are verified
before extraction.

## Instruction benchmarks

Each script accepts the word width and CPU thread count:

```bash
cd PULSAR-artifact/PULSAR-CPU/.build/openfhe
./run_pulsar_add.sh 128 12
./run_pulsar_gt.sh 128 12
./run_pulsar_eq.sh 128 12
./run_pulsar_xor.sh 128 12
./run_pulsar_mixed.sh 128 12 1
./run_pulsar_mul.sh 128 12 1
```

Supported widths are `16`, `32`, `64`, `128`, and `256`. The generic Boolean
operator interface is:

```bash
./run_pulsar_cpu_operator.sh <add|gt|eq|xor|mixed> \
  <word_bits> <threads> [output_refresh]
```

`output_refresh` defaults to `1`. The default measurement excludes setup and
input refresh, evaluates the requested benchmark once, and includes a shared
output Boolean-to-Boolean refresh. Set the argument to `0` only to inspect the
raw instruction circuit.

For `mixed`, the benchmark evaluates `ADD(A,B)`, XORs the sum with a third
operand, and compares the result with `A` using `GT`. Any intermediate refresh
selected before `GT` and the requested output refresh are included in the
reported latency.

Multiplication has a separate interface:

```bash
./run_pulsar_mul.sh <word_bits> <threads> [repeats]
```

## Scheduler and applications

The scheduler module was incorporated in all the application tests.

Application commands are:

```bash
OMP_NUM_THREADS=12 ./run_pulsar_transfer.sh 128 0.01
OMP_NUM_THREADS=12 ./run_pulsar_auction.sh 128 0.01
./run_pulsar_auction_16_256.sh 0.01
./run_pulsar_vault.sh 256 3 3 12 1
./run_pulsar_sha256.sh 1 0.01 12
./run_pulsar_sha256.sh 64 0.01 12
```

The Vault interface is

```bash
./run_pulsar_vault.sh \
  <word_bits> <public_exchange_rate> <public_rate_scale> <threads> [repeats]
```

## Correctness checks

Run checks that do not require an OpenFHE build:

```bash
python3 tests/single_round_plain_test.py
python3 tests/multiplier_plain_test.py
python3 tests/sha256_plain_test.py
```

