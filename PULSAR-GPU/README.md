# PULSAR-GPU

This package contains the PULSAR GPU operators and the modified FIDESlib
source required to build them. PULSAR represents fixed-width integers as
Boolean CKKS slots and uses a ring dimension of `2^17`.

## Requirements

- Linux x86-64
- NVIDIA GPU with CUDA 12 or newer
- GCC/G++ 11 or newer
- CMake 3.25.2 or newer
- Git and GNU Make
- Internet access while installing OpenFHE
- NCCL is optional and enables FIDESlib multi-GPU support when installed

The tested environment is Ubuntu 22.04 with an NVIDIA A100, CUDA 12.4,
GCC 11.4, and CMake 3.31.1.

On Ubuntu, the non-CUDA build tools can be installed with:

```bash
sudo apt install -y build-essential git
```

CUDA must be installed separately and available at `/usr/local/cuda`. Set
`CUDA_HOME` before installation when CUDA is installed elsewhere.

## Install

From the extracted `PULSAR-GPU` directory, run:

```bash
chmod +x install_gpu.sh env.sh
JOBS=24 ./install_gpu.sh
```

The script performs the complete build:

1. clones the pinned OpenFHE `v1.5.1` source;
2. applies the included PULSAR OpenFHE patch;
3. builds and installs OpenFHE;
4. builds the bundled PULSAR-modified FIDESlib; and
5. builds and installs all PULSAR-GPU operators and applications.

Sources remain unchanged. Downloaded dependencies and build files are stored
in `.build`, while installed headers, libraries, and executables are stored
in `.local`. Both directories are local to this artifact.

The script detects the compute capabilities reported by `nvidia-smi`. To set
the target explicitly, use, for example:

```bash
FIDESLIB_ARCH=80-real JOBS=24 ./install_gpu.sh
```

When the default `cmake` is older than 3.25.2, specify a newer executable:

```bash
CMAKE_BIN=/opt/cmake-3.31.1/bin/cmake JOBS=24 ./install_gpu.sh
```

## Run

Load the installed environment and run an operator directly. For example,
the following command executes 256-bit addition on one GPU:

```bash
source ./env.sh
CUDA_VISIBLE_DEVICES=0 pulsar_gpu_add --bits 256 --gpus 1
```

Every executable accepts `--help`. Installed operators are:

- `pulsar_gpu_add`, `pulsar_gpu_sub`, and `pulsar_gpu_mul`
- `pulsar_gpu_scalar_mul`, `pulsar_gpu_div`, and `pulsar_gpu_scalar_div`
- `pulsar_gpu_gt`, `pulsar_gpu_ge`, `pulsar_gpu_lt`, `pulsar_gpu_le`,
  `pulsar_gpu_eq`, and `pulsar_gpu_ne`
- `pulsar_gpu_and`, `pulsar_gpu_or`, `pulsar_gpu_xor`, and `pulsar_gpu_not`
- `pulsar_gpu_shl`, `pulsar_gpu_shr`, `pulsar_gpu_scalar_shl`, and
  `pulsar_gpu_scalar_shr`

## Run Applications

After installation, load the environment once:

```bash
source ./env.sh
```

The five application workloads used in the evaluation are invoked as follows.
Transfer uses all GPUs exposed by `CUDA_VISIBLE_DEVICES`; Auction and Vault
require three visible GPUs.

### Transfer (White Paper)

```bash
CUDA_VISIBLE_DEVICES=0,1,2 pulsar_gpu_transfer whitepaper
```

### Transfer (No CMUX)

```bash
CUDA_VISIBLE_DEVICES=0,1,2 pulsar_gpu_transfer no_cmux
```

### Transfer (Overflow)

```bash
CUDA_VISIBLE_DEVICES=0,1,2 pulsar_gpu_transfer overflow
```

All three Transfer workloads can alternatively be run consecutively with
`pulsar_gpu_transfer all`.

### Auction

```bash
CUDA_VISIBLE_DEVICES=0,1,2 pulsar_gpu_auction
```

### Confidential Vault

The arguments are the public exchange rate and the nonzero public rate scale.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 pulsar_gpu_vault 3 3
```

## Layout

- `src`: PULSAR-GPU operator and application implementations
- `third_party/FIDESlib`: modified FIDESlib source required by PULSAR
- `patches`: provenance patch for FIDESlib and build patch for OpenFHE
- `tests`: plaintext correctness tests
- `install_gpu.sh`: complete source installation
- `env.sh`: runtime environment
Exact dependency revisions are recorded in `DEPENDENCIES.md`.
