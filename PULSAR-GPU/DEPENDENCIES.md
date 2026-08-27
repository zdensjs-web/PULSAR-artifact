# Dependency provenance

This artifact bundles the PULSAR-modified FIDESlib source in
`third_party/FIDESlib`.

- FIDESlib base commit: `0ec405544b179fdfbe01fe4e0421a7cd6336f35c`
- OpenFHE version: `v1.5.1`
- OpenFHE commit: `1306d14f8c26bb6150d3e6ad54f28dfe1007689e`
- Known build environment: Ubuntu 22.04, GCC 11.4, CUDA 12.4, CMake 3.31.1
- Known GPU architecture: NVIDIA A100, compute capability 8.0 (`80-real`)

`patches/fideslib-pulsar-0ec4055.patch` records the PULSAR changes relative
to the FIDESlib base commit. The bundled FIDESlib source already contains
these changes. `patches/openfhe-pulsar-v1.5.1.patch` is applied automatically
to a clean OpenFHE `v1.5.1` checkout by `install_gpu.sh`.

The FIDESlib and OpenFHE license files remain in their respective source
trees. No prebuilt FIDESlib or OpenFHE libraries are included.
