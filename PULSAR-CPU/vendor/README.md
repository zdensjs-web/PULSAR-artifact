# Vendored sources

The setup script builds the following pinned source archives without network
access:

- `fhe-simd-alu-08f1eb8.tar.gz`: FHE-SIMD-ALU commit
  `08f1eb87434e7be072cba889270a8400bbffc08e`, including cereal commit
  `984e3f194862b17916536b5fade40cba6e47a6fe` and gperftools commit
  `83edb60836d87cf1b406e8846b9059c03031e8f5`.
- `hexl-1.2.6.tar.gz`: Intel HEXL 1.2.6, commit
  `75a60b8cc908fb4c166315dd8b55b2e1ee7faea7`.
- `cpu-features-32b49eb.tar.gz`: Google cpu_features commit
  `32b49eb5e7809052a28422cfde2f2745fbb0eb76`.

`SHA256SUMS` records the archive checksums verified by `setup_cpu.sh`. Each
archive includes its upstream license and attribution files.
