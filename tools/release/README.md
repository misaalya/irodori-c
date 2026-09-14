# Irodori C — prebuilt Linux x86-64 release

Contents:

- `bin/irodori-onemkl` — engine with the oneMKL backend (FP32 + int8 paths;
  int8 needs a CPU with AVX-512 VNNI, e.g. Intel Ice Lake or newer, AMD Zen 4).
- `bin/irodori-blas` — engine on OpenBLAS (FP32 only, any x86-64-v3 CPU).
- `bin/irodori` — portable scalar oracle (slow; for testing).
- `lib/` — bundled runtime libraries (oneMKL subset, Intel OpenMP, OpenBLAS).
- `demo/` — minimal web UI (needs python3, standard library only).
- `download-model.sh`, `run-demo.sh` — helpers.

Requirements: Linux x86-64 with AVX2/FMA (x86-64-v3), glibc 2.35 or newer,
`curl` for the download script, `python3` for the demo. 2–3 GB RAM per engine.

## Setup

```sh
tar xzf irodori-c-assets-<version>.tar.gz --strip-components=1   # tokenizer.bin + codec (into weights/)
./download-model.sh                                             # model.safetensors, ~3 GB
./run-demo.sh                                                   # http://127.0.0.1:8080
```

Command line:

```sh
IRO_NUM_THREADS=2 bin/irodori-onemkl --text 'こんにちは。今日はいい天気ですね。' \
  --model weights/model.safetensors --tokenizer weights/tokenizer.bin \
  --decoder weights/dacvae_decoder.safetensors --encoder weights/dacvae_encoder.safetensors \
  --dit-precision int8 --codec-precision int8 --steps 8 --out out.wav
```

Drop the `--*-precision` flags (or use `bin/irodori-blas`) for the FP32
reference path. Set `IRO_NUM_THREADS` to the number of physical cores.
See `README-engine.md` for all options, quality notes and benchmarks.

Third-party components: Intel oneMKL and Intel OpenMP (Intel Simplified
Software License, `licenses/`), OpenBLAS (BSD-3), TBB malloc (Apache-2.0).
