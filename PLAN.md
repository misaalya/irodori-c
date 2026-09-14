# PLAN — Irodori C Engine

Target produk: engine inference pure-C untuk Irodori-TTS-v4.1-Small — no Python,
no PyTorch saat runtime. Filosofi mengikuti qwen-asr (antirez) / qwen3-tts
(mmap safetensors, C + BLAS, golden-reference tests).

Prinsip:
1. **Golden test sebelum optimasi** — setiap komponen C dibandingkan bit-eksak
   (toleransi fp32 ~1e-4) dengan dump referensi PyTorch. Tidak ada gate yang dilompati.
2. **Urutan komponen = urutan pipeline** — kalau komponen awal salah, semua setelahnya
   ikut salah; verifikasi berantai.
3. **Scalar dulu, SIMD belakangan** — kebenaran dulu, kecepatan kemudian.
4. **Setiap fase menghasilkan artefak yang bisa dites** — bukan big-bang.

---

## P0 — Fondasi & Ground Truth (✅ SELESAI)

| Item | Status |
|---|---|
| MODEL.md (source of truth arsitektur) | ✅ (✅/⚠️ ter-update dari golden trace) |
| C skeleton: Makefile, main.c, safetensors mmap loader | ✅ |
| `--list-tensors` + `--info` + `--check` terhadap checkpoint asli | ✅ gate lolos: 714 tensor, 0 masalah |
| `tools/dump_dacvae_safetensors.py` — konversi codec .pth → safetensors + kwargs JSON | ✅ (317 tensor, 0 masalah; hop 1920 @48kHz = 25Hz) |
| `tools/dump_golden.py` — golden trace per komponen | ✅ (41 file: token_ids → backbone → projector → duration → x_t/cond_embed/velocity per step → codec decode → waveform) |
| Inventory nama tensor final | ✅ (section 11 MODEL.md) |

Temuan penting dari golden trace (sudah masuk MODEL.md):
- `cond_module(t_embed)` = satu-satunya input AdaLN conditioning (512→1280→1280→3840)
- Text di-pad kanan ke 256 token
- CFG uncond di-skip saat t < cfg_min_t 0.5 (batch 2 → 1 di tengah sampling)

## P1 — Text-Only Path (fp32 scalar C)

Komponen yang diimplementasi + gate per komponen:
1. **Unigram tokenizer** — ✅ gate lolos: **20/20 kalimat uji identik dengan HF**
   (emoji, byte fallback 𩸽, Metaspace ▁, angka, campuran JP/latin).
   Aset: `tools/compile_tokenizer.py` → `weights/tokenizer.bin` (1.5MB, mmap).
2. **Text normalization** — ✅ gate lolos bersama tokenizer (20/20, input raw):
   simple-replace + 4 regex (scanner codepoint) + bracket strip + NFKC
   (vendored **utf8proc v2.10.0**, MIT) + aturan "..." → "…".
3. **ModernBERT backbone** (25L) — gate: max abs err < 1e-3 vs golden dump
   (sliding window attention perlu implementasi benar — cek window size!)
4. **Projector residual_mlp** — gate: < 1e-4
5. **Duration predictor** — gate: frame count sama (integer) untuk 20 kalimat uji
6. **DiT forward 12 blocks + half-RoPE + AdaLN + joint attention** —
   gate: velocity max err < 1e-3 per step (5 timestep sampel)
7. **Euler CFG loop** — gate: latent final err < 1e-2; WAV hasil C vs referensi:
   tidak bisa bit-identik (RNG + fp accumulation), gate = korelasi spektrogram > 0.99
   ATAU dump noise dari golden → bit-identik
8. **DACVAE decoder** — gate: waveform err < 1e-3 vs golden decode
9. **WAV writer + trim tail** — gate: identik

Endpoint P1: `./irodori --text "こんにちは" --seed 42 --out out.wav`
(text-only, tanpa ref/caption) — hasil terdengar benar & terukur mirip referensi.

## P2 — Voice Cloning + Caption

1. DACVAE **encoder** port (deterministic mean path) — gate: latent err < 1e-3
2. WAV loader (PCM16/24/32, resample ke codec rate ⚠️ cek codec rate 44.1k/48k)
3. Loudness normalize −16 dB + peak safety (per klip) — gate: identik
4. **Speaker encoder** (8L, 768) — gate: < 1e-3
5. Cabang caption (backbone reuse + projector kedua) — gate: < 1e-3
6. Duration + DiT dengan conditioning penuh — gate: kombinasi 3 cabang

## P3 — Performa

1. BLAS (OpenBLAS/Accelerate) untuk matmul besar — target: RTF < 1 @ fp32 (CPU modern)
2. Layout memori: weights interleaved/packed per konsumsi, posix_memalign(64)
3. SIMD AVX2/FMA untuk kernel kecil (rmsnorm, rotary, softmax, gelu)
4. KV-cache context precompute (text/speaker/caption sekali, reuse 40 steps —
   struktur runtime sudah begitu di Python, replikasi layout-nya)
5. Benchmark harness (bench/bench.sh) + tabel RTF per komponen

## P4 — Kuantisasi (INT8/INT4 weight-only)

1. Tool konversi: int8/int4 blob + scale, satu file + header
2. Kernel dequant-on-the-fly (fused gemm atau pre-dequant per blok)
3. Gate kualitas: C-int8 vs C-fp32 — kana-CER tidak boleh lebih buruk
   dari beda fp32↔torchao-int8 (margin < 0.5%)
4. Target memori: RSS < 1.2 GB (model int8 ~0.9 GB)

## P5 — Produk

1. CLI final: `--text --caption --ref (multi) --voice-preset --seed --steps --out`
2. Library: `libirodori.h` API bersih (init/feed/free) untuk embedding
3. Integrasi opsional ke webui (ganti backend Python, UI tetap)
4. CI golden tests, cross-platform Makefile (Linux/macOS/Windows-MSVC)
5. Docs + blog notes per problem seru (pattern qwen3-tts)

---

## Keputusan Desain Tercatat

| # | Keputusan | Alasan |
|---|---|---|
| D1 | mmap safetensors langsung, tanpa konversi format (fp32) | pattern qwen-asr; startup instan |
| D2 | Noise RNG: ekstrak z dari golden dump, bukan port MT19937-torch di P1 | bit-identik lebih penting daripada port RNG; port RNG jadi item P3 opsional |
| D3 | Tokenizer: Viterbi unigram murni C, vocab dari tokenizer.json | normalizer null → tidak perlu ICU |
| D4 | DACVAE weights dikonversi sekali ke safetensors via tool Python | torch .pth (pickle) tidak bisa di-mmap aman dari C |
| D5 | fp32 scalar → BLAS/SIMD → int8, per gate | correctness dulu |
| D6 | Folder `irodori-c/` terpisah dari webui/ | batas tanggung jawab jelas |

## Risiko Terbuka

- **Sliding window attention ModernBERT** — beda dari LLM biasa; harus match persis.
- **Half-RoPE & AdaLN mapping** — verifikasi dari kode, bukan asumsi.
- **fp accumulation order** — C vs PyTorch beda urutan sum → tolerance, bukan bit-identik.
- **DACVAE decoder** — conv net besar (paling banyak kerja per-sampel); butuh im2col/gemm
  yang benar; ini komponen perf terpenting setelah DiT.
