# Codec DACVAE int8 (W8A8) — laporan 2026-09-14

Opsi baru `--codec-precision int8` (engine: `IroEngineConfig.codec_precision`,
env `IRO_CODEC_PRECISION`), independen dari `--dit-precision`. Default produk
tetap fp32.

## 1. Implementasi

- `dacvae.c`: `IroDACVAEInt8` milik engine (initial Conv7 1024→1536, 4×
  ConvTranspose upsample, 12× residual Conv7 + Conv1; payload 62,6 MiB).
  Conv7/Conv1 dikuantisasi per output channel langsung dari layout
  safetensors `[out,kernel,in]` → `[N, kernel*in]`; ConvTranspose ditranspos
  saat init.
- `conv1d_int8_forward`: im2col ditulis sebagai **uint8** dengan skala/zero
  point **per baris output** (gabungan rentang 7 baris input yang menyumbang),
  padding = zero point; satu GEMM int8 K = kernel·in per blok (≈6–8 k baris),
  akumulasi int32 eksak. Term residual opsional: sisa kuantisasi punya rentang
  ±s/2 sehingga skala keduanya deterministik (s/255, zp 128) tanpa statistik
  tambahan. Conv1 dan upsample memakai `iro_int8_quantize_rows(_residual)`.
- Snake: kernel per baris dengan `restrict` + 1/α di-hoist → libmvec
  `_ZGVeN16v_sinf` (sebelumnya `snake_bias_forward` in-place tidak
  tervektorisasi sama sekali); `dacvae_blas.o` dikompilasi
  `-mprefer-vector-width=512` (+25% pada snake in-place, microbench).
- Halaman FP32 `decoder.model.*` dilepas setelah kuantisasi; self-check
  saturasi backend sama seperti DiT; stats `codec_precision`/`codec_int8_bytes`;
  worker/harness (`bench_speed_tradeoff.py --experiment codec-int8`) dan
  `int8_quality_corpus.py` (config `codec-int8`, `codec-int8-plain`,
  `full-int8`) diperluas. Test: `test-int8-engine-onemkl` kini juga mencakup
  codec int8 (determinisme, SNR vs fp32 ≥20 dB) dan lolos ASan/UBSan.

## 2. Kualitas — gate langsung (latent identik, DiT fp32)

Berbeda dari DiT, decoder deterministik terhadap latent, sehingga SNR waveform
int8 vs fp32 mengukur error codec murni.

Atribusi pada satu sampel (teks "greeting", noise golden):

| Konfigurasi codec | SNR dB | LSD dB | STOI |
|---|---:|---:|---:|
| W8A8 plain (semua grup) | 31,7 | 4,02 | 0,999 |
| hanya initial / upsample / conv7 / conv1 int8 | 47,6 / 38,5 / 35,9 / 35,5 | 0,47 / 1,08 / 3,27 / 2,32 | 1,000 |
| residual aktivasi di semua grup (≈ W8 saja) | 33,9 | 0,74 | 1,000 |
| residual conv7+conv1 **hanya stage 3** (default) | 31,7 | 1,37 | 0,999 |

Bacaan: SNR dibatasi kuantisasi **bobot** (~34 dB bahkan dengan aktivasi
≈16-bit); LSD (jarak log-mel, sensitif pada bin frekuensi tinggi berenergi
rendah) ditentukan kuantisasi aktivasi di **stage terakhir** (96 ch). Error
bersifat signal-dependent (korelasi 0,98 dengan energi frame; frame sunyi
tetap −87 dBFS), bukan noise floor. Default memilih residual stage 3 saja:
LSD 4,0 → 1,4 dengan ≈¼ FLOP conv7 tambahan.

Korpus 4 mode × 6 teks, 8-step (`quality-*/`, `tables-generated.md`):

| mode | codec int8 plain: SNR mean/min · LSD mean/max | **codec int8 default**: SNR · LSD · STOI min | full int8 (DiT+codec): LSD mean/max · STOI mean/min |
|---|---|---|---|
| text-only | 32,2/31,4 · 4,05/6,27 | 32,2/31,4 · **1,30/2,52** · 1,000 | 2,07/3,19 · 0,983/0,954 |
| caption-only | 32,3/31,5 · 3,97/6,04 | 32,4/31,5 · 1,39/2,53 · 0,996 | 2,20/2,97 · 0,980/0,958 |
| clone | 32,7/30,2 · 1,84/2,66 | 32,7/30,2 · 0,62/0,91 · 0,995 | 1,68/2,27 · 0,965/0,943 |
| clone+caption | 33,1/30,7 · 1,75/2,74 | 33,1/30,7 · 0,70/1,24 · 0,991 | 1,76/2,36 · 0,963/0,952 |

Pembanding: DiT int8 saja (laporan sebelumnya) LSD 1,45–1,70 / STOI
0,965–0,983 → codec int8 menambah ≈0,3–0,6 dB LSD dan tidak mengubah STOI.

**ASR CER** (kotoba-whisper-v2.0, 48 klip): codec-int8 dan full-int8
**0,032 di keempat mode, median 0, tidak ada klip >0,2** — identik dengan fp32.

Blind A/B: `blind-ab-full-int8/` (fp32 vs DiT+codec int8) dan
`blind-ab-codec-only/` (fp32 vs codec int8 saja), kunci terpisah — belum
dinilai pengguna.

## 3. Kecepatan

Profil per-op decode 112 frame (diagnostik, host sibuk): conv7 GEMM+dekuant
3,58 → 0,99 s (3,6×), upsample GEMM 0,66 → 0,22 s, conv1 0,81 → 0,41 s,
im2col+kuantisasi 0,38 s, snake ~0,3 s. Non-GEMM kini ≈50% waktu decode int8.
Angka formal: lihat §4 (diisi setelah host idle).

## 4. Kecepatan formal

### 4.1 C/C: codec int8 di atas DiT int8 — `speed-codec-int8/`

`bench_speed_tradeoff.py --experiment codec-int8`, kedua varian
`IRO_DIT_PRECISION=int8` + `MKL_CBWR=AVX512_E1`, baseline = codec fp32,
kandidat = codec int8 (default). 3 pair bergantian, proses fresh + warm-up,
CPU 0,1, idle ≥90% sebelum setiap sampel (terukur 90–95%). Kandidat lebih
cepat pada 12/12 pair.

| mode | steps | codec fp32 p50 s (min–max) | codec int8 p50 s (min–max) | speedup E2E | decode p50 s | peak RSS MiB | semua pair lebih cepat |
|---|---:|---:|---:|---:|---:|---:|---|
| text-only | 8 | 7.06 (6.91–7.39) | 5.03 (4.90–5.19) | **1.40×** | 4.45 → 2.44 | 1811 → 1610 | True |
| text-only | 40 | 18.99 (18.28–19.05) | 16.42 (16.19–17.35) | **1.16×** | 5.05 → 2.58 | 1810 → 1610 | True |
| clone+caption | 8 | 15.02 (14.97–15.23) | 12.53 (12.49–13.06) | **1.20×** | 5.25 → 2.71 | 2312 → 2123 | True |
| clone+caption | 40 | 35.07 (34.22–35.70) | 32.81 (32.22–33.19) | **1.07×** | 5.35 → 2.96 | 2313 → 2124 | True |

Run pertama (`attempts-aborted/speed-codec-int8-run1`) memberi angka yang sama
kecuali satu outlier kandidat 55 s yang bertepatan dengan tekanan memori host;
run bersih di atas menggantikannya.

### 4.2 Python vs C, DiT int8 + codec int8 (8-step, 5 repeat, idle 98–99%)

Lihat `../pyc-3scenarios-20260914/REPORT.md` lengan C. p50: text-only Python
25,09 → C 5,51 s (**4,55×**, RTF 1,23); caption-only 29,11 → 7,22 (4,03×);
clone 37,25 → 12,30 (3,03×); clone+caption 40,46 → 12,64 (3,20×). Peak RSS
C 1,6–2,1 GB (−61…−68% vs Python).

## 5. Ringkasan trade-off

| | codec fp32 | codec int8 (default) |
|---|---|---|
| decode 112–116 frame | 5,2–5,5 s | 2,5–2,8 s |
| E2E text-only 8-step (dengan DiT int8) | 7,1 s | **5,1 s** (RTF 1,2) |
| SNR vs fp32, latent identik | – | 32–33 dB (min 30,2) |
| LSD / STOI vs fp32 | – | 0,6–1,4 dB / ≥0,991 |
| CER ASR | 0,032 | 0,032 |
| Payload tambahan | – | 62,6 MiB (halaman FP32 decoder dilepas → RSS −200 MiB) |
| Reproduksi bit-level | ya | tidak |

Batas berikutnya (decode int8 2,6 s): snake ≈0,3 s, im2col+kuantisasi
≈0,35 s, conv1 ≈0,4 s, GEMM ≈1,3 s; pada clone, encode reference FP32 5 s.
