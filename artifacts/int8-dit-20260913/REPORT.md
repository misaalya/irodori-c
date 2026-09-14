# DiT int8 (W8A8) — laporan sesi 2026-09-13

Status: **implementasi selesai dan lolos test/sanitizer; kualitas dievaluasi
dengan metrik audio + ASR CER; timing kecepatan pada sesi ini diagnostik**
(host tidak idle sepanjang sesi, load 2–5 pada CPU 4-thread). Tidak ada
perubahan default: `--dit-precision int8` opt-in.

## 1. Mengapa jalur FP32 dihentikan: roofline host

Pengukuran sesi ini (`sgemm_roof.c`, `int8_pack.c` di direktori ini):

| Fakta | Nilai |
|---|---|
| CPU | i3-1005G1, 2 core / 4 thread, AVX-512 + VNNI |
| Frekuensi saat beban AVX 2-core | **2,00 GHz konstan** — `intel_pstate/max_perf_pct=60`, `scaling_max_freq=2000000`, EPP `balance_power` (konfigurasi host, bukan thermal; `no_turbo=0`) |
| Roofline FP32 | 2 core × 32 FLOP/cycle × 2,0 GHz = **128 GFLOPS** |
| MKL SGEMM 2048³, 2 thread | 110–113 GFLOPS (AVX2 / AVX512 / auto sama) |
| GEMM DiT M=224,N=3680,K=1280 | 102–108 GFLOPS |
| Conv7 codec M=8192,N=192,K=192 | 99–109 GFLOPS |
| Engine text-only 8-step (≈1,2 TFLOP: DiT ~0,73 + codec ~0,49) | 11,7 s idle vs minimum teoretis 10,9 s → **~94% roofline** |

Konsekuensi: seluruh eksperimen kernel FP32 yang ditolak sebelumnya memang
tidak punya headroom. 40-step FP32 ≈ 4,1 TFLOP → minimum ~37 s untuk 4,5 s
audio; **RTF<1 tidak dapat dicapai dalam FP32 di host ini** apa pun kernelnya.
Satu-satunya tuas besar yang tersisa adalah menaikkan ops/cycle (VNNI int8 =
4× FP32) atau mengurangi FLOP (mengubah model/step, di luar scope).

Catatan host: melepas cap 60% (`max_perf_pct`) akan menaikkan roofline hingga
~1,5× bila thermal mengizinkan; ini keputusan pengguna, tidak diubah sesi ini,
dan semua angka historis diukur dengan cap tersebut.

## 2. Bench VNNI lama vs MKL int8

Kesimpulan lama "VNNI tidak menarik" (`tests/bench_vnni_linear.c`) berasal
dari kernel naif yang store-bound (akumulator `acc[bm]` dengan trip count
variabel → GCC tidak menaruhnya di register): 105 GOPS ≈ FP32. MKL
`cblas_gemm_s8u8s32` (oneMKL 2026.1 runtime yang sudah ada di artefak V2):

| Shape | FP32 ms | int8 ms | speedup |
|---|---:|---:|---:|
| 2048³ | 170,7 | 40,9 | 4,18× |
| 224×3680×1280 (W1/W3, B=2) | 23,8 | 5,4 | 4,41× |
| 224×1280×1280 (QKVG/Wo) | 7,8 | 1,7 | 4,66× |
| 224×1280×3680 (W2) | 20,6 | 6,1 | 3,40× |
| 448×3680×1280 (B=4) | 41,7 | 10,5 | 3,97× |
| 8192×192×192 (codec) | 5,7 | 1,4 | 4,14× |

Packed-B (`cblas_gemm_s8u8s32_pack/compute`) hanya +10–15% dan
`pack_get_size` melaporkan 12–17 MiB per matriks (~1,4 GB seluruh DiT); pack
bisa dipakai lintas M (bit-exact). Tidak dipakai; raw int8 (257 MiB) menjadi
default.

## 3. Implementasi

- `ops.c/h`: `IroInt8Weight` (int8 per output channel + scale + colsum),
  `iro_int8_quantize_rows` (uint8 per baris asimetris, zero point per baris),
  `iro_int8_linear[_ex]` (MKL s8u8s32 di build oneMKL; referensi skalar di
  build lain; akumulasi int32 eksak → **MKL bit-exact dengan referensi**),
  `iro_int8_quantize_rows_residual` (opsi dua-term, lihat §5).
- `dit.c/h`: `IroDiTInt8` (8 proyeksi × 12 layer, 256,8 MiB), Q/K/V/gate
  berbagi satu kuantisasi input, W1/W3 juga; AdaLN, conditioner, in/out,
  K/V konteks, QKᵀ/PV attention tetap FP32. Pemilihan layer identik dengan
  profil `core` rilis upstream `Irodori-TTS-v4.1-Small-Quantized/int8-dynamic`.
- `generate.c/h`: `IroEngineConfig.dit_precision`, kuantisasi saat init
  (~0,4 s), halaman FP32 bobot yang digantikan dilepas (`iro_st_drop_prefix`),
  `IroGenerateStats.dit_precision/dit_int8_bytes`.
- CLI `--dit-precision fp32|int8` / `IRO_DIT_PRECISION`; worker
  `--dit-precision`; harness `bench_speed_tradeoff.py --experiment int8`.
- Test: `test-int8-ops` (skalar), `test-int8-ops-onemkl` (MKL bit-exact,
  cross-M), `test-int8-engine-onemkl`, `test-int8-engine-sanitize`
  (ASan/UBSan, leak detection off seperti audit existing). `make audit-warnings
  audit-static`, `git diff --check` PASS. Output int8 deterministik antar run
  dan antar request warm (byte-identik).
- Knob diagnostik (bukan produk): `IRO_INT8_MASK`, `IRO_INT8_RESIDUAL`,
  `IRO_INT8_EMULATE`, `IRO_INT8_STATS`, `IRO_INT8_DUMP`.

## 4. Kualitas: mengapa golden FP32 tidak bisa dipakai

Pasangan kalibrasi pada teks/seed/noise sama (`attribution-e1.txt`, satu sampel
"greeting" dengan noise golden — sampel yang kebetulan sulit):

| Pasangan | SNR dB | corr | LSD dB | MCD | STOI |
|---|---:|---:|---:|---:|---:|
| FP32 AVX2 vs FP32 AVX512 (hanya rounding) | 82,3 | 1,000 | 0,25 | 0,02 | 1,000 |
| FP32 seed 42 vs seed 43 (sampel lain) | −3,1 | −0,02 | 20,8 | 139,8 | 0,245 |
| FP32 vs int8 W8A8 plain | −0,2 | 0,48 | 4,09 | 27,5 | 0,803 |
| FP32 vs bobot-int8 saja (emulasi) | 4,7 | 0,83 | 1,76 | 8,3 | 0,951 |
| FP32 vs aktivasi-int8 saja (emulasi) | −0,4 | 0,45 | 4,37 | 27,5 | 0,799 |

Kuantisasi **aktivasi** mendominasi. Statistik aktivasi (`IRO_INT8_STATS`):
input W2 (produk SwiGLU) max/rms 15–21, input lain 5–8; channel outlier hanya
sebagian konsisten. Simulasi NumPy pada aktivasi nyata (`sim_quant.py`,
error relatif per GEMM rata-rata): W8 saja 0,78%; A8 per-row asimetris
(implementasi ini) 1,75%; A8 simetris per-row (= torchao int8-dynamic
upstream) 2,12%; K-group 128 ~1,0%; SmoothQuant 1,40%; residual 2-term 0,78%.

Sampler Euler 8-step kaotik: perturbasi 1e-6 → LSD 0,1 dB, ~1% → LSD 2–5.
Golden max-abs FP32 karena itu tidak berlaku; output int8 adalah *sampel valid
lain* yang dekat (STOI 0,95–0,98 pada korpus) — bukan versi terdegradasi.
Gate yang dipakai: korpus 4 mode × 6 teks (jarak + kalibrasi), **ASR CER
terhadap teks target**, dan paket blind A/B untuk pendengar.

### 4.1 Temuan correctness: MKL int8 saturasi pada branch non-VNNI

`cblas_gemm_s8u8s32` **tidak eksak** pada `MKL_CBWR=AVX2`/`AVX512`
(intermediate int16 jenuh: 255×127×2 > 32767) — unit test bit-exact FAIL pada
kedua branch, PASS pada `AVX512_E1`/`AUTO` (VNNI). Semua korpus int8 awal sesi
ini berjalan di AVX2 dan tercemar; hasilnya disimpan di `old-avx2-int8/`
sebagai catatan, dan seluruh angka di bawah diulang pada `AVX512_E1`. Engine
kini menjalankan `iro_int8_selfcheck()` saat init dan **menolak** backend yang
jenuh; `iro_int8_prepare_backend()` memilih `AVX512_E1` bila `MKL_CBWR` kosong.

### 4.2 Hasil korpus (E1, 8-step, 6 teks per mode)

| mode | config | LSD mean/max | STOI mean/min | Euler median s |
|---|---|---:|---:|---:|
| text-only | rounding (fp32-avx512) | 0,12 / 0,20 | 1,000 | 6,8 |
| text-only | int8 plain | 1,91 / 3,05 | 0,969 / 0,935 | 2,27 |
| text-only | **int8 default (residual W2)** | **1,45 / 2,38** | **0,983 / 0,954** | 2,65 |
| caption-only | int8 plain | 2,06 / 3,02 | 0,970 / 0,950 | 3,13 |
| caption-only | int8 default | 1,58 / 2,32 | 0,981 / 0,961 | 3,73 |
| clone | int8 plain | 1,94 / 2,59 | 0,952 / 0,933 | 3,33 |
| clone | int8 default | 1,56 / 2,11 | 0,965 / 0,945 | 3,95 |
| clone+caption | int8 plain | 1,90 / 2,47 | 0,961 / 0,945 | 4,12 |
| clone+caption | int8 default | 1,70 / 2,10 | 0,965 / 0,954 | 5,13 |

Varian residual text-only (`quality-text8-residual-e1/`): W2 1,45/0,983;
W2+W13 1,36/0,981 (Euler 3,38 s); semua 1,43/0,978 (4,17 s). Residual W2
dipilih sebagai default: hampir seluruh gain dengan ~+20% waktu sampling.

**ASR CER** (kotoba-whisper-v2.0, `asr-fp32.json`, `asr-int8.json`, 48 klip):
fp32 dan int8 default **0,032 rata-rata di keempat mode, median 0, tidak ada
klip >0,2**; setiap "error" tersisa adalah ortografi ASR yang sama di kedua
sisi (三→3, さまざま→様々, 第二→第2).

## 5. Keputusan desain yang diuji dan ditolak

| Kandidat | Hasil | Keputusan |
|---|---|---|
| Residual 2-term pada W1/W3 + W2 atau semua grup | +0,1 LSD lebih baik dari W2 saja, Euler +50–85% | tersedia via `IRO_INT8_RESIDUAL`, bukan default |
| Packed-B MKL int8 | +10–15% GEMM, ~1,4 GB payload | tidak dipakai |
| K-group / SmoothQuant | butuh K/G GEMM + traffic int32, atau kalibrasi statis; simulasi tidak lebih baik dari residual | tidak diimplementasikan |
| Branch MKL AVX2/AVX512 untuk int8 | tidak eksak (saturasi) dan 2× lebih lambat | ditolak oleh self-check |

## 6. Kecepatan formal (cap 2,0 GHz, profil Balanced) — `speed-int8-4modes/`

Protokol: `tools/bench_speed_tradeoff.py --experiment int8`, worker
`irodori-bench-worker-onemkl` (oneMKL 2026.1), `MKL_CBWR=AVX512_E1` untuk
**kedua** varian (kecepatan FP32 tidak bergantung branch di host ini; lihat
§1), CPU 0,1, 2 thread, proses fresh + warm-up sebelum setiap sampel,
urutan varian bergantian, 5 pair per sel, idle host ≥90% sebelum setiap
sampel (min 90–93%). Kandidat lebih cepat pada **40/40 pair**.

| mode | steps | fp32 p50 s | int8 p50 s | speedup E2E | sampling fp32/int8 s | decode s | peak RSS fp32/int8 MiB | RTF fp32/int8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| text-only | 8 | 13,40 | 8,25 | **1,62×** | 7,7 / 2,5 | 5,5 | 2560 / 1808 | 2,99 / 1,84 |
| text-only | 40 | 44,82 | 18,13 | **2,47×** | 39,0 / 12,5 | 5,5 | 2560 / 1809 | 10,00 / 4,05 |
| caption-only | 8 | 16,51 | 9,56 | 1,73× | 10,4 / 3,5 | 5,7 | 2627 / 1881 | 3,53 / 2,04 |
| caption-only | 40 | 59,76 | 23,43 | 2,55× | 53,5 / 17,3 | 5,9 | 2627 / 1880 | 12,77 / 5,01 |
| clone | 8 | 21,92 | 14,75 | 1,49× | 10,7 / 3,6 | 5,8 | 3012 / 2256 | 4,61 / 3,10 |
| clone | 40 | 68,19 | 31,49 | 2,17× | 56,6 / 19,6 | 6,1 | 3012 / 2256 | 14,33 / 6,62 |
| clone+caption | 8 | 24,44 | 16,14 | 1,51× | 13,4 / 4,2 | 5,6 | 3090 / 2319 | 5,27 / 3,48 |
| clone+caption | 40 | 75,67 | 31,69 | 2,39× | 64,6 / 20,7 | 5,6 | 3089 / 2319 | 16,31 / 6,83 |

- Sampling DiT 2,9–3,1× lebih cepat di semua mode; sisa waktu sampling int8
  (≈12,5 s / 40 step text-only) = GEMM int8 (~4,3 s) + AdaLN/attention/norm
  FP32 + kuantisasi/dekuantisasi.
- Peak RSS turun ~750 MiB karena halaman FP32 bobot DiT yang digantikan
  dilepas; init engine 0,39 s (kuantisasi) vs 0,015 s.
- p95 mengikuti p50 kecuali clone+caption 8-step baseline (satu sampel
  72,0 s vs p50 24,4 — outlier fp32, bukan kandidat).
- Bottleneck berikutnya: decode codec FP32 ~5,5 s (67% waktu text-only
  8-step int8) dan encode reference ~7 s pada clone.
- Gate handoff (p50 turun ≥2%, semua pair lebih cepat, p95 tidak memburuk
  >5%, mode lain tidak regresi) **terpenuhi untuk kecepatan**; P3 historis
  (rasio C/Python) tidak diukur ulang sesi ini.

## 7. Eksperimen cap frekuensi dilepas — `speed-int8-perfprofile/`

Atas izin pengguna, `system76-power profile performance` (max_perf_pct 100).
Microbench burst pendek naik: clock 2,7–3,4 GHz, SGEMM 2048³ 152 GFLOPS
(+38%), int8 554–597 GOPS. Namun pada beban berkelanjutan (satu request
8–75 s) suhu mencapai **89–91 °C** dan thermald menurunkan clock kembali ke
2,0 GHz, sehingga E2E praktis sama dengan run 2 GHz (3 pair, idle ≥90%):

| mode | steps | fp32 p50 s | int8 p50 s | speedup | (2 GHz: fp32 / int8) |
|---|---:|---:|---:|---:|---|
| text-only | 8 | 12,03 | 7,06 | 1,70× | 13,40 / 8,25 |
| text-only | 40 | 42,81 | 16,76 | 2,55× | 44,82 / 18,13 |
| clone+caption | 8 | 23,18 | 14,83 | 1,56× | 24,44 / 16,14 |
| clone+caption | 40 | 74,69 | 31,62 | 2,36× | 75,67 / 31,69 |

Kesimpulan: cap 60% yang dipasang pengguna ≈ frekuensi yang memang bisa
ditahan laptop ini secara termal; melepasnya hanya memberi 5–10% pada
request pendek dengan biaya suhu 90 °C. Profil dikembalikan ke Balanced;
`max_perf_pct` kembali ke 60 memerlukan root (dicatat ke pengguna).

## 8. Kecepatan konfigurasi default (residual W2) — `speed-int8-default-resw2/`

3 pair, cap 2,0 GHz kembali aktif secara termal (lihat §7), idle ≥91%,
semua pair kandidat lebih cepat:

| mode | steps | fp32 p50 s | int8 default p50 s | speedup | sampling fp32/int8 s | (int8 plain, §6) |
|---|---:|---:|---:|---:|---:|---|
| text-only | 8 | 13,09 | 7,86 | **1,67×** | 7,5 / 2,6 | 8,25 s |
| text-only | 40 | 44,22 | 19,57 | **2,26×** | 38,6 / 14,1 | 18,13 s |
| clone+caption | 8 | 23,37 | 15,74 | 1,48× | 12,6 / 4,9 | 16,14 s |
| clone+caption | 40 | 74,74 | 34,73 | 2,15× | 63,9 / 24,0 | 31,69 s |

Biaya residual W2 vs plain: +8% (8-step) sampai +10% (40-step) E2E, ditukar
dengan LSD −0,3…−0,5 dB dan STOI +0,01…+0,015 di semua mode (§4.2).
`IRO_INT8_RESIDUAL=0` mengembalikan profil plain.

## 9. Ringkasan trade-off untuk pengguna

| | FP32 (accepted) | int8 default (residual W2) | int8 plain |
|---|---|---|---|
| Kecepatan E2E 8-step / 40-step (text-only) | 1× | 1,67× / 2,26× | 1,62× / 2,47× |
| Sampling DiT | 1× | 2,7–2,9× | 3,0–3,1× |
| Peak RSS | 2,56 GB | 1,81 GB | 1,81 GB |
| Init engine | 0,02 s | +0,4 s kuantisasi | +0,4 s |
| Reproduksi golden FP32 bit-level | ya | tidak (by design) | tidak |
| Jarak audio ke FP32 (LSD / STOI) | 0,1 / 1,00 (rounding) | 1,45–1,70 / 0,965–0,983 | 1,9–2,1 / 0,95–0,97 |
| ASR CER vs teks target | 0,032 | 0,032 | 0,032 (branch E1) |
| Dependency | OpenBLAS atau MKL | oneMKL runtime + CPU VNNI (dicek saat init) | sama |
| Human listening | — | pending (`blind-ab/`) | — |

Yang belum: penilaian blind A/B oleh pengguna; codec masih FP32 (decode
~5,5 s kini 67% waktu text-only 8-step); kernel VNNI mandiri tanpa oneMKL;
perbandingan Python/C ulang dengan int8.

## 10. Reproduksi

```sh
cd /home/kenobu/development/irodori/irodori-c
MKL_ROOT=$(realpath ../benchmark-results/v2-speed-20260913-003700/onemkl-runtime)
MODEL=/home/kenobu/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4.1-Small/snapshots/2b28324dc263ed5e6638b3cf3dd94c82ead07b4b/model.safetensors
REF=/home/kenobu/development/kana-alya/assets/voices/kana-default.wav
make irodori-onemkl irodori-bench-worker-onemkl MKL_ROOT=$MKL_ROOT
make test-int8-ops test-int8-ops-onemkl test-int8-engine-onemkl test-int8-engine-sanitize MKL_ROOT=$MKL_ROOT MODEL=$MODEL
IRO_NUM_THREADS=2 ./irodori-onemkl --text 'こんにちは。' --model $MODEL --dit-precision int8 --out int8.wav
../Irodori-TTS/.venv/bin/python tools/bench_speed_tradeoff.py --model $MODEL --worker ./irodori-bench-worker-onemkl \
  --backend onemkl-sgemm --experiment int8 --modes text-only,caption-only,clone,clone+caption --steps 8,40 --pairs 5 --out OUT
../Irodori-TTS/.venv/bin/python tools/int8_quality_corpus.py --binary ./irodori-onemkl --model $MODEL --steps 8 \
  --configs fp32,fp32-avx512,int8,int8-plain --work-dir OUT/quality-text8      # tambah --caption / --ref $REF untuk mode lain
../Irodori-TTS/.venv/bin/python tools/asr_cer.py manifest.json --json out.json  # manifest: [{wav,text,label,group}]
../Irodori-TTS/.venv/bin/python tools/int8_report.py artifacts/int8-dit-20260913 > tables.md
```
