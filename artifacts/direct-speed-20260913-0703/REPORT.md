# Direct-speed continuation — 2026-09-13 07:03 WIB

Status: **tidak ada kandidat baru yang dipromosikan ke default**. Jalur
standalone oneMKL 2026 ditolak untuk direct-speed umum karena quality/speed
trade-off. Kandidat outer-parallel Conv7 juga ditolak setelah integrasi opt-in
sementara: golden dan WAV exact, tetapi A/B end-to-end berbalik arah dan median
latency memburuk. Timing akhir tidak memenuhi preflight host idle >=90%, sehingga
semua angka outer-parallel di bawah berstatus diagnostik.

## Provenance

- Accepted baseline backend: `torch-mkl-sgemm`, `MKL_CBWR=AVX2`, 2 thread,
  affinity CPU `0,1` (dua core fisik berbeda).
- PyTorch CPU wheel: 2.10.0+cpu, oneMKL **2024.2** build 20240605.
- Standalone oneMKL experiment: **2026.1.0** runtime dari artefak V2.
- Baseline source/status/binary/topology/hash disimpan di `baseline/`.
- Input clone+caption memakai seed/noise existing, reference
  `kana-default.wav`, 8 Euler step, dan golden
  `golden/kana_clone_caption_seed42_steps8`.
- Eksperimen tidak mengubah tolerance golden.

## A. Diagnosis standalone oneMKL tanpa packing

Hipotesis: kegagalan packed cache mungkin hanya berasal dari packing W1/W3.
Kontrol `packed_mib=0` dibandingkan dengan backend accepted pada input identik.

Hasil:

- Front-end identik antar-backend sampai `reference_latent`, `speaker_state`,
  `caption_state`, `duration_log_frames`, dan `noise`.
- Drift mulai di DiT sampling: accepted-vs-standalone max abs sekitar
  `1.8e-6` pada velocity step 0 dan tumbuh menjadi `3.532e-4` pada step 7.
- Accepted torch-MKL tetap PASS: velocity step 7 `0.0049690604` dengan limit
  `0.005`; waveform `0.0017798543` dengan limit `0.002`.
- Standalone oneMKL AVX2 tanpa packing FAIL: velocity step 7
  `0.0053222328`; waveform masih PASS `0.0019268841`.

Hipotesis kedua: perbedaan berasal dari entry point `cblas_sgemm` vs `sgemm_`.
Standalone 2026 sementara dipaksa memakai Fortran `sgemm_` yang sama dengan
backend torch-MKL. Output tetap sama secara praktis dan step 7 masih
`0.0053222328`; patch ini kemudian direvert.

`MKL_CBWR=COMPATIBLE` pada standalone 2026 memperbaiki quality seluruh golden:
step 7 menjadi `0.0015963912`. Namun single diagnostic run clone+caption naik
dari sekitar **45.501 s** (AVX2) menjadi **153.557 s** (COMPATIBLE), sehingga
mode ini tidak layak sebagai kandidat speed.

Kesimpulan: packed cache bukan akar kegagalan quality. Perbedaan runtime MKL
2024.2 vs 2026.1 adalah perbedaan implementasi utama yang tersisa, tetapi belum
dibuktikan sebagai sebab tunggal. Standalone oneMKL 2026 tidak dipromosikan.
`tensor-localization/backend-diff.json` menyimpan per-tensor delta.

## B. Codec hotspot

Profil accepted backend (`IRO_DACVAE_OP_PROFILE=1`) pada golden 112 latent frame:

- median decoder: **10.414 s**, best **9.062 s**;
- golden PASS, worst intermediate `3.61204e-05`;
- pada representative pass, residual Conv7 linear stage2 + stage3 sekitar
  `2.980 + 2.157 = 5.137 s`, hampir separuh decoder;
- bias/scatter dan allocation/copy bukan bucket dominan; reuse arena tidak
  mempunyai Amdahl budget yang sebanding dengan Conv7 SGEMM.

Log lengkap: `codec-profile.log`.

## C. Outer-parallel Conv7

Hipotesis: blok output direct-tap 8192 row saling independen. Dua outer worker
dapat memproses blok berbeda sambil mempertahankan urutan tujuh tap per output.
Prototype hanya dibuat di artefak `bench_codec_conv7_outer.c`; jalur produk
`dacvae.c` tidak diubah.

Konfigurasi awal kandidat memakai **MKL 1-thread + outer2** agar total budget
tetap dua core. Probe layer menunjukkan hasil campuran; total sum-of-medians
enam Conv7 adalah baseline **4.072 s** vs kandidat **4.438 s** pada run awal.
Direct-only fresh-process screening berikutnya justru memberi 3/3 arah menang:

| Pair | Baseline MKL2+outer1 | Candidate MKL1+outer2 | Ratio baseline/candidate | Idle preflight |
|---|---:|---:|---:|---:|
| 1 | 4.203 s | 3.884 s | 1.082x | 39.6% / 53.6% |
| 2 | 4.585 s | 3.994 s | 1.148x | 29.4% / 30.3% |
| 3 | 5.028 s | 4.557 s | 1.103x | 47.2% / 45.8% |

Semua pair di atas invalid untuk acceptance karena host jauh di bawah 90% idle.
Grafik diagnostik ada di `codec-outer-diagnostic.png`.

Runtime torch-MKL mengekspor dua nama setter thread lokal. Simbol lowercase
`mkl_set_num_threads_local` crash bahkan dari main thread; reproduksi kecil
disimpan di `test_mkl_local_threads_lowercase.c`. Entry point publik uppercase
`MKL_Set_Num_Threads_Local` bekerja. Dengan setter itu, main/DiT tetap dua
thread sementara masing-masing outer worker memakai MKL satu thread. Smoke enam
Conv7 turun menjadi **3.266 s** dengan seluruh checksum identik.

Kandidat lalu diintegrasikan sementara sebagai env opt-in dan diuji golden.
Codec tetap PASS dengan worst `3.61204e-05`; diagnostic internal median berubah
dari **7.506 s** ke **7.073 s**. Namun A/B end-to-end text-only 8-step tiga pair
memberi speedup **0.916 / 1.062 / 0.938x**: kandidat hanya menang satu pair,
p50 memburuk **21.068 → 21.261 s**, dan median decode memburuk
**8.275 → 8.679 s**. WAV seluruh run byte-identik. Peak RSS naik sekitar
**13.3 MiB**. Host idle hanya 51–66%, jadi angka ini diagnostik, tetapi reversal
dan median regression sudah cukup untuk menghentikan kandidat sebelum formal
run. Perubahan produk dan env opt-in direvert; artefak prototype dipertahankan.

## Verification dan state akhir

- Temporary standalone `sgemm_` patch sudah direvert.
- Temporary outer-parallel integration dan local-thread ops sudah direvert.
- `irodori-bench-worker-onemkl` dibuild ulang dari source yang sudah direvert.
- Binary torch-MKL dan standalone oneMKL final byte-identik dengan snapshot
  baseline masing-masing (SHA-256 sama).
- `git diff --check`, static analyzer, 10+2+4 harness tests, dan ASan/UBSan
  model-I/O serta audio PASS. `make audit` utuh berhenti hanya karena
  LeakSanitizer tidak dapat berjalan di bawah ptrace pada environment Codex;
  sanitizer yang sama dijalankan ulang dengan `detect_leaks=0` dan PASS.
- Cleanup `Worker.close()` pada `bench_speed_tradeoff.py` diperkeras agar
  startup-error child tidak membuat `BrokenPipeError` kedua saat menutup pipe.
- Tidak ada tolerance, precision, jumlah step, atau default runtime yang diubah.
- Tidak ada corpus 162-case atau long benchmark dijalankan karena tidak ada
  kandidat yang melewati screening dan host timing sedang sibuk.

## Langkah berikutnya

Prioritas direct FP32 berikutnya harus tetap menargetkan Conv7 192/96-channel,
tetapi harus mempertahankan MKL 2-thread untuk DiT. Kandidat baru perlu berupa
backend/microkernel yang memberi >=5% pada shape Conv7 nyata dan punya Amdahl
budget >=2% E2E; jangan mengulang block-size, repack, zero-fill, Snake thread,
atau outer-parallel di atas tanpa hipotesis baru. Formal timing baru boleh
dimulai setelah preflight host >=90% idle.
