| Skenario | Arm | p50 s | p95 s | min–max s | CV % | Speedup vs Python | Latency cut | RTF | Peak RSS MiB | RAM cut | PCM16 max LSB | deterministik |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Clone | Python fp32 (run 1) | 36.338 | 37.532 | 34.95–37.59 | 2.6 | 1.00× | 0.0% | 7.63 | 5552 | 0.0% | – | True |
| Clone | C fp32 (torch-MKL AVX2) | 21.055 | 21.704 | 20.67–21.82 | 1.8 | 1.73× | 42.1% | 4.42 | 3058 | 44.9% | 3 | True |
| Clone | Python fp32 (run 2) | 37.384 | 38.063 | 37.23–38.18 | 0.9 | 1.00× | 0.0% | 7.85 | 5476 | 0.0% | – | True |
| Clone | C int8 DiT (oneMKL VNNI) | 14.904 | 15.426 | 14.82–15.49 | 1.7 | 2.51× | 60.1% | 3.13 | 2248 | 59.0% | 27956 | True |
| Clone | Python fp32 (run 3) | 37.246 | 37.866 | 36.85–37.94 | 1.0 | 1.00× | 0.0% | 7.82 | 5442 | 0.0% | – | True |
| Clone | C int8 DiT+codec | 12.302 | 12.443 | 11.65–12.46 | 2.6 | 3.03× | 67.0% | 2.58 | 2048 | 62.4% | 23368 | True |

8 Euler step, seed 42, 2 thread pada 2 core fisik, 1 warm-up + 5 repeat per backend/skenario, Python/C interleaved, noise awal identik, idle host ≥90% sebelum setiap skenario. Whisker = min–max, titik = repeat.
