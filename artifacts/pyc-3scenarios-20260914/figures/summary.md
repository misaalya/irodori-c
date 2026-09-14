| Skenario | Arm | p50 s | p95 s | min–max s | CV % | Speedup vs Python | Latency cut | RTF | Peak RSS MiB | RAM cut | PCM16 max LSB | deterministik |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Text only | Python fp32 (run 1) | 24.696 | 25.247 | 23.40–25.28 | 2.7 | 1.00× | 0.0% | 5.51 | 5134 | 0.0% | – | True |
| Text only | C fp32 (torch-MKL AVX2) | 12.558 | 12.839 | 12.35–12.89 | 1.5 | 1.97× | 49.1% | 2.80 | 2617 | 49.0% | 21 | True |
| Caption only | Python fp32 (run 1) | 29.413 | 29.682 | 28.77–29.69 | 1.1 | 1.00× | 0.0% | 6.28 | 4923 | 0.0% | – | True |
| Caption only | C fp32 (torch-MKL AVX2) | 16.260 | 16.417 | 15.91–16.44 | 1.1 | 1.81× | 44.7% | 3.47 | 2687 | 45.4% | 6 | True |
| Clone + caption | Python fp32 (run 1) | 41.139 | 42.210 | 40.52–42.44 | 1.6 | 1.00× | 0.0% | 8.87 | 5545 | 0.0% | – | True |
| Clone + caption | C fp32 (torch-MKL AVX2) | 24.524 | 24.622 | 24.30–24.63 | 0.5 | 1.68× | 40.4% | 5.29 | 3134 | 43.5% | 58 | True |
| Text only | Python fp32 (run 2) | 25.391 | 25.643 | 24.42–25.65 | 1.8 | 1.00× | 0.0% | 5.67 | 5127 | 0.0% | – | True |
| Text only | C int8 DiT (oneMKL VNNI) | 8.224 | 8.363 | 7.90–8.39 | 2.0 | 3.09× | 67.6% | 1.84 | 1814 | 64.6% | 34908 | True |
| Caption only | Python fp32 (run 2) | 29.643 | 29.803 | 29.22–29.84 | 0.7 | 1.00× | 0.0% | 6.33 | 5061 | 0.0% | – | True |
| Caption only | C int8 DiT (oneMKL VNNI) | 9.690 | 9.833 | 9.52–9.86 | 1.3 | 3.06× | 67.3% | 2.07 | 1884 | 62.8% | 42152 | True |
| Clone + caption | Python fp32 (run 2) | 41.117 | 41.442 | 40.94–41.44 | 0.5 | 1.00× | 0.0% | 8.86 | 5552 | 0.0% | – | True |
| Clone + caption | C int8 DiT (oneMKL VNNI) | 15.947 | 16.222 | 15.54–16.28 | 1.5 | 2.58× | 61.2% | 3.44 | 2315 | 58.3% | 35801 | True |
| Text only | Python fp32 (run 3) | 25.092 | 25.412 | 24.16–25.41 | 1.8 | 1.00× | 0.0% | 5.60 | 4983 | 0.0% | – | True |
| Text only | C int8 DiT+codec | 5.510 | 5.586 | 5.39–5.60 | 1.4 | 4.55× | 78.0% | 1.23 | 1610 | 67.7% | 34847 | True |
| Caption only | Python fp32 (run 3) | 29.109 | 29.148 | 28.98–29.16 | 0.2 | 1.00× | 0.0% | 6.22 | 5059 | 0.0% | – | True |
| Caption only | C int8 DiT+codec | 7.215 | 7.260 | 7.13–7.26 | 0.7 | 4.03× | 75.2% | 1.54 | 1684 | 66.7% | 42068 | True |
| Clone + caption | Python fp32 (run 3) | 40.460 | 40.941 | 40.35–41.01 | 0.6 | 1.00× | 0.0% | 8.72 | 5505 | 0.0% | – | True |
| Clone + caption | C int8 DiT+codec | 12.637 | 12.718 | 12.53–12.74 | 0.5 | 3.20× | 68.8% | 2.72 | 2124 | 61.4% | 38709 | True |

8 Euler step, seed 42, 2 thread pada 2 core fisik, 1 warm-up + 5 repeat per backend/skenario, Python/C interleaved, noise awal identik, idle host ≥90% sebelum setiap skenario. Whisker = min–max, titik = repeat.
