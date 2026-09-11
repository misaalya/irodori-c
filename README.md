# Irodori C

Implementasi inference CPU dalam C untuk
[`Aratako/Irodori-TTS-v4.1-Small`](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small).
Engine membaca bobot FP32 langsung dari safetensors dengan `mmap` dan dapat
menghasilkan audio text-to-speech, caption/style-conditioned speech, serta
voice cloning dari WAV referensi.

Runtime utama tidak memerlukan Python, PyTorch, ONNX, atau framework inference.
Python hanya dipakai sekali untuk mengunduh dan menyiapkan aset model, serta
untuk benchmark/parity pengembangan. Output berupa WAV mono PCM16 48 kHz.

> Status: proyek masih dalam pengembangan. Jalur text, caption, clone, dan
> clone+caption sudah berfungsi, tetapi inference FP32 40-step belum realtime
> pada CPU kelas laptop dua core.

## Fitur

- Text normalization dan tokenizer Unigram/byte fallback dalam C.
- ModernBERT-ja encoder, duration predictor, RF-DiT, dan Euler CFG dalam C.
- DACVAE encoder/decoder untuk voice cloning dan audio 48 kHz.
- Independent CFG untuk text, speaker reference, dan caption.
- Backend scalar portabel serta backend CBLAS berperforma tinggi.
- Engine reusable untuk beberapa request tanpa memuat ulang semua bobot.
- Pembacaan safetensors melalui `mmap` dan pelepasan halaman bobot yang sudah
  tidak dibutuhkan untuk menekan peak RSS.

## Kebutuhan sistem

Untuk build Linux yang direkomendasikan:

```sh
sudo apt update
sudo apt install build-essential libopenblas-dev python3 python3-venv git
```

Di macOS, pasang Command Line Tools. Target `make blas` otomatis memakai
Accelerate.framework.

RAM yang dibutuhkan bergantung mode. Siapkan beberapa GiB RAM kosong dan ruang
disk sekitar 2 GB untuk checkpoint utama, codec asli, dan aset hasil ekspor.
Repository ini sengaja tidak menyertakan model weights atau golden tensor.

## Quick start

### 1. Clone source

```sh
git clone https://github.com/misaalya/irodori-c.git
git clone https://github.com/Aratako/Irodori-TTS.git
cd irodori-c
```

Letakkan kedua repository sebagai sibling seperti berikut:

```text
workspace/
├── irodori-c/
└── Irodori-TTS/
```

Source upstream diperlukan oleh tool persiapan tokenizer dan benchmark. Ia
tidak diperlukan oleh executable C setelah semua aset selesai dibuat.

### 2. Buat environment untuk menyiapkan aset

Contoh berikut memakai environment di repository upstream agar target MKL
opsional juga dapat menemukannya:

```sh
python3 -m venv ../Irodori-TTS/.venv
IRO_PY=../Irodori-TTS/.venv/bin/python

"$IRO_PY" -m pip install --upgrade pip
"$IRO_PY" -m pip install huggingface-hub safetensors
"$IRO_PY" -m pip install torch \
  --index-url https://download.pytorch.org/whl/cpu
```

### 3. Unduh model dan codec

```sh
mkdir -p downloads/irodori downloads/dacvae weights

../Irodori-TTS/.venv/bin/hf download \
  Aratako/Irodori-TTS-v4.1-Small \
  model.safetensors tokenizer/tokenizer.json \
  --local-dir downloads/irodori

../Irodori-TTS/.venv/bin/hf download \
  Aratako/Semantic-DACVAE-Japanese-32dim \
  weights.pth \
  --local-dir downloads/dacvae
```

### 4. Siapkan aset runtime C

```sh
IRO_PY=../Irodori-TTS/.venv/bin/python

cp downloads/irodori/model.safetensors weights/model.safetensors

"$IRO_PY" tools/compile_tokenizer.py \
  downloads/irodori/tokenizer/tokenizer.json \
  weights/tokenizer.bin \
  weights/tokenizer_vectors.json

"$IRO_PY" tools/export_dacvae_decoder.py \
  downloads/dacvae/weights.pth \
  weights/dacvae_decoder.safetensors

"$IRO_PY" tools/export_dacvae_encoder.py \
  downloads/dacvae/weights.pth \
  weights/dacvae_encoder.safetensors
```

File runtime yang dihasilkan:

```text
weights/model.safetensors
weights/tokenizer.bin
weights/dacvae_decoder.safetensors
weights/dacvae_encoder.safetensors
```

### 5. Build

Build berperforma tinggi yang direkomendasikan:

```sh
make blas
```

Executable-nya adalah `./irodori-blas`. Build scalar dapat dibuat dengan
`make`; executable `./irodori` terutama berguna sebagai oracle portabel dan
akan jauh lebih lambat untuk inference penuh.

## Cara memakai

Contoh di bawah memakai dua thread. Ganti sesuai jumlah core fisik CPU:

```sh
export IRO_NUM_THREADS=2
export IRO_MODEL="$PWD/weights/model.safetensors"
export IRO_TOKENIZER="$PWD/weights/tokenizer.bin"
export IRO_DECODER="$PWD/weights/dacvae_decoder.safetensors"
export IRO_ENCODER="$PWD/weights/dacvae_encoder.safetensors"
```

Environment variable tersebut opsional. Semua path juga dapat diberikan
langsung melalui argumen CLI.

### Text-to-speech

```sh
./irodori-blas \
  --text 'こんにちは。今日はいい天気ですね。' \
  --steps 40 \
  --seed 42 \
  --out result.wav
```

### Caption atau style conditioning

```sh
./irodori-blas \
  --text 'こんにちは。今日はいい天気ですね。' \
  --caption '落ち着いた自然な女性の声で、やわらかく話す。' \
  --steps 40 \
  --seed 42 \
  --out styled.wav
```

Caption menjelaskan gaya, emosi, atau karakter suara. Caption-only tidak
memerlukan reference WAV.

### Voice cloning

```sh
./irodori-blas \
  --text 'この声を使って、新しい文章を読み上げます。' \
  --ref reference.wav \
  --steps 40 \
  --seed 42 \
  --out cloned.wav
```

WAV referensi dapat berupa PCM16/24/32 atau float32 dan boleh multichannel.
Engine melakukan downmix, resampling ke 48 kHz, loudness normalization, DACVAE
encoding, dan speaker conditioning. Referensi yang bersih dengan satu pembicara
memberi hasil paling stabil.

### Voice cloning dengan caption

```sh
./irodori-blas \
  --text 'この声を使って、新しい文章を読み上げます。' \
  --ref reference.wav \
  --caption '明るく元気な声で、少し速めに話す。' \
  --steps 40 \
  --seed 42 \
  --out styled-clone.wav
```

### Memberikan semua path lewat CLI

```sh
IRO_NUM_THREADS=2 ./irodori-blas \
  --text '生成する文章' \
  --model weights/model.safetensors \
  --tokenizer weights/tokenizer.bin \
  --encoder weights/dacvae_encoder.safetensors \
  --decoder weights/dacvae_decoder.safetensors \
  --ref reference.wav \
  --steps 40 \
  --out output.wav
```

Opsi utama:

| Opsi | Arti | Default |
|---|---|---|
| `--text` | Teks yang akan dibacakan | wajib |
| `--caption` | Deskripsi gaya/emosi suara | tidak aktif |
| `--ref` | WAV untuk voice cloning | tidak aktif |
| `--steps` | Jumlah Euler sampling step | `40` |
| `--seed` | Seed noise deterministik | `42` |
| `--out` | Lokasi WAV hasil | `out.wav` |
| `--model` | Checkpoint Irodori FP32 | `IRO_MODEL` atau `weights/model.safetensors` |
| `--tokenizer` | Tokenizer binary | `IRO_TOKENIZER` atau `weights/tokenizer.bin` |
| `--decoder` | DACVAE decoder | `IRO_DECODER` atau path default |
| `--encoder` | DACVAE encoder untuk clone | `IRO_ENCODER` atau path default |
| `--noise` | Noise float32 eksternal untuk regression | tidak aktif |
| `--dump-dir` | Simpan intermediate tensor untuk diagnosis | tidak aktif |

Nilai `--steps 8` berguna untuk test cepat, sedangkan 40 adalah konfigurasi
default kualitas. Seed yang sama menghasilkan noise yang sama dalam backend C;
hasil antarbackend BLAS dapat memiliki perbedaan rounding FP32 kecil.

## Backend MKL opsional

Jika environment `../Irodori-TTS/.venv` berisi PyTorch CPU wheel, build berikut
memakai oneMKL yang dibundel oleh wheel tersebut:

```sh
make mkl
IRO_NUM_THREADS=2 ./irodori-mkl --text 'こんにちは。' --out mkl.wav
```

Backend ini merupakan target pengembangan dan bergantung pada library PyTorch
CPU saat runtime. Launcher menetapkan mode kompatibilitas AVX2 karena mode MKL
otomatis belum memenuhi seluruh regression tolerance pada mesin pengembangan.
Gunakan `irodori-blas` untuk runtime CBLAS mandiri.

## Preprocess reference saja

Untuk menghasilkan mean latent DACVAE dari WAV tanpa menjalankan TTS:

```sh
./irodori-blas --encode-reference \
  weights/dacvae_encoder.safetensors \
  reference.wav \
  reference.f32
```

File `.f32` ini merupakan artefak diagnosis, bukan WAV yang bisa diputar.

## Test

Test yang tidak membutuhkan model besar:

```sh
make test-audio
make test-model-io
make audit-warnings
```

Tokenizer boundary test dapat dijalankan setelah `weights/tokenizer.bin` dibuat:

```sh
make test-tokenizer-boundaries
```

Golden tensor lengkap tidak disertakan karena ukurannya besar. Tool `dump_*.py`
di direktori `tools/` dapat membuat fixture dari runtime Python upstream untuk
pengembangan parity. Jangan memakai golden yang dibuat oleh kandidat C sebagai
referensi kualitas kandidat itu sendiri.

## Benchmark

Microbenchmark backend linear:

```sh
make bench-linear
```

Benchmark Python vs C empat mode memerlukan upstream environment lengkap,
checkpoint, codec, reference WAV, dan golden fixtures:

```sh
MODEL="$PWD/weights/model.safetensors"
REF=/path/to/reference.wav

make irodori-bench-worker
../Irodori-TTS/.venv/bin/python tools/bench_four_modes.py \
  --model "$MODEL" \
  --ref "$REF" \
  --binary ./irodori-blas \
  --c-worker ./irodori-bench-worker \
  --threads 2 \
  --steps 8,40 \
  --repeats 5 \
  --work-dir /tmp/irodori-benchmark
```

Harness menolak benchmark formal bila host kurang dari 90% idle. Gunakan p50,
p95, RTF, peak RSS, determinisme, dan parity audio dari summary; jangan memakai
best-run tunggal sebagai klaim performa.

## Struktur source

```text
main.c                 CLI dan test dispatcher
generate.c             orchestration text/reference → WAV
backbone.c              ModernBERT-ja encoder
duration.c              duration predictor
dit.c / sampler.c       RF-DiT dan Euler CFG
dacvae.c                codec encoder/decoder
speaker.c               speaker encoder
audio.c                 WAV, resampler, loudness, PCM output
safetensors.c           mmap safetensors loader
ops.c                   scalar/CBLAS kernels
tools/                  persiapan aset, golden, dan benchmark
tests/                  unit, boundary, dan kernel benchmark
vendor/utf8proc/        Unicode normalization dependency
```

## Catatan keamanan input

Engine memvalidasi header dan ukuran safetensors/WAV, tetapi model serta aset
sebaiknya tetap diunduh dari sumber tepercaya. Gunakan checkpoint FP32 yang
sesuai; checkpoint dengan layout atau precision berbeda akan ditolak.
