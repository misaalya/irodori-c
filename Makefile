# Irodori C Engine — scalar oracle + optional high-performance CBLAS build
CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11
LDFLAGS ?= -lm
BLAS_CFLAGS ?=
FAST_CFLAGS ?= -O3 -march=native -ffast-math -Wall -Wextra -std=c11
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
BLAS_LIBS ?= -framework Accelerate
else
SYSTEM_OPENBLAS := $(shell $(CC) -print-file-name=libopenblas.so)
SCIPY_OPENBLAS ?= $(firstword $(wildcard ../Irodori-TTS/.venv/lib/python*/site-packages/scipy.libs/libscipy_openblas*.so))
ifneq ($(SYSTEM_OPENBLAS),libopenblas.so)
BLAS_LIBS ?= -lopenblas
BLAS_CFLAGS += -DIRO_USE_OPENBLAS
else ifneq ($(SCIPY_OPENBLAS),)
# Development fallback: SciPy wheels ship the same OpenBLAS kernels but prefix
# exported CBLAS/thread-control symbols with "scipy_".  This keeps local
# performance builds usable when the distro runtime has BLAS but not the
# libopenblas-dev linker symlink.  Production packages should install/use a
# normal system OpenBLAS.
BLAS_LIBS ?= $(SCIPY_OPENBLAS) -Wl,-rpath,$(abspath $(dir $(SCIPY_OPENBLAS)))
BLAS_CFLAGS += -DIRO_USE_SCIPY_OPENBLAS
else
$(warning OpenBLAS not found; install libopenblas-dev for the high-performance backend)
BLAS_LIBS ?= -lblas
endif
endif

SRC_COMMON := safetensors.c tokenizer.c normalize.c backbone.c condition.c duration.c dit.c sampler.c dacvae.c speaker.c audio.c generate.c vendor/utf8proc/utf8proc.c
OBJ_COMMON := main.o $(SRC_COMMON:.c=.o)
OBJ_BLAS_COMMON := main_blas.o $(SRC_COMMON:.c=_blas.o)
OBJ := $(OBJ_COMMON) ops.o
HDR := irodori.h tokenizer.h normalize.h backbone.h condition.h duration.h dit.h sampler.h dacvae.h speaker.h audio.h generate.h ops.h

CFLAGS_EXTRA := -Ivendor
TOKENIZER_BIN ?= weights/tokenizer.bin
TOKENIZER_GOLDEN ?= golden/tokenizer_vectors.tsv
BACKBONE_GOLDEN ?= golden/seed42_steps8
DURATION_GOLDEN ?= golden/duration_vectors.bin
CODEC_WEIGHTS ?= weights/dacvae_decoder.safetensors
ENCODER_WEIGHTS ?= weights/dacvae_encoder.safetensors
ENCODER_GOLDEN ?= golden/encoder_test
CLONE_GOLDEN ?= golden/kana_clone_seed42_steps8
CLONE_CAPTION_GOLDEN ?= golden/kana_clone_caption_seed42_steps8
CAPTION_GOLDEN ?= golden/caption_seed42_steps8
PYTHON ?= ../Irodori-TTS/.venv/bin/python
REF ?=

# Optional development backend: PyTorch CPU wheels bundle oneMKL inside
# libtorch_cpu.so.  Keeping it as a separate target avoids making the normal
# OpenBLAS build depend on PyTorch while still allowing the faster FP32 SGEMM
# kernels on machines where that wheel is already installed.
TORCH_LIBDIR ?= $(firstword $(wildcard ../Irodori-TTS/.venv/lib/python*/site-packages/torch/lib))
TORCH_CPU_LIB ?= $(TORCH_LIBDIR)/libtorch_cpu.so
TORCH_GOMP_LIB ?= $(TORCH_LIBDIR)/libgomp.so.1
TORCH_MKL_LIBS = $(TORCH_CPU_LIB) $(TORCH_GOMP_LIB) -Wl,-rpath,$(abspath $(TORCH_LIBDIR))

all: irodori

blas: irodori-blas

mkl: irodori-mkl

irodori: $(OBJ) $(HDR)
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA) -o $@ $(OBJ) $(LDFLAGS)

irodori-blas: $(OBJ_BLAS_COMMON) ops_blas.o $(HDR)
	$(CC) $(FAST_CFLAGS) $(CFLAGS_EXTRA) $(BLAS_CFLAGS) -o $@ $(OBJ_BLAS_COMMON) ops_blas.o $(LDFLAGS) $(BLAS_LIBS)

irodori-mkl: irodori-mkl.bin Makefile
	@printf '%s\n' \
		'#!/bin/sh' \
		'threads=$${IRO_NUM_THREADS:-2}' \
		'export MKL_NUM_THREADS="$$threads"' \
		'export OMP_NUM_THREADS="$$threads"' \
		'export MKL_CBWR="$${MKL_CBWR:-AVX2}"' \
		'exec "$$(dirname "$$0")/irodori-mkl.bin" "$$@"' > $@
	@chmod +x $@

irodori-mkl.bin: $(OBJ_BLAS_COMMON) ops_mkl.o $(HDR) Makefile
	@test -f "$(TORCH_CPU_LIB)" -a -f "$(TORCH_GOMP_LIB)" || \
		(echo "PyTorch CPU wheel dengan bundled MKL tidak ditemukan"; exit 2)
	$(CC) $(FAST_CFLAGS) $(CFLAGS_EXTRA) -o $@ $(OBJ_BLAS_COMMON) ops_mkl.o \
		$(LDFLAGS) $(TORCH_MKL_LIBS)

ops_blas.o: ops.c $(HDR)
	$(CC) $(FAST_CFLAGS) $(CFLAGS_EXTRA) $(BLAS_CFLAGS) -DIRO_USE_CBLAS -c -o $@ $<

ops_mkl.o: ops.c $(HDR)
	@test -f "$(TORCH_CPU_LIB)" -a -f "$(TORCH_GOMP_LIB)" || \
		(echo "PyTorch CPU wheel dengan bundled MKL tidak ditemukan"; exit 2)
	$(CC) $(FAST_CFLAGS) $(CFLAGS_EXTRA) -DIRO_USE_CBLAS -DIRO_USE_TORCH_MKL -c -o $@ $<

%_blas.o: %.c $(HDR)
	$(CC) $(FAST_CFLAGS) $(CFLAGS_EXTRA) $(BLAS_CFLAGS) -c -o $@ $<

%.o: %.c $(HDR)
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA) -c -o $@ $<

test: test-tokenizer test-tokenizer-boundaries test-audio test-model-io

test-tokenizer: irodori
	./irodori --test-tokenizer $(TOKENIZER_BIN) $(TOKENIZER_GOLDEN)

test-tokenizer-boundaries: /tmp/irodori-test-tokenizer-boundaries
	/tmp/irodori-test-tokenizer-boundaries $(TOKENIZER_BIN)

/tmp/irodori-test-tokenizer-boundaries: tests/test_tokenizer_boundaries.c tokenizer.c tokenizer.h
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA) -I. -o $@ tests/test_tokenizer_boundaries.c tokenizer.c $(LDFLAGS)

test-audio: /tmp/irodori-test-audio
	/tmp/irodori-test-audio

/tmp/irodori-test-audio: tests/test_audio.c audio.c audio.h
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA) -I. -o $@ tests/test_audio.c audio.c $(LDFLAGS)

test-model-io: /tmp/irodori-test-model-io
	/tmp/irodori-test-model-io

/tmp/irodori-test-model-io: tests/test_model_io.c safetensors.c tokenizer.c irodori.h tokenizer.h
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA) -I. -o $@ tests/test_model_io.c safetensors.c tokenizer.c $(LDFLAGS)

audit-warnings:
	$(CC) -O2 -Wall -Wextra -Werror -std=c11 $(CFLAGS_EXTRA) -I. -fsyntax-only \
		main.c $(SRC_COMMON) ops.c

audit-sanitize: tests/test_model_io.c tests/test_audio.c
	$(CC) -O1 -g -Wall -Wextra -Werror -std=c11 -fsanitize=address,undefined \
		-fno-omit-frame-pointer $(CFLAGS_EXTRA) -I. -o /tmp/irodori-test-model-io-san \
		tests/test_model_io.c safetensors.c tokenizer.c $(LDFLAGS)
	ASAN_OPTIONS=detect_leaks=1 /tmp/irodori-test-model-io-san
	$(CC) -O1 -g -Wall -Wextra -Werror -std=c11 -fsanitize=address,undefined \
		-fno-omit-frame-pointer $(CFLAGS_EXTRA) -I. -o /tmp/irodori-test-audio-san \
		tests/test_audio.c audio.c $(LDFLAGS)
	ASAN_OPTIONS=detect_leaks=1 /tmp/irodori-test-audio-san

audit-static:
	$(CC) -O0 -Wall -Wextra -std=c11 -fanalyzer $(CFLAGS_EXTRA) -I. \
		-c tokenizer.c -o /tmp/irodori-tokenizer-analyzer.o
	$(CC) -O0 -Wall -Wextra -std=c11 -fanalyzer $(CFLAGS_EXTRA) -I. \
		-c safetensors.c -o /tmp/irodori-safetensors-analyzer.o
	$(CC) -O0 -Wall -Wextra -std=c11 -fanalyzer $(CFLAGS_EXTRA) -I. \
		-c audio.c -o /tmp/irodori-audio-analyzer.o

audit: audit-warnings test-model-io test-tokenizer-boundaries test-audio audit-sanitize audit-static

test-backbone: irodori
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	./irodori --test-backbone $(MODEL) $(BACKBONE_GOLDEN)

test-projector: irodori
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	./irodori --test-projector $(MODEL) $(BACKBONE_GOLDEN)

test-duration: irodori
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	./irodori --test-duration-vectors $(MODEL) $(DURATION_GOLDEN)

test-dit-prepare: irodori
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	./irodori --test-dit-prepare $(MODEL) $(BACKBONE_GOLDEN)

test-dit-adaln: irodori
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	./irodori --test-dit-adaln $(MODEL) $(BACKBONE_GOLDEN)

test-dit-attention: irodori
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	./irodori --test-dit-attention $(MODEL) $(BACKBONE_GOLDEN)

test-dit-mlp: irodori
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	./irodori --test-dit-mlp $(MODEL) $(BACKBONE_GOLDEN)

test-dit-attention-blas: irodori-blas
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	./irodori-blas --test-dit-attention $(MODEL) $(BACKBONE_GOLDEN)

test-dit-mlp-blas: irodori-blas
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	./irodori-blas --test-dit-mlp $(MODEL) $(BACKBONE_GOLDEN)

test-dit-forward-blas: irodori-blas
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	./irodori-blas --test-dit-forward $(MODEL) $(BACKBONE_GOLDEN)

test-euler-blas: irodori-blas
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	./irodori-blas --test-euler $(MODEL) $(BACKBONE_GOLDEN)

test-euler-speaker-blas: irodori-blas
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	./irodori-blas --test-euler-speaker $(MODEL) golden/speaker_test

test-codec-blas: irodori-blas
	./irodori-blas --test-codec $(CODEC_WEIGHTS) $(BACKBONE_GOLDEN)

test-encoder-blas: irodori-blas
	./irodori-blas --test-encoder $(ENCODER_WEIGHTS) $(ENCODER_GOLDEN)

test-speaker-blas: irodori-blas
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	./irodori-blas --test-speaker $(MODEL) golden/speaker_test

test-wav: irodori
	./irodori --test-wav $(BACKBONE_GOLDEN) /tmp/irodori-golden.wav

test-pipeline-blas: irodori-blas
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	./irodori-blas --test-pipeline $(MODEL) $(CODEC_WEIGHTS) $(BACKBONE_GOLDEN) /tmp/irodori-pipeline.wav

test-generate-blas: irodori-blas
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	./irodori-blas --text 'こんにちは、色とりどりの世界へようこそ。' \
		--model $(MODEL) --tokenizer $(TOKENIZER_BIN) --decoder $(CODEC_WEIGHTS) \
		--seed 42 --steps 8 --noise $(BACKBONE_GOLDEN)/x_t_step000.f32 \
		--out /tmp/irodori-text-e2e.wav

test-engine-reuse-blas: irodori-blas
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	$(CC) $(FAST_CFLAGS) $(CFLAGS_EXTRA) $(BLAS_CFLAGS) -I. \
		-o /tmp/irodori-test-engine-reuse tests/test_engine_reuse.c \
		$(SRC_COMMON) ops.c $(LDFLAGS) $(BLAS_LIBS) -DIRO_USE_CBLAS
	IRO_NUM_THREADS=2 /tmp/irodori-test-engine-reuse "$(MODEL)" \
		"$(TOKENIZER_BIN)" "$(CODEC_WEIGHTS)" "$(BACKBONE_GOLDEN)/x_t_step000.f32"

bench-linear: irodori irodori-blas
	./irodori --bench-linear
	./irodori-blas --bench-linear

bench-generate-python:
	$(PYTHON) tools/bench_generate.py --threads 2 --steps 8 --repeats 3

bench-clone-python:
	@test -n "$(REF)" || (echo "REF=/path/to/reference.wav wajib diisi"; exit 2)
	$(PYTHON) tools/bench_generate.py --threads 2 --steps 8 --repeats 3 --ref "$(REF)"

golden-caption:
	$(PYTHON) tools/dump_clone_golden.py --out-dir "$(CAPTION_GOLDEN)" --steps 8 \
		--caption '落ち着いた自然な女性の声で、やわらかく話す。'

test-caption-parity-blas: irodori-blas
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	@test -f "$(CAPTION_GOLDEN)/noise.f32" || \
		(echo "caption golden belum ada; jalankan make golden-caption"; exit 2)
	rm -rf /tmp/irodori-caption-parity
	mkdir -p /tmp/irodori-caption-parity
	IRO_NUM_THREADS=2 ./irodori-blas --text 'こんにちは、色とりどりの世界へようこそ。' \
		--caption '落ち着いた自然な女性の声で、やわらかく話す。' \
		--model "$(MODEL)" --tokenizer "$(TOKENIZER_BIN)" \
		--decoder "$(CODEC_WEIGHTS)" --seed 42 --steps 8 \
		--noise "$(CAPTION_GOLDEN)/noise.f32" \
		--dump-dir /tmp/irodori-caption-parity --out /tmp/irodori-caption-parity.wav
	$(PYTHON) tools/compare_clone_golden.py "$(CAPTION_GOLDEN)" /tmp/irodori-caption-parity

bench-four-modes: irodori-blas
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	@test -n "$(REF)" || (echo "REF=/path/to/reference.wav wajib diisi"; exit 2)
	$(MAKE) irodori-bench-worker
	$(PYTHON) tools/bench_four_modes.py --model "$(MODEL)" --ref "$(REF)" \
		--threads 2 --steps 8,40 --repeats 5

bench-four-modes-mkl: irodori-mkl irodori-bench-worker-mkl
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	@test -n "$(REF)" || (echo "REF=/path/to/reference.wav wajib diisi"; exit 2)
	MKL_CBWR=AVX2 $(PYTHON) tools/bench_four_modes.py --model "$(MODEL)" --ref "$(REF)" \
		--binary ./irodori-mkl --c-worker ./irodori-bench-worker-mkl \
		--threads 2 --steps 8,40 --repeats 5

irodori-bench-worker: tools/bench_c_worker.c $(SRC_COMMON:.c=_blas.o) ops_blas.o $(HDR)
	$(CC) $(FAST_CFLAGS) $(CFLAGS_EXTRA) $(BLAS_CFLAGS) -I. -o $@ \
		tools/bench_c_worker.c $(SRC_COMMON:.c=_blas.o) ops_blas.o $(LDFLAGS) $(BLAS_LIBS)

irodori-bench-worker-mkl: irodori-bench-worker-mkl.bin Makefile
	@printf '%s\n' \
		'#!/bin/sh' \
		'threads=$${IRO_NUM_THREADS:-2}' \
		'export MKL_NUM_THREADS="$$threads"' \
		'export OMP_NUM_THREADS="$$threads"' \
		'export MKL_CBWR="$${MKL_CBWR:-AVX2}"' \
		'exec "$$(dirname "$$0")/irodori-bench-worker-mkl.bin" "$$@"' > $@
	@chmod +x $@

irodori-bench-worker-mkl.bin: tools/bench_c_worker.c $(SRC_COMMON:.c=_blas.o) ops_mkl.o $(HDR) Makefile
	@test -f "$(TORCH_CPU_LIB)" -a -f "$(TORCH_GOMP_LIB)" || \
		(echo "PyTorch CPU wheel dengan bundled MKL tidak ditemukan"; exit 2)
	$(CC) $(FAST_CFLAGS) $(CFLAGS_EXTRA) -I. -o $@ \
		tools/bench_c_worker.c $(SRC_COMMON:.c=_blas.o) ops_mkl.o \
		$(LDFLAGS) $(TORCH_MKL_LIBS)

test-clone-parity-blas: irodori-blas
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	@test -n "$(REF)" || (echo "REF=/path/to/reference.wav wajib diisi"; exit 2)
	mkdir -p /tmp/irodori-clone-parity
	IRO_NUM_THREADS=2 ./irodori-blas --text 'こんにちは、色とりどりの世界へようこそ。' \
		--model "$(MODEL)" --tokenizer "$(TOKENIZER_BIN)" \
		--encoder "$(ENCODER_WEIGHTS)" --decoder "$(CODEC_WEIGHTS)" \
		--ref "$(REF)" --seed 42 --steps 8 \
		--noise "$(CLONE_GOLDEN)/noise.f32" \
		--dump-dir /tmp/irodori-clone-parity --out /tmp/irodori-clone-parity.wav
	$(PYTHON) tools/compare_clone_golden.py "$(CLONE_GOLDEN)" /tmp/irodori-clone-parity

test-clone-caption-parity-blas: irodori-blas
	@test -n "$(MODEL)" || (echo "MODEL=/path/to/model.safetensors wajib diisi"; exit 2)
	@test -n "$(REF)" || (echo "REF=/path/to/reference.wav wajib diisi"; exit 2)
	mkdir -p /tmp/irodori-clone-caption-parity
	IRO_NUM_THREADS=2 ./irodori-blas --text 'こんにちは、色とりどりの世界へようこそ。' \
		--caption '落ち着いた自然な女性の声で、やわらかく話す。' \
		--model "$(MODEL)" --tokenizer "$(TOKENIZER_BIN)" \
		--encoder "$(ENCODER_WEIGHTS)" --decoder "$(CODEC_WEIGHTS)" \
		--ref "$(REF)" --seed 42 --steps 8 \
		--noise "$(CLONE_CAPTION_GOLDEN)/noise.f32" \
		--dump-dir /tmp/irodori-clone-caption-parity \
		--out /tmp/irodori-clone-caption-parity.wav
	$(PYTHON) tools/compare_clone_golden.py "$(CLONE_CAPTION_GOLDEN)" \
		/tmp/irodori-clone-caption-parity

bench-resample: /tmp/irodori-bench-resample
	/tmp/irodori-bench-resample

/tmp/irodori-bench-resample: tests/bench_resample.c audio.c audio.h
	$(CC) $(FAST_CFLAGS) -I. -o $@ tests/bench_resample.c audio.c $(LDFLAGS)

clean:
	rm -f $(OBJ_COMMON) $(OBJ_BLAS_COMMON) ops.o ops_blas.o ops_mkl.o \
		irodori irodori-blas irodori-mkl irodori-mkl.bin \
		irodori-bench-worker irodori-bench-worker-mkl \
		irodori-bench-worker-mkl.bin

.PHONY: all blas mkl clean test test-tokenizer test-tokenizer-boundaries test-audio test-model-io audit audit-warnings audit-sanitize audit-static test-backbone test-projector test-duration test-dit-prepare test-dit-adaln test-dit-attention test-dit-mlp test-dit-attention-blas test-dit-mlp-blas test-dit-forward-blas test-euler-blas test-euler-speaker-blas test-codec-blas test-encoder-blas test-speaker-blas test-wav test-pipeline-blas test-generate-blas test-engine-reuse-blas test-caption-parity-blas test-clone-parity-blas test-clone-caption-parity-blas golden-caption bench-linear bench-generate-python bench-clone-python bench-four-modes bench-four-modes-mkl bench-resample
