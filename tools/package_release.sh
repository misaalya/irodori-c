#!/usr/bin/env bash
# Build a self-contained Linux x86-64 release of the Irodori C engine.
#
#   tools/package_release.sh VERSION MKL_ROOT [SCIPY_OPENBLAS_SO]
#
# Produces dist/irodori-c-VERSION-linux-x86_64.tar.gz (binaries + runtime
# libraries + demo web UI + helper scripts) and, when weights/ holds the
# tokenizer and codec exports, dist/irodori-c-assets-VERSION.tar.gz.
# Binaries are compiled for the x86-64-v3 baseline (AVX2/FMA); AVX-512 and
# VNNI kernels are dispatched at runtime by oneMKL, so one build serves both.
set -euo pipefail
VERSION=${1:?version, e.g. v0.2.0}
MKL_ROOT=${2:?path to oneMKL runtime (include/mkl.h, lib/libmkl_rt.so.3)}
OPENBLAS_SO=${3:-$(ls ../Irodori-TTS/.venv/lib/python*/site-packages/scipy.libs/libscipy_openblas*.so 2>/dev/null | head -1)}
cd "$(dirname "$0")/.."
NAME=irodori-c-$VERSION-linux-x86_64
STAGE=dist/$NAME
rm -rf "$STAGE" && mkdir -p "$STAGE/bin" "$STAGE/lib" "$STAGE/licenses" dist

RELEASE_CFLAGS='-O3 -march=x86-64-v3 -mtune=generic -ffast-math -Wall -Wextra -std=c11'
RPATH='-Wl,-rpath,\$$ORIGIN/../lib'

# Objects are rebuilt with portable flags into the source tree; rebuild
# normally afterwards if you keep a -march=native development build.
make -s clean >/dev/null 2>&1 || rm -f *.o vendor/utf8proc/*.o
# DT_NEEDED records the soname only, so linking against the absolute library
# path with an $ORIGIN rpath yields a relocatable binary.
make -s irodori-onemkl FAST_CFLAGS="$RELEASE_CFLAGS" MKL_ROOT="$MKL_ROOT" \
  ONEMKL_LIBS="$MKL_ROOT/lib/libmkl_rt.so.3 $RPATH -lpthread -ldl" 2>&1 | grep -v '^cc' || true
[ -x irodori-onemkl ] || { echo "irodori-onemkl build failed"; exit 1; }
if [ -n "$OPENBLAS_SO" ] && [ -f "$OPENBLAS_SO" ]; then
  make -s irodori-blas FAST_CFLAGS="$RELEASE_CFLAGS" \
    BLAS_LIBS="$OPENBLAS_SO $RPATH" BLAS_CFLAGS="-DIRO_USE_SCIPY_OPENBLAS" 2>&1 | grep -v '^cc' || true
  [ -x irodori-blas ] || { echo "irodori-blas build failed"; exit 1; }
fi
make -s irodori CFLAGS="-O2 -march=x86-64-v3 -Wall -Wextra -std=c11" >/dev/null

cp irodori-onemkl "$STAGE/bin/irodori-onemkl"
[ -x irodori-blas ] && cp irodori-blas "$STAGE/bin/irodori-blas"
cp irodori "$STAGE/bin/irodori"
# oneMKL runtime subset actually loaded through libmkl_rt (verified with
# LD_DEBUG=libs): dispatcher, core, LP64 interface, Intel OpenMP threading,
# AVX-512 and AVX2 kernels (+ VML) and the generic fallback.
for f in libmkl_rt.so.3 libmkl_core.so.3 libmkl_intel_lp64.so.3 libmkl_intel_thread.so.3 \
         libmkl_avx512.so.3 libmkl_vml_avx512.so.3 libmkl_avx2.so.3 libmkl_vml_avx2.so.3 \
         libmkl_def.so.3 libmkl_vml_def.so.3 libiomp5.so libtbbmalloc.so.2; do
  cp "$MKL_ROOT/lib/$f" "$STAGE/lib/"
done
if [ -n "$OPENBLAS_SO" ] && [ -f "$OPENBLAS_SO" ]; then
  cp "$OPENBLAS_SO" "$STAGE/lib/"
  for dep in libgfortran libquadmath; do cp "$(dirname "$OPENBLAS_SO")"/$dep*.so* "$STAGE/lib/" 2>/dev/null || true; done
fi
strip --strip-unneeded "$STAGE"/bin/* 2>/dev/null || true
cp "$MKL_ROOT"/mkl-*.dist-info/LICENSE.txt "$STAGE/licenses/oneMKL-LICENSE.txt"
cp "$MKL_ROOT"/intel_openmp-*.dist-info/LICENSE.txt "$STAGE/licenses/intel-openmp-LICENSE.txt"
cp "$MKL_ROOT"/tbb-*.dist-info/LICENSE.txt "$STAGE/licenses/tbb-LICENSE.txt" 2>/dev/null || true
cp README.md "$STAGE/README-engine.md"
[ -d demo ] && rsync -a --exclude .git --exclude outputs --exclude __pycache__ demo/ "$STAGE/demo/"
cp tools/release/download-model.sh tools/release/run-demo.sh tools/release/README.md "$STAGE/"
chmod +x "$STAGE"/*.sh
tar -C dist -czf "dist/$NAME.tar.gz" "$NAME"

if [ -f weights/tokenizer.bin ] && [ -f weights/dacvae_decoder.safetensors ]; then
  A=irodori-c-assets-$VERSION
  rm -rf "dist/$A" && mkdir -p "dist/$A/weights"
  cp weights/tokenizer.bin weights/dacvae_decoder.safetensors "dist/$A/weights/"
  [ -f weights/dacvae_encoder.safetensors ] && cp weights/dacvae_encoder.safetensors "dist/$A/weights/"
  cp tools/release/ASSETS-README.md "dist/$A/README.md"
  tar -C dist -czf "dist/$A.tar.gz" "$A"
fi
(cd dist && sha256sum *.tar.gz > SHA256SUMS)
ls -la dist/*.tar.gz dist/SHA256SUMS
