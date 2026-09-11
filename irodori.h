/* irodori.h — Irodori-TTS pure C inference engine
 *
 * Phase 0: safetensors mmap loader (pattern: qwen-asr / qwen3-tts).
 * No Python, no PyTorch. Weights are mmapped read-only from the original
 * safetensors files published on Hugging Face.
 */
#ifndef IRODORI_H
#define IRODORI_H

#include <stdint.h>
#include <stddef.h>

#define IRO_MAX_DIMS 8

typedef enum {
    IRO_ST_F32 = 0,
    IRO_ST_F16,
    IRO_ST_BF16,
    IRO_ST_I64,
    IRO_ST_I32,
    IRO_ST_I16,
    IRO_ST_I8,
    IRO_ST_U8,
    IRO_ST_BOOL,
    IRO_ST_UNKNOWN
} IroDType;

typedef struct {
    char      *name;
    IroDType   dtype;
    int        ndims;
    uint64_t   shape[IRO_MAX_DIMS];
    uint64_t   data_begin;  /* byte offset dalam file */
    uint64_t   nbytes;
    void      *data;         /* pointer ke mmap (siap dipakai) */
} IroTensor;

typedef struct {
    int        fd;
    uint64_t   map_size;
    void      *map;
    char      *header;        /* salinan header JSON (NUL-terminated) */
    IroTensor *tensors;
    int        n_tensors;
    /* metadata __metadata__ (mis. config_json checkpoint) */
    char     **meta_keys;
    char     **meta_values;
    int        n_meta;
} IroSafetensors;

/* Load safetensors read-only via mmap. Return 0 on success, -1 on error. */
int  iro_st_load(const char *path, IroSafetensors *st);

/* Cari tensor by name. NULL jika tidak ada. */
const IroTensor *iro_st_get(const IroSafetensors *st, const char *name);

/* Ambil metadata value by key. NULL jika tidak ada. */
const char *iro_st_meta(const IroSafetensors *st, const char *key);

/* Evict clean mmap pages belonging to tensors with this name prefix. Pointers
   remain valid and the kernel transparently faults pages back if reused. */
uint64_t iro_st_drop_prefix(const IroSafetensors *st, const char *prefix);

void iro_st_free(IroSafetensors *st);

/* Utilitas */
const char *iro_dtype_name(IroDType dt);
int64_t     iro_tensor_numel(const IroTensor *t);
/* Return 1 when every F32 element is finite, 0 when NaN/Inf is present,
   and -1 for an invalid/non-F32 tensor. Intended for offline validation. */
int         iro_tensor_f32_all_finite(const IroTensor *t);

#endif /* IRODORI_H */
