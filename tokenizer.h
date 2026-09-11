/* tokenizer.h — Unigram tokenizer (metaspace + Viterbi + byte fallback) */
#ifndef IRO_TOKENIZER_H
#define IRO_TOKENIZER_H

#include <stdint.h>

#define IRO_TOK_MAX_IDS 1024

typedef struct {
    int      fd;
    void    *map;
    uint64_t map_size;

    int32_t  unk_id;
    uint8_t  byte_fallback;
    uint8_t  prepend_bos;      /* reserved (BOS ditangani caller) */
    char     replacement[8];   /* Metaspace replacement (mis. "▁") */
    uint16_t rep_len;
    uint8_t  prepend_scheme;   /* 0 = never */
    uint8_t  split;

    uint32_t vocab_count;
    /* per entry: pointer ke byte pertama piece (di dalam mmap) */
    uint32_t *offsets;        /* [vocab_count] offset dari pieces_base */
    const char *pieces_base;   /* awal blob piece */
    const uint16_t *lens;      /* [vocab_count] — pointer ke u16 len tiap entry */
    float   *scores;           /* [vocab_count] salinan score (aligned aman) */

    int32_t *htab;             /* hash piece->id, -1 kosong */
    uint32_t htab_cap;
    uint32_t max_piece_len;
    float    min_score;
} IroTokenizer;

int  iro_tok_load(const char *path, IroTokenizer *t);
/* encode teks (sudah dinormalisasi) -> ids; add_bos menambah BOS di depan.
   return jumlah id, -1 jika error. */
int  iro_tok_encode(IroTokenizer *t, const char *utf8, int add_bos,
                    int32_t *out_ids, int max_ids);
const char *iro_tok_piece(IroTokenizer *t, int32_t id);
void iro_tok_free(IroTokenizer *t);

#endif

