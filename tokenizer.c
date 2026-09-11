/* tokenizer.c — Unigram Viterbi tokenizer (port dari HF tokenizers unigram.rs).

Pipeline yang direplikasi (tokenizer.json modernbert-ja-310m):
  1. Metaspace pre-tokenizer: ' ' -> "▁" (prepend_scheme never, split false)
  2. Viterbi Unigram: edge per piece yang cocok; unk per-char
     (hanya jika tidak ada piece 1-char di posisi itu), score unk = min_score - 10
  3. byte_fallback: unk span -> piece <0xXX> per byte UTF-8
  4. prepend BOS (id 1, add_bos=true dari runtime)
*/
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "tokenizer.h"

#define K_UNK_PENALTY 10.0f

static int utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1; /* invalid: treat 1 byte */
}

static uint32_t fnv1a(const char *s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 16777619u;
    }
    return h;
}

static int checked_bytes(size_t count, size_t item_size, size_t *bytes) {
    if (!bytes || (count != 0 && item_size > SIZE_MAX / count)) return -1;
    *bytes = count * item_size;
    return 0;
}

const char *iro_tok_piece(IroTokenizer *t, int32_t id) {
    if (id < 0 || (uint32_t)id >= t->vocab_count) return NULL;
    return t->pieces_base + t->offsets[id];
}

/* cari id piece untuk byte s[0..n) */
static int32_t tok_lookup(const IroTokenizer *t, const char *s, size_t n) {
    uint32_t cap = t->htab_cap;
    uint32_t i = fnv1a(s, n) & (cap - 1);
    for (;;) {
        int32_t id = t->htab[i];
        if (id < 0) return -1;
        if ((size_t)t->lens[id] == n && memcmp(t->pieces_base + t->offsets[id], s, n) == 0)
            return id;
        i = (i + 1) & (cap - 1);
    }
}

static void htab_insert(IroTokenizer *t, int32_t id) {
    const char *p = t->pieces_base + t->offsets[id];
    size_t n = t->lens[id];
    uint32_t cap = t->htab_cap;
    uint32_t i = fnv1a(p, n) & (cap - 1);
    while (t->htab[i] >= 0) i = (i + 1) & (cap - 1);
    t->htab[i] = id;
}

/* ------------------------------------------------------------------ */
/* Load tokenizer.bin                                                  */
/* ------------------------------------------------------------------ */

int iro_tok_load(const char *path, IroTokenizer *t) {
    if (!path || !t) return -1;
    memset(t, 0, sizeof(*t));
    t->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open tokenizer"); return -1; }
    struct stat sb;
    if (fstat(fd, &sb) < 0) { perror("fstat"); close(fd); return -1; }
    if (sb.st_size <= 0 || (uintmax_t)sb.st_size > SIZE_MAX) {
        fprintf(stderr, "iro_tok: ukuran file tidak valid\n");
        close(fd);
        return -1;
    }
    size_t map_size = (size_t)sb.st_size;
    uint8_t *map = mmap(NULL, map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); close(fd); return -1; }

    /* Publish ownership immediately so every later failure can use one cleanup. */
    t->fd = fd;
    t->map = map;
    t->map_size = (uint64_t)map_size;

    if (map_size < 24 || memcmp(map, "IRTK", 4) != 0) {
        fprintf(stderr, "iro_tok: bukan file tokenizer.bin\n");
        goto fail;
    }

    size_t pos = 4;
    uint32_t version;
    memcpy(&version, map + pos, 4); pos += 4;
    if (version != 1) {
        fprintf(stderr, "iro_tok: versi %u tidak didukung\n", version);
        goto fail;
    }

    int32_t unk_id;
    memcpy(&unk_id, map + pos, 4); pos += 4;
    t->unk_id = unk_id;
    t->byte_fallback = map[pos]; t->prepend_bos = 0; pos += 4;
    if (pos + 2 > map_size) goto malformed;
    uint16_t raw_rep_len = (uint16_t)(map[pos] | (map[pos + 1] << 8)); pos += 2;
    if (raw_rep_len == 0 || raw_rep_len > sizeof(t->replacement) ||
        (size_t)raw_rep_len > map_size - pos) {
        goto malformed;
    }
    t->rep_len = raw_rep_len;
    memcpy(t->replacement, map + pos, t->rep_len); pos += raw_rep_len;
    if (map_size - pos < 8) goto malformed;
    t->prepend_scheme = map[pos]; t->split = map[pos + 1]; pos += 4;

    uint32_t count;
    memcpy(&count, map + pos, 4); pos += 4;
    if (count == 0 || (size_t)count > (map_size - pos) / 6u) goto malformed;
    t->vocab_count = count;

    size_t offsets_bytes, lens_bytes, scores_bytes;
    if (checked_bytes((size_t)count, sizeof(*t->offsets), &offsets_bytes) != 0 ||
        checked_bytes((size_t)count, sizeof(*t->lens), &lens_bytes) != 0 ||
        checked_bytes((size_t)count, sizeof(*t->scores), &scores_bytes) != 0)
        goto malformed;
    t->offsets = malloc(offsets_bytes);
    t->lens = malloc(lens_bytes);
    t->scores = malloc(scores_bytes);
    if (!t->offsets || !t->lens || !t->scores) goto fail;
    t->pieces_base = (const char *)(map + pos);

    float min_score = 1e30f;
    uint32_t max_len = 0;
    size_t cur = pos;
    for (uint32_t i = 0; i < count; i++) {
        if (map_size - cur < 2) goto malformed;
        uint16_t len = (uint16_t)(map[cur] | (map[cur + 1] << 8)); cur += 2;
        ((uint16_t *)t->lens)[i] = len;
        if (cur - pos > UINT32_MAX || (size_t)len > map_size - cur ||
            map_size - (cur + (size_t)len) < sizeof(float))
            goto malformed;
        t->offsets[i] = (uint32_t)(cur - pos);
        if (len > max_len) max_len = len;
        cur += len;
        float score;
        memcpy(&score, map + cur, 4); cur += 4;
        t->scores[i] = score;
        if (score < min_score) min_score = score;
    }
    t->max_piece_len = max_len;
    t->min_score = min_score;

    /* hash table piece -> id */
    uint32_t cap = 262144;
    if (count > UINT32_MAX / 2u) goto malformed;
    uint32_t desired = count * 2u;
    while (cap < desired) {
        if (cap > UINT32_MAX / 2u) goto malformed;
        cap <<= 1;
    }
    size_t htab_bytes;
    if (checked_bytes((size_t)cap, sizeof(*t->htab), &htab_bytes) != 0)
        goto malformed;
    t->htab = malloc(htab_bytes);
    if (!t->htab) goto fail;
    memset(t->htab, 0xFF, htab_bytes);
    t->htab_cap = cap;
    for (uint32_t i = 0; i < count; i++) htab_insert(t, (int32_t)i);

    return 0;

malformed:
    fprintf(stderr, "iro_tok: tokenizer.bin rusak atau terpotong\n");
fail:
    iro_tok_free(t);
    return -1;
}

void iro_tok_free(IroTokenizer *t) {
    free(t->offsets);
    free((void *)t->lens);
    free(t->scores);
    free(t->htab);
    if (t->map) munmap(t->map, (size_t)t->map_size);
    if (t->fd >= 0) close(t->fd);
    memset(t, 0, sizeof(*t));
    t->fd = -1;
}



/* ------------------------------------------------------------------ */
/* Encode: metaspace -> Viterbi -> byte fallback -> BOS                 */
/* ------------------------------------------------------------------ */

typedef struct {
    int32_t id;      /* -1 = unk */
    int32_t start;   /* char index awal */
    int32_t next;    /* char index target */
    float   score;
} Edge;

int iro_tok_encode(IroTokenizer *t, const char *utf8, int add_bos,
                   int32_t *out_ids, int max_ids) {
    if (!t || !utf8 || !out_ids || max_ids <= 0 || !t->map ||
        !t->htab || t->htab_cap == 0)
        return -1;
    size_t in_len = strlen(utf8);

    /* --- 1. Metaspace: ' ' -> replacement (▁) --- */
    size_t expansion = t->rep_len > 1 ? (size_t)t->rep_len : 1u;
    if (in_len > (SIZE_MAX - 1u) / expansion) return -1;
    size_t cap_buf = in_len * expansion + 1u;
    char *buf = malloc(cap_buf);
    if (!buf) return -1;
    size_t n = 0;
    for (size_t i = 0; i < in_len; i++) {
        if (utf8[i] == ' ') {
            memcpy(buf + n, t->replacement, t->rep_len);
            n += t->rep_len;
        } else {
            buf[n++] = utf8[i];
        }
    }
    buf[n] = '\0';

    /* --- 2. peta posisi char UTF-8 --- */
    int nchars = 0;
    size_t cpos[IRO_TOK_MAX_IDS * 4];
    const int cpos_cap = (int)(sizeof(cpos) / sizeof(cpos[0]));
    for (size_t i = 0; i < n; ) {
        if (nchars >= cpos_cap - 1) {
            free(buf);
            return -1;
        }
        cpos[nchars++] = i;
        size_t step = (size_t)utf8_len((unsigned char)buf[i]);
        if (step > n - i) step = 1;
        i += step;
    }
    cpos[nchars] = n;
    if (nchars == 0) { free(buf); return add_bos ? (out_ids[0] = 1, 1) : 0; }

    /* --- 3. Viterbi (DP maju, backtrack mundur) --- */
    float unk_score = t->min_score - K_UNK_PENALTY;
    int max_edges = nchars * 32 + 16;
    Edge *edges = malloc(sizeof(Edge) * (size_t)max_edges);
    int *edge_start = malloc(sizeof(int) * (size_t)(nchars + 1));  /* head per posisi */
    int *edge_count = malloc(sizeof(int) * (size_t)(nchars + 1));
    if (!edges || !edge_start || !edge_count) { free(buf); free(edges); free(edge_start); free(edge_count); return -1; }
    memset(edge_count, 0, sizeof(int) * (size_t)(nchars + 1));

    int ne = 0;
    for (int s = 0; s < nchars; s++) {
        size_t b0 = cpos[s];
        int has_single = 0;
        int head = ne;
        /* kandidat piece: dari char s ke char e (byte b0..cpos[e]) */
        for (int e = s + 1; e <= nchars; e++) {
            size_t b1 = cpos[e];
            size_t plen = b1 - b0;
            if (plen > t->max_piece_len) break;
            int32_t id = tok_lookup(t, buf + b0, plen);
            if (id >= 0) {
                if (e == s + 1) has_single = 1;
                if (ne >= max_edges) {
                    if (max_edges > INT_MAX / 2) goto encode_fail;
                    int new_max = max_edges * 2;
                    Edge *grown = realloc(edges, sizeof(*edges) * (size_t)new_max);
                    if (!grown) goto encode_fail;
                    edges = grown;
                    max_edges = new_max;
                }
                edges[ne].id = id;
                edges[ne].start = s;
                edges[ne].next = e;
                edges[ne].score = t->scores[id];
                ne++;
            }
        }
        /* unk edge per-char — hanya jika tidak ada piece 1-char dari sini */
        if (!has_single) {
            if (ne >= max_edges) {
                if (max_edges > INT_MAX / 2) goto encode_fail;
                int new_max = max_edges * 2;
                Edge *grown = realloc(edges, sizeof(*edges) * (size_t)new_max);
                if (!grown) goto encode_fail;
                edges = grown;
                max_edges = new_max;
            }
            edges[ne].id = -1;
            edges[ne].start = s;
            edges[ne].next = s + 1;
            edges[ne].score = unk_score;
            ne++;
        }
        edge_start[s] = head;
        edge_count[s] = ne - head;
    }

    /* DP: best[c] = skor terbaik mencapai char index c */
    float *best = malloc(sizeof(float) * (size_t)(nchars + 1));
    int *back = malloc(sizeof(int) * (size_t)(nchars + 1)); /* edge index terpilih */
    if (!best || !back) {
        free(best); free(back);
        goto encode_fail;
    }
    for (int c = 0; c <= nchars; c++) {
        best[c] = -1e30f;
        back[c] = -1;
    }
    best[0] = 0.0f;
    for (int s = 0; s < nchars; s++) {
        if (best[s] <= -1e29f) continue;
        for (int k = 0; k < edge_count[s]; k++) {
            Edge *e = &edges[edge_start[s] + k];
            float cand = best[s] + e->score;
            if (cand > best[e->next]) {
                best[e->next] = cand;
                back[e->next] = edge_start[s] + k;
            }
        }
    }

    /* --- 4. backtrack: nchars -> 0 (simpan index edge, bukan id saja) --- */
    int32_t *rev_edge = malloc(sizeof(int32_t) * (size_t)(nchars + 1));
    if (!rev_edge) {
        free(best); free(back);
        goto encode_fail;
    }
    int nout = 0;
    int c = nchars;
    while (c > 0 && nout <= nchars) {
        int ei = back[c];
        if (ei < 0) {
            free(rev_edge); free(best); free(back); free(edges);
            free(edge_start); free(edge_count); free(buf);
            return -1;
        }
        rev_edge[nout++] = ei;
        c = edges[ei].start;
    }
    if (c != 0) {
        free(rev_edge); free(best); free(back); free(edges);
        free(edge_start); free(edge_count); free(buf);
        return -1;
    }

    /* --- 5. susun output (urut maju) + byte fallback --- */
    int nids = 0;
    int ok = 1;
    int32_t bos = tok_lookup(t, "<s>", 3);
    if (bos < 0) bos = 1;
    if (add_bos) {
        if (nids < max_ids) out_ids[nids++] = bos;
        else ok = 0;
    }
    for (int i = nout - 1; i >= 0 && ok; i--) {
        Edge *e = &edges[rev_edge[i]];
        if (e->id >= 0) {
            if (nids < max_ids) out_ids[nids++] = e->id;
            else ok = 0;
        } else if (t->byte_fallback) {
            /* span byte char unknown -> piece <0xXX> per byte UTF-8 */
            size_t b0 = cpos[e->start], b1 = cpos[e->next];
            for (size_t b = b0; b < b1 && ok; b++) {
                char piece[8];
                snprintf(piece, sizeof(piece), "<0x%02X>", (unsigned char)buf[b]);
                int32_t id = tok_lookup(t, piece, strlen(piece));
                if (id >= 0) {
                    if (nids < max_ids) out_ids[nids++] = id;
                    else ok = 0;
                } else {
                    ok = 0;
                }
            }
        } else {
            if (nids < max_ids) out_ids[nids++] = t->unk_id;
            else ok = 0;
        }
    }

    free(rev_edge);
    free(best);
    free(back);
    free(edges);
    free(edge_start);
    free(edge_count);
    free(buf);
    return ok ? nids : -1;

encode_fail:
    free(edges);
    free(edge_start);
    free(edge_count);
    free(buf);
    return -1;
}
