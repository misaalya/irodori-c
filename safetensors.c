#define _DEFAULT_SOURCE
/* safetensors.c — mmap loader untuk HF safetensors.
 *
 * Format: [8-byte LE header length][header JSON UTF-8][tensor data ...]
 * Header: { "<tensor name>": {"dtype": "F32", "shape": [d0, d1, ...],
 *          "data_offsets": [begin, end]}, ..., "__metadata__": {k: v, ...} }
 */
#include <fcntl.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "irodori.h"

/* ------------------------------------------------------------------ */
/* Minimal JSON tokenizer (jsmn-style, implementasi sendiri)           */
/* ------------------------------------------------------------------ */

typedef enum { JT_PRIMITIVE = 0, JT_STRING, JT_OBJECT, JT_ARRAY } JType;

typedef struct {
    JType type;
    int   start;   /* offset token dalam buffer (string: tanpa kutip) */
    int   end;
    int   parent;
    int   size;    /* jumlah child langsung */
} JTok;

typedef struct {
    const char *js;
    size_t      len;
    size_t      pos;
    JTok       *toks;
    int         ntoks, max_toks;
    int         toksuper; /* index token pembuka saat ini, -1 root */
} JParser;

static int j_alloc(JParser *p, JType type) {
    if (p->ntoks >= p->max_toks) return -1;
    JTok *t = &p->toks[p->ntoks++];
    t->type = type;
    t->start = (int)p->pos;
    t->end = -1;
    t->parent = p->toksuper;
    t->size = 0;
    return (int)(t - p->toks);
}

/* buka object/array baru */
static void j_open(JParser *p, JType type) {
    int idx = j_alloc(p, type);
    if (p->toksuper != -1) p->toks[p->toksuper].size++;
    if (idx >= 0) p->toksuper = idx;
}

/* tutup object/array. pos harus di '}' atau ']' — loop utama yang majukan. */
static void j_close(JParser *p) {
    JTok *t = &p->toks[p->toksuper];
    t->end = (int)p->pos;
    p->toksuper = t->parent;
}

static int j_parse_string(JParser *p) {
    int idx = j_alloc(p, JT_STRING);
    if (idx < 0) return -1;
    p->pos++; /* lewati kutip pembuka */
    size_t start = p->pos;
    while (p->pos < p->len) {
        char c = p->js[p->pos];
        if (c == '"') {
            /* berhenti DI kutip penutup; loop utama yang majukan */
            p->toks[idx].start = (int)start;
            p->toks[idx].end = (int)p->pos;
            p->toks[idx].size = 0;
            if (p->toksuper != -1) p->toks[p->toksuper].size++;
            return 0;
        }
        if (c == '\\') {
            p->pos++;
            if (p->pos >= p->len) return -1;
            char e = p->js[p->pos];
            if (e == 'u') {
                if (p->pos + 4 >= p->len) return -1;
                for (int k = 1; k <= 4; k++) {
                    char h = p->js[p->pos + k];
                    int ok = (h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') ||
                             (h >= 'A' && h <= 'F');
                    if (!ok) return -1;
                }
                p->pos += 4; /* pos di digit hex terakhir */
            } else if (e != '"' && e != '\\' && e != '/' && e != 'b' && e != 'f' &&
                       e != 'n' && e != 'r' && e != 't') {
                return -1;
            }
        } else if ((unsigned char)c < 32) {
            return -1; /* kontrol ilegal */
        }
        p->pos++;
    }
    return -1; /* unterminated */
}

static int j_parse_primitive(JParser *p) {
    int idx = j_alloc(p, JT_PRIMITIVE);
    if (idx < 0) return -1;
    if (p->pos >= p->len) return -1;
    size_t last = p->pos;
    for (size_t i = p->pos; i < p->len; i++) {
        char c = p->js[i];
        if (c == ':' || c == ',' || c == ']' || c == '}' || c == ' ' || c == '\t' ||
            c == '\r' || c == '\n') break;
        if (c == '"' || c == '{' || c == '[' || (unsigned char)c < 32) return -1;
        last = i;
    }
    p->toks[idx].start = (int)p->pos;
    p->toks[idx].end = (int)(last + 1);
    p->toks[idx].size = 0;
    if (p->toksuper != -1) p->toks[p->toksuper].size++;
    p->pos = last; /* berhenti di char primitive terakhir; loop utama maju ke delimiter */
    return 0;
}

static int j_parse(JParser *p) {
    for (; p->pos < p->len; p->pos++) {
        char c = p->js[p->pos];
        switch (c) {
        case '{': case '[':
            j_open(p, c == '{' ? JT_OBJECT : JT_ARRAY);
            break;
        case '}': case ']':
            if (p->toksuper == -1) return -1;
            j_close(p);
            break;
        case '"':
            if (j_parse_string(p) < 0) return -1;
            break;
        case ' ': case '\t': case '\r': case '\n': case ':': case ',':
            break; /* whitespace/separator */
        default:
            if (j_parse_primitive(p) < 0) return -1;
            break;
        }
    }
    return (p->toksuper == -1) ? 0 : -1;
}


/* ------------------------------------------------------------------ */
/* Token helpers                                                      */
/* ------------------------------------------------------------------ */

static int j_tok_len(const JParser *p, int idx) {
    const JTok *t = &p->toks[idx];
    return t->end - t->start;
}

static int j_tok_eq(const JParser *p, int idx, const char *key, size_t keylen) {
    return (size_t)j_tok_len(p, idx) == keylen &&
           memcmp(p->js + p->toks[idx].start, key, keylen) == 0;
}

static int j_tok_int(const JParser *p, int idx, int64_t *out) {
    if (!p || !out || idx < 0 || idx >= p->ntoks) return -1;
    const JTok *t = &p->toks[idx];
    int len = t->end - t->start;
    if (t->type != JT_PRIMITIVE || len <= 0 || len >= 64) return -1;
    char buf[64];
    memcpy(buf, p->js + t->start, (size_t)len);
    buf[len] = '\0';
    char *end = NULL;
    errno = 0;
    long long value = strtoll(buf, &end, 10);
    if (errno == ERANGE || end == buf || *end != '\0') return -1;
    *out = (int64_t)value;
    return 0;
}

/* panjang subtree (jumlah token) milik token idx — children selalu
   sekuensial setelah parent, jadi subtree = 1 + Σ subtree child */
static int j_subtree_len(const JParser *p, int idx) {
    const JTok *t = &p->toks[idx];
    if (t->type != JT_OBJECT && t->type != JT_ARRAY) return 1;
    int len = 1;
    int c = idx + 1;
    for (int i = 0; i < t->size; i++) {
        int sub = j_subtree_len(p, c);
        len += sub;
        c += sub;
    }
    return len;
}

/* child ke-i (0-based) dari array; untuk object = value dari pasangan ke-i */
static int j_child(const JParser *p, int parent, int i) {
    int idx = parent + 1;
    int count = 0;
    for (int c = 0; c < p->toks[parent].size; c++) {
        int sub = j_subtree_len(p, idx);
        if (p->toks[parent].type == JT_ARRAY) {
            if (count == i) return idx;
        } else {
            /* object: children = key,value,key,value... value = child ganjil */
            if (count == 2 * i + 1) return idx;
        }
        idx += sub;
        count++;
    }
    return -1;
}

/* cari value token untuk key dalam object. return token index value, -1 jika tak ada */
static int j_obj_get(const JParser *p, int obj, const char *key) {
    int idx = obj + 1;
    size_t keylen = strlen(key);
    for (int i = 0; i < p->toks[obj].size; i++) {
        int sub = j_subtree_len(p, idx);
        if ((i % 2) == 0 && p->toks[idx].type == JT_STRING &&
            j_tok_eq(p, idx, key, keylen)) {
            return idx + 1; /* key selalu 1 token */
        }
        idx += sub;
    }
    return -1;
}


static IroDType iro_dtype_from_str(const char *s) {
    if (!strcmp(s, "F32")) return IRO_ST_F32;
    if (!strcmp(s, "F16")) return IRO_ST_F16;
    if (!strcmp(s, "BF16")) return IRO_ST_BF16;
    if (!strcmp(s, "I64")) return IRO_ST_I64;
    if (!strcmp(s, "I32")) return IRO_ST_I32;
    if (!strcmp(s, "I16")) return IRO_ST_I16;
    if (!strcmp(s, "I8")) return IRO_ST_I8;
    if (!strcmp(s, "U8")) return IRO_ST_U8;
    if (!strcmp(s, "BOOL")) return IRO_ST_BOOL;
    return IRO_ST_UNKNOWN;
}

const char *iro_dtype_name(IroDType dt) {
    switch (dt) {
    case IRO_ST_F32: return "F32";
    case IRO_ST_F16: return "F16";
    case IRO_ST_BF16: return "BF16";
    case IRO_ST_I64: return "I64";
    case IRO_ST_I32: return "I32";
    case IRO_ST_I16: return "I16";
    case IRO_ST_I8: return "I8";
    case IRO_ST_U8: return "U8";
    case IRO_ST_BOOL: return "BOOL";
    default: return "?";
    }
}

int64_t iro_tensor_numel(const IroTensor *t) {
    if (!t || t->ndims < 0 || t->ndims > IRO_MAX_DIMS) return -1;
    uint64_t n = 1;
    for (int i = 0; i < t->ndims; i++) {
        if (t->shape[i] == 0) return 0;
        if (n > (uint64_t)INT64_MAX / t->shape[i]) return -1;
        n *= t->shape[i];
    }
    return (int64_t)n;
}

int iro_tensor_f32_all_finite(const IroTensor *t) {
    if (!t || t->dtype != IRO_ST_F32 || !t->data) return -1;
    int64_t numel = iro_tensor_numel(t);
    if (numel < 0 || (uint64_t)numel > SIZE_MAX / sizeof(float) ||
        (uint64_t)numel * sizeof(float) != t->nbytes)
        return -1;
    const unsigned char *bytes = t->data;
    for (int64_t i = 0; i < numel; i++) {
        float value;
        memcpy(&value, bytes + (size_t)i * sizeof(value), sizeof(value));
        if (!isfinite(value)) return 0;
    }
    return 1;
}

static uint64_t dtype_size(IroDType dtype) {
    switch (dtype) {
    case IRO_ST_F32: case IRO_ST_I32: return 4;
    case IRO_ST_F16: case IRO_ST_BF16: case IRO_ST_I16: return 2;
    case IRO_ST_I64: return 8;
    case IRO_ST_I8: case IRO_ST_U8: case IRO_ST_BOOL: return 1;
    default: return 0;
    }
}

/* ------------------------------------------------------------------ */
/* Safetensors loader                                                 */
/* ------------------------------------------------------------------ */

int iro_st_load(const char *path, IroSafetensors *st) {
    if (!path || !st) return -1;
    memset(st, 0, sizeof(*st));
    st->fd = -1;

    char *header = NULL;
    JTok *toks = NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return -1; }

    struct stat sb;
    if (fstat(fd, &sb) < 0) { perror("fstat"); close(fd); return -1; }
    if (sb.st_size < 0 || (uintmax_t)sb.st_size > SIZE_MAX ||
        (uint64_t)sb.st_size < 10) {
        fprintf(stderr, "iro: file terlalu kecil atau terlalu besar\n");
        close(fd);
        return -1;
    }

    size_t map_size = (size_t)sb.st_size;
    uint8_t *map = mmap(NULL, map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); close(fd); return -1; }

    /* Publish mmap ownership early so every parser/allocation failure shares
       the same cleanup path. */
    st->fd = fd;
    st->map = map;
    st->map_size = (uint64_t)map_size;

    /* 8-byte little-endian header length */
    uint64_t header_len = 0;
    for (int i = 0; i < 8; i++) header_len |= (uint64_t)map[i] << (8 * i);
    if (header_len == 0 || header_len > (uint64_t)map_size - 8u ||
        header_len > SIZE_MAX - 1u) {
        fprintf(stderr, "iro: header length tidak valid (%llu)\n",
                (unsigned long long)header_len);
        goto fail;
    }

    header = malloc((size_t)header_len + 1u);
    if (!header) goto fail;
    memcpy(header, map + 8, (size_t)header_len);
    header[header_len] = '\0';

    /* parse JSON: root object berisi nama tensor + __metadata__ */
    int max_toks = 1024 * 1024;
    toks = malloc(sizeof(*toks) * (size_t)max_toks);
    if (!toks) goto fail;

    JParser p = { .js = header, .len = (size_t)header_len, .pos = 0,
                  .toks = toks, .ntoks = 0, .max_toks = max_toks, .toksuper = -1 };
    if (j_parse(&p) < 0 || p.ntoks == 0 || p.toks[0].type != JT_OBJECT) {
        fprintf(stderr, "iro: gagal parse header JSON (ntoks=%d pos=%zu/%zu)\n",
                p.ntoks, p.pos, p.len);
        goto fail;
    }

    int root = 0;
    if ((p.toks[root].size & 1) != 0) {
        fprintf(stderr, "iro: root safetensors bukan pasangan key/value lengkap\n");
        goto fail;
    }
    int meta_tok = j_obj_get(&p, root, "__metadata__");
    int n_tensors = p.toks[root].size / 2 - (meta_tok >= 0 ? 1 : 0);
    if (n_tensors < 0) goto fail;

    IroTensor *tensors = calloc((size_t)n_tensors, sizeof(IroTensor));
    if (n_tensors > 0 && !tensors) {
        fprintf(stderr, "iro: alloc gagal\n");
        goto fail;
    }
    st->tensors = tensors;

    int ti = 0;
    int idx = root + 1;
    const uint64_t data_start = 8u + header_len;
    const uint64_t data_bytes = (uint64_t)map_size - data_start;
    for (int i = 0; i < p.toks[root].size; i += 2) {
        if (idx < 0 || idx + 1 >= p.ntoks || p.toks[idx].type != JT_STRING)
            goto malformed;
        int key_tok = idx;
        int val_tok = idx + 1; /* key string selalu 1 token */
        int keylen = j_tok_len(&p, key_tok);
        int val_sub = j_subtree_len(&p, val_tok);
        if (keylen <= 0 || val_sub <= 0 || val_tok + val_sub > p.ntoks)
            goto malformed;

        static const char k_meta_key[] = "__metadata__";
        int is_meta = j_tok_eq(&p, key_tok, k_meta_key, sizeof(k_meta_key) - 1);
        if (!is_meta) {
            if (p.toks[val_tok].type != JT_OBJECT) goto malformed;
            /* entri tensor: {"dtype", "shape", "data_offsets"} */
            char *name = malloc((size_t)keylen + 1);
            if (!name) goto fail;
            memcpy(name, header + p.toks[key_tok].start, (size_t)keylen);
            name[keylen] = '\0';

            if (ti >= n_tensors) { free(name); goto malformed; }
            IroTensor *t = &tensors[ti++];
            t->name = name;
            st->n_tensors = ti;

            int dt_tok = j_obj_get(&p, val_tok, "dtype");
            if (dt_tok < 0 || p.toks[dt_tok].type != JT_STRING) goto malformed;
            char buf[16];
            int dl = j_tok_len(&p, dt_tok);
            if (dl <= 0 || (size_t)dl >= sizeof(buf)) goto malformed;
            memcpy(buf, header + p.toks[dt_tok].start, (size_t)dl);
            buf[dl] = '\0';
            t->dtype = iro_dtype_from_str(buf);
            uint64_t item_size = dtype_size(t->dtype);
            if (item_size == 0) goto malformed;

            int shape_tok = j_obj_get(&p, val_tok, "shape");
            if (shape_tok < 0 || p.toks[shape_tok].type != JT_ARRAY ||
                p.toks[shape_tok].size > IRO_MAX_DIMS)
                goto malformed;
            int nd = p.toks[shape_tok].size;
            t->ndims = nd;
            uint64_t numel = 1;
            for (int d = 0; d < nd; d++) {
                int child = j_child(&p, shape_tok, d);
                int64_t v = 0;
                if (child < 0 || j_tok_int(&p, child, &v) != 0 || v < 0)
                    goto malformed;
                t->shape[d] = (uint64_t)v;
                if (v == 0) numel = 0;
                else if (numel != 0) {
                    if (numel > UINT64_MAX / (uint64_t)v) goto malformed;
                    numel *= (uint64_t)v;
                }
            }

            int off_tok = j_obj_get(&p, val_tok, "data_offsets");
            if (off_tok < 0 || p.toks[off_tok].type != JT_ARRAY ||
                p.toks[off_tok].size != 2)
                goto malformed;
            int64_t b = 0, e = 0;
            if (j_tok_int(&p, j_child(&p, off_tok, 0), &b) != 0 ||
                j_tok_int(&p, j_child(&p, off_tok, 1), &e) != 0 ||
                b < 0 || e < b || (uint64_t)e > data_bytes)
                goto malformed;
            t->data_begin = (uint64_t)b;
            t->nbytes = (uint64_t)(e - b);
            if (numel > UINT64_MAX / item_size || numel * item_size != t->nbytes)
                goto malformed;
            t->data = map + data_start + t->data_begin;
        } else if (p.toks[val_tok].type != JT_OBJECT) {
            goto malformed;
        }
        idx = val_tok + val_sub; /* maju sepanjang subtree value */
    }
    if (ti != n_tensors) goto malformed;

    /* metadata __metadata__ */
    char **mk = NULL, **mv = NULL;
    int n_meta = 0;
    if (meta_tok >= 0 && p.toks[meta_tok].type == JT_OBJECT) {
        n_meta = p.toks[meta_tok].size / 2;
        if ((p.toks[meta_tok].size & 1) != 0) goto malformed;
        mk = calloc((size_t)n_meta, sizeof(char *));
        mv = calloc((size_t)n_meta, sizeof(char *));
        if (n_meta > 0 && (!mk || !mv)) {
            free(mk); free(mv);
            goto fail;
        }
        st->meta_keys = mk;
        st->meta_values = mv;
        int midx = meta_tok + 1;
        int m = 0;
        for (int i = 0; i < p.toks[meta_tok].size; i += 2) {
            if (midx < 0 || midx + 1 >= p.ntoks ||
                p.toks[midx].type != JT_STRING || p.toks[midx + 1].type != JT_STRING)
                goto malformed;
            int kt = midx, vt = midx + 1;
            int kl = j_tok_len(&p, kt), vl = j_tok_len(&p, vt);
            int vt_sub = j_subtree_len(&p, vt);
            if (kl < 0 || vl < 0 || vt_sub <= 0) goto malformed;
            mk[m] = malloc((size_t)kl + 1u);
            mv[m] = malloc((size_t)vl + 1u);
            if (!mk[m] || !mv[m]) {
                free(mk[m]); free(mv[m]);
                mk[m] = NULL; mv[m] = NULL;
                goto fail;
            }
            memcpy(mk[m], header + p.toks[kt].start, (size_t)kl); mk[m][kl] = '\0';
            memcpy(mv[m], header + p.toks[vt].start, (size_t)vl); mv[m][vl] = '\0';
            m++;
            st->n_meta = m;
            midx = vt + vt_sub;
        }
        n_meta = m;
    }

    free(toks);
    free(header);
    toks = NULL;
    header = NULL;
    st->n_meta = n_meta;
    return 0;

malformed:
    fprintf(stderr, "iro: header safetensors rusak atau tensor di luar batas file\n");
fail:
    free(toks);
    free(header);
    iro_st_free(st);
    return -1;
}

const IroTensor *iro_st_get(const IroSafetensors *st, const char *name) {
    for (int i = 0; i < st->n_tensors; i++)
        if (!strcmp(st->tensors[i].name, name)) return &st->tensors[i];
    return NULL;
}

const char *iro_st_meta(const IroSafetensors *st, const char *key) {
    for (int i = 0; i < st->n_meta; i++)
        if (!strcmp(st->meta_keys[i], key)) return st->meta_values[i];
    return NULL;
}

uint64_t iro_st_drop_prefix(const IroSafetensors *st, const char *prefix) {
    if (!st || !st->map || !prefix || !*prefix) return 0;
    uintptr_t first = UINTPTR_MAX, last = 0;
    size_t prefix_len = strlen(prefix);
    for (int i = 0; i < st->n_tensors; i++) {
        const IroTensor *tensor = &st->tensors[i];
        if (strncmp(tensor->name, prefix, prefix_len)) continue;
        uintptr_t begin = (uintptr_t)tensor->data;
        uintptr_t end = begin + (uintptr_t)tensor->nbytes;
        if (begin < first) first = begin;
        if (end > last) last = end;
    }
    if (first == UINTPTR_MAX || last <= first) return 0;
    long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) return 0;
    uintptr_t page_size = (uintptr_t)page_size_long;
    uintptr_t map_begin = (uintptr_t)st->map;
    uintptr_t map_end = map_begin + (uintptr_t)st->map_size;
    uintptr_t aligned_begin = first & ~(page_size - 1u);
    uintptr_t aligned_end = (last + page_size - 1u) & ~(page_size - 1u);
    if (aligned_begin < map_begin) aligned_begin = map_begin;
    if (aligned_end > map_end) aligned_end = map_end;
    if (aligned_end <= aligned_begin ||
        madvise((void *)aligned_begin, (size_t)(aligned_end - aligned_begin),
                MADV_DONTNEED) != 0)
        return 0;
    return (uint64_t)(aligned_end - aligned_begin);
}

void iro_st_free(IroSafetensors *st) {
    for (int i = 0; i < st->n_tensors; i++) free(st->tensors[i].name);
    free(st->tensors);
    for (int i = 0; i < st->n_meta; i++) { free(st->meta_keys[i]); free(st->meta_values[i]); }
    free(st->meta_keys); free(st->meta_values);
    if (st->map) munmap(st->map, (size_t)st->map_size);
    if (st->fd >= 0) close(st->fd);
    memset(st, 0, sizeof(*st));
    st->fd = -1;
}
