/* normalize.c — port irodori_tts/text_normalization.py (gate: tokenizer vectors).

Urutan PERSIS seperti Python:
  1. SIMPLE_REPLACE_MAP (literal): "\t"->"", "[n]"->"", "\[n\]"->"", "　"->"",
     "？"->"?", "！"->"!", "♥"->"♡", "●"->"○", "◯"->"○", "〇"->"○"
  2. REGEX_REPLACE_MAP (4): remove [;▼♀♂《》≪≫①②③④⑤⑥];
     remove [\u02d7 \u2010-\u2015 \u2043 \u2212 \u23af \u23e4 \u2500 \u2501 \u2e3a \u2e3b];
     [\uff5e\u301C] -> "ー"; "…"{3,} -> "……"
  3. strip_outer_brackets (「」『』（）【】() dengan cek enclosing + depth)
  4. NFKC (utf8proc)
  5. "..." -> "…", ".." -> "…" (kiri-ke-kanan non-overlap, seperti str.replace)
*/
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "vendor/utf8proc/utf8proc.h"

typedef struct {
    char   *buf;
    size_t  len, cap;
} SB;

static int sb_init(SB *s) {
    s->cap = 256;
    s->len = 0;
    s->buf = malloc(s->cap);
    return s->buf ? 0 : -1;
}

static int sb_grow(SB *s, size_t need) {
    if (s->len + need > s->cap) {
        while (s->len + need > s->cap) s->cap *= 2;
        s->buf = realloc(s->buf, s->cap);
        return s->buf ? 0 : -1;
    }
    return 0;
}

static void sb_put(SB *s, const char *p, size_t n) {
    if (sb_grow(s, n) == 0) {
        memcpy(s->buf + s->len, p, n);
        s->len += n;
    }
}

/* decode satu codepoint UTF-8; return panjang byte (0 = invalid) */
static int u8_next(const char *p, size_t avail, uint32_t *cp) {
    if (avail == 0) return 0;
    unsigned char c = (unsigned char)p[0];
    if (c < 0x80) { *cp = c; return 1; }
    int n;
    uint32_t v;
    if ((c & 0xE0) == 0xC0) { n = 2; v = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { n = 3; v = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { n = 4; v = c & 0x07; }
    else return 0;
    if ((size_t)n > avail) return 0;
    for (int i = 1; i < n; i++) {
        if (((unsigned char)p[i] & 0xC0) != 0x80) return 0;
        v = (v << 6) | ((unsigned char)p[i] & 0x3F);
    }
    *cp = v;
    return n;
}

/* ------------------------------------------------------------------ */
/* 1+2. simple replace + regex class (single pass)                      */
/* ------------------------------------------------------------------ */

static int utf8_starts_with(const char *s, size_t avail, const char *pat, size_t plen) {
    return avail >= plen && memcmp(s, pat, plen) == 0;
}

/* langkah 1+2 dalam satu pass kiri-ke-kanan */
static char *pass_simple_regex(const char *text) {
    size_t len = strlen(text);
    SB out;
    if (sb_init(&out) != 0) return NULL;

    size_t i = 0;
    while (i < len) {
        /* --- SIMPLE_REPLACE_MAP (literal, seperti dict Python) --- */
        if (text[i] == '\t') { i++; continue; }                     /* "\t" -> "" */
        if (utf8_starts_with(text + i, len - i, "[n]", 3)) { i += 3; continue; }
        if (utf8_starts_with(text + i, len - i, "\\[n\\]", 5)) { i += 5; continue; }
        if (utf8_starts_with(text + i, len - i, "\xE3\x80\x80", 3)) { i += 3; continue; } /* U+3000 */
        if (utf8_starts_with(text + i, len - i, "\xEF\xBC\x9F", 3)) { /* ？ -> ? */
            sb_put(&out, "?", 1); i += 3; continue;
        }
        if (utf8_starts_with(text + i, len - i, "\xEF\xBC\x81", 3)) { /* ！ -> ! */
            sb_put(&out, "!", 1); i += 3; continue;
        }
        if (utf8_starts_with(text + i, len - i, "\xE2\x99\xA5", 3)) { /* ♥ -> ♡ */
            sb_put(&out, "\xE2\x99\xA1", 3); i += 3; continue;
        }
        if (utf8_starts_with(text + i, len - i, "\xE2\x97\x8F", 3) ||   /* ● */
            utf8_starts_with(text + i, len - i, "\xE2\x97\xAF", 3) ||   /* ◯ */
            utf8_starts_with(text + i, len - i, "\xE3\x80\x87", 3)) {   /* 〇 */
            sb_put(&out, "\xE2\x97\x8B", 3); i += 3; continue;          /* -> ○ */
        }

        uint32_t cp;
        int cn = u8_next(text + i, len - i, &cp);
        if (cn == 0) { sb_put(&out, text + i, 1); i++; continue; }

        /* --- REGEX_REPLACE_MAP --- */
        int skip = 0;
        if (cp == ';' || cp == 0x25BC /*▼*/ || cp == 0x2640 /*♀*/ || cp == 0x2642 /*♂*/ ||
            cp == 0x300A /*《*/ || cp == 0x300B /*《*/ || cp == 0x226A /*≪*/ ||
            cp == 0x226B /*≫*/ || (cp >= 0x2460 && cp <= 0x2465) /*①..⑥*/) {
            skip = 1;
        } else if (cp == 0x02D7 || (cp >= 0x2010 && cp <= 0x2015) || cp == 0x2043 ||
                   cp == 0x2212 || cp == 0x23AF || cp == 0x23E4 || cp == 0x2500 ||
                   cp == 0x2501 || cp == 0x2E3A || cp == 0x2E3B) {
            skip = 1;
        } else if (cp == 0xFF5E || cp == 0x301C) {
            sb_put(&out, "\xE3\x83\xBC", 3); /* -> "ー" U+30FC */
            skip = 1;
        }
        if (skip) { i += (size_t)cn; continue; }

        /* --- "…"{3,} -> "……" --- */
        if (cp == 0x2026) {
            size_t run = 0, j = i;
            while (j < len) {
                uint32_t c2;
                int n2 = u8_next(text + j, len - j, &c2);
                if (n2 > 0 && c2 == 0x2026) { run++; j += (size_t)n2; }
                else break;
            }
            if (run >= 3) {
                sb_put(&out, "\xE2\x80\xA6", 3);
                sb_put(&out, "\xE2\x80\xA6", 3);
                i = j;
                continue;
            }
        }

        sb_put(&out, text + i, (size_t)cn);
        i += (size_t)cn;
    }
    sb_put(&out, "\0", 1);
    return out.buf;
}


/* ------------------------------------------------------------------ */
/* 3. strip_outer_brackets                                             */
/* ------------------------------------------------------------------ */

static uint32_t bracket_pair(uint32_t cp) {
    switch (cp) {
    case 0x300C: return 0x300D; /* 「」 */
    case 0x300E: return 0x300F; /* 『』 */
    case 0xFF08: return 0xFF09; /* （） */
    case 0x3010: return 0x3011; /* 【】 */
    case '(':    return ')';
    default: return 0;
    }
}

static char *strip_outer_brackets(char *text) {
    for (;;) {
        size_t len = strlen(text);
        if (len < 2) break;

        uint32_t first_cp, last_cp;
        int fn = u8_next(text, len, &first_cp);
        if (fn == 0) break;
        /* posisi byte char terakhir */
        size_t last_start = len;
        while (last_start > 0 && (((unsigned char)text[last_start - 1] & 0xC0) == 0x80))
            last_start--;
        int ln = u8_next(text + last_start, len - last_start, &last_cp);
        if (ln == 0) break;

        uint32_t closer = bracket_pair(first_cp);
        if (!closer || closer != last_cp) break;

        /* mirror Python: scan semua char; enclosing = depth kembali 0
           HANYA di char terakhir */
        int depth = 0, enclosing = 1;
        size_t pos = 0;
        while (pos < len) {
            uint32_t c;
            int n = u8_next(text + pos, len - pos, &c);
            if (n == 0) { enclosing = 0; break; }
            if (c == first_cp) depth++;
            else if (c == last_cp) depth--;
            if (depth == 0 && pos + (size_t)n < len) { enclosing = 0; break; }
            pos += (size_t)n;
        }
        if (enclosing && depth == 0) {
            memmove(text, text + (size_t)fn, len - (size_t)fn - (size_t)ln);
            text[len - (size_t)fn - (size_t)ln] = '\0';
            continue;
        }
        break;
    }
    return text;
}

/* ------------------------------------------------------------------ */
/* 5. "..." -> "…", ".." -> "…" (non-overlap kiri-ke-kanan)            */
/* ------------------------------------------------------------------ */

static char *pass_dots(char *text) {
    size_t len = strlen(text);
    SB out;
    if (sb_init(&out) != 0) return text;
    for (size_t i = 0; i < len; ) {
        if (i + 3 <= len && text[i] == '.' && text[i + 1] == '.' && text[i + 2] == '.') {
            sb_put(&out, "\xE2\x80\xA6", 3);
            i += 3;
        } else {
            sb_put(&out, text + i, 1);
            i++;
        }
    }
    sb_put(&out, "\0", 1);

    /* pass 2 atas hasil pass 1 */
    size_t l2 = out.len - 1;
    SB out2;
    if (sb_init(&out2) != 0) { free(out.buf); return text; }
    for (size_t i = 0; i < l2; ) {
        if (i + 2 <= l2 && out.buf[i] == '.' && out.buf[i + 1] == '.') {
            sb_put(&out2, "\xE2\x80\xA6", 3);
            i += 2;
        } else {
            sb_put(&out2, out.buf + i, 1);
            i++;
        }
    }
    sb_put(&out2, "\0", 1);
    free(out.buf);
    free(text);
    return out2.buf;
}

/* ------------------------------------------------------------------ */
/* API                                                                 */
/* ------------------------------------------------------------------ */

char *iro_normalize(const char *text) {
    char *s = pass_simple_regex(text);
    if (!s) return NULL;
    s = strip_outer_brackets(s);

    /* 4. NFKC via utf8proc */
    uint8_t *nfkc = NULL;
    utf8proc_ssize_t r = utf8proc_map(
        (const uint8_t *)s, (utf8proc_ssize_t)strlen(s), &nfkc,
        (utf8proc_option_t)(UTF8PROC_COMPOSE | UTF8PROC_COMPAT));
    free(s);
    if (r < 0 || !nfkc) return NULL;

    return pass_dots((char *)nfkc);
}
