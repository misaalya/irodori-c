#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer.h"

static char *fallback_text(int emoji_count, int add_snowman) {
    static const char emoji[] = "😀";   /* 4-byte fallback in project tokenizer */
    static const char snowman[] = "☃"; /* 3-byte fallback in project tokenizer */
    size_t bytes = (size_t)emoji_count * (sizeof(emoji) - 1u) +
                   (add_snowman ? sizeof(snowman) - 1u : 0u);
    char *text = malloc(bytes + 1u);
    if (!text) return NULL;
    char *p = text;
    for (int i = 0; i < emoji_count; i++) {
        memcpy(p, emoji, sizeof(emoji) - 1u);
        p += sizeof(emoji) - 1u;
    }
    if (add_snowman) {
        memcpy(p, snowman, sizeof(snowman) - 1u);
        p += sizeof(snowman) - 1u;
    }
    *p = '\0';
    return text;
}

static int check_count(IroTokenizer *tok, const char *label,
                       int emoji_count, int add_snowman, int expected) {
    char *text = fallback_text(emoji_count, add_snowman);
    if (!text) return -1;
    int32_t ids[IRO_TOK_MAX_IDS];
    int count = iro_tok_encode(tok, text, 1, ids, IRO_TOK_MAX_IDS);
    free(text);
    int pass = count == expected;
    printf("%s %s: %d tokens\n", pass ? "PASS" : "FAIL", label, count);
    return pass ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s tokenizer.bin\n", argv[0]);
        return 2;
    }
    IroTokenizer tok = {.fd = -1};
    if (iro_tok_load(argv[1], &tok) != 0) return 1;
    int failed = 0;
    /* BOS + 63*4-byte fallback + 3-byte fallback = 256. */
    failed |= check_count(&tok, "text max boundary", 63, 1, 256) != 0;
    failed |= check_count(&tok, "text over boundary", 64, 0, 257) != 0;
    /* BOS + 127*4-byte fallback + 3-byte fallback = 512. */
    failed |= check_count(&tok, "caption max boundary", 127, 1, 512) != 0;
    failed |= check_count(&tok, "caption over boundary", 128, 0, 513) != 0;

    char *text = fallback_text(63, 1);
    int32_t ids[256];
    int overflow = text ? iro_tok_encode(&tok, text, 1, ids, 255) : 0;
    int pass_capacity = text && overflow == -1;
    printf("%s tokenizer output-capacity guard\n",
           pass_capacity ? "PASS" : "FAIL");
    failed |= !pass_capacity;
    free(text);
    iro_tok_free(&tok);
    return failed ? 1 : 0;
}
