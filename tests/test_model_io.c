#define _POSIX_C_SOURCE 200809L
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "irodori.h"
#include "tokenizer.h"

static void put_u32(unsigned char *p, uint32_t value) {
    p[0] = (unsigned char)value;
    p[1] = (unsigned char)(value >> 8);
    p[2] = (unsigned char)(value >> 16);
    p[3] = (unsigned char)(value >> 24);
}

static int write_all(int fd, const void *data, size_t size) {
    const unsigned char *p = data;
    while (size > 0) {
        ssize_t n = write(fd, p, size);
        if (n <= 0) return -1;
        p += (size_t)n;
        size -= (size_t)n;
    }
    return 0;
}

static int expect_bad_tokenizer(const char *label,
                                const unsigned char *data, size_t size) {
    char path[] = "/tmp/irodori-tokenizer-bad-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return -1;
    int failed = write_all(fd, data, size) != 0 || close(fd) != 0;
    IroTokenizer tok = {.fd = -1};
    int result = failed ? -1 : iro_tok_load(path, &tok);
    unlink(path);
    int pass = result != 0 && tok.fd == -1 && tok.map == NULL &&
               tok.offsets == NULL && tok.htab == NULL;
    iro_tok_free(&tok);
    printf("%s tokenizer %s\n", pass ? "PASS" : "FAIL", label);
    return pass ? 0 : -1;
}

static int write_safetensors_fixture(const char *path, const char *json,
                                     const void *payload, size_t payload_size) {
    int fd = open(path, O_WRONLY | O_TRUNC);
    if (fd < 0) return -1;
    uint64_t header_len = (uint64_t)strlen(json);
    unsigned char prefix[8];
    for (int i = 0; i < 8; i++) prefix[i] = (unsigned char)(header_len >> (8 * i));
    int failed = write_all(fd, prefix, sizeof(prefix)) != 0 ||
                 write_all(fd, json, (size_t)header_len) != 0 ||
                 (payload_size > 0 && write_all(fd, payload, payload_size) != 0) ||
                 close(fd) != 0;
    return failed ? -1 : 0;
}

static int check_safetensors_case(const char *label, const char *json,
                                  const void *payload, size_t payload_size,
                                  int should_pass) {
    char path[] = "/tmp/irodori-safetensors-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return -1;
    close(fd);
    int result = write_safetensors_fixture(path, json, payload, payload_size);
    IroSafetensors st = {.fd = -1};
    if (result == 0) result = iro_st_load(path, &st);
    unlink(path);
    int pass = should_pass ? result == 0 : result != 0;
    if (should_pass && pass) {
        const IroTensor *tensor = iro_st_get(&st, "x");
        pass = tensor && tensor->dtype == IRO_ST_F32 && tensor->ndims == 1 &&
               tensor->shape[0] == 1 && tensor->nbytes == 4 &&
               iro_tensor_numel(tensor) == 1;
    }
    if (!should_pass && pass)
        pass = st.fd == -1 && st.map == NULL && st.tensors == NULL;
    iro_st_free(&st);
    printf("%s safetensors %s\n", pass ? "PASS" : "FAIL", label);
    return pass ? 0 : -1;
}

int main(void) {
    int failed = 0;

    unsigned char wrong_version[24] = {0};
    memcpy(wrong_version, "IRTK", 4);
    put_u32(wrong_version + 4, 2);
    failed |= expect_bad_tokenizer("unsupported version", wrong_version,
                                   sizeof(wrong_version)) != 0;

    unsigned char truncated_replacement[24] = {0};
    memcpy(truncated_replacement, "IRTK", 4);
    put_u32(truncated_replacement + 4, 1);
    truncated_replacement[12] = 1;
    truncated_replacement[16] = 8;
    failed |= expect_bad_tokenizer("truncated replacement", truncated_replacement,
                                   sizeof(truncated_replacement)) != 0;

    unsigned char truncated_vocab[29] = {0};
    memcpy(truncated_vocab, "IRTK", 4);
    put_u32(truncated_vocab + 4, 1);
    truncated_vocab[12] = 1;
    truncated_vocab[16] = 3;
    memcpy(truncated_vocab + 18, "abc", 3);
    put_u32(truncated_vocab + 25, 1);
    failed |= expect_bad_tokenizer("truncated vocab", truncated_vocab,
                                   sizeof(truncated_vocab)) != 0;

    float payload = 1.25f;
    failed |= check_safetensors_case(
        "valid scalar buffer",
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}",
        &payload, sizeof(payload), 1) != 0;
    failed |= check_safetensors_case(
        "out-of-bounds data",
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,100]}}",
        NULL, 0, 0) != 0;
    failed |= check_safetensors_case(
        "negative offset",
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[-1,3]}}",
        &payload, sizeof(payload), 0) != 0;
    failed |= check_safetensors_case(
        "negative shape",
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[-1],\"data_offsets\":[0,4]}}",
        &payload, sizeof(payload), 0) != 0;
    failed |= check_safetensors_case(
        "shape-byte mismatch",
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}",
        &payload, sizeof(payload), 0) != 0;

    char finite_path[] = "/tmp/irodori-safetensors-finite-XXXXXX";
    int finite_fd = mkstemp(finite_path);
    if (finite_fd < 0) return 1;
    close(finite_fd);
    float nonfinite_payload = 0.0f / 0.0f;
    int finite_result = write_safetensors_fixture(
        finite_path,
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}",
        &nonfinite_payload, sizeof(nonfinite_payload));
    IroSafetensors finite_st = {.fd = -1};
    if (finite_result == 0) finite_result = iro_st_load(finite_path, &finite_st);
    unlink(finite_path);
    const IroTensor *finite_tensor = finite_result == 0 ? iro_st_get(&finite_st, "x") : NULL;
    int pass_nonfinite = finite_tensor && iro_tensor_f32_all_finite(finite_tensor) == 0;
    printf("%s non-finite F32 detection\n", pass_nonfinite ? "PASS" : "FAIL");
    failed |= !pass_nonfinite;
    iro_st_free(&finite_st);

    IroTensor huge = {.ndims = 2, .shape = {UINT64_MAX, 2}};
    int pass_overflow = iro_tensor_numel(&huge) == -1;
    printf("%s tensor numel overflow\n", pass_overflow ? "PASS" : "FAIL");
    failed |= !pass_overflow;

    return failed ? 1 : 0;
}
