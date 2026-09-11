#define _POSIX_C_SOURCE 200809L
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "audio.h"

static void put_u16(unsigned char *p, uint16_t value) {
    p[0] = (unsigned char)value;
    p[1] = (unsigned char)(value >> 8);
}

static void put_u32(unsigned char *p, uint32_t value) {
    p[0] = (unsigned char)value;
    p[1] = (unsigned char)(value >> 8);
    p[2] = (unsigned char)(value >> 16);
    p[3] = (unsigned char)(value >> 24);
}

static int write_fixture(const char *path, int format, int bits,
                         const unsigned char *data, uint32_t data_bytes) {
    FILE *file = fopen(path, "wb");
    if (!file) return -1;
    unsigned char header[12] = "RIFF\0\0\0\0WAVE";
    put_u32(header + 4, 48u + data_bytes);
    unsigned char junk[12] = {'J','U','N','K',3,0,0,0,1,2,3,0};
    unsigned char fmt[24] = {'f','m','t',' ',16,0,0,0};
    put_u16(fmt + 8, (uint16_t)format);
    put_u16(fmt + 10, 2);
    put_u32(fmt + 12, 48000);
    uint16_t align = (uint16_t)(2 * bits / 8);
    put_u32(fmt + 16, 48000u * align);
    put_u16(fmt + 20, align);
    put_u16(fmt + 22, (uint16_t)bits);
    unsigned char data_header[8] = {'d','a','t','a'};
    put_u32(data_header + 4, data_bytes);
    int failed = fwrite(header, 1, sizeof(header), file) != sizeof(header) ||
                 fwrite(junk, 1, sizeof(junk), file) != sizeof(junk) ||
                 fwrite(fmt, 1, sizeof(fmt), file) != sizeof(fmt) ||
                 fwrite(data_header, 1, sizeof(data_header), file) != sizeof(data_header) ||
                 fwrite(data, 1, data_bytes, file) != data_bytes;
    if (fclose(file) != 0) failed = 1;
    return failed ? -1 : 0;
}

static int check_case(const char *label, int format, int bits,
                      const unsigned char *data, uint32_t data_bytes,
                      float expected0, float expected1) {
    char path[] = "/tmp/irodori-audio-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return -1;
    close(fd);
    int result = write_fixture(path, format, bits, data, data_bytes);
    IroAudio audio;
    if (result == 0) result = iro_load_wav_mono(path, &audio);
    unlink(path);
    if (result != 0) {
        fprintf(stderr, "FAIL %s: load\n", label);
        return -1;
    }
    int pass = audio.sample_count == 2 && audio.sample_rate == 48000 &&
               audio.source_channels == 2 && audio.source_bits == bits &&
               fabsf(audio.samples[0] - expected0) < 2e-7f &&
               fabsf(audio.samples[1] - expected1) < 2e-7f;
    printf("%s %s: %.8f %.8f\n", pass ? "PASS" : "FAIL", label,
           audio.samples[0], audio.samples[1]);
    iro_audio_free(&audio);
    return pass ? 0 : -1;
}

static int check_resample(const char *label, int source_rate, int target_rate,
                          const float *expected, size_t expected_count) {
    const float input[17] = {
        -1.f, 0.272727281f, -0.545454562f, 0.727272749f, -0.0909090936f,
        -0.909090936f, 0.363636374f, -0.454545468f, 0.818181813f, 0.f,
        -0.818181813f, 0.454545468f, -0.363636374f, 0.909090936f,
        0.0909090936f, -0.727272749f, 0.545454562f,
    };
    float *output = NULL;
    size_t count = 0;
    if (iro_resample_sinc(input, 17, source_rate, target_rate,
                          &output, &count) != 0) {
        fprintf(stderr, "FAIL %s: resample\n", label);
        return -1;
    }
    float worst = 0.0f;
    for (size_t i = 0; i < count && i < expected_count; i++) {
        float error = fabsf(output[i] - expected[i]);
        if (error > worst) worst = error;
    }
    int pass = count == expected_count && worst < 2e-6f;
    printf("%s %s: %zu sample, max err %.8g\n",
           pass ? "PASS" : "FAIL", label, count, worst);
    free(output);
    return pass ? 0 : -1;
}

static int check_normalize(const char *label, int kind, size_t count,
                           float expected_lufs, float expected_gain,
                           float expected_peak) {
    const float pi = 3.14159265358979323846f;
    float *samples = calloc(count, sizeof(float));
    if (!samples) return -1;
    for (size_t i = 0; i < count; i++) {
        float t = (float)i / 48000.0f;
        if (kind == 0)
            samples[i] = 0.31f * sinf(2 * pi * 233 * t) +
                         0.17f * sinf(2 * pi * 3101 * t) +
                         0.03f * cosf(2 * pi * 47 * t);
        else if (kind == 1)
            samples[i] = 0.2f * sinf(2 * pi * 440 * t);
    }
    if (kind == 2) samples[100] = 0.01f;
    float lufs = 0.0f, gain = 0.0f;
    int result = iro_normalize_loudness(samples, count, 48000, -16.0f,
                                        &lufs, &gain);
    float peak = 0.0f;
    for (size_t i = 0; i < count; i++)
        if (fabsf(samples[i]) > peak) peak = fabsf(samples[i]);
    int pass = result == 0 && fabsf(lufs - expected_lufs) < 0.02f &&
               fabsf(gain - expected_gain) < 0.003f &&
               fabsf(peak - expected_peak) < 0.001f;
    printf("%s %s: LUFS %.5f gain %.6f peak %.6f\n",
           pass ? "PASS" : "FAIL", label, lufs, gain, peak);
    free(samples);
    return pass ? 0 : -1;
}

static int check_bad_wav(const char *label, const unsigned char *data, size_t size) {
    char path[] = "/tmp/irodori-audio-bad-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return -1;
    int failed = size > 0 && write(fd, data, size) != (ssize_t)size;
    if (close(fd) != 0) failed = 1;
    IroAudio audio = {0};
    int result = failed ? -1 : iro_load_wav_mono(path, &audio);
    unlink(path);
    int pass = result != 0 && audio.samples == NULL && audio.sample_count == 0;
    iro_audio_free(&audio);
    printf("%s malformed WAV %s\n", pass ? "PASS" : "FAIL", label);
    return pass ? 0 : -1;
}

static int check_short_reference(const unsigned char *data, uint32_t data_bytes) {
    char path[] = "/tmp/irodori-reference-short-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return -1;
    close(fd);
    int result = write_fixture(path, 1, 16, data, data_bytes);
    IroAudio audio = {0};
    float lufs = 0.0f, gain = 0.0f;
    if (result == 0)
        result = iro_prepare_reference_wav(path, -16.0f, &audio, &lufs, &gain);
    unlink(path);
    int pass = result == 0 && audio.sample_count == 2 &&
               audio.sample_rate == 48000 && isfinite(lufs) && isfinite(gain);
    for (size_t i = 0; pass && i < audio.sample_count; i++)
        pass = isfinite(audio.samples[i]);
    printf("%s very-short reference: %zu samples\n",
           pass ? "PASS" : "FAIL", audio.sample_count);
    iro_audio_free(&audio);
    return pass ? 0 : -1;
}

static int check_nonfinite_wav_write(void) {
    char path[] = "/tmp/irodori-wav-nonfinite-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return -1;
    close(fd);
    const float samples[4] = {NAN, INFINITY, -INFINITY, 0.25f};
    int result = iro_write_wav_pcm16(path, samples, 4, 48000);
    IroAudio audio = {0};
    if (result == 0) result = iro_load_wav_mono(path, &audio);
    unlink(path);
    int pass = result == 0 && audio.sample_count == 4 &&
               audio.samples[0] == 0.0f && audio.samples[1] == 0.0f &&
               audio.samples[2] == 0.0f && fabsf(audio.samples[3] - 0.25f) < 1e-5f;
    printf("%s non-finite WAV output sanitization\n", pass ? "PASS" : "FAIL");
    iro_audio_free(&audio);
    return pass ? 0 : -1;
}

static int check_wav_write_failure(void) {
    if (access("/dev/full", F_OK) != 0) {
        printf("PASS WAV write failure: /dev/full unavailable (skipped)\n");
        return 0;
    }
    const float sample = 0.0f;
    int pass = iro_write_wav_pcm16("/dev/full", &sample, 1, 48000) != 0;
    printf("%s WAV write failure propagation\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : -1;
}

int main(void) {
    unsigned char pcm16[8];
    put_u16(pcm16 + 0, (uint16_t)32767); put_u16(pcm16 + 2, (uint16_t)-32768);
    put_u16(pcm16 + 4, (uint16_t)-16384); put_u16(pcm16 + 6, (uint16_t)16384);

    unsigned char pcm24[12] = {
        0xff,0xff,0x7f, 0x00,0x00,0x80,
        0x00,0x00,0x40, 0x00,0x00,0xc0,
    };
    unsigned char pcm32[16];
    put_u32(pcm32 + 0, 0x7fffffffu); put_u32(pcm32 + 4, 0x80000000u);
    put_u32(pcm32 + 8, 0x40000000u); put_u32(pcm32 + 12, 0xc0000000u);

    const float fvalues[4] = {0.75f, -0.25f, -1.0f, 0.5f};
    unsigned char float32[16];
    for (int i = 0; i < 4; i++) {
        uint32_t raw;
        memcpy(&raw, &fvalues[i], sizeof(raw));
        put_u32(float32 + i * 4, raw);
    }
    const float golden_44100_48000[19] = {
        -0.98008436f, 0.221726686f, -0.459363163f, 0.35580349f,
        0.542465568f, -1.02525485f, -0.0839734077f, 0.0373734832f,
        -0.208208293f, 0.971961737f, -0.385911644f, -0.672030687f,
        0.438629955f, -0.342039734f, 0.71911937f, 0.521736205f,
        -0.832132936f, 0.141387343f, 0.415387809f,
    };
    const float golden_16000_48000[51] = {
        -0.980084479f, -0.586860657f, -0.0381887369f, 0.252235562f,
        0.0991813466f, -0.296321094f, -0.528318107f, -0.334096968f,
        0.198151097f, 0.717401743f, 0.876784205f, 0.555798233f,
        -0.0907776356f, -0.736243606f, -1.0558207f, -0.900165856f,
        -0.390381396f, 0.138282388f, 0.349526882f, 0.148318931f,
        -0.239289865f, -0.440439463f, -0.229253158f, 0.299410611f,
        0.809252799f, 0.964749992f, 0.644979f, 5.66575342e-10f,
        -0.644979119f, -0.964749932f, -0.809252799f, -0.299410552f,
        0.229253262f, 0.440438837f, 0.239140049f, -0.14875935f,
        -0.349324316f, -0.135572165f, 0.394962311f, 0.898727238f,
        1.04055166f, 0.716480851f, 0.0947450548f, -0.511378169f,
        -0.823374808f, -0.724529207f, -0.300548404f, 0.209076032f,
        0.538406014f, 0.547401607f, 0.301757008f,
    };
    const float golden_48000_16000[6] = {
        -0.26627022f, -0.052214209f, -0.0816919133f,
        0.00413893722f, 0.0698599443f, 0.0864886194f,
    };

    int failed = 0;
    failed |= check_case("PCM16 stereo", 1, 16, pcm16, sizeof(pcm16),
                         -1.0f / 65536.0f, 0.0f) != 0;
    failed |= check_case("PCM24 stereo", 1, 24, pcm24, sizeof(pcm24),
                         -1.0f / 16777216.0f, 0.0f) != 0;
    failed |= check_case("PCM32 stereo", 1, 32, pcm32, sizeof(pcm32),
                         -1.0f / 4294967296.0f, 0.0f) != 0;
    failed |= check_case("float32 stereo", 3, 32, float32, sizeof(float32),
                         0.25f, -0.25f) != 0;
    failed |= check_resample("44.1k -> 48k", 44100, 48000,
                             golden_44100_48000, 19) != 0;
    failed |= check_resample("16k -> 48k", 16000, 48000,
                             golden_16000_48000, 51) != 0;
    failed |= check_resample("48k -> 16k", 48000, 16000,
                             golden_48000_16000, 6) != 0;
    failed |= check_normalize("normalize tone", 0, 48000,
                              -11.639428f, 0.605301f, 0.308470f) != 0;
    failed |= check_normalize("normalize short", 1, 4800,
                              -23.739218f, 2.437591f, 0.487518f) != 0;
    failed |= check_normalize("normalize peak safety", 2, 48000,
                              -70.0f, 100.0f, 1.0f) != 0;
    failed |= check_short_reference(pcm16, sizeof(pcm16)) != 0;
    failed |= check_nonfinite_wav_write() != 0;
    failed |= check_wav_write_failure() != 0;
    const unsigned char truncated_riff[] = {'R','I','F','F',0,0,0,0,'W','A'};
    failed |= check_bad_wav("empty", NULL, 0) != 0;
    failed |= check_bad_wav("truncated RIFF", truncated_riff,
                            sizeof(truncated_riff)) != 0;
    return failed ? 1 : 0;
}
