/* audio.c — exact Irodori trim heuristic and portable little-endian WAV. */
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "audio.h"

static uint16_t read_u16le_bytes(const unsigned char *p) {
    return (uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8);
}

static uint32_t read_u32le_bytes(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int read_exact(FILE *file, void *dst, size_t bytes) {
    return bytes == 0 || fread(dst, 1, bytes, file) == bytes ? 0 : -1;
}

static int skip_bytes(FILE *file, uint32_t bytes) {
    while (bytes) {
        long step = bytes > (uint32_t)LONG_MAX ? LONG_MAX : (long)bytes;
        if (fseek(file, step, SEEK_CUR) != 0) return -1;
        bytes -= (uint32_t)step;
    }
    return 0;
}

static float decode_wav_sample(const unsigned char *p, int format, int bits) {
    if (format == 3) {
        uint32_t raw = read_u32le_bytes(p);
        float value;
        memcpy(&value, &raw, sizeof(value));
        return isfinite(value) ? value : 0.0f;
    }
    if (bits == 16)
        return (float)(int16_t)read_u16le_bytes(p) * (1.0f / 32768.0f);
    if (bits == 24) {
        int32_t value = (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                                  ((uint32_t)p[2] << 16));
        if (value & 0x00800000) value |= (int32_t)0xff000000;
        return (float)value * (1.0f / 8388608.0f);
    }
    return (float)(int32_t)read_u32le_bytes(p) * (1.0f / 2147483648.0f);
}

void iro_audio_free(IroAudio *audio) {
    if (!audio) return;
    free(audio->samples);
    memset(audio, 0, sizeof(*audio));
}

static int rate_gcd(int a, int b) {
    while (b) {
        int next = a % b;
        a = b;
        b = next;
    }
    return a;
}

int iro_resample_sinc(const float *input, size_t input_count,
                      int source_rate, int target_rate,
                      float **output, size_t *output_count) {
    enum { LOWPASS_WIDTH = 6 };
    const float rolloff = 0.99f;
    if (!input || input_count == 0 || source_rate <= 0 || target_rate <= 0 ||
        !output || !output_count)
        return -1;
    *output = NULL;
    *output_count = 0;
    int divisor = rate_gcd(source_rate, target_rate);
    int orig = source_rate / divisor;
    int dest = target_rate / divisor;
    if (input_count > (SIZE_MAX - ((size_t)orig - 1u)) / (size_t)dest)
        return -1;
    size_t count = (input_count * (size_t)dest + (size_t)orig - 1u) /
                   (size_t)orig;
    if (count > SIZE_MAX / sizeof(float)) return -1;
    float *resampled = malloc(sizeof(float) * count);
    if (!resampled) return -1;
    if (source_rate == target_rate) {
        memcpy(resampled, input, sizeof(float) * input_count);
        *output = resampled;
        *output_count = input_count;
        return 0;
    }

    float base = (float)(orig < dest ? orig : dest) * rolloff;
    double width_value = ceil((double)LOWPASS_WIDTH * (double)orig /
                              (double)base);
    if (width_value > (double)(INT_MAX - orig) * 0.5) {
        free(resampled);
        return -1;
    }
    int width = (int)width_value;
    int kernel_size = 2 * width + orig;
    if ((size_t)dest > SIZE_MAX / (size_t)kernel_size ||
        (size_t)dest * (size_t)kernel_size > SIZE_MAX / sizeof(float)) {
        free(resampled);
        return -1;
    }
    size_t kernel_count = (size_t)dest * (size_t)kernel_size;
    float *kernel = malloc(sizeof(float) * kernel_count);
    if (!kernel) {
        free(resampled);
        return -1;
    }
    const float pi = 3.14159265358979323846f;
    for (int phase = 0; phase < dest; phase++) {
        float *phase_kernel = kernel + (size_t)phase * kernel_size;
        for (int tap = 0; tap < kernel_size; tap++) {
            float t = ((float)(tap - width) / (float)orig -
                       (float)phase / (float)dest) * base;
            if (t < -(float)LOWPASS_WIDTH) t = -(float)LOWPASS_WIDTH;
            if (t > (float)LOWPASS_WIDTH) t = (float)LOWPASS_WIDTH;
            float window = cosf(t * pi / (float)LOWPASS_WIDTH * 0.5f);
            window *= window;
            float angle = t * pi;
            float sinc = angle == 0.0f ? 1.0f : sinf(angle) / angle;
            phase_kernel[tap] = sinc * window * (base / (float)orig);
        }
    }
    for (size_t index = 0; index < count; index++) {
        size_t block = index / (size_t)dest;
        int phase = (int)(index % (size_t)dest);
        const float *phase_kernel = kernel + (size_t)phase * kernel_size;
        float sum = 0.0f;
        size_t anchor = block * (size_t)orig;
        int tap_begin = 0;
        size_t source_begin = anchor;
        if (anchor < (size_t)width) {
            tap_begin = width - (int)anchor;
            source_begin = 0;
        } else {
            source_begin -= (size_t)width;
        }
        if (tap_begin < kernel_size && source_begin < input_count) {
            size_t taps = (size_t)(kernel_size - tap_begin);
            size_t available = input_count - source_begin;
            if (taps > available) taps = available;
            const float *source = input + source_begin;
            const float *weights = phase_kernel + tap_begin;
            for (size_t tap = 0; tap < taps; tap++)
                sum += source[tap] * weights[tap];
        }
        resampled[index] = sum;
    }
    free(kernel);
    *output = resampled;
    *output_count = count;
    return 0;
}

static void biquad_inplace(float *samples, size_t count,
                           const float b[3], const float a[3]) {
    float x1 = 0.0f, x2 = 0.0f, y1 = 0.0f, y2 = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float x0 = samples[i];
        float y0 = b[0] * x0 + b[1] * x1 + b[2] * x2 -
                   a[1] * y1 - a[2] * y2;
        samples[i] = y0;
        x2 = x1; x1 = x0;
        y2 = y1; y1 = y0;
    }
}

int iro_normalize_loudness(float *samples, size_t sample_count, int sample_rate,
                           float target_db, float *measured_db,
                           float *applied_gain) {
    enum { RATE = 48000, BLOCK = 19200, STRIDE = 4800, MIN_SAMPLES = 24000 };
    static const float shelf_b[3] = {
        1.5351828864f, -2.6918040302f, 1.1984262633f,
    };
    static const float shelf_a[3] = {
        1.0f, -1.6906995866f, 0.7325047061f,
    };
    static const float highpass_b[3] = {
        0.9950442970f, -1.9900885940f, 0.9950442970f,
    };
    static const float highpass_a[3] = {
        1.0f, -1.9900762840f, 0.9901009041f,
    };
    if (!samples || sample_count == 0 || sample_rate != RATE ||
        !isfinite(target_db))
        return -1;
    size_t filtered_count = sample_count < MIN_SAMPLES ? MIN_SAMPLES : sample_count;
    if (filtered_count > SIZE_MAX / sizeof(float)) return -1;
    float *filtered = calloc(filtered_count, sizeof(float));
    if (!filtered) return -1;
    memcpy(filtered, samples, sizeof(float) * sample_count);
    biquad_inplace(filtered, filtered_count, shelf_b, shelf_a);
    biquad_inplace(filtered, filtered_count, highpass_b, highpass_a);

    size_t extra = filtered_count > BLOCK ? filtered_count - BLOCK : 0;
    size_t blocks = 1u + (extra + STRIDE - 1u) / STRIDE;
    double *energy = malloc(sizeof(double) * blocks);
    float *loudness = malloc(sizeof(float) * blocks);
    if (!energy || !loudness) {
        free(energy); free(loudness); free(filtered);
        return -1;
    }
    double absolute_sum = 0.0;
    size_t absolute_count = 0;
    for (size_t block = 0; block < blocks; block++) {
        size_t begin = block * STRIDE;
        double sumsq = 0.0;
        for (size_t i = 0; i < BLOCK; i++) {
            size_t index = begin + i;
            float value = index < filtered_count ? filtered[index] : 0.0f;
            sumsq += (double)value * value;
        }
        energy[block] = sumsq / (double)BLOCK;
        loudness[block] = energy[block] > 0.0
                              ? -0.691f + 10.0f * log10f((float)energy[block])
                              : -INFINITY;
        if (loudness[block] > -70.0f) {
            absolute_sum += energy[block];
            absolute_count++;
        }
    }
    double absolute_mean = absolute_count ? absolute_sum / absolute_count : 0.0;
    float relative_gate = absolute_mean > 0.0
                              ? -10.691f + 10.0f * log10f((float)absolute_mean)
                              : -INFINITY;
    double gated_sum = 0.0;
    size_t gated_count = 0;
    for (size_t block = 0; block < blocks; block++) {
        if (loudness[block] > -70.0f && loudness[block] > relative_gate) {
            gated_sum += energy[block];
            gated_count++;
        }
    }
    double gated_mean = gated_count ? gated_sum / gated_count : 0.0;
    float lufs = gated_mean > 0.0
                     ? -0.691f + 10.0f * log10f((float)gated_mean)
                     : -70.0f;
    if (!isfinite(lufs) || lufs < -70.0f) lufs = -70.0f;
    free(energy); free(loudness); free(filtered);

    float gain = expf((target_db - lufs) * 0.11512925464970229f);
    float peak = 0.0f;
    for (size_t i = 0; i < sample_count; i++) {
        samples[i] *= gain;
        float magnitude = fabsf(samples[i]);
        if (magnitude > peak) peak = magnitude;
    }
    if (isfinite(peak) && peak > 1.0f) {
        float safety = 1.0f / peak;
        for (size_t i = 0; i < sample_count; i++) samples[i] *= safety;
        gain *= safety;
    }
    if (measured_db) *measured_db = lufs;
    if (applied_gain) *applied_gain = gain;
    return 0;
}

int iro_prepare_reference_wav(const char *path, float target_db,
                              IroAudio *audio, float *measured_db,
                              float *applied_gain) {
    enum { CODEC_RATE = 48000 };
    if (!audio || iro_load_wav_mono(path, audio) != 0) return -1;
    if (audio->sample_rate != CODEC_RATE) {
        float *resampled = NULL;
        size_t resampled_count = 0;
        if (iro_resample_sinc(audio->samples, audio->sample_count,
                              audio->sample_rate, CODEC_RATE,
                              &resampled, &resampled_count) != 0) {
            iro_audio_free(audio);
            return -1;
        }
        free(audio->samples);
        audio->samples = resampled;
        audio->sample_count = resampled_count;
        audio->sample_rate = CODEC_RATE;
    }
    if (iro_normalize_loudness(audio->samples, audio->sample_count,
                               CODEC_RATE, target_db,
                               measured_db, applied_gain) != 0) {
        iro_audio_free(audio);
        return -1;
    }
    return 0;
}

int iro_load_wav_mono(const char *path, IroAudio *audio) {
    if (!path || !audio) return -1;
    memset(audio, 0, sizeof(*audio));
    FILE *file = fopen(path, "rb");
    if (!file) return -1;

    unsigned char riff[12];
    if (read_exact(file, riff, sizeof(riff)) != 0 ||
        memcmp(riff, "RIFF", 4) || memcmp(riff + 8, "WAVE", 4)) {
        fclose(file);
        return -1;
    }

    int format = 0, channels = 0, sample_rate = 0, bits = 0, block_align = 0;
    long data_offset = -1;
    uint32_t data_bytes = 0;
    for (;;) {
        unsigned char chunk[8];
        if (fread(chunk, 1, sizeof(chunk), file) != sizeof(chunk)) break;
        uint32_t size = read_u32le_bytes(chunk + 4);
        if (!memcmp(chunk, "fmt ", 4)) {
            unsigned char fmt[40] = {0};
            size_t kept = size < sizeof(fmt) ? size : sizeof(fmt);
            if (size < 16 || read_exact(file, fmt, kept) != 0 ||
                skip_bytes(file, size - (uint32_t)kept) != 0) {
                fclose(file);
                return -1;
            }
            format = read_u16le_bytes(fmt);
            channels = read_u16le_bytes(fmt + 2);
            uint32_t rate = read_u32le_bytes(fmt + 4);
            sample_rate = rate <= INT_MAX ? (int)rate : 0;
            block_align = read_u16le_bytes(fmt + 12);
            bits = read_u16le_bytes(fmt + 14);
            /* WAVE_FORMAT_EXTENSIBLE stores the real format tag at byte 24. */
            if (format == 0xfffe && size >= 40 && read_u16le_bytes(fmt + 16) >= 22)
                format = read_u16le_bytes(fmt + 24);
        } else if (!memcmp(chunk, "data", 4)) {
            data_offset = ftell(file);
            data_bytes = size;
            if (skip_bytes(file, size) != 0) {
                fclose(file);
                return -1;
            }
        } else if (skip_bytes(file, size) != 0) {
            fclose(file);
            return -1;
        }
        if ((size & 1u) && fseek(file, 1, SEEK_CUR) != 0) {
            fclose(file);
            return -1;
        }
    }

    int bytes_per_sample = bits / 8;
    int valid_format = format == 1 && (bits == 16 || bits == 24 || bits == 32);
    valid_format |= format == 3 && bits == 32;
    if (!valid_format || channels <= 0 || sample_rate <= 0 ||
        bytes_per_sample <= 0 || block_align != channels * bytes_per_sample ||
        data_offset < 0 || data_bytes == 0 || data_bytes % (uint32_t)block_align) {
        fclose(file);
        return -1;
    }
    size_t frames = data_bytes / (uint32_t)block_align;
    if (frames > SIZE_MAX / sizeof(float) ||
        fseek(file, data_offset, SEEK_SET) != 0) {
        fclose(file);
        return -1;
    }
    float *mono = malloc(sizeof(float) * frames);
    unsigned char *frame = malloc((size_t)block_align);
    if (!mono || !frame) {
        free(mono); free(frame); fclose(file);
        return -1;
    }
    for (size_t i = 0; i < frames; i++) {
        if (read_exact(file, frame, (size_t)block_align) != 0) {
            free(mono); free(frame); fclose(file);
            return -1;
        }
        double sum = 0.0;
        for (int channel = 0; channel < channels; channel++)
            sum += decode_wav_sample(frame + channel * bytes_per_sample,
                                     format, bits);
        mono[i] = (float)(sum / (double)channels);
    }
    free(frame);
    fclose(file);
    audio->samples = mono;
    audio->sample_count = frames;
    audio->sample_rate = sample_rate;
    audio->source_channels = channels;
    audio->source_bits = bits;
    return 0;
}

int iro_find_flattening_point(const float *latent, int frames, int channels,
                              float target_value, int window_size,
                              float std_threshold, float mean_threshold) {
    if (!latent || frames <= 0 || channels <= 0 || window_size <= 0)
        return frames > 0 ? frames : 0;

    /* Match torch.std(unbiased=False) and torch.mean over each [W,D]
       window.  Samples beyond the latent sequence are the same zero pad as
       inference_runtime.find_flattening_point. */
    const size_t window_values = (size_t)window_size * channels;
    for (int start = 0; start < frames; start++) {
        double sum = 0.0;
        double sumsq = 0.0;
        for (int offset = 0; offset < window_size; offset++) {
            int time = start + offset;
            if (time >= frames) continue;
            const float *row = latent + (size_t)time * channels;
            for (int channel = 0; channel < channels; channel++) {
                double value = row[channel];
                sum += value;
                sumsq += value * value;
            }
        }
        double mean = sum / (double)window_values;
        double variance = sumsq / (double)window_values - mean * mean;
        if (variance < 0.0) variance = 0.0;
        if (sqrt(variance) < std_threshold &&
            fabs(mean - target_value) < mean_threshold)
            return start;
    }
    return frames;
}

size_t iro_trimmed_sample_count(const float *latent, int frames, int channels,
                                size_t target_samples, int hop_length,
                                int window_size, float std_threshold,
                                float mean_threshold) {
    if (hop_length <= 0) return 0;
    int point = iro_find_flattening_point(
        latent, frames, channels, 0.0f, window_size,
        std_threshold, mean_threshold);
    size_t flattening_samples = (size_t)point * (size_t)hop_length;
    return flattening_samples < target_samples ? flattening_samples
                                                : target_samples;
}

static int write_u16le(FILE *file, uint16_t value) {
    unsigned char bytes[2] = {
        (unsigned char)(value & 0xffu),
        (unsigned char)((value >> 8) & 0xffu),
    };
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes) ? 0 : -1;
}

static int write_u32le(FILE *file, uint32_t value) {
    unsigned char bytes[4] = {
        (unsigned char)(value & 0xffu),
        (unsigned char)((value >> 8) & 0xffu),
        (unsigned char)((value >> 16) & 0xffu),
        (unsigned char)((value >> 24) & 0xffu),
    };
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes) ? 0 : -1;
}

static int16_t pcm16(float sample) {
    if (!isfinite(sample)) sample = 0.0f;
    if (sample > 1.0f) sample = 1.0f;
    if (sample < -1.0f) sample = -1.0f;
    return (int16_t)lrintf(sample * 32767.0f);
}

int iro_write_wav_pcm16(const char *path, const float *samples,
                        size_t sample_count, int sample_rate) {
    if (!path || !samples || sample_rate <= 0 ||
        sample_count > (UINT32_MAX - 36u) / 2u)
        return -1;
    FILE *file = fopen(path, "wb");
    if (!file) return -1;
    uint32_t data_bytes = (uint32_t)sample_count * 2u;
    int failed =
        fwrite("RIFF", 1, 4, file) != 4 ||
        write_u32le(file, 36u + data_bytes) != 0 ||
        fwrite("WAVEfmt ", 1, 8, file) != 8 ||
        write_u32le(file, 16u) != 0 ||
        write_u16le(file, 1u) != 0 ||
        write_u16le(file, 1u) != 0 ||
        write_u32le(file, (uint32_t)sample_rate) != 0 ||
        write_u32le(file, (uint32_t)sample_rate * 2u) != 0 ||
        write_u16le(file, 2u) != 0 ||
        write_u16le(file, 16u) != 0 ||
        fwrite("data", 1, 4, file) != 4 ||
        write_u32le(file, data_bytes) != 0;
    for (size_t i = 0; !failed && i < sample_count; i++)
        failed = write_u16le(file, (uint16_t)pcm16(samples[i])) != 0;
    if (fclose(file) != 0) failed = 1;
    return failed ? -1 : 0;
}
