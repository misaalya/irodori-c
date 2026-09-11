/* audio.h — tail trimming and dependency-free PCM WAV output. */
#ifndef IRO_AUDIO_H
#define IRO_AUDIO_H

#include <stddef.h>

typedef struct {
    float *samples;
    size_t sample_count;
    int sample_rate;
    int source_channels;
    int source_bits;
} IroAudio;

/* Load RIFF/WAVE PCM16/24/32 or IEEE float32 and downmix channels to mono.
   The caller owns audio->samples and releases it with iro_audio_free(). */
int iro_load_wav_mono(const char *path, IroAudio *audio);
void iro_audio_free(IroAudio *audio);

/* Hann-windowed sinc resampler compatible with torchaudio's default
   lowpass_filter_width=6 and rolloff=0.99. Output is always newly allocated. */
int iro_resample_sinc(const float *input, size_t input_count,
                      int source_rate, int target_rate,
                      float **output, size_t *output_count);

/* In-place ITU-R BS.1770-4 normalization used by audiotools at 48 kHz,
   followed by peak safety. measured_db/applied_gain may be NULL. */
int iro_normalize_loudness(float *samples, size_t sample_count, int sample_rate,
                           float target_db, float *measured_db,
                           float *applied_gain);

/* Complete reference front-end used before DACVAE encode. */
int iro_prepare_reference_wav(const char *path, float target_db,
                              IroAudio *audio, float *measured_db,
                              float *applied_gain);

int iro_find_flattening_point(const float *latent, int frames, int channels,
                              float target_value, int window_size,
                              float std_threshold, float mean_threshold);

size_t iro_trimmed_sample_count(const float *latent, int frames, int channels,
                                size_t target_samples, int hop_length,
                                int window_size, float std_threshold,
                                float mean_threshold);

int iro_write_wav_pcm16(const char *path, const float *samples,
                        size_t sample_count, int sample_rate);

#endif
