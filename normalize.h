/* normalize.h — port irodori_tts/text_normalization.py */
#ifndef IRO_NORMALIZE_H
#define IRO_NORMALIZE_H

/* return string malloc'd (UTF-8) hasil normalisasi; caller free() */
char *iro_normalize(const char *text);

#endif
