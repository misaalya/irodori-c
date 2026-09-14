# Irodori C — runtime assets

- `weights/tokenizer.bin` — compiled ModernBERT-ja Unigram tokenizer.
- `weights/dacvae_decoder.safetensors`, `weights/dacvae_encoder.safetensors` —
  Semantic-DACVAE-Japanese-32dim codec exported to the engine layout.

Derived from [Aratako/Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small)
and [Aratako/Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim)
(both MIT). The main checkpoint `model.safetensors` is not included; use
`download-model.sh` from the binary release or the Hugging Face page.
Extract next to the binary release so that `weights/` sits beside `bin/`.
