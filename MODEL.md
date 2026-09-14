# Irodori-TTS — Model Reference (C Engine Port)

Model: `Aratako/Irodori-TTS-v4.1-Small` (0.8B params, FP32 safetensors, 714 tensor)
Codec: `Aratako/Semantic-DACVAE-Japanese-32dim` (DACVAE, latent 32-dim, output 48 kHz)
Tokenizer: `sbintuisions/modernbert-ja-310m` fast tokenizer (Unigram, dibundel di repo model)

Dokumen ini adalah **single source of truth** untuk port pure-C — semua angka berasal dari
metadata `config_json` checkpoint asli (bukan tebakan). Status verifikasi ditandai:
✅ terverifikasi dari kode/checkpoint, ⚠️ perlu verifikasi saat implementasi.

---

## 1. Pipeline Inference

```
text (JP) ──► normalize_text ──► Unigram tokenizer (+BOS) ──► ModernBERT-ja backbone (25L, 768d)
                            │                                              │
                            │                             residual_mlp projector (768→512)
                            │                                              │
                            │                ┌─────────────────────────────┼──────────┐
                            │                ▼                             ▼          ▼
                            │         text_state [S,512]   caption_state [Sc,512]  speaker_state [Sr,768]
                            │                │                             │          ▲
                            ▼                │                             │          │ ref audio → DACVAE encode
                duration predictor (token_sum_dual_adarn_zero, 3L, 1024d)  │          │ → speaker encoder 8L
                            │  → frame count N                              │          │ (opsional, voice cloning)
                            ▼                                                │          │
                x0 ~ N(0,1) [N,32] (seed)                                    │          │
                            │                                               │          │
                Euler RF sampling: t = 1→0, default 40 steps, CFG independent │          │
                    per-step: DiT forward (12 blocks, 1280d) → velocity v    │          │
                    x_t += v * (t_next − t)                                 │          │
                            │                                                │          │
                latent final [N,32] ◄──────── conditioning context KV ───────┘──────────┘
                            │
                DACVAE decoder (conv net) → waveform 48 kHz mono
                            │
                trim tail (window=20, std<0.05, |mean|<0.1) ──► WAV
```

Tiga sumber conditioning DiT (v4.1 menyatukan base + VoiceDesign):
- **text** (wajib): output projector teks
- **speaker** (opsional): reference audio → latents → speaker encoder; tanpa ref → null
- **caption** (opsional): teks gaya/emosi → backbone + projector sama (tensor terpisah)

## 2. Konfigurasi Inti (dari `config_json` checkpoint — ✅)

| Parameter | Nilai |
|---|---|
| latent_dim | 32 |
| latent_patch_size | 1 |
| model_dim (DiT) | 1280 |
| DiT num_layers | 12 |
| DiT num_heads | 20 (head_dim 64) |
| mlp_ratio | 2.875 → hidden 3680 (SwiGLU) |
| timestep_embed_dim | 512 |
| adaln_rank | 192 |
| norm_eps | 1e-5 |
| text_vocab_size | 102400 |
| text_dim (output projector) | 512 |
| caption_dim | 512 (backbone & tokenizer sama dengan text) |
| speaker_dim | 768, speaker_layers 8, speaker_heads 12, speaker_patch_size 4 |
| use_duration_predictor | true, hidden 1024, layers 3, attn heads 8 |
| duration_architecture | `token_sum_dual_adarn_zero_no_aux` |
| duration_token_init_frames | 9.0 |
| duration_aux_dim | 14 (training only, tidak dipakai inference) |
| max_text_len / max_caption_len | 256 / 512 |
| ref_max_seconds | 120.0 |

## 3. ModernBERT-ja Backbone (text + caption encoder)

`sbintuisions/modernbert-ja-310m`, `ModernBertForMaskedLM` backbone (✅ dari `text_encoder_config_json`):

| Parameter | Nilai |
|---|---|
| hidden_size | 768 |
| num_hidden_layers | 25 |
| num_attention_heads | 12 (head_dim 64) |
| intermediate_size | 3072 |
| vocab_size | 102400 |
| norm_eps | 1e-5, norm_bias false |
| hidden_activation | gelu |
| attention_bias | false |
| layer_types | `full_attention` tiap layer ke-3; lainnya `sliding_attention` |
| sliding window | 64 posisi ke kiri/kanan (total konfigurasi `local_attention=128`) |
| positional embedding | split-half RoPE; theta full=160000, sliding=10000 |

Struktur layer ModernBERT: **Wqkv fused + Wo** (bukan Q/K/V terpisah), LayerNorm,
exact GELU MLP (Wi fused + Wo), residual. Output `last_hidden_state [S,768]` lalu di-mask:
`state * mask` (✅ `PretrainedTextBackbone.forward`).

Detail rotary, urutan norm, embedding, dan seluruh 25 layer sudah diverifikasi di P1
dengan golden dump per-tensor; final max abs err C vs PyTorch **1.91e-5**.

## 4. Projector (residual_mlp, 768→512)

`PretrainedConditionProjector`, `projector_type="residual_mlp"`, `hidden_ratio=2.0` (✅):
- `projector`: Linear 768→512 (bias)
- `residual_norm`: RMSNorm(768)
- `residual_up`: Linear 768→1024 (bias)
- `residual_down`: Linear 1024→512 (bias)
- forward: `out = projector(x) + residual_down(silu(residual_up(RMSNorm(x))))`,
  lalu mask; hasilnya masuk `text_norm` RMSNorm (✅ golden P1).

Dipakai 2×: cabang text dan cabang caption.

## 5. DiT Blocks (12×)

Per block (✅ dari nama tensor checkpoint):

**Attention (joint, 3-cabang context):**
- self: `wq/wk/wv/wo` [1280,1280] tanpa bias
- `q_norm`/`k_norm`: RMSNorm per-head [20,64] (QK-norm)
- rotary: **half-RoPE** — `_apply_rotary_half` membagi dimensi head-index
  (`dim=-2`), jadi 10 head pertama di-rotate dan 10 head terakhir passthrough
- context projections: `wk_text/wv_text` [1280,512], `wk_speaker/wv_speaker` [1280,768],
  `wk_caption/wv_caption` [1280,512]
- `gate`: Linear [1280,1280] — output attention × `sigmoid(gate(x))` (Echo-style)
- SDPA: q/k/v self + concat context [k_self,k_text,(k_speaker),(k_caption)], mask per-segmen

**MLP:** SwiGLU `w1/w3` [3680,1280], `w2` [1280,3680]

## 6. Duration Predictor (`token_sum_dual_adarn_zero_no_aux`)

Menentukan panjang output otomatis (✅ config + struktur kode):
- `token_input_proj`: Linear 512→1024
- 3× `DurationSwiGLUBlock` (dim 1024): SwiGLU + AdaRN-Zero conditioning dari
  speaker (768) dan caption (512, masked_mean pooling)
- `token_out_norm`: RMSNorm(1024), `token_out_proj`: Linear 1024→1
- per token: `frames_i = softplus(token_out_proj(token_out_norm(h_i)))`
- output predictor: `log_frames = log1p(Σ frames_i * text_mask_i)`
- runtime: `N = round(expm1(log_frames) * duration_scale)`, lalu clamp ke
  `ceil(min_seconds * 25)` .. `floor(max_seconds * 25)` (✅ 20/20 golden P1)

## 7. Sampling (Rectified Flow Euler + CFG)

`sample_euler_rf_cfg` (rf.py):
- x0 = randn [N,32] — ⚠️ keputusan P1: noise diekstrak dari golden dump PyTorch
  (port RNG MT19937-torch opsional nanti untuk seed-identik)
- schedule dimulai dari `init_scale=0.999`, jadi timestep pertama adalah 0.999
  (bukan 1.0; ✅ `cond_module` golden P1)
- schedule: t = 1→0; mode `linear` / `sway` (sway_coeff −1.0); formula sway
  sudah diverifikasi dari `rf.py`, sementara P1 C mengimplementasikan jalur linear
- per step: `v_cond = forward(x_t, t, context penuh)`; CFG `independent`:
  `v = v_cond + Σ_b scale_b·(v_cond − v_uncond_b)`; jalur text-only
  dan transisi batch 2→1 sudah lolos golden Euler C
  (default scales: text 3.0, caption 3.0, speaker 5.0; branch tanpa kondisi di-skip)
- **Optimisasi CFG aktif default (✅ golden trace):** uncond di-skip saat `t < cfg_min_t`
  (0.5) — batch cond+uncond (2) untuk t ≥ 0.5, batch cond saja (1) untuk t < 0.5
- **Text di-pad kanan ke `max_text_len` = 256 token** (✅ golden trace: text_state (1,256,512))
- `x_t += v·(t_next − t)`; hasil akhir = x di t=0
- trim tail: index pertama dengan trailing window flat (window 20, std<0.05, |mean|<0.1)

## 8. DACVAE Codec (32-dim, 48 kHz) — ✅ kwargs terverifikasi dari weights.pth

- `encoder_dim 64, encoder_rates [2,8,10,12]` (hop 1920), `decoder_dim 1536`,
  `decoder_rates [12,10,8,2]`, `latent_dim 1024` (intermediate), `codebook_dim 32`,
  `n_codebooks 16, codebook_size 1024`, `sample_rate 48000`
- Latent rate: 48000/1920 = **25 Hz** (40 ms/frame) (✅ dari kwargs + run nyata)
- Decoder aktif: latents [N,32] → quantizer.out_proj 1×1 → Conv1d 7
  (1024→1536) → empat stage dengan rate [12,10,8,2]. Tiap stage berisi
  Snake → ConvTranspose1d → tiga residual Snake/Conv1d dengan dilation
  [1,3,9]. Output 96 channel → Snake → Conv1d 7 → tanh → mono 48 kHz.
- Cabang watermark/LSTM tidak masuk graph inference Irodori karena decoder
  memakai `alpha=0`; passthrough akhirnya tetap menjalankan Snake+Conv1d+tanh.
- Tool `export_dacvae_decoder.py` melebur weight normalization secara offline dan
  menyimpan 91 tensor aktif (250 MiB). Runtime C memakai layout aktivasi
  time-major, im2col terpotong, SGEMM, dan overlap-add ConvTranspose.
- **Encoder** (voice cloning): waveform → encoder blocks → quantizer.in_proj → mean
  (deterministic_encode=True) (✅ codec.py)
- `weights.pth` (torch pickle, 317 tensor) dikonversi → `irodori-c/weights/dacvae.safetensors`
  via `tools/dump_dacvae_safetensors.py` (✅ teruji; loader C membaca 317 tensor, 0 masalah)

## 9. Tokenizer (Unigram) — ✅ inspeksi `tokenizer.json`

- model: **Unigram**, vocab 102400, byte_fallback, unk_id
- normalizer: **null** (normalisasi Jepang dilakukan `normalize_text` sebelum tokenisasi)
- pre_tokenizer: Metaspace (`▁`, prepend_scheme "never", split=false)
- added_tokens: 18 special tokens; BOS `<s>` di-prepend (add_bos=true)
- `PretrainedTextTokenizer.encode`: `add_special_tokens=False` + insert BOS manual (✅)
- Port C: Viterbi unigram (log-prob) + byte fallback (hex `<0xXX>` → byte) + prepend BOS

## 10. Text Normalization (✅ text_normalization.py, murni regex)

Port C: `SIMPLE_REPLACE_MAP` + `REGEX_REPLACE_MAP` + `strip_outer_brackets` + normalisasi
Unicode NFKC via vendored utf8proc, termasuk aturan `"..." → "…"` (✅ 20/20 golden).

## 11. Inventory Weight (checkpoint utama — 714 tensor, ✅ terverifikasi via loader C)

```
blocks.{0..11}.attention.{wq,wk,wv,wo,gate}.weight        [1280,1280]
blocks.{0..11}.attention.{wk,wv}_{text,caption}.weight    [1280,512]
blocks.{0..11}.attention.{wk,wv}_speaker.weight           [1280,768]
blocks.{0..11}.attention.{q,k}_norm.weight                [20,64]
blocks.{0..11}.attention_adaln.{shift,scale,gate}_down.weight [192,1280]
blocks.{0..11}.attention_adaln.{shift,scale,gate}_up.{weight,bias} [1280,192]/[1280]
blocks.{0..11}.mlp.{w1,w3}.weight [3680,1280]  w2.weight [1280,3680]
blocks.{0..11}.mlp_adaln.*        (sama dengan attention_adaln)

in_proj.weight/bias [1280,32]        # latent → dim 1280
out_norm.weight [1280]
out_proj.weight/bias [32,1280]       # → velocity 32
cond_module.{0,2,4}.weight           # 512→1280→1280→3840 (AdaLN cond MLP)
text_norm.weight / caption_norm.weight [512] / speaker_norm.weight [768]

pretrained_text_backbone.backbone.embeddings.tok_embeddings.weight [102400,768]
pretrained_text_backbone.backbone.embeddings.norm.weight [768]
pretrained_text_backbone.backbone.layers.{0..24}.attn.{Wqkv,Wo}.weight  [2304,768]/[768,768]
pretrained_text_backbone.backbone.layers.{0..24}.attn_norm.weight [768]
pretrained_text_backbone.backbone.layers.{0..24}.mlp.{Wi,Wo}.weight   [6144,768]/[768,3072]
pretrained_text_backbone.backbone.layers.{0..24}.mlp_norm.weight [768]
pretrained_text_backbone.backbone.final_norm.weight [768]

text_encoder.{projector.weight/bias, residual_norm, residual_up, residual_down}.*   # 768→512
caption_encoder.*  (struktur identik, tensor terpisah)

speaker_encoder.in_proj.weight/bias [768,128]           # patch 4×32=128 → 768
speaker_encoder.blocks.{0..7}.attention.{wq,wk,wv,wo,gate}.weight [768,768]
speaker_encoder.blocks.{0..7}.attention.{q,k}_norm.weight [12,64]
speaker_encoder.blocks.{0..7}.{attention_norm,mlp_norm}.weight [768]
speaker_encoder.blocks.{0..7}.mlp.{w1,w3}.weight [1996,768]  w2 [768,1996]
(catatan: speaker encoder TIDAK punya context projections — self-attn only)

duration_predictor.token_input_proj.weight/bias [1024,512]
duration_predictor.token_blocks.{0..2}.norm.weight [1024]
duration_predictor.token_blocks.{0..2}.modulation.weight/bias [3072,768]        # speaker cond
duration_predictor.token_blocks.{0..2}.caption_modulation.weight/bias [3072,512] # caption cond
duration_predictor.token_blocks.{0..2}.mlp.{w1,w3}.weight [1024,1024]  w2 [1024,1024]
duration_predictor.token_out_norm.weight [1024]
duration_predictor.token_out_proj.weight/bias [1,1024]/[1]
duration_predictor.null_speaker [768]  null_caption [512]
```


**AdaLN (per sub-block, 2×/layer):** `attention_adaln` + `mlp_adaln` — LowRankAdaLN rank 192:
- `shift_down/scale_down/gate_down` [192,1280] no-bias
- `shift_up/scale_up/gate_up` [1280,192] **dengan bias**
- sumber conditioning (✅ terverifikasi golden trace + kode `forward_with_encoded_conditions`):
  `cond_embed = cond_module(t_embed(t))` dengan t_embed sinusoidal 512-d;
  `cond_module` = Sequential(Linear 512→1280, Linear 1280→1280, Linear 1280→3840)
  → cond_embed [B,3840] broadcast ke semua blok; tiap sub-block LowRankAdaLN memecah
  shift/scale/gate dari cond_embed ini

**Final layers (✅ terverifikasi dari inventory):**
- `in_proj` [1280,32] + bias: latent 32 → model dim 1280 (patch size 1)
- `out_norm` [1280] (RMSNorm) → `out_proj` [32,1280] + bias: → velocity dim 32
- `cond_module.{0,2,4}`: Sequential Linear 512→1280, 1280→1280, 1280→3840 (3840 = 3×1280
  shift/scale/gate per sub-block); inputnya hanya sinusoidal timestep embedding 512-d
  (✅ kode `forward_with_encoded_conditions` + golden trace)
- `text_norm` / `caption_norm` [512]: RMSNorm setelah projector (masuk ke kondisi)
- `speaker_norm` [768]: RMSNorm keluaran speaker encoder

| max_position_embeddings (backbone) | 8192 |
