#!/usr/bin/env python3
"""Create deterministic per-layer fixtures for the reference speaker encoder."""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Irodori-TTS"))

from irodori_tts.config import ModelConfig  # noqa: E402
from irodori_tts.model import DurationPredictor, RMSNorm, ReferenceLatentEncoder  # noqa: E402

DEFAULT_MODEL = Path.home() / (
    ".cache/huggingface/hub/models--Aratako--Irodori-TTS-v4.1-Small/"
    "snapshots/2b28324dc263ed5e6638b3cf3dd94c82ead07b4b/model.safetensors"
)
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "golden" / "speaker_test"


def save(path: Path, tensor: torch.Tensor) -> None:
    tensor.detach().float().squeeze(0).contiguous().cpu().numpy().tofile(path)
    print(f"  {path.stem}: {tuple(tensor.shape)}")


def main() -> int:
    checkpoint = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else 17
    if frames < 4:
        raise ValueError("latent frame count must be at least 4")
    output.mkdir(parents=True, exist_ok=True)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        raw_cfg = json.loads(metadata["config_json"])
        allowed = {field.name for field in fields(ModelConfig)}
        cfg = ModelConfig(**{key: value for key, value in raw_cfg.items() if key in allowed})
        state = {
            key.removeprefix("speaker_encoder."): handle.get_tensor(key)
            for key in handle.keys()
            if key.startswith("speaker_encoder.")
        }
        norm_weight = handle.get_tensor("speaker_norm.weight")
        duration_state = {
            key.removeprefix("duration_predictor."): handle.get_tensor(key)
            for key in handle.keys()
            if key.startswith("duration_predictor.")
        }
        dit_speaker_wk = handle.get_tensor("blocks.0.attention.wk_speaker.weight")
        dit_speaker_wv = handle.get_tensor("blocks.0.attention.wv_speaker.weight")
        dit_k_norm = handle.get_tensor("blocks.0.attention.k_norm.weight")

    encoder = ReferenceLatentEncoder(cfg).eval()
    encoder.load_state_dict(state)
    norm = RMSNorm(cfg.speaker_dim, eps=cfg.norm_eps).eval()
    norm.weight.data.copy_(norm_weight)
    duration = DurationPredictor(
        text_dim=cfg.text_dim,
        aux_dim=cfg.duration_aux_dim,
        hidden_dim=cfg.duration_hidden_dim,
        layers=cfg.duration_layers,
        dropout=cfg.duration_dropout,
        speaker_dim=cfg.speaker_dim,
        speaker_fusion=cfg.duration_speaker_fusion,
        caption_dim=cfg.caption_dim_resolved,
        caption_fusion=cfg.duration_caption_fusion,
        caption_pooling=cfg.duration_caption_pooling,
        attention_heads=cfg.duration_attention_heads,
        norm_eps=cfg.norm_eps,
        architecture=cfg.duration_architecture,
        token_init_frames=cfg.duration_token_init_frames,
    ).eval()
    duration.load_state_dict(duration_state)

    # The default 17 frames deliberately checks truncation to four full patches.
    t = torch.arange(frames, dtype=torch.float32)[:, None]
    d = torch.arange(cfg.latent_dim, dtype=torch.float32)[None, :]
    latent = (0.31 * torch.sin(t * 0.17 + d * 0.11) +
              0.07 * torch.cos(t * 0.031 - d * 0.23))
    latent.numpy().tofile(output / "latent.f32")
    usable = (frames // cfg.speaker_patch_size) * cfg.speaker_patch_size
    patched = latent[:usable].reshape(1, usable // cfg.speaker_patch_size, -1)
    mask = torch.ones(patched.shape[:2], dtype=torch.bool)

    with torch.inference_mode():
        x = encoder.in_proj(patched) / 6.0
        save(output / "speaker_in.f32", x)
        freqs = encoder._rope_freqs(x.shape[1], x.device)  # noqa: SLF001
        for index, block in enumerate(encoder.blocks):
            x = block(x, mask=mask, freqs_cis=freqs)
            save(output / f"speaker_b{index}.f32", x)
        x = norm(x)
        save(output / "speaker_norm.f32", x)
        mean = x.mean(dim=1, keepdim=True)
        speaker_state = torch.cat([mean, x], dim=1)
        save(output / "speaker_state.f32", speaker_state)
        text_golden = ROOT / "irodori-c" / "golden" / "seed42_steps8"
        token_count = (text_golden / "token_ids.i64").stat().st_size // 8
        text_array = np.fromfile(text_golden / "text_state.f32", dtype=np.float32)
        text_state = torch.from_numpy(text_array.reshape(1, 256, cfg.text_dim).copy())
        text_state = text_state[:, :token_count]
        save(output / "duration_text.f32", text_state)
        duration_out = duration(
            text_state,
            text_mask=torch.ones((1, token_count), dtype=torch.bool),
            aux_features=torch.zeros((1, cfg.duration_aux_dim)),
            speaker_state=speaker_state,
            speaker_mask=torch.ones(speaker_state.shape[:2], dtype=torch.bool),
            has_speaker=torch.ones(1, dtype=torch.bool),
            caption_state=None,
            caption_mask=None,
            has_caption=torch.zeros(1, dtype=torch.bool),
        )
        save(output / "duration_speaker_out.f32", duration_out)
        speaker_k = F.linear(speaker_state, dit_speaker_wk).reshape(
            1, speaker_state.shape[1], cfg.num_heads, cfg.model_dim // cfg.num_heads
        )
        speaker_k = speaker_k * torch.rsqrt(
            speaker_k.square().mean(dim=-1, keepdim=True) + cfg.norm_eps
        ) * dit_k_norm
        speaker_v = F.linear(speaker_state, dit_speaker_wv).reshape(
            1, speaker_state.shape[1], cfg.model_dim
        )
        save(output / "dit_speaker_k.f32", speaker_k.reshape(1, speaker_state.shape[1], -1))
        save(output / "dit_speaker_v.f32", speaker_v)
    print(f"saved speaker golden: {output} ({frames} latent -> {speaker_state.shape[1]} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
