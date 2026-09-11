#!/usr/bin/env python3
"""Create a compact independent text+speaker CFG Euler fixture."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Irodori-TTS"))

from irodori_tts.inference_runtime import (  # noqa: E402
    RuntimeKey,
    download_hf_checkpoint,
    get_cached_runtime,
)

CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
STEPS = 2
LATENT_FRAMES = 8


def save(path: Path, tensor: torch.Tensor) -> None:
    tensor.detach().float().contiguous().cpu().numpy().tofile(path)
    print(f"  {path.stem}: {tuple(tensor.shape)}")


def main() -> int:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "irodori-c" / "golden" / "speaker_test"
    )
    checkpoint = download_hf_checkpoint("Aratako/Irodori-TTS-v4.1-Small")
    torch.set_num_threads(2)
    runtime, _ = get_cached_runtime(RuntimeKey(
        checkpoint=checkpoint,
        model_device="cpu",
        codec_repo=CODEC_REPO,
        model_precision="fp32",
        codec_device="cpu",
        codec_precision="fp32",
    ))
    model = runtime.model
    text_raw = np.fromfile(output / "duration_text.f32", dtype=np.float32)
    speaker_raw = np.fromfile(output / "speaker_state.f32", dtype=np.float32)
    text = torch.from_numpy(text_raw.reshape(1, -1, 512).copy())
    speaker = torch.from_numpy(speaker_raw.reshape(1, -1, 768).copy())
    text_tokens, speaker_tokens = text.shape[1], speaker.shape[1]

    text_cfg = torch.cat([text, torch.zeros_like(text), text], dim=0)
    text_mask_cfg = torch.cat([
        torch.ones((1, text_tokens), dtype=torch.bool),
        torch.zeros((1, text_tokens), dtype=torch.bool),
        torch.ones((1, text_tokens), dtype=torch.bool),
    ])
    speaker_cfg = torch.cat([speaker, speaker, torch.zeros_like(speaker)], dim=0)
    speaker_mask_cfg = torch.cat([
        torch.ones((2, speaker_tokens), dtype=torch.bool),
        torch.zeros((1, speaker_tokens), dtype=torch.bool),
    ])
    caption_cfg = torch.zeros((3, 1, 512), dtype=torch.float32)
    caption_mask_cfg = torch.zeros((3, 1), dtype=torch.bool)
    with torch.inference_mode():
        cache_cfg = model.build_context_kv_cache(text_cfg, speaker_cfg, caption_cfg)

    generator = torch.Generator(device="cpu").manual_seed(20260911)
    state = torch.randn((1, LATENT_FRAMES, 32), generator=generator)
    save(output / "speaker_euler_noise.f32", state)
    for step in range(STEPS):
        timestep = (1.0 - step / STEPS) * 0.999
        next_timestep = (1.0 - (step + 1) / STEPS) * 0.999
        use_cfg = timestep >= 0.5
        batch = 3 if use_cfg else 1
        x_batch = state.expand(batch, -1, -1).contiguous()
        save(output / f"speaker_euler_x_step{step:03d}.f32", x_batch)
        if use_cfg:
            step_text, step_text_mask = text_cfg, text_mask_cfg
            step_speaker, step_speaker_mask = speaker_cfg, speaker_mask_cfg
            step_caption, step_caption_mask = caption_cfg, caption_mask_cfg
            step_cache = cache_cfg
        else:
            step_text, step_text_mask = text, text_mask_cfg[:1]
            step_speaker, step_speaker_mask = speaker, speaker_mask_cfg[:1]
            step_caption, step_caption_mask = caption_cfg[:1], caption_mask_cfg[:1]
            step_cache = [tuple(value[:1] for value in layer) for layer in cache_cfg]
        with torch.inference_mode():
            velocity = model.forward_with_encoded_conditions(
                x_t=x_batch,
                t=torch.full((batch,), timestep),
                text_state=step_text,
                text_mask=step_text_mask,
                speaker_state=step_speaker,
                speaker_mask=step_speaker_mask,
                caption_state=step_caption,
                caption_mask=step_caption_mask,
                context_kv_cache=step_cache,
            )
        save(output / f"speaker_euler_velocity_step{step:03d}.f32", velocity)
        guided = velocity[0]
        if use_cfg:
            guided = guided + 3.0 * (velocity[0] - velocity[1])
            guided = guided + 5.0 * (velocity[0] - velocity[2])
        state = state + guided.unsqueeze(0) * (next_timestep - timestep)
    save(output / "speaker_euler_final.f32", state)
    print(f"saved speaker Euler golden: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
