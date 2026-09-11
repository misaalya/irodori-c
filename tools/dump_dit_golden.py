#!/usr/bin/env python3
"""Dump focused RF-DiT component fixtures without re-running full sampling."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Irodori-TTS"))

from irodori_tts.inference_runtime import (  # noqa: E402
    RuntimeKey,
    download_hf_checkpoint,
    get_cached_runtime,
)

CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"


def load_f32(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    array = np.fromfile(path, dtype=np.float32)
    return torch.from_numpy(array.reshape(shape).copy())


def save_f32(path: Path, tensor: torch.Tensor) -> None:
    tensor.detach().cpu().contiguous().numpy().tofile(path)
    print(f"  {path.stem}: {tuple(tensor.shape)} {tensor.dtype}")


def save_u8(path: Path, tensor: torch.Tensor) -> None:
    tensor.detach().cpu().to(torch.uint8).contiguous().numpy().tofile(path)
    print(f"  {path.stem}: {tuple(tensor.shape)} uint8")


def main() -> int:
    golden = ROOT / "irodori-c" / "golden" / "seed42_steps8"
    checkpoint = download_hf_checkpoint("Aratako/Irodori-TTS-v4.1-Small")
    key = RuntimeKey(
        checkpoint=checkpoint,
        model_device="cpu",
        codec_repo=CODEC_REPO,
        model_precision="fp32",
        codec_device="cpu",
        codec_precision="fp32",
    )
    print("[dit-golden] loading runtime (fp32)...")
    runtime, _ = get_cached_runtime(key)
    model = runtime.model

    cond = load_f32(golden / "cond_embed_step000.f32", (2, 3840))
    x = load_f32(golden / "in_proj_out_step000.f32", (2, 112, 1280))
    text_one = load_f32(golden / "text_state.f32", (1, 256, 512))
    token_count = (golden / "token_ids.i64").stat().st_size // 8
    text = torch.cat([text_one, torch.zeros_like(text_one)], dim=0)
    text_mask = torch.zeros((2, 256), dtype=torch.bool)
    text_mask[0, :token_count] = True

    with torch.inference_mode():
        block = model.blocks[0]
        attention = block.attention
        h, gate = block.attention_adaln(x, cond[:, None, :])

        # The no-ref/no-caption fixture masks those contexts completely. Keeping
        # only self + text is mathematically equivalent; the focused C gate also
        # omits those branches and skips masked text entries during attention.
        k_text = attention.wk_text(text).reshape(2, 256, 20, 64)
        v_text = attention.wv_text(text).reshape(2, 256, 20, 64)
        k_text = attention.k_norm(k_text)

        q = attention.wq(h).reshape(2, 112, 20, 64)
        k_self = attention.wk(h).reshape(2, 112, 20, 64)
        v_self = attention.wv(h).reshape(2, 112, 20, 64)
        gate_logits = attention.gate(h)
        q = attention.q_norm(q)
        k_self = attention.k_norm(k_self)
        freqs = model._rope_freqs(112, h.device)
        q = attention._apply_rotary_half(q, freqs)
        k_self = attention._apply_rotary_half(k_self, freqs)

        joined_k = torch.cat([k_self, k_text], dim=1)
        joined_v = torch.cat([v_self, v_text], dim=1)
        joined_mask = torch.cat(
            [torch.ones((2, 112), dtype=torch.bool), text_mask], dim=1
        )
        mix = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            joined_k.transpose(1, 2),
            joined_v.transpose(1, 2),
            attn_mask=joined_mask[:, None, None, :],
            is_causal=False,
        ).transpose(1, 2).reshape(2, 112, 1280)
        attention_out = attention.wo(mix * torch.sigmoid(gate_logits))
        x_after_attention = x + gate * attention_out
        mlp_h, mlp_gate = block.mlp_adaln(x_after_attention, cond[:, None, :])
        mlp_out = block.mlp(mlp_h)
        block_out = x_after_attention + mlp_gate * mlp_out

        dummy_speaker = torch.zeros((2, 1, 768), dtype=h.dtype)
        dummy_caption = torch.zeros((2, 1, 512), dtype=h.dtype)
        dummy_mask = torch.zeros((2, 1), dtype=torch.bool)
        context_cache = model.build_context_kv_cache(
            text_state=text,
            speaker_state=dummy_speaker,
            caption_state=dummy_caption,
        )
        module_out = block(
            x=x,
            cond_embed=cond[:, None, :],
            text_state=text,
            text_mask=text_mask,
            speaker_state=dummy_speaker,
            speaker_mask=dummy_mask,
            caption_state=dummy_caption,
            caption_mask=dummy_mask,
            freqs_cis=freqs,
            context_kv=context_cache[0],
        )
        equivalence_err = (module_out - block_out).abs().max().item()
        if equivalence_err >= 1e-5:
            raise RuntimeError(
                f"compact masked-context block mismatch: {equivalence_err:.6g}"
            )
        print(f"  masked-context omission equivalence: {equivalence_err:.6g}")

        layer_outputs = [block_out]
        state = block_out
        for layer_index, next_block in enumerate(model.blocks[1:], start=1):
            state = next_block(
                x=state,
                cond_embed=cond[:, None, :],
                text_state=text,
                text_mask=text_mask,
                speaker_state=dummy_speaker,
                speaker_mask=dummy_mask,
                caption_state=dummy_caption,
                caption_mask=dummy_mask,
                freqs_cis=freqs,
                context_kv=context_cache[layer_index],
            )
            layer_outputs.append(state)
        velocity = model.out_proj(model.out_norm(state))
        velocity_ref = load_f32(golden / "velocity_step000.f32", (2, 112, 32))
        velocity_err = (velocity - velocity_ref).abs().max().item()
        if velocity_err >= 1e-4:
            raise RuntimeError(f"full DiT reconstruction mismatch: {velocity_err:.6g}")
        print(f"  full DiT reconstruction: velocity err {velocity_err:.6g}")

    save_f32(golden / "dit_b0_attn_adaln_h.f32", h)
    save_f32(golden / "dit_b0_attn_adaln_gate.f32", gate)
    save_u8(golden / "dit_b0_text_mask.u8", text_mask)
    save_f32(golden / "dit_b0_text_k.f32", k_text)
    save_f32(golden / "dit_b0_text_v.f32", v_text)
    save_f32(golden / "dit_b0_attn_q.f32", q)
    save_f32(golden / "dit_b0_attn_k_self.f32", k_self)
    save_f32(golden / "dit_b0_attn_v_self.f32", v_self)
    save_f32(golden / "dit_b0_attn_gate_logits.f32", gate_logits)
    save_f32(golden / "dit_b0_attn_mix.f32", mix)
    save_f32(golden / "dit_b0_attn_out.f32", attention_out)
    save_f32(golden / "dit_b0_x_after_attention.f32", x_after_attention)
    save_f32(golden / "dit_b0_mlp_adaln_h.f32", mlp_h)
    save_f32(golden / "dit_b0_mlp_adaln_gate.f32", mlp_gate)
    save_f32(golden / "dit_b0_mlp_out.f32", mlp_out)
    save_f32(golden / "dit_b0_out.f32", block_out)
    for layer_index, layer_output in enumerate(layer_outputs):
        save_f32(golden / f"dit_b{layer_index:02d}_out.f32", layer_output)
    save_f32(golden / "dit_velocity_step000.f32", velocity)
    print(f"[dit-golden] done -> {golden}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
