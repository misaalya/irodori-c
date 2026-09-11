#!/usr/bin/env python3
"""Golden reference dumper — dump intermediate tensor PyTorch untuk verifikasi engine C.

Setiap komponen C (P1) dibandingkan dengan file golden ini (toleransi fp32 ~1e-4).
Pakai checkpoint fp32 agar golden = referensi penuh.

Output: irodori-c/golden/<prefix>/ berisi .pt files (torch.save) + manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Irodori-TTS"))

from irodori_tts.inference_runtime import (  # noqa: E402
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
    get_cached_runtime,
)
from irodori_tts.text_normalization import normalize_text  # noqa: E402

CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
TEXT = "こんにちは、色とりどりの世界へようこそ。"
SEED = 42
STEPS = 8


def save(out_dir: Path, name: str, tensor) -> None:
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    torch.save(tensor, out_dir / f"{name}.pt")
    # export biner untuk engine C: .f32 + meta shape
    if isinstance(tensor, torch.Tensor) and tensor.dtype.is_floating_point:
        arr = tensor.contiguous().numpy()
        arr.tofile(out_dir / f"{name}.f32")
    elif isinstance(tensor, torch.Tensor) and tensor.dtype == torch.int64:
        tensor.contiguous().numpy().tofile(out_dir / f"{name}.i64")
    shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else None
    print(f"  {name}: {shape} {tensor.dtype if isinstance(tensor, torch.Tensor) else type(tensor)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--text", default=TEXT)
    parser.add_argument(
        "--backbone-only",
        action="store_true",
        help="dump the padded ModernBERT path without running duration/DiT/codec",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "golden" / f"seed{SEED}_steps{args.steps}"
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized = normalize_text(args.text)
    print(f"[golden] text: {args.text!r}")
    print(f"[golden] normalized: {normalized!r}")
    (out_dir / "text.json").write_text(json.dumps({
        "raw": args.text, "normalized": normalized, "seed": SEED, "steps": args.steps,
    }, ensure_ascii=False, indent=2))

    checkpoint = download_hf_checkpoint("Aratako/Irodori-TTS-v4.1-Small")
    key = RuntimeKey(
        checkpoint=checkpoint, model_device="cpu", codec_repo=CODEC_REPO,
        model_precision="fp32", codec_device="cpu", codec_precision="fp32",
    )
    print("[golden] loading runtime (fp32)...")
    runtime, _ = get_cached_runtime(key)
    model = runtime.model

    # ---- 1. tokenizer ----
    ids = runtime.tokenizer.encode(normalized)
    save(out_dir, "token_ids", ids)

    # ---- 2. hooks untuk menangkap intermediate ----
    captured: dict[str, list] = {}

    def hook(name, store_first=False):
        def fn(module, inputs, output):
            if store_first:
                # The backbone is shared by text and caption.  The old dumper
                # overwrote the text result with the all-padding caption pass,
                # producing an invalid all-zero backbone golden.
                captured.setdefault(name, output)
            else:
                captured.setdefault(name, []).append(
                    output if isinstance(output, torch.Tensor) else output[0]
                )
        return fn

    handles = [
        model.pretrained_text_backbone.register_forward_hook(hook("backbone_out", True)),
        model.pretrained_text_backbone.backbone.embeddings.register_forward_hook(
            hook("bb_embed", True)
        ),
        *[
            model.pretrained_text_backbone.backbone.layers[i].register_forward_hook(
                hook(f"bb_layer{i}", True)
            )
            for i in range(25)
        ],
        model.pretrained_text_backbone.backbone.layers[0].attn.Wqkv.register_forward_hook(
            hook("bb0_qkv", True)
        ),
        model.pretrained_text_backbone.backbone.layers[0].attn.Wo.register_forward_hook(
            hook("bb0_attn_out", True)
        ),
        model.pretrained_text_backbone.backbone.layers[0].mlp_norm.register_forward_hook(
            hook("bb0_mlp_norm", True)
        ),
        model.pretrained_text_backbone.backbone.layers[0].mlp.Wi.register_forward_hook(
            hook("bb0_mlp_wi", True)
        ),
        model.pretrained_text_backbone.backbone.layers[0].mlp.Wo.register_forward_pre_hook(
            lambda module, inputs: captured.setdefault("bb0_mlp_act", inputs[0])
        ),
        model.pretrained_text_backbone.backbone.layers[0].mlp.Wo.register_forward_hook(
            hook("bb0_mlp_out", True)
        ),
        model.text_encoder.register_forward_hook(hook("text_projector_out", True)),
        model.text_norm.register_forward_hook(hook("text_state", True)),
        model.cond_module.register_forward_hook(hook("cond_module_out")),
        model.in_proj.register_forward_hook(hook("in_proj_out")),
        model.in_proj.register_forward_pre_hook(
            lambda mod, args: captured.setdefault("x_t_in", []).append(args[0])
        ),
        model.out_proj.register_forward_hook(hook("velocity")),
        model.duration_predictor.register_forward_hook(hook("duration_out", True)),
        runtime.codec.model.decoder.register_forward_hook(hook("codec_decoder_out", True)),
    ]

    if args.backbone_only:
        print("[golden] running padded ModernBERT path only...")
        max_len = int(runtime.default_text_max_len)
        text_ids, text_mask = runtime.tokenizer.batch_encode(
            [normalized], max_length=max_len
        )
        text_ids = text_ids.to(runtime.model_device)
        text_mask = text_mask.to(runtime.model_device)
        with torch.inference_mode():
            model.pretrained_text_backbone(text_ids, text_mask)
        for h in handles:
            h.remove()
        for name in (
            "bb_embed",
            "bb0_qkv",
            "bb0_attn_out",
            "bb0_mlp_norm",
            "bb0_mlp_wi",
            "bb0_mlp_act",
            "bb0_mlp_out",
            *(f"bb_layer{i}" for i in range(25)),
            "backbone_out",
        ):
            if name in captured:
                save(out_dir, name, captured[name])
        shapes = {}
        for p in sorted(out_dir.glob("*.pt")):
            tensor = torch.load(p, weights_only=True)
            if isinstance(tensor, torch.Tensor):
                shapes[p.stem] = {
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                }
        (out_dir / "shapes.json").write_text(json.dumps(shapes, indent=1))
        print(f"[golden] backbone done -> {out_dir}")
        return 0

    # ---- 3. run synthesize (text-only, no ref, no caption) ----
    print(f"[golden] synthesize: {args.steps} steps...")
    result = runtime.synthesize(
        SamplingRequest(
            text=args.text, no_ref=True, num_steps=args.steps, seed=SEED,
            cfg_guidance_mode="independent", cfg_scale_text=3.0,
            cfg_scale_caption=0.0, cfg_scale_speaker=0.0,
        ),
        log_fn=lambda msg: print(f"  [rt] {msg}"),
    )
    for h in handles:
        h.remove()

    # ---- 4. simpan hasil ----
    print("[golden] saving tensors...")
    for name in ("backbone_out", "text_projector_out", "text_state", "duration_out",
                 "codec_decoder_out", "bb_embed", "bb0_qkv", "bb0_attn_out",
                 "bb0_mlp_norm", "bb0_mlp_wi", "bb0_mlp_act", "bb0_mlp_out",
                 *(f"bb_layer{i}" for i in range(25))):
        if name in captured:
            save(out_dir, name, captured[name])
    for i, t in enumerate(captured.get("x_t_in", [])):
        save(out_dir, f"x_t_step{i:03d}", t)
    for i, t in enumerate(captured.get("in_proj_out", [])):
        save(out_dir, f"in_proj_out_step{i:03d}", t)
    for i, v in enumerate(captured.get("velocity", [])):
        save(out_dir, f"velocity_step{i:03d}", v)
    for i, c in enumerate(captured.get("cond_module_out", [])):
        save(out_dir, f"cond_embed_step{i:03d}", c)
    # The sampler returns after the last Euler update, after the final model
    # hook has fired. Reconstruct and persist that state for the C loop gate.
    last_x = captured["x_t_in"][-1][:1]
    last_velocity_raw = captured["velocity"][-1]
    if last_velocity_raw.shape[0] == 2:
        last_velocity = last_velocity_raw[:1] + 3.0 * (
            last_velocity_raw[:1] - last_velocity_raw[1:2]
        )
    else:
        last_velocity = last_velocity_raw
    schedule = (
        1.0
        - torch.linspace(
            0.0, 1.0, args.steps + 1, device=last_x.device, dtype=last_x.dtype
        )
    ) * 0.999
    latent_final = last_x + last_velocity * (schedule[-1] - schedule[-2])
    save(out_dir, "latent_final", latent_final)
    save(out_dir, "waveform", result.audios[0])
    # manifest shape untuk engine C
    shapes = {}
    for p in sorted(out_dir.glob("*.pt")):
        t = torch.load(p, weights_only=True)
        if isinstance(t, torch.Tensor):
            shapes[p.stem] = {"dtype": str(t.dtype), "shape": list(t.shape)}
    (out_dir / "shapes.json").write_text(json.dumps(shapes, indent=1))
    manifest = {
        "used_seed": result.used_seed, "sample_rate": result.sample_rate,
        "num_steps": args.steps, "text": args.text,
        "messages": result.messages, "stage_timings": result.stage_timings,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[golden] done -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
