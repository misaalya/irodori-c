#!/usr/bin/env python3
"""Compile tokenizer.json (HF fast tokenizer) -> tokenizer.bin untuk engine C.

Format bin (little-endian):
  magic "IRTK" u32, version u32
  unk_id i32, byte_fallback u8, prepend_bos u8, pad[2]
  replacement_len u16 + replacement bytes (Metaspace "▁")
  prepend_scheme u8 (0=never), split u8, pad[2]
  vocab_count u32
  per entry (urut id): u16 len + utf-8 bytes + f32 score

Juga menghasilkan test_vectors.json: hasil normalize+tokenize kalimat uji
(sebagai gate referensi untuk implementasi C).
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Irodori-TTS"))

from irodori_tts.text_normalization import normalize_text  # noqa: E402

SENTENCES = [
    "こんにちは、色とりどりの世界へようこそ。",
    "今日はとても良い天気ですね。",
    "声のクローンテストです。この声は似ていますか？",
    "速度比較テストです。どのモデルが一番速いでしょうか。",
    "AI音声合成の未来へようこそ！",
    "東京都渋谷区で開発されています。",
    " Hello world と日本語の混合テキスト。",
    "緊急時は110番へ電話してください。",
    "感情表現のテストです😊楽しいですね！",
    "小説の朗読：『雪国』——国境の長いトンネルを抜けると雪国であった。",
    "0000-00-00 12:34:56 数値と記号!@#$%^&*()",
    "ゔぁゔぃっぽぃぽん",
    "𩸽の漁師と䲢",
    "hogetaro@example.com",
    "3.14159265358979",
    "この文章は、機械学習モデルによって生成されました。",
    "　全角スペースと\n改行を含む\tテキスト。",
    "カタカナ、ひらがな、漢字、Alphabet、数字12345",
    "「引用符」の中身と、カッコ（括弧）のテスト。",
    "repeat repeat repeat 繰り返し繰り返し",
]


def main() -> int:
    tokenizer_json = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path.home() / ".cache/huggingface/hub/models--Aratako--Irodori-TTS-v4.1-Small/"
        "snapshots/2b28324dc263ed5e6638b3cf3dd94c82ead07b4b/tokenizer/tokenizer.json"
    )
    out_bin = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path(__file__).resolve().parent.parent / "weights" / "tokenizer.bin"
    )
    out_vec = Path(sys.argv[3]) if len(sys.argv) > 3 else (
        Path(__file__).resolve().parent.parent / "golden" / "tokenizer_vectors.json"
    )

    tj = json.loads(tokenizer_json.read_text())
    model = tj["model"]
    assert model["type"] == "Unigram", f"bukan Unigram: {model['type']}"
    vocab = model["vocab"]
    unk_id = model["unk_id"]
    byte_fallback = bool(model.get("byte_fallback", False))
    pre = tj.get("pre_tokenizer") or {}
    replacement = pre.get("replacement", "▁") if pre.get("type") == "Metaspace" else "▁"
    prepend_scheme = 0 if pre.get("prepend_scheme", "never") == "never" else 1
    split = 1 if pre.get("split", False) else 0

    # tambahan token khusus (added_tokens) tidak diikutsertakan sebagai vocab entry —
    # encode() pakai add_special_tokens=False; BOS ditangani runtime.
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(b"IRTK")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<iBB2x", unk_id, 1 if byte_fallback else 0, 0))
        rep = replacement.encode("utf-8")
        f.write(struct.pack("<H", len(rep)))
        f.write(rep)
        f.write(struct.pack("<BB2x", prepend_scheme, split))
        f.write(struct.pack("<I", len(vocab)))
        for piece, score in vocab:
            b = piece.encode("utf-8")
            f.write(struct.pack("<H", len(b)))
            f.write(b)
            f.write(struct.pack("<f", score))
    print(f"saved: {out_bin} ({out_bin.stat().st_size / 1e6:.1f} MB)")
    print(f"vocab: {len(vocab)} | unk_id: {unk_id} | byte_fallback: {byte_fallback}")

    # test vectors via HF tokenizer (pipeline lengkap runtime: normalize -> encode -> BOS)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        str(tokenizer_json.parent), use_fast=True, trust_remote_code=False
    )
    vectors = []
    for raw in SENTENCES:
        normalized = normalize_text(raw)
        ids = tok.encode(normalized, add_special_tokens=False)
        ids = [tok.bos_token_id] + list(ids)  # runtime prepend BOS manual
        vectors.append({
            "raw": raw,
            "normalized": normalized,
            "ids": ids,
            "tokens": tok.convert_ids_to_tokens(ids),
        })
        print(f"  {normalized!r} -> {ids}")
    out_vec.parent.mkdir(parents=True, exist_ok=True)
    out_vec.write_text(json.dumps(vectors, ensure_ascii=False, indent=1))
    print(f"saved: {out_vec} ({len(vectors)} kalimat uji)")

    # format TSV untuk gate C: "<ids spasi>\t<hex normalized>\t<hex raw>"
    tsv = out_vec.with_suffix(".tsv")
    with open(tsv, "w") as f:
        for v in vectors:
            ids = " ".join(map(str, v["ids"]))
            hx = v["normalized"].encode("utf-8").hex()
            hraw = v["raw"].encode("utf-8").hex()
            f.write(f"{ids}\t{hx}\t{hraw}\n")
    print(f"saved: {tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
