#!/usr/bin/env python3
"""Matched PyTorch vs pure-C benchmark for all four conditioning modes.

Each mode first asks the PyTorch runtime to export the exact initial sampler
noise, then reuses that raw float32 tensor in the C endpoint.  Timed runs use
the same checkpoint, text, caption/reference, seed, step count, and thread
count.  Results are persisted as JSON so a single fast run cannot become the
performance claim by accident.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np


C_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_PYTHON = C_ROOT.parent / "Irodori-TTS" / ".venv" / "bin" / "python"
PY_BENCH = C_ROOT / "tools" / "bench_generate.py"
PY_WORKER = C_ROOT / "tools" / "bench_python_worker.py"
C_WORKER = C_ROOT / "irodori-bench-worker"
DEFAULT_TEXT = "こんにちは、色とりどりの世界へようこそ。"
DEFAULT_CAPTION = "落ち着いた自然な女性の声で、やわらかく話す。"
DEFAULT_CAPTION_B = "明るく元気に、少し速めのテンポで話す。"
STAGE_RE = re.compile(
    r"encode\s+([0-9.]+)\s+s\s+\|\s+Euler\s+([0-9.]+)\s+s\s+\|\s+"
    r"decode\s+([0-9.]+)\s+s"
)
RSS_RE = re.compile(r"__IRO_MAXRSS__=(\d+)")
WORKER_RESULT_PREFIX = "__IRO_RESULT__="


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def parse_steps(value: str) -> list[int]:
    try:
        steps = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--steps must be comma-separated positive integers") from exc
    if not steps or any(step <= 0 for step in steps):
        raise argparse.ArgumentTypeError("--steps must contain positive integers")
    return list(dict.fromkeys(steps))


def cpu_core_identity(cpu: int) -> tuple[str, str] | None:
    topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        package = (topo / "physical_package_id").read_text().strip()
        core = (topo / "core_id").read_text().strip()
    except OSError:
        return None
    return package, core


def auto_physical_cpus(count: int) -> list[int] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    available = sorted(os.sched_getaffinity(0))
    selected: list[int] = []
    seen: set[tuple[str, str]] = set()
    for cpu in available:
        key = cpu_core_identity(cpu)
        if key is None:
            continue
        if key in seen:
            continue
        seen.add(key)
        selected.append(cpu)
        if len(selected) == count:
            return selected
    return None


def affinity_uses_distinct_physical_cores(cpus: list[int] | None, count: int) -> bool:
    if cpus is None or len(cpus) != count:
        return False
    identities = [cpu_core_identity(cpu) for cpu in cpus]
    return all(identity is not None for identity in identities) and len(set(identities)) == count


def sample_idle_fraction(seconds: float) -> dict[str, object]:
    def read_cpu() -> tuple[int, int]:
        first = Path("/proc/stat").read_text().splitlines()[0].split()
        if not first or first[0] != "cpu" or len(first) < 6:
            raise RuntimeError("unexpected /proc/stat cpu line")
        values = [int(value) for value in first[1:9]]
        idle = values[3] + values[4]
        total = sum(values)
        return idle, total

    try:
        idle0, total0 = read_cpu()
        loadavg = list(os.getloadavg())
        time.sleep(seconds)
        idle1, total1 = read_cpu()
    except (OSError, RuntimeError, ValueError):
        return {"available": False, "sample_seconds": seconds}
    delta_total = total1 - total0
    delta_idle = idle1 - idle0
    fraction = float(delta_idle / delta_total) if delta_total > 0 else 0.0
    return {
        "available": True,
        "sample_seconds": seconds,
        "idle_fraction": fraction,
        "busy_fraction": 1.0 - fraction,
        "loadavg": loadavg,
    }


def parse_cpus(value: str, threads: int) -> list[int] | None:
    if value == "none":
        return None
    if value == "auto":
        cpus = auto_physical_cpus(threads)
        if cpus is None or len(cpus) < threads:
            raise RuntimeError(f"could not find {threads} CPUs for affinity")
        return cpus
    try:
        cpus = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise RuntimeError("--cpus must be 'auto', 'none', or a comma-separated CPU list") from exc
    if len(cpus) != threads or any(cpu < 0 for cpu in cpus) or len(set(cpus)) != len(cpus):
        raise RuntimeError(f"--cpus must contain exactly {threads} unique non-negative CPU ids")
    return cpus


class PersistentWorker:
    def __init__(
        self,
        command: list[str],
        *,
        env: dict[str, str],
        cpus: list[int] | None,
        label: str,
    ) -> None:
        self.label = label
        preexec_fn = None
        if cpus is not None and hasattr(os, "sched_setaffinity"):
            cpu_set = set(cpus)

            def set_affinity() -> None:
                os.sched_setaffinity(0, cpu_set)

            preexec_fn = set_affinity
        self.process = subprocess.Popen(
            command,
            cwd=C_ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=preexec_fn,
        )
        self._read_until("__IRO_READY__")

    def _read_until(self, marker: str) -> str:
        assert self.process.stdout is not None
        while True:
            line = self.process.stdout.readline()
            if line == "":
                code = self.process.poll()
                raise RuntimeError(f"{self.label} worker exited before {marker} (code={code})")
            stripped = line.rstrip("\r\n")
            if stripped == marker or stripped.startswith(marker):
                return stripped

    def request(self, command: str) -> dict:
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        line = self._read_until(WORKER_RESULT_PREFIX)
        return json.loads(line[len(WORKER_RESULT_PREFIX) :])

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        assert self.process.stdin is not None
        self.process.stdin.write("QUIT\n")
        self.process.stdin.flush()
        self.process.wait(timeout=30)

    def __enter__(self) -> "PersistentWorker":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            return
        self.close()


def wav_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise RuntimeError(f"expected mono PCM16 WAV: {path}")
        rate = wav.getframerate()
        data = wav.readframes(wav.getnframes())
    return np.frombuffer(data, dtype="<i2").copy(), rate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mode_args(mode: str, caption: str, reference: Path) -> list[str]:
    extra: list[str] = []
    if mode in {"caption-only", "clone+caption"}:
        extra += ["--caption", caption]
    if mode in {"clone", "clone+caption"}:
        extra += ["--ref", str(reference)]
    return extra


def run_python_mode(
    *,
    python: Path,
    mode: str,
    model: Path,
    reference: Path,
    caption: str,
    text: str,
    seed: int,
    steps: int,
    threads: int,
    repeats: int,
    directory: Path,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "python.wav"
    noise = directory / "noise.f32"
    metrics = directory / "python.json"
    command = [
        str(python),
        str(PY_BENCH),
        "--checkpoint",
        str(model),
        "--threads",
        str(threads),
        "--steps",
        str(steps),
        "--repeats",
        str(repeats),
        "--text",
        text,
        "--seed",
        str(seed),
        "--out",
        str(output),
        "--noise-out",
        str(noise),
        "--json-out",
        str(metrics),
        *mode_args(mode, caption, reference),
    ]
    subprocess.run(command, cwd=C_ROOT, check=True)
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    payload["wav_path"] = str(output)
    payload["wav_sha256"] = sha256(output)
    payload["noise_path"] = str(noise)
    return payload


def c_command(
    *,
    binary: Path,
    model: Path,
    tokenizer: Path,
    decoder: Path,
    encoder: Path,
    reference: Path,
    caption: str,
    text: str,
    seed: int,
    steps: int,
    mode: str,
    noise: Path,
    output: Path,
    dump_dir: Path | None = None,
) -> list[str]:
    command = [
        str(binary),
        "--text",
        text,
        "--model",
        str(model),
        "--tokenizer",
        str(tokenizer),
        "--decoder",
        str(decoder),
        "--encoder",
        str(encoder),
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--noise",
        str(noise),
        "--out",
        str(output),
        *mode_args(mode, caption, reference),
    ]
    if dump_dir is not None:
        command += ["--dump-dir", str(dump_dir)]
    return command


def run_c_once(command: list[str], env: dict[str, str]) -> tuple[float, int | None, str]:
    time_binary = Path("/usr/bin/time")
    wrapped = command
    use_external_time = time_binary.is_file()
    if use_external_time:
        wrapped = [str(time_binary), "-f", "__IRO_MAXRSS__=%M", *command]
    begin = time.perf_counter()
    completed = subprocess.run(
        wrapped,
        cwd=C_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.perf_counter() - begin
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError(f"C benchmark failed with exit code {completed.returncode}")
    rss = None
    if use_external_time:
        match = RSS_RE.search(completed.stderr)
        if match:
            rss = int(match.group(1))
    return elapsed, rss, completed.stdout


def run_c_mode(
    *,
    mode: str,
    binary: Path,
    model: Path,
    tokenizer: Path,
    decoder: Path,
    encoder: Path,
    reference: Path,
    caption: str,
    text: str,
    seed: int,
    steps: int,
    threads: int,
    repeats: int,
    noise: Path,
    directory: Path,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["IRO_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)

    warm_output = directory / "c-warm.wav"
    warm_command = c_command(
        binary=binary,
        model=model,
        tokenizer=tokenizer,
        decoder=decoder,
        encoder=encoder,
        reference=reference,
        caption=caption,
        text=text,
        seed=seed,
        steps=steps,
        mode=mode,
        noise=noise,
        output=warm_output,
    )
    run_c_once(warm_command, env)

    samples: list[float] = []
    rss_values: list[int] = []
    hashes: list[str] = []
    stages: list[tuple[float, float, float]] = []
    last_output: Path | None = None
    for index in range(repeats):
        output = directory / f"c-run-{index + 1}.wav"
        command = c_command(
            binary=binary,
            model=model,
            tokenizer=tokenizer,
            decoder=decoder,
            encoder=encoder,
            reference=reference,
            caption=caption,
            text=text,
            seed=seed,
            steps=steps,
            mode=mode,
            noise=noise,
            output=output,
        )
        elapsed, rss, stdout = run_c_once(command, env)
        samples.append(elapsed)
        if rss is not None:
            rss_values.append(rss)
        hashes.append(sha256(output))
        stage_match = STAGE_RE.search(stdout)
        if stage_match:
            stages.append(tuple(float(stage_match.group(i)) for i in range(1, 4)))
        last_output = output
        print(f"  [c {mode}] run {index + 1}: {elapsed:.3f} s", flush=True)

    assert last_output is not None
    audio, rate = wav_pcm16(last_output)
    p50 = statistics.median(samples)
    stage_payload = None
    if stages:
        stage_payload = {
            "encode": statistics.median(row[0] for row in stages),
            "sample": statistics.median(row[1] for row in stages),
            "decode": statistics.median(row[2] for row in stages),
        }
    return {
        "backend": "c",
        "mode": mode,
        "threads": threads,
        "steps": steps,
        "seed": seed,
        "samples_seconds": samples,
        "p50_seconds": p50,
        "p95_seconds": percentile(samples, 95),
        "best_seconds": min(samples),
        "audio_seconds": len(audio) / rate,
        "rtf_p50": p50 / (len(audio) / rate),
        "peak_rss_kib": max(rss_values) if rss_values else None,
        "sample_count": int(audio.size),
        "stage_timings": stage_payload,
        "deterministic": len(set(hashes)) == 1,
        "wav_sha256": hashes[-1],
        "wav_path": str(last_output),
    }


def compare_wavs(python_wav: Path, c_wav: Path) -> dict:
    py_audio, py_rate = wav_pcm16(python_wav)
    c_audio, c_rate = wav_pcm16(c_wav)
    if py_rate != c_rate:
        return {"same_rate": False, "python_rate": py_rate, "c_rate": c_rate}
    count = min(py_audio.size, c_audio.size)
    max_lsb = None
    mae_lsb = None
    if count:
        delta = np.abs(py_audio[:count].astype(np.int32) - c_audio[:count].astype(np.int32))
        max_lsb = int(delta.max())
        mae_lsb = float(delta.mean())
    return {
        "same_rate": True,
        "same_length": py_audio.size == c_audio.size,
        "python_samples": int(py_audio.size),
        "c_samples": int(c_audio.size),
        "max_lsb": max_lsb,
        "mae_lsb": mae_lsb,
    }


def summarize_worker_samples(
    *, backend: str, mode: str, steps: int, threads: int, seed: int,
    samples: list[dict], hashes: list[str], output: Path, noise: Path | None,
) -> dict:
    elapsed = [float(sample["elapsed_seconds"]) for sample in samples]
    audio_seconds = [
        float(sample.get("audio_seconds", int(sample["output_samples"]) / 48000.0))
        for sample in samples
    ]
    p50 = statistics.median(elapsed)
    stage_payload = {
        name: statistics.median(float(sample[f"{name}_seconds"]) for sample in samples)
        for name in ("encode", "sample", "decode")
    }
    rss_values = [int(sample["peak_rss_kib"]) for sample in samples if sample.get("peak_rss_kib")]
    return {
        "backend": backend,
        "mode": mode,
        "threads": threads,
        "steps": steps,
        "seed": seed,
        "samples_seconds": elapsed,
        "p50_seconds": p50,
        "p95_seconds": percentile(elapsed, 95),
        "best_seconds": min(elapsed),
        "audio_seconds": statistics.median(audio_seconds),
        "rtf_p50": p50 / statistics.median(audio_seconds),
        "peak_rss_kib": max(rss_values) if rss_values else None,
        "sample_count": int(samples[-1]["output_samples"]),
        "stage_timings": stage_payload,
        "deterministic": len(set(hashes)) == 1,
        "wav_sha256": hashes[-1],
        "wav_path": str(output),
        "noise_path": None if noise is None else str(noise),
    }


def python_worker_command(
    *, python: Path, model: Path, reference: Path, caption: str, text: str,
    seed: int, steps: int, threads: int, mode: str,
) -> list[str]:
    return [
        str(python), str(PY_WORKER), "--checkpoint", str(model),
        "--threads", str(threads), "--steps", str(steps), "--text", text,
        "--seed", str(seed), *mode_args(mode, caption, reference),
    ]


def c_worker_command(
    *, worker: Path, model: Path, tokenizer: Path, decoder: Path, encoder: Path,
    reference: Path, caption: str, text: str, seed: int, steps: int,
    mode: str, noise: Path,
) -> list[str]:
    command = [
        str(worker), "--model", str(model), "--tokenizer", str(tokenizer),
        "--decoder", str(decoder), "--text", text, "--noise", str(noise),
        "--seed", str(seed), "--steps", str(steps),
    ]
    if mode in {"clone", "clone+caption"}:
        command += ["--encoder", str(encoder), "--ref", str(reference)]
    if mode in {"caption-only", "clone+caption"}:
        command += ["--caption", caption]
    return command


def run_interleaved_mode(
    *, python: Path, c_worker: Path, mode: str, model: Path, tokenizer: Path,
    decoder: Path, encoder: Path, reference: Path, caption: str, text: str,
    seed: int, steps: int, threads: int, repeats: int, cpus: list[int] | None,
    directory: Path,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    noise = directory / "noise.f32"
    py_warm = directory / "python-warm.wav"
    c_warm = directory / "c-warm.wav"
    env = os.environ.copy()
    env["IRO_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)

    py_command = python_worker_command(
        python=python, model=model, reference=reference, caption=caption, text=text,
        seed=seed, steps=steps, threads=threads, mode=mode,
    )
    py_samples: list[dict] = []
    c_samples: list[dict] = []
    py_hashes: list[str] = []
    c_hashes: list[str] = []
    py_last = py_warm
    c_last = c_warm

    with PersistentWorker(py_command, env=env, cpus=cpus, label=f"python/{mode}/{steps}") as py:
        py.request(f"WARM\t{py_warm}\t{noise}")
        c_command_line = c_worker_command(
            worker=c_worker, model=model, tokenizer=tokenizer, decoder=decoder,
            encoder=encoder, reference=reference, caption=caption, text=text,
            seed=seed, steps=steps, mode=mode, noise=noise,
        )
        with PersistentWorker(
            c_command_line, env=env, cpus=cpus, label=f"c/{mode}/{steps}"
        ) as c:
            c.request(f"WARM\t{c_warm}")
            for index in range(repeats):
                order = ("python", "c") if index % 2 == 0 else ("c", "python")
                for backend in order:
                    if backend == "python":
                        py_last = directory / f"python-run-{index + 1}.wav"
                        sample = py.request(f"RUN\t{py_last}")
                        py_samples.append(sample)
                        py_hashes.append(sha256(py_last))
                        print(
                            f"  [python {mode} step={steps}] run {index + 1}: "
                            f"{sample['elapsed_seconds']:.3f} s",
                            flush=True,
                        )
                    else:
                        c_last = directory / f"c-run-{index + 1}.wav"
                        sample = c.request(f"RUN\t{c_last}")
                        c_samples.append(sample)
                        c_hashes.append(sha256(c_last))
                        print(
                            f"  [c {mode} step={steps}] run {index + 1}: "
                            f"{sample['elapsed_seconds']:.3f} s",
                            flush=True,
                        )

    python_metrics = summarize_worker_samples(
        backend="python", mode=mode, steps=steps, threads=threads, seed=seed,
        samples=py_samples, hashes=py_hashes, output=py_last, noise=noise,
    )
    c_metrics = summarize_worker_samples(
        backend="c", mode=mode, steps=steps, threads=threads, seed=seed,
        samples=c_samples, hashes=c_hashes, output=c_last, noise=noise,
    )
    parity = compare_wavs(py_last, c_last)
    speedup = python_metrics["p50_seconds"] / c_metrics["p50_seconds"]
    latency_ratio = c_metrics["p50_seconds"] / python_metrics["p50_seconds"]
    rss_ratio = None
    if python_metrics["peak_rss_kib"] and c_metrics["peak_rss_kib"]:
        rss_ratio = c_metrics["peak_rss_kib"] / python_metrics["peak_rss_kib"]
    return {
        "python": python_metrics,
        "c": c_metrics,
        "pcm16_parity": parity,
        "speedup_python_over_c": speedup,
        "gate": {
            "latency_ratio_c_over_python": latency_ratio,
            "latency_pass": latency_ratio <= 0.80,
            "rss_ratio_c_over_python": rss_ratio,
            "rss_pass": rss_ratio is not None and rss_ratio <= 0.80,
            "rtf_pass": c_metrics["rtf_p50"] < 1.0,
        },
    }


def check_caption_effect(
    *,
    python: Path,
    binary: Path,
    model: Path,
    tokenizer: Path,
    decoder: Path,
    encoder: Path,
    reference: Path,
    text: str,
    caption_a: str,
    caption_b: str,
    seed: int,
    steps: int,
    threads: int,
    base_caption_metrics: dict,
    directory: Path,
) -> dict:
    """Prove two captions are observed and each caption is deterministic in C."""
    directory.mkdir(parents=True, exist_ok=True)
    b_python = run_python_mode(
        python=python,
        mode="caption-only",
        model=model,
        reference=reference,
        caption=caption_b,
        text=text,
        seed=seed,
        steps=steps,
        threads=threads,
        repeats=2,
        directory=directory / "python-b",
    )
    env = os.environ.copy()
    env["IRO_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)

    def dumped_run(label: str, caption: str, noise: Path) -> tuple[Path, Path]:
        dump_dir = directory / label
        output = directory / f"{label}.wav"
        command = c_command(
            binary=binary,
            model=model,
            tokenizer=tokenizer,
            decoder=decoder,
            encoder=encoder,
            reference=reference,
            caption=caption,
            text=text,
            seed=seed,
            steps=steps,
            mode="caption-only",
            noise=noise,
            output=output,
            dump_dir=dump_dir,
        )
        run_c_once(command, env)
        return output, dump_dir

    noise_a = Path(base_caption_metrics["python"]["noise_path"])
    noise_b = Path(b_python["noise_path"])
    a_output, a_dump = dumped_run("a-check", caption_a, noise_a)
    b1_output, b1_dump = dumped_run("b-check-1", caption_b, noise_b)
    b2_output, b2_dump = dumped_run("b-check-2", caption_b, noise_b)

    a_hash = sha256(a_output)
    b1_hash = sha256(b1_output)
    b2_hash = sha256(b2_output)
    a_caption = (a_dump / "caption_state.f32").read_bytes()
    b_caption = (b1_dump / "caption_state.f32").read_bytes()
    b_caption_repeat = (b2_dump / "caption_state.f32").read_bytes()
    return {
        "caption_a_deterministic": bool(base_caption_metrics["c"]["deterministic"]),
        "caption_b_deterministic": b1_hash == b2_hash and b_caption == b_caption_repeat,
        "conditioning_differs": a_caption != b_caption,
        "output_differs": a_hash != b1_hash,
        "python_caption_b_deterministic": bool(b_python["deterministic"]),
        "same_initial_noise": noise_a.read_bytes() == noise_b.read_bytes(),
        "passed": (
            bool(base_caption_metrics["c"]["deterministic"])
            and b1_hash == b2_hash
            and b_caption == b_caption_repeat
            and a_caption != b_caption
            and a_hash != b1_hash
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=UPSTREAM_PYTHON if UPSTREAM_PYTHON.is_file() else Path(sys.executable),
        help="Python interpreter with the upstream Irodori-TTS/PyTorch environment",
    )
    parser.add_argument("--binary", type=Path, default=C_ROOT / "irodori-blas")
    parser.add_argument("--c-worker", type=Path, default=C_WORKER)
    parser.add_argument("--tokenizer", type=Path, default=C_ROOT / "weights" / "tokenizer.bin")
    parser.add_argument(
        "--decoder", type=Path, default=C_ROOT / "weights" / "dacvae_decoder.safetensors"
    )
    parser.add_argument(
        "--encoder", type=Path, default=C_ROOT / "weights" / "dacvae_encoder.safetensors"
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--caption", default=DEFAULT_CAPTION)
    parser.add_argument("--caption-b", default=DEFAULT_CAPTION_B)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--steps",
        type=parse_steps,
        default=[8, 40],
        help="comma-separated sampler step counts; formal acceptance uses 8,40",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--cpus",
        default="auto",
        help="CPU affinity list, 'auto' for distinct physical cores, or 'none'",
    )
    parser.add_argument(
        "--idle-sample-seconds",
        type=float,
        default=1.0,
        help="preflight /proc/stat sampling window used to verify an idle benchmark host",
    )
    parser.add_argument(
        "--min-idle-fraction",
        type=float,
        default=0.90,
        help="minimum aggregate CPU idle fraction required for a formal P3 run",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/irodori-four-mode-bench"))
    parser.add_argument("--skip-caption-effect", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0 or args.repeats <= 0:
        parser.error("--threads and --repeats must be positive")
    if args.idle_sample_seconds <= 0:
        parser.error("--idle-sample-seconds must be positive")
    if not 0.0 <= args.min_idle_fraction <= 1.0:
        parser.error("--min-idle-fraction must be between 0 and 1")
    try:
        cpus = parse_cpus(args.cpus, args.threads)
    except RuntimeError as exc:
        parser.error(str(exc))

    paths = [
        args.model,
        args.ref,
        args.python,
        args.binary,
        args.c_worker,
        PY_WORKER,
        args.tokenizer,
        args.decoder,
        args.encoder,
    ]
    missing = [str(path) for path in paths if not path.expanduser().is_file()]
    if missing:
        parser.error("missing required file(s): " + ", ".join(missing))
    model = args.model.expanduser().absolute()
    reference = args.ref.expanduser().resolve()
    python = args.python.expanduser().absolute()
    binary = args.binary.expanduser().resolve()
    c_worker = args.c_worker.expanduser().resolve()
    tokenizer = args.tokenizer.expanduser().resolve()
    decoder = args.decoder.expanduser().resolve()
    encoder = args.encoder.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    idle_preflight = sample_idle_fraction(args.idle_sample_seconds)
    idle_fraction = idle_preflight.get("idle_fraction")
    idle_pass = bool(
        idle_preflight.get("available")
        and isinstance(idle_fraction, float)
        and idle_fraction >= args.min_idle_fraction
    )
    distinct_physical_cores = affinity_uses_distinct_physical_cores(cpus, args.threads)
    print(
        "preflight: "
        f"cpus={cpus} distinct_physical_cores={distinct_physical_cores} "
        f"idle_fraction={idle_fraction if idle_fraction is not None else 'unavailable'} "
        f"required={args.min_idle_fraction:.2f}",
        flush=True,
    )

    modes = ("text-only", "caption-only", "clone", "clone+caption")
    report: dict[str, object] = {
        "model": str(model),
        "reference": str(reference),
        "text": args.text,
        "caption": args.caption,
        "seed": args.seed,
        "steps": args.steps,
        "threads": args.threads,
        "repeats": args.repeats,
        "protocol": {
            "warmups_per_backend_per_mode_step": 1,
            "persistent_workers": True,
            "interleaved_python_c": True,
            "alternating_first_backend": True,
            "minimum_formal_repeats": 5,
            "formal_repeat_count_met": args.repeats >= 5,
            "cpu_affinity": cpus,
            "two_threads": args.threads == 2,
            "distinct_physical_cores": distinct_physical_cores,
            "idle_preflight": idle_preflight,
            "minimum_idle_fraction": args.min_idle_fraction,
            "idle_preflight_pass": idle_pass,
            "idle_checks": [],
        },
        "step_runs": {},
    }
    report_path = work_dir / "summary.json"

    def abort_for_idle(reason: str) -> int:
        report["acceptance"] = {
            "formal_protocol_met": False,
            "p3_performance_gate_pass": False,
            "aborted": True,
            "abort_reason": reason,
        }
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"benchmark aborted: {reason}", file=sys.stderr, flush=True)
        print(f"summary: {report_path}", flush=True)
        return 2

    if not idle_pass:
        value = "unavailable" if idle_fraction is None else f"{idle_fraction:.4f}"
        return abort_for_idle(
            f"preflight idle fraction {value} is below required "
            f"{args.min_idle_fraction:.4f}"
        )

    for steps in args.steps:
        print(f"\n######## {steps}-step ########", flush=True)
        step_report: dict[str, object] = {"modes": {}}
        report["step_runs"][str(steps)] = step_report
        for mode in modes:
            print(f"\n== {mode} ==", flush=True)
            idle_check = sample_idle_fraction(args.idle_sample_seconds)
            check_fraction = idle_check.get("idle_fraction")
            check_pass = bool(
                idle_check.get("available")
                and isinstance(check_fraction, float)
                and check_fraction >= args.min_idle_fraction
            )
            idle_check.update({"steps": steps, "mode": mode, "passed": check_pass})
            report["protocol"]["idle_checks"].append(idle_check)
            print(
                "  idle check: "
                f"fraction={check_fraction if check_fraction is not None else 'unavailable'} "
                f"passed={check_pass}",
                flush=True,
            )
            if not check_pass:
                value = "unavailable" if check_fraction is None else f"{check_fraction:.4f}"
                return abort_for_idle(
                    f"idle fraction before {steps}-step {mode} mode is {value}, "
                    f"below required {args.min_idle_fraction:.4f}"
                )
            mode_report = run_interleaved_mode(
                python=python,
                c_worker=c_worker,
                mode=mode,
                model=model,
                tokenizer=tokenizer,
                decoder=decoder,
                encoder=encoder,
                reference=reference,
                caption=args.caption,
                text=args.text,
                seed=args.seed,
                steps=steps,
                threads=args.threads,
                repeats=args.repeats,
                cpus=cpus,
                directory=work_dir / f"steps-{steps}" / mode,
            )
            step_report["modes"][mode] = mode_report
            python_metrics = mode_report["python"]
            c_metrics = mode_report["c"]
            parity = mode_report["pcm16_parity"]
            print(
                f"  p50 Python={python_metrics['p50_seconds']:.3f}s "
                f"C={c_metrics['p50_seconds']:.3f}s "
                f"speedup={mode_report['speedup_python_over_c']:.3f}x "
                f"RTF C={c_metrics['rtf_p50']:.3f} "
                f"PCM max LSB={parity.get('max_lsb')}",
                flush=True,
            )

    if not args.skip_caption_effect:
        print("\n== caption effect/determinism ==", flush=True)
        effect_steps = args.steps[0]
        base_caption_metrics = report["step_runs"][str(effect_steps)]["modes"]["caption-only"]
        caption_check = check_caption_effect(
            python=python,
            binary=binary,
            model=model,
            tokenizer=tokenizer,
            decoder=decoder,
            encoder=encoder,
            reference=reference,
            text=args.text,
            caption_a=args.caption,
            caption_b=args.caption_b,
            seed=args.seed,
            steps=effect_steps,
            threads=args.threads,
            base_caption_metrics=base_caption_metrics,
            directory=work_dir / "caption-effect",
        )
        report["caption_effect"] = caption_check
        print(json.dumps(caption_check, indent=2), flush=True)

    caption_ok = args.skip_caption_effect or bool(report.get("caption_effect", {}).get("passed"))
    caption_verified = not args.skip_caption_effect and bool(
        report.get("caption_effect", {}).get("passed")
    )
    mode_reports = [
        report["step_runs"][str(steps)]["modes"][mode]
        for steps in args.steps
        for mode in modes
    ]
    deterministic = all(
        bool(mode_report[backend]["deterministic"])
        for mode_report in mode_reports
        for backend in ("python", "c")
    )
    parity_shapes = all(
        bool(mode_report["pcm16_parity"].get("same_rate"))
        and bool(mode_report["pcm16_parity"].get("same_length"))
        for mode_report in mode_reports
    )
    formal_protocol = (
        args.repeats >= 5
        and 8 in args.steps
        and 40 in args.steps
        and args.threads == 2
        and distinct_physical_cores
        and idle_pass
        and all(bool(check["passed"]) for check in report["protocol"]["idle_checks"])
    )
    latency_pass = all(bool(mode_report["gate"]["latency_pass"]) for mode_report in mode_reports)
    rss_pass = all(bool(mode_report["gate"]["rss_pass"]) for mode_report in mode_reports)
    rtf_pass = all(bool(mode_report["gate"]["rtf_pass"]) for mode_report in mode_reports)
    report["acceptance"] = {
        "formal_protocol_met": formal_protocol,
        "deterministic": deterministic,
        "pcm16_shape_parity": parity_shapes,
        "caption_effect": caption_ok,
        "caption_effect_verified": caption_verified,
        "latency_all_modes_steps": latency_pass,
        "rss_all_modes_steps": rss_pass,
        "rtf_all_modes_steps": rtf_pass,
        "p3_performance_gate_pass": (
            formal_protocol
            and deterministic
            and parity_shapes
            and caption_verified
            and latency_pass
            and rss_pass
            and rtf_pass
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nsummary: {report_path}")
    print(json.dumps(report["acceptance"], indent=2), flush=True)
    return 0 if caption_ok and deterministic and parity_shapes else 1


if __name__ == "__main__":
    raise SystemExit(main())
