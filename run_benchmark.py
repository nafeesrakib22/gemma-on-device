"""
run_benchmark.py
----------------
Runs benchmark.py across multiple concurrency levels,
polls nvidia-smi for GPU stats during each run,
and writes a clean summary report (CSV + Markdown).

Usage:
    python run_benchmark.py [--url URL] [--levels 1,5,10] [--json INPUT_JSON]

NOTE: Start server.py with SKIP_WARMUP=1 before running this script
      to ensure Turn 1 TTFT reflects cold (no pre-warm) latency.
"""

import asyncio
import httpx
import json
import time
import statistics
import argparse
import threading
import subprocess
import csv
import sys
from pathlib import Path
from datetime import datetime

# ── Re-use existing benchmark logic ──────────────────────────────────────────
from benchmark import run_session


# ── GPU Monitoring ────────────────────────────────────────────────────────────

class GPUMonitor:
    """Polls nvidia-smi in a background thread to track peak GPU util and VRAM."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.peak_util = 0.0    # %
        self.peak_vram = 0.0    # MiB

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    text=True
                ).strip()
                for line in out.splitlines():
                    parts = line.split(",")
                    if len(parts) == 2:
                        util = float(parts[0].strip())
                        vram = float(parts[1].strip())
                        self.peak_util = max(self.peak_util, util)
                        self.peak_vram = max(self.peak_vram, vram)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self.peak_util = 0.0
        self.peak_vram = 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)


# ── Single concurrency run ────────────────────────────────────────────────────

async def run_concurrency_level(url, concurrency, queries, output_json):
    monitor = GPUMonitor(interval=0.5)
    monitor.start()

    async with httpx.AsyncClient(timeout=None) as client:
        tasks = [
            run_session(client, url, queries, f"user_{i+1}")
            for i in range(concurrency)
        ]
        all_sessions = await asyncio.gather(*tasks)

    monitor.stop()

    # Aggregate metrics
    turn1_ttfts, steady_ttfts, ttf_puncs, gen_times = [], [], [], []
    for session in all_sessions:
        perf = session["performance"]
        if not perf:
            continue
        if perf[0]["client_ttft"]:
            turn1_ttfts.append(perf[0]["client_ttft"])
        for turn in perf[1:]:
            if turn["client_ttft"]: steady_ttfts.append(turn["client_ttft"])
            if turn.get("ttf_punc"): ttf_puncs.append(turn["ttf_punc"])
            if turn["total_time"]:  gen_times.append(turn["total_time"])

    def avg(lst):
        return round(statistics.mean(lst), 3) if lst else None

    row = {
        "concurrency":        concurrency,
        "avg_turn1_ttft":     avg(turn1_ttfts),
        "avg_steady_ttft":    avg(steady_ttfts),
        "avg_ttf_punc":       avg(ttf_puncs),
        "avg_gen_time":       avg(gen_times),
        "peak_gpu_util_pct":  round(monitor.peak_util, 1),
        "peak_vram_mib":      round(monitor.peak_vram, 1),
    }

    # Also save full session data alongside summary
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"summary": row, "sessions": all_sessions}, f, ensure_ascii=False, indent=2)
    print(f"  [saved] {output_json}")

    return row


# ── Report writing ────────────────────────────────────────────────────────────

COLUMNS = [
    "concurrency",
    "avg_turn1_ttft",
    "avg_steady_ttft",
    "avg_ttf_punc",
    "avg_gen_time",
    "peak_gpu_util_pct",
    "peak_vram_mib",
]

HEADERS = {
    "concurrency":       "Concurrency",
    "avg_turn1_ttft":    "Avg Turn1 TTFT (s)",
    "avg_steady_ttft":   "Avg Steady TTFT (s)",
    "avg_ttf_punc":      "Avg TTF Punc (s)",
    "avg_gen_time":      "Avg Gen Time (s)",
    "peak_gpu_util_pct": "Peak GPU Util (%)",
    "peak_vram_mib":     "Peak VRAM (MiB)",
}


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path, model, context, slots, warmup):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Benchmark Report — {ts}",
        "",
        "## Configuration",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Model | `{model}` |",
        f"| Context | {context} tokens |",
        f"| KV Slots | {slots} |",
        f"| GPU | NVIDIA GeForce RTX 4090 (24 GB) |",
        f"| Warmup | {'Enabled' if not warmup else 'Disabled (cold)'} |",
        "",
        "## Results",
        "",
    ]

    # Header row
    header = "| " + " | ".join(HEADERS[c] for c in COLUMNS) + " |"
    sep    = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    lines += [header, sep]

    for row in rows:
        cells = []
        for c in COLUMNS:
            v = row[c]
            cells.append(str(v) if v is not None else "N/A")
        lines.append("| " + " | ".join(cells) + " |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Multi-concurrency benchmark runner")
    parser.add_argument("--url",      default="http://localhost:7860/chat")
    parser.add_argument("--levels",   default="1,5,10",
                        help="Comma-separated concurrency levels (default: 1,5,10)")
    parser.add_argument("--json",     default="convo_gemini/test-gemma-convo.json",
                        help="Input conversation JSON for benchmark")
    parser.add_argument("--out-dir",  default=".",
                        help="Directory to write report files (default: current dir)")
    parser.add_argument("--model",    default="google_gemma-4-E2B-it-Q4_K_M.gguf")
    parser.add_argument("--context",  default=16384, type=int)
    parser.add_argument("--slots",    default=10, type=int)
    parser.add_argument("--no-warmup", action="store_true",
                        help="Mark report as cold/no-warmup run")
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load queries
    with open(args.json, "r") as f:
        data = json.load(f)
    queries = [item["text"] for item in data["conversation"] if item["speaker"] == "user"]
    print(f"Loaded {len(queries)} user turns from {args.json}")

    rows = []
    for level in levels:
        print(f"\n{'='*50}")
        print(f"Running concurrency={level} ...")

        # Reset server state before each run to ensure cold start
        try:
            resp = httpx.post(f"{args.url.replace('/chat', '')}/reset", timeout=5)
            if resp.status_code == 200:
                print("  [reset] Server state cleared.")
            else:
                print(f"  [warning] Reset failed (status {resp.status_code})")
        except Exception as e:
            print(f"  [warning] Could not connect to /reset: {e}")

        json_out = out_dir / f"benchmark_c{level}.json"
        row = await run_concurrency_level(args.url, level, queries, json_out)
        rows.append(row)

        print(f"  Turn1 TTFT:    {row['avg_turn1_ttft']}s")
        print(f"  Steady TTFT:   {row['avg_steady_ttft']}s")
        print(f"  TTF Punc:      {row['avg_ttf_punc']}s")
        print(f"  Gen Time:      {row['avg_gen_time']}s")
        print(f"  Peak GPU:      {row['peak_gpu_util_pct']}%")
        print(f"  Peak VRAM:     {row['peak_vram_mib']} MiB")

    # Write summary report
    csv_path = out_dir / "benchmark_summary.csv"
    md_path  = out_dir / "benchmark_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path,
                   model=args.model,
                   context=args.context,
                   slots=args.slots,
                   warmup=args.no_warmup)

    print(f"\n{'='*50}")
    print(f"[done] Summary CSV:      {csv_path}")
    print(f"[done] Summary Markdown: {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
