"""
measure_wer_cer.py
------------------
Evaluates the Gemma 4 E2B model's Khmer audio transcription performance
using Word Error Rate (WER) and Character Error Rate (CER) metrics.

Reuses the same LiteRT-LM engine, model, and audio pre-processing
logic from benchmark_khmer_asr.py.
"""

import csv
import os
import sys
import time
import tempfile
from pydub import AudioSegment
import jiwer
import litert_lm
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

MODEL_PATH       = os.getenv(
    "MODEL_PATH",
    "/home/moriarty4k/.litert-lm/models/gemma-e4b/model.litertlm",
)
AUDIO_DIR_PATH   = os.getenv("AUDIO_DIR_PATH", "./audio_folder")
TSV_FILE_PATH    = os.getenv("TSV_FILE_PATH",   "./line_index.tsv") # Updated based on list_dir
num_env = os.getenv("NUM_FILES_TO_TEST", "5")
try:
    NUM_FILES_TO_TEST = int(num_env)
except ValueError:
    NUM_FILES_TO_TEST = 5
REPORT_PATH      = os.getenv("WER_CER_REPORT_PATH", "wer_cer_report.csv")

ASR_PROMPT       = "Transcribe the provided Khmer audio into Khmer script."
ASR_SYSTEM_PROMPT = """You are a specialized Khmer Automatic Speech Recognition (ASR) system.
Your ONLY task is to transcribe Khmer audio into accurate Khmer script.
- DO NOT translate to English.
- DO NOT output any English text.
- DO NOT add commentary or explanations.
- Output ONLY the Khmer transcription.
"""

# ---------------------------------------------------------------------------
# Validate critical config
# ---------------------------------------------------------------------------
if not os.path.isfile(TSV_FILE_PATH):
    sys.exit(f"[ERROR] TSV file not found: {TSV_FILE_PATH}")

# ---------------------------------------------------------------------------
# Initialise clients
# ---------------------------------------------------------------------------
print("[INFO] Initialising Gemma 4 E2B engine (litert-lm) …")
engine = litert_lm.Engine(
    MODEL_PATH,
    backend=litert_lm.Backend.CPU,
    audio_backend=litert_lm.Backend.CPU,
)
print("[INFO] Engine ready.")

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
dataset: list[dict] = []
with open(TSV_FILE_PATH, newline="", encoding="utf-8") as f:
    # File has no header; columns are: id <TAB> (empty) <TAB> sentence
    reader = csv.DictReader(f, fieldnames=["file_name", "_blank", "sentence"], delimiter="\t")
    for row in reader:
        file_id = row["file_name"].strip()
        sentence = row["sentence"].strip()
        if file_id:
            dataset.append({"file_name": file_id + ".wav", "ground_truth": sentence})

total_available = len(dataset)
if NUM_FILES_TO_TEST == -1:
    limit = total_available
else:
    limit = min(NUM_FILES_TO_TEST, total_available)
print(f"[INFO] Dataset: {total_available} entries — testing {limit}.")

# ---------------------------------------------------------------------------
# Audio Pre-processing (Exact copy from benchmark_khmer_asr.py)
# ---------------------------------------------------------------------------
def preprocess_audio(input_path: str) -> str:
    """Converts audio to 16kHz Mono WAV as required by LiteRT-LM."""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(temp_file.name, format="wav")
    return temp_file.name

# ---------------------------------------------------------------------------
# Metrics Calculation
# ---------------------------------------------------------------------------
def calculate_metrics(ground_truth: str, hypothesis: str):
    """Calculates WER and CER using jiwer."""
    # WER: uses strings as is (assuming space segmentation represents words)
    wer = jiwer.wer(ground_truth, hypothesis)
    
    # CER: remove spaces to compare characters only (standard for unsegmented Khmer)
    gt_no_space = ground_truth.replace(" ", "")
    hyp_no_space = hypothesis.replace(" ", "")
    cer = jiwer.cer(gt_no_space, hyp_no_space)
    
    return wer, cer

# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------
report_fields = [
    "file_name",
    "wer",
    "cer",
    "gemma_transcription",
    "ground_truth",
]

report_file = open(REPORT_PATH, "w", newline="", encoding="utf-8")
writer = csv.DictWriter(report_file, fieldnames=report_fields)
writer.writeheader()

# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
results = []

for idx, entry in enumerate(dataset[:limit], start=1):
    file_name    = entry["file_name"]
    ground_truth = entry["ground_truth"]
    audio_path   = os.path.join(AUDIO_DIR_PATH, file_name)

    print(f"\n[{idx}/{limit}] Processing: {file_name}")

    if not os.path.isfile(audio_path):
        print(f"  [WARN] Audio file not found, skipping: {audio_path}")
        continue

    # Step 0: Pre-process audio
    converted_path = None
    try:
        converted_path = preprocess_audio(audio_path)
        inference_audio_path = converted_path
    except Exception as exc:
        print(f"  [ERROR] Audio pre-processing failed: {exc}")
        continue

    # Step 1: Transcribe with Gemma
    gemma_transcription = ""
    try:
        user_message = {
            "role": "user",
            "content": [
                {"type": "audio", "path": inference_audio_path},
                {"type": "text",  "text": ASR_PROMPT},
            ],
        }
        system_message = {
            "role": "system",
            "content": [{"type": "text", "text": ASR_SYSTEM_PROMPT}],
        }
        with engine.create_conversation(messages=[system_message]) as conversation:
            t0 = time.perf_counter()
            response = conversation.send_message(user_message)
            elapsed = time.perf_counter() - t0

        if isinstance(response, dict) and "content" in response:
            for item in response["content"]:
                if item.get("type") == "text":
                    gemma_transcription += item.get("text", "")
        else:
            gemma_transcription = str(response)

        gemma_transcription = gemma_transcription.strip()
        print(f"  Gemma ({elapsed:.2f}s): {gemma_transcription}")

    except Exception as exc:
        print(f"  [ERROR] Gemma transcription failed: {exc}")
        if converted_path and os.path.exists(converted_path):
            os.remove(converted_path)
        continue

    # Step 2: Calculate Metrics
    try:
        wer, cer = calculate_metrics(ground_truth, gemma_transcription)
        print(f"  WER: {wer:.4f} | CER: {cer:.4f}")
    except Exception as exc:
        print(f"  [ERROR] Metrics calculation failed: {exc}")
        wer, cer = None, None

    # Cleanup
    if converted_path and os.path.exists(converted_path):
        os.remove(converted_path)

    # Write result
    writer.writerow({
        "file_name":           file_name,
        "wer":                 f"{wer:.4f}" if wer is not None else "N/A",
        "cer":                 f"{cer:.4f}" if cer is not None else "N/A",
        "gemma_transcription": gemma_transcription,
        "ground_truth":        ground_truth,
    })
    report_file.flush()
    
    if wer is not None:
        results.append((wer, cer))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
report_file.close()

print("\n" + "=" * 60)
print(f"BENCHMARK COMPLETE — {len(results)} files processed")
print(f"Report saved to: {REPORT_PATH}")

if results:
    avg_wer = sum(r[0] for r in results) / len(results)
    avg_cer = sum(r[1] for r in results) / len(results)
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Average CER: {avg_cer:.4f}")
print("=" * 60)
