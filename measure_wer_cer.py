"""
measure_wer_cer.py
------------------
Evaluates the Gemma 4 E2B model's Khmer audio transcription performance
using Word Error Rate (WER) and Character Error Rate (CER) metrics.

Reuses the same LiteRT-LM engine, model, and audio pre-processing
logic from benchmark_khmer_asr.py.
"""

import csv
import json
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
    "/home/moriarty4k/.litert-lm/models/gemma-e2b/model.litertlm",
)
AUDIO_DIR_PATH   = os.getenv("AUDIO_DIR_PATH", "./audio_folder")
TSV_FILE_PATH    = os.getenv("TSV_FILE_PATH",   "./line_index.tsv") # Updated based on list_dir
num_env = os.getenv("NUM_FILES_TO_TEST", "5")
try:
    NUM_FILES_TO_TEST = int(num_env)
except ValueError:
    NUM_FILES_TO_TEST = 5

REPORT_PATH      = os.getenv("WER_CER_REPORT_PATH", "wer_cer_report.json")

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
# Limit calculation moved inside main()

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
# Reporting Helper
# ---------------------------------------------------------------------------
def save_report(results, stats, path):
    """Calculates summary and saves the JSON report."""
    output = {
        "results": results,
        "summary": {
            "total_files": len(results),
            "average_wer_pct": None,
            "average_cer_pct": None
        }
    }
    
    if stats:
        # For the averages, we cap individual values at 100% (1.0) 
        # to prevent looping/hallucinations from skewing the overall report.
        capped_wer = [min(s[0], 1.0) for s in stats]
        capped_cer = [min(s[1], 1.0) for s in stats]
        
        avg_wer = (sum(capped_wer) / len(stats)) * 100
        avg_cer = (sum(capped_cer) / len(stats)) * 100
        
        output["summary"]["average_wer_pct"] = round(avg_wer, 2)
        output["summary"]["average_cer_pct"] = round(avg_cer, 2)
        
        # New: Filtered averages (excluding catastrophic failures)
        non_catastrophic = [s for s in stats if s[0] <= 1.0 and s[1] <= 1.0]
        if non_catastrophic:
            f_avg_wer = (sum(s[0] for s in non_catastrophic) / len(non_catastrophic)) * 100
            f_avg_cer = (sum(s[1] for s in non_catastrophic) / len(non_catastrophic)) * 100
            output["summary"]["filtered_average_wer_pct"] = round(f_avg_wer, 2)
            output["summary"]["filtered_average_cer_pct"] = round(f_avg_cer, 2)
            output["summary"]["filtered_total_files"] = len(non_catastrophic)
        
        # Count failures where error > 100% (likely looping)
        failures = [
            results[i]["file_name"] 
            for i, s in enumerate(stats) 
            if s[0] > 1.0 or s[1] > 1.0
        ]
        output["summary"]["catastrophic_failures"] = len(failures)
        output["summary"]["catastrophic_failure_files"] = failures
        
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# Main benchmark execution
# ---------------------------------------------------------------------------
def main():
    processed_results = []
    summary_stats = []
    
    # Check dataset size
    total_available = len(dataset)
    if NUM_FILES_TO_TEST == -1:
        limit = total_available
    else:
        limit = min(NUM_FILES_TO_TEST, total_available)
    print(f"[INFO] Dataset: {total_available} entries — testing {limit}.")

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
        finally:
            # Cleanup temp audio
            if converted_path and os.path.exists(converted_path):
                os.remove(converted_path)

        # Step 2: Calculate Metrics
        try:
            wer, cer = calculate_metrics(ground_truth, gemma_transcription)
            print(f"  WER: {wer*100:.2f}% | CER: {cer*100:.2f}%")
        except Exception as exc:
            print(f"  [ERROR] Metrics calculation failed: {exc}")
            wer, cer = None, None

        # Collect result
        result_entry = {
            "file_name":           file_name,
            "wer_pct":             round(wer * 100, 2) if wer is not None else None,
            "cer_pct":             round(cer * 100, 2) if cer is not None else None,
            "gemma_transcription": gemma_transcription,
            "ground_truth":        ground_truth,
        }
        processed_results.append(result_entry)
        
        if wer is not None:
            summary_stats.append((wer, cer))

        # Incremental Save
        save_report(processed_results, summary_stats, REPORT_PATH)

    # Final Summary and Output
    print("\n" + "=" * 60)
    print(f"BENCHMARK COMPLETE — {len(processed_results)} files processed")

    if summary_stats:
        # We display the CAPPED average to the terminal as it's more representative of quality
        capped_wer = [min(r[0], 1.0) for r in summary_stats]
        capped_cer = [min(r[1], 1.0) for r in summary_stats]
        
        avg_wer = (sum(capped_wer) / len(summary_stats)) * 100
        avg_cer = (sum(capped_cer) / len(summary_stats)) * 100
        loops = sum(1 for r in summary_stats if r[0] > 1.0 or r[1] > 1.0)
        
        print("\nSUMMARY (Capped at 100%):")
        print(f"  Average WER: {avg_wer:.2f}%")
        print(f"  Average CER: {avg_cer:.2f}%")
        
        non_catastrophic = [s for s in summary_stats if s[0] <= 1.0 and s[1] <= 1.0]
        if non_catastrophic:
            f_avg_wer = (sum(s[0] for s in non_catastrophic) / len(non_catastrophic)) * 100
            f_avg_cer = (sum(s[1] for s in non_catastrophic) / len(non_catastrophic)) * 100
            print("\nSUMMARY (Filtered — Excludes failures):")
            print(f"  Average WER: {f_avg_wer:.2f}%")
            print(f"  Average CER: {f_avg_cer:.2f}%")
            print(f"  Based on {len(non_catastrophic)} files.")

        if loops > 0:
            print(f"\nDetected {loops} files with possible model looping (Errors > 100%)")

    print(f"Final report saved to: {REPORT_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()
