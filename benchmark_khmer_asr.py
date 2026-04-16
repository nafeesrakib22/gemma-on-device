"""
benchmark_khmer_asr.py
----------------------
Evaluates the Gemma 4 E2B model's Khmer audio transcription performance.

Pipeline:
  1. Load audio files listed in a .tsv dataset.
  2. Transcribe each file using Gemma 4 E2B via the litert-lm API.
  3. Grade each transcription against the ground truth using Gemini 2.5 Flash
     as an expert Khmer linguist judge.
  4. Write per-file results to a CSV report incrementally.
  5. Print a final summary with mean accuracy.
"""

import csv
import json
import os
import sys
import time

from google import genai
from google.genai import types
import litert_lm
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
MODEL_PATH       = os.getenv(
    "MODEL_PATH",
    "/home/moriarty4k/.litert-lm/models/gemma-e2b/model.litertlm",
)
AUDIO_DIR_PATH   = os.getenv("AUDIO_DIR_PATH", "./audio_folder")
TSV_FILE_PATH    = os.getenv("TSV_FILE_PATH",   "./data/line_index.tsv")
NUM_FILES_TO_TEST = int(os.getenv("NUM_FILES_TO_TEST", "5"))
REPORT_PATH      = os.getenv("REPORT_PATH", "benchmark_report.csv")

JUDGE_MODEL = "gemini-3.1-flash-lite-preview"
ASR_PROMPT       = "Transcribe the provided Khmer audio into Khmer script."

JUDGE_SYSTEM_PROMPT = """You are an expert Khmer linguist evaluating automatic speech recognition output.
You will receive a Ground Truth transcription and a Model Transcription.
Your task:
- Compare them as a Khmer language expert.
- Ignore white-space differences (Khmer is unsegmented).
- Identify words that are misspelled, missing, or hallucinated by the model.
- Score accuracy from 0 to 100 (100 = perfect match).
Return ONLY a valid JSON object with exactly these keys:
  accuracy_score  : integer 0-100
  wrong_words     : list of strings (Khmer words that are wrong/missing/hallucinated)
  explanation     : short string explaining the errors
Example:
{"accuracy_score": 85, "wrong_words": ["ពាក្យ", "ខុស"], "explanation": "Two words were substituted."}
"""

# ---------------------------------------------------------------------------
# Validate critical config
# ---------------------------------------------------------------------------
if not GEMINI_API_KEY:
    sys.exit("[ERROR] GEMINI_API_KEY is not set. Check your .env file.")

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

judge_client = genai.Client(api_key=GEMINI_API_KEY)
print(f"[INFO] Judge model: {JUDGE_MODEL}")

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
limit = min(NUM_FILES_TO_TEST, total_available)
print(f"[INFO] Dataset: {total_available} entries — testing {limit}.")

# ---------------------------------------------------------------------------
# CSV report — written incrementally row-by-row
# ---------------------------------------------------------------------------
report_fields = [
    "file_name",
    "accuracy_score",
    "wrong_words",
    "gemma_transcription",
    "ground_truth",
    "explanation",
]

report_file = open(REPORT_PATH, "w", newline="", encoding="utf-8")
writer = csv.DictWriter(report_file, fieldnames=report_fields)
writer.writeheader()

# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
scores: list[float] = []

for idx, entry in enumerate(dataset[:limit], start=1):
    file_name    = entry["file_name"]
    ground_truth = entry["ground_truth"]
    audio_path   = os.path.join(AUDIO_DIR_PATH, file_name)

    print(f"\n[{idx}/{limit}] Processing: {file_name}")

    # ── Guard: file must exist ──────────────────────────────────────────────
    if not os.path.isfile(audio_path):
        print(f"  [WARN] Audio file not found, skipping: {audio_path}")
        writer.writerow({
            "file_name": file_name,
            "accuracy_score": "N/A",
            "wrong_words": "FILE_NOT_FOUND",
            "gemma_transcription": "",
            "ground_truth": ground_truth,
            "explanation": "Audio file missing from AUDIO_DIR_PATH.",
        })
        report_file.flush()
        continue

    # ── Step 1: Transcribe with Gemma 4 E2B ────────────────────────────────
    gemma_transcription = ""
    try:
        user_message = {
            "role": "user",
            "content": [
                {"type": "audio", "path": audio_path},
                {"type": "text",  "text": ASR_PROMPT},
            ],
        }
        with engine.create_conversation() as conversation:
            t0 = time.perf_counter()
            response = conversation.send_message(user_message)
            elapsed = time.perf_counter() - t0

        # Extract text from response
        if isinstance(response, dict) and "content" in response:
            for item in response["content"]:
                if item.get("type") == "text":
                    gemma_transcription += item.get("text", "")
        else:
            gemma_transcription = str(response)

        gemma_transcription = gemma_transcription.strip()
        print(f"  Gemma ({elapsed:.2f}s): {gemma_transcription[:80]} …")

    except Exception as exc:
        print(f"  [ERROR] Gemma transcription failed: {exc}")
        writer.writerow({
            "file_name": file_name,
            "accuracy_score": "N/A",
            "wrong_words": "",
            "gemma_transcription": f"ERROR: {exc}",
            "ground_truth": ground_truth,
            "explanation": "Gemma API call failed.",
        })
        report_file.flush()
        continue

    # ── Step 2: Judge with Gemini ──────────────────────────────────────────
    accuracy_score = None
    wrong_words    = []
    explanation    = ""
    max_retries    = 5
    base_delay     = 2

    for attempt in range(max_retries):
        try:
            judge_prompt = (
                f"Ground Truth:\n{ground_truth}\n\n"
                f"Model Transcription:\n{gemma_transcription}"
            )
            judge_response = judge_client.models.generate_content(
                model=JUDGE_MODEL,
                contents=judge_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=JUDGE_SYSTEM_PROMPT,
                ),
            )
            raw_json = judge_response.text.strip()

            # Strip markdown code fences if present
            if raw_json.startswith("```"):
                raw_json = raw_json.split("```")[1]
                if raw_json.startswith("json"):
                    raw_json = raw_json[4:]

            parsed = json.loads(raw_json)
            accuracy_score = int(parsed.get("accuracy_score", 0))
            wrong_words    = parsed.get("wrong_words", [])
            explanation    = parsed.get("explanation", "")
            scores.append(accuracy_score)
            print(f"  Score: {accuracy_score}%  |  Wrong words: {wrong_words}")
            break  # Success!

        except Exception as exc:
            exc_str = str(exc)
            # Retry only on transient "overloaded" or "unavailable" errors
            if any(msg in exc_str for msg in ["503", "UNAVAILABLE", "high demand", "Deadline Exceeded"]):
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"  [WARN] Judge busy/overloaded. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                    continue
            
            print(f"  [ERROR] Judge call failed: {exc}")
            explanation = f"Judge error: {exc}"
            break

    # ── Write result row immediately ────────────────────────────────────────
    writer.writerow({
        "file_name":           file_name,
        "accuracy_score":      accuracy_score if accuracy_score is not None else "N/A",
        "wrong_words":         "; ".join(wrong_words),
        "gemma_transcription": gemma_transcription,
        "ground_truth":        ground_truth,
        "explanation":         explanation,
    })
    report_file.flush()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
report_file.close()

print("\n" + "=" * 60)
print(f"BENCHMARK COMPLETE — {limit} files tested")
print(f"Report saved to: {REPORT_PATH}")

if scores:
    mean_acc = sum(scores) / len(scores)
    print(f"Mean Accuracy : {mean_acc:.1f}%  (over {len(scores)} scored files)")
else:
    print("No files were scored successfully.")
print("=" * 60)
