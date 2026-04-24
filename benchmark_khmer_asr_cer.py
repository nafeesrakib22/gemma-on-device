"""
benchmark_khmer_asr_cer.py
--------------------------
Evaluates the Gemma 4 E2B model's Khmer audio transcription performance using
Character Error Rate (CER) as the primary metric.
"""

import csv
import json
import os
import sys
import time
import tempfile
from pydub import AudioSegment
from google import genai
from google.genai import types
import litert_lm
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
MODEL_PATH       = os.getenv("MODEL_PATH", "/home/moriarty4k/.litert-lm/models/gemma-e4b/model.litertlm")
AUDIO_DIR_PATH   = os.getenv("AUDIO_DIR_PATH", "./audio_folder")
TSV_FILE_PATH    = os.getenv("TSV_FILE_PATH", "./data/line_index.tsv")
NUM_FILES_TO_TEST = int(os.getenv("NUM_FILES_TO_TEST", "5"))
REPORT_PATH      = "benchmark_cer_report.csv" # Specific report for CER

JUDGE_MODEL = "gemini-3.1-flash-lite-preview"
ASR_PROMPT       = "Transcribe the provided Khmer audio into Khmer script."
ASR_SYSTEM_PROMPT = """You are a specialized Khmer Automatic Speech Recognition (ASR) system.
Your ONLY task is to transcribe Khmer audio into accurate Khmer script.
- DO NOT translate to English.
- DO NOT output any English text.
- DO NOT add commentary or explanations.
- Output ONLY the Khmer transcription.
"""

JUDGE_SYSTEM_PROMPT = """You are an expert Khmer linguist evaluating automatic speech recognition output at a character level.
You will receive a Ground Truth transcription and a Model Transcription.
Your task:
- Compare them as a Khmer language expert, focusing on character-level precision.
- Pay close attention to diacritics and vowel placement.
- Identify characters that are misspelled, missing, or hallucinated.
- Score character accuracy from 0 to 100 (100 = perfect match).
Return ONLY a valid JSON object with exactly these keys:
  accuracy_score  : integer 0-100
  wrong_characters: list of strings (individual Khmer characters/diacritics that are wrong)
  explanation     : short string explaining the character-level errors
"""

# ---------------------------------------------------------------------------
# CER Calculation (Levenshtein)
# ---------------------------------------------------------------------------
def calculate_cer(reference: str, hypothesis: str) -> tuple[int, float]:
    """Calculates Character Error Rate (CER) using Levenshtein distance."""
    # Simple Levenshtein distance implementation
    n, m = len(reference), len(hypothesis)
    if n == 0: return m, 1.0 # 100% error if reference is empty
    
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i-1] == hypothesis[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j] + 1,    # Deletion
                              dp[i][j-1] + 1,    # Insertion
                              dp[i-1][j-1] + 1)  # Substitution
    
    distance = dp[n][m]
    cer = distance / n
    return distance, cer

# ---------------------------------------------------------------------------
# Initialise clients
# ---------------------------------------------------------------------------
if not GEMINI_API_KEY:
    sys.exit("[ERROR] GEMINI_API_KEY is not set.")

print("[INFO] Initialising Gemma 4 E2B engine...")
engine = litert_lm.Engine(MODEL_PATH, backend=litert_lm.Backend.CPU, audio_backend=litert_lm.Backend.CPU)
judge_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
dataset = []
with open(TSV_FILE_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, fieldnames=["file_name", "_blank", "sentence"], delimiter="\t")
    for row in reader:
        file_id = row["file_name"].strip()
        if file_id:
            dataset.append({"file_name": file_id + ".wav", "ground_truth": row["sentence"].strip()})

limit = min(NUM_FILES_TO_TEST, len(dataset))
print(f"[INFO] Testing {limit} files.")

def preprocess_audio(input_path):
    audio = AudioSegment.from_file(input_path).set_frame_rate(16000).set_channels(1)
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(temp_file.name, format="wav")
    return temp_file.name

# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
report_fields = ["file_name", "cer", "cer_percent", "accuracy_score", "wrong_characters", "gemma_transcription", "ground_truth", "explanation"]
report_file = open(REPORT_PATH, "w", newline="", encoding="utf-8")
writer = csv.DictWriter(report_file, fieldnames=report_fields)
writer.writeheader()

cer_values = []
accuracy_scores = []

for idx, entry in enumerate(dataset[:limit], start=1):
    file_name = entry["file_name"]
    ground_truth = entry["ground_truth"]
    audio_path = os.path.join(AUDIO_DIR_PATH, file_name)

    print(f"\n[{idx}/{limit}] Processing: {file_name}")

    if not os.path.isfile(audio_path):
        continue

    try:
        temp_audio = preprocess_audio(audio_path)
        
        # Transcribe
        system_message = {"role": "system", "content": [{"type": "text", "text": ASR_SYSTEM_PROMPT}]}
        user_message = {"role": "user", "content": [{"type": "audio", "path": temp_audio}, {"type": "text", "text": ASR_PROMPT}]}
        
        with engine.create_conversation(messages=[system_message]) as convo:
            response = convo.send_message(user_message)
            gemma_transcription = "".join([item["text"] for item in response["content"] if item["type"] == "text"]).strip()
        
        os.remove(temp_audio)
        print(f"  Transcription: {gemma_transcription}")

        # Calculate CER
        dist, cer = calculate_cer(ground_truth, gemma_transcription)
        cer_values.append(cer)
        print(f"  CER: {cer:.4f} ({dist} errors)")

        # Judge with Gemini
        judge_prompt = f"Ground Truth:\n{ground_truth}\n\nModel Transcription:\n{gemma_transcription}"
        judge_response = judge_client.models.generate_content(
            model=JUDGE_MODEL,
            contents=judge_prompt,
            config=types.GenerateContentConfig(system_instruction=JUDGE_SYSTEM_PROMPT),
        )
        
        raw_json = judge_response.text.strip()
        if "```json" in raw_json: raw_json = raw_json.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_json: raw_json = raw_json.split("```")[1].strip()
        
        parsed = json.loads(raw_json)
        acc = parsed.get("accuracy_score", 0)
        accuracy_scores.append(acc)

        writer.writerow({
            "file_name": file_name,
            "cer": dist,
            "cer_percent": f"{cer*100:.2f}%",
            "accuracy_score": acc,
            "wrong_characters": "; ".join(parsed.get("wrong_characters", [])),
            "gemma_transcription": gemma_transcription,
            "ground_truth": ground_truth,
            "explanation": parsed.get("explanation", ""),
        })
        report_file.flush()

    except Exception as e:
        print(f"  [ERROR] {e}")

report_file.close()

print("\n" + "="*60)
print(f"CER BENCHMARK COMPLETE")
if cer_values:
    print(f"Mean CER      : {sum(cer_values)/len(cer_values):.4f} ({sum(cer_values)/len(cer_values)*100:.2f}%)")
    print(f"Mean Accuracy : {sum(accuracy_scores)/len(accuracy_scores):.1f}%")
print(f"Report saved to: {REPORT_PATH}")
print("="*60)
