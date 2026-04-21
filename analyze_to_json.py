import litert_lm
import json
import os
import time

# Configuration
MODEL_PATH = "/home/moriarty4k/.litert-lm/models/gemma-e2b/model.litertlm"
INPUT_FOLDER = "/home/moriarty4k/Documents/Gemma/transcriptions"
INPUT_FILE = os.path.join(INPUT_FOLDER, "Results.json")
OUTPUT_FILE = os.path.join(INPUT_FOLDER, "AnalysisResults.json")

# Refined System Prompt for JSON output
SYSTEM_PROMPT = """You are an expert call center analyst. Your task is to analyze the following conversation transcript and produce a structured output in JSON format.

**Instructions:**
1. **Diarization:** The text may or may not be diarized. If it is not diarized, you must diarize it with proper speaker identification (customer/agent) by fully understanding the context.
2. **Rating:** Rate each dialogue turn from 0 (very negative) to 9 (very positive).
3. **Title:** Provide a suitable title focusing on the main topic of the conversation.
4. **Summary:** Provide a concise summary of the conversation.
5. **Tags:** 
    - Identify relevant tags for the conversation.
    - Prefer using these tags: GPS, Payment, Scheduling, Network, Support.
    - Introduce new tags only if absolutely necessary.

**Output Format:**
You MUST return ONLY a valid JSON object with the following structure:
{{
  "title": "Conversation Title",
  "summary": "Conversation Summary",
  "tags": ["tag1", "tag2"],
  "dialogues": [
    {{
      "text": "dialogue text",
      "speaker": "customer or agent",
      "rating": 0-9
    }},
    ...
  ]
}}
"""

def analyze_conversation(engine, conversation_data):
    # Extract transcription
    try:
        dialogues_raw = conversation_data.get("dialogues", "[]")
        if isinstance(dialogues_raw, str):
            dialogues = json.loads(dialogues_raw)
        else:
            dialogues = dialogues_raw
    except Exception as e:
        print(f"[ERROR] Failed to load dialogues: {e}")
        dialogues = []

    # Prepare prompt inputs
    prompt_input = {
        "transcription": json.dumps(dialogues, ensure_ascii=False, indent=2),
        "agent_reference": conversation_data.get("agent_reference", "N/A"),
        "call_meta_id": conversation_data.get("call_meta_id", "N/A")
    }

    user_message = f"**Input Data:**\n- **Transcription:** {prompt_input['transcription']}\n- **Agent Reference:** {prompt_input['agent_reference']}\n- **Call Meta ID:** {prompt_input['call_meta_id']}"

    # Initialize conversation
    ctx = engine.create_conversation(messages=[
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    ])
    
    with ctx as convo:
        response_text = ""
        for chunk in convo.send_message_async({"role": "user", "content": [{"type": "text", "text": user_message}]}):
            for item in chunk.get("content", []):
                if item.get("type") == "text":
                    response_text += item["text"]
        
        # Parse JSON response
        try:
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:-3].strip()
            elif json_str.startswith("```"):
                json_str = json_str[3:-3].strip()
            
            return json.loads(json_str)
        except Exception as e:
            print(f"[ERROR] Failed to parse model response: {e}")
            return {
                "title": "Error",
                "summary": "Model output parsing failed",
                "tags": ["Error"],
                "dialogues": dialogues # Return original on failure
            }

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        return

    print(f"[INFO] Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    print(f"[INFO] Initializing Gemma Engine...")
    engine = litert_lm.Engine(MODEL_PATH, backend=litert_lm.Backend.CPU)

    all_results = []
    
    # Debug limit for testing
    DEBUG_LIMIT = 2
    data_to_process = data[:DEBUG_LIMIT] if DEBUG_LIMIT else data

    for i, entry in enumerate(data_to_process):
        print(f"[INFO] Analyzing conversation {i+1} of {len(data_to_process)}...")
        analysis = analyze_conversation(engine, entry)
        all_results.append(analysis)
        
        # Incremental save
        if (i + 1) % 5 == 0:
            save_results(all_results)

    save_results(all_results)
    print(f"[INFO] Analysis complete. Output saved to {OUTPUT_FILE}")

def save_results(results):
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
