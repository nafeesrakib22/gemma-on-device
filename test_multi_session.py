import litert_lm
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/home/moriarty4k/.litert-lm/models/gemma-e2b/model.litertlm",
)

engine = litert_lm.Engine(MODEL_PATH)

print("Starting Session 1...")
ctx1 = engine.create_conversation(messages=[{"role": "user", "content": "Hello"}])
conv1 = ctx1.__enter__()
print("Session 1 active.")

try:
    print("Starting Session 2...")
    ctx2 = engine.create_conversation(messages=[{"role": "user", "content": "Hi"}])
    conv2 = ctx2.__enter__()
    print("Session 2 active.")
except Exception as e:
    print(f"FAILED to start Session 2: {e}")

finally:
    ctx1.__exit__(None, None, None)
    try:
        ctx2.__exit__(None, None, None)
    except:
        pass
