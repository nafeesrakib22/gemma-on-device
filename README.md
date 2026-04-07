# Gemma 2b Voice-Enabled Chat

A responsive, multi-modal chat application using the LiteRT-LM (formerly TensorFlow Lite) Python API and the Gemma 2b model. This application features a Gradio-based interface that supports text, image, and audio inputs (including live microphone recording) with integrated web search capabilities.

## Features
- **Multi-modal Interaction**: Chat using text, images, or audio.
- **Voice-Enabled**: Direct audio perception using LiteRT-LM.
- **Web Search**: Integrated DuckDuckGo search for real-time information.
- **Premium UI**: Modern, responsive interface built with Gradio.

## Prerequisites
- **Python 3.10+** (tested with Python 3.12).
- **LiteRT-LM Model**: You need the `model.litertlm` file for Gemma 2b.
- **System Dependencies**: Ensure you have library support for audio (e.g., `libsndfile` on Linux).

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Gemma
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Model**:
   Choose the model version you wish to use and run the corresponding import command:

   - **For Gemma 2b (E2B)**:
     ```bash
     litert-lm import --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm gemma-e2b .
     ```
   - **For Gemma 4b (E4B)**:
     ```bash
     litert-lm import --from-huggingface-repo=litert-community/gemma-4-E4B-it-litert-lm gemma-4-E4B-it.litertlm gemma-e4b .
     ```

   After downloading, you can list the available models using:
   ```bash
   litert-lm list
   ```

## Configuration

Before running the application, you must update the model path in `app.py`. 

**Tip:** You can find the absolute path to your downloaded models by running:
```bash
litert-lm list
```

1. Locate the `MODEL_PATH` variable in `app.py` (line 7):
   ```python
   MODEL_PATH = "path/to/your/model.litertlm"
   ```
2. Update it to the absolute path where your Gemma 2b LiteRT-LM model is stored.

## Running the Application

Start the Gradio server:
```bash
python app.py
```

The application will launch and provide a local URL (usually `http://127.0.0.1:7860`) and a public shareable URL if enabled.

## Project Structure
- `app.py`: Main Gradio application.
- `tools/web_search.py`: Tool for internet search integration.
- `requirements.txt`: List of Python dependencies.
