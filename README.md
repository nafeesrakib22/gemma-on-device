# Gemma 2b On-Device Chat

A responsive, text-only chat application using the LiteRT-LM (formerly TensorFlow Lite) Python API and the Gemma 2b model. This application features a Gradio-based interface with persistent conversation context and performance metrics.

## Features
- **Text Interaction**: Optimized for fast text-based chat.
- **Persistent Context**: Uses KV caching to maintain conversation history efficiently across turns.
- **Performance Metrics**: Reports Time to First Token (TTFT) and Total Generation Time.
- **Premium UI**: Modern, responsive interface built with Gradio.

## Prerequisites
- **Python 3.12+**
- **Docker & Docker Compose** (optional, for containerized deployment)
- **LiteRT-LM Model**: You need the `model.litertlm` file for Gemma 2b.

## Running with Docker (Recommended)

1. **Prepare the Model**: Ensure your model file is located in a directory accessible to Docker (e.g., `~/.litert-lm/models/gemma-e2b/model.litertlm`).

2. **Run with Docker Compose**:
   ```bash
   docker compose up -d
   ```
   *Note: The `docker-compose.yml` is configured to mount `/home/moriarty4k/.litert-lm/models` by default. Adjust the volume mapping in the file if your models are elsewhere.*

3. **Access the Application**:
   Open `http://localhost:7860` in your browser.

## Manual Installation (Local)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nafeesrakib22/gemma-on-device.git
   cd gemma-on-device
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Model Path**:
   Update the `MODEL_PATH` in `app.py` or set the `MODEL_PATH` environment variable.

4. **Start the Application**:
   ```bash
   python app.py
   ```

## Project Structure
- `app.py`: Main Gradio application.
- `Dockerfile`: Container definition with security enhancements and healthchecks.
- `docker-compose.yml`: Simplified deployment configuration.
- `requirements.txt`: Python dependencies.
