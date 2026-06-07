# Ender3Monitor

A 3D print failure detection system that watches your printer through a USB webcam, uses Claude AI vision to spot problems in real time, fires email alerts when something goes wrong, and records a timelapse of every print.

---

## Features

- **Live failure detection** — captures a frame every 30 seconds and sends it to Claude for analysis
- **Detects six failure types** — spaghetti/stringing, layer shifts, bed detachment, stopped extrusion, nozzle collisions, and warping
- **Email alerts** — sends an SMTP notification with the failure type, confidence score, and an attached snapshot the moment a problem is found
- **Prometheus metrics** — exposes a `/metrics` endpoint for frames analyzed, failure counts by type, and confidence score trends
- **Grafana dashboard** — pre-built JSON dashboard ready to import; shows failure rate, confidence trends, frame history, and failure type breakdown
- **Timelapse** — saves a frame every 60 seconds and compiles them into an MP4 when monitoring stops
- **Multi-camera support** — auto-detects all USB cameras and prompts you to pick one if there are multiple

---

## Requirements

- Python 3.11+
- A USB webcam aimed at your printer
- An [Anthropic API key](https://console.anthropic.com/) **or** [Ollama](https://ollama.com) running locally
- An SMTP-capable email account (Gmail with an App Password works well)
- Prometheus + Grafana (optional, for the metrics dashboard)

---

## Installation

```bash
git clone https://github.com/btgendler2005/Ender3Monitor.git
cd Ender3Monitor
pip install -r requirements.txt
```

---

## Configuration

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `ANALYZER_BACKEND` | `anthropic` (default) or `ollama` |
| `ANTHROPIC_API_KEY` | Your Anthropic API key (only required for `anthropic` backend) |
| `ANTHROPIC_MODEL` | Model to use (default: `claude-sonnet-4-6`) |
| `OLLAMA_MODEL` | Ollama model to use (default: `llava:7b`) |
| `OLLAMA_HOST` | Ollama server URL (default: `http://localhost:11434`) |
| `SMTP_HOST` | SMTP server (e.g. `smtp.gmail.com`) |
| `SMTP_PORT` | SMTP port (usually `587` for TLS) |
| `SMTP_USERNAME` | Login username for your email account |
| `SMTP_PASSWORD` | Email password or App Password |
| `SMTP_SENDER` | From address for alert emails |
| `SMTP_RECIPIENT` | Where to send alerts |
| `CAMERA_INDEX` | Camera index to use (`-1` to pick at startup) |
| `CONFIDENCE_THRESHOLD` | Alert threshold `0.0–1.0` (default `0.70`) |
| `METRICS_PORT` | Port for Prometheus metrics (default `8000`) |
| `TIMELAPSE_DIR` | Folder to store timelapse frames (default `timelapse_frames`) |

### Ollama setup (free, runs locally)

```bash
# Install Ollama: https://ollama.com
ollama pull llava:7b        # ~4.1 GB — recommended for 8 GB RAM machines
```

Then in `.env`:

```
ANALYZER_BACKEND=ollama
OLLAMA_MODEL=llava:7b
```

**M2 MacBook Air (8 GB) model guide:**

| Model | Size | Fits on 8 GB? | Quality |
|---|---|---|---|
| `llava:7b` | ~4.1 GB | ✅ Recommended | Good |
| `moondream` | ~1.4 GB | ✅ Easily | Limited |
| `llama3.2-vision:11b` | ~6.2 GB | ⚠️ Will swap | Better |

`llava:7b` is the best balance for 8 GB — fits alongside macOS overhead and
runs on the M2 GPU via Metal. Consider raising `CONFIDENCE_THRESHOLD` to `0.80`
with local models, as they can be overconfident compared to Claude.

### Gmail setup

Gmail requires an App Password when 2FA is enabled:
1. Go to **Google Account → Security → 2-Step Verification → App passwords**
2. Generate a password for "Mail"
3. Use that as `SMTP_PASSWORD`

---

## Usage

```bash
python monitor.py
```

The terminal interface accepts these commands:

| Command | Action |
|---|---|
| `start` | Open the camera and begin monitoring |
| `stop` | Stop monitoring and release the camera |
| `timelapse` | Compile saved frames into an MP4 |
| `status` | Print current status, last result, and frame count |
| `quit` | Stop monitoring and exit |

While monitoring, a live status line updates after every analysis:

```
  [14:23:01] Status: Monitoring…              | Frames:   12 | Failures:   0 | Confidence: 8.3% | Type: none
```

---

## Prometheus & Grafana

### Prometheus

Add this job to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: ender3monitor
    static_configs:
      - targets: ['localhost:8000']
```

Metrics exposed:

| Metric | Type | Description |
|---|---|---|
| `printer_frames_analyzed_total` | Counter | Total frames sent for analysis |
| `printer_failures_detected_total{failure_type}` | Counter | Failures detected, labelled by type |
| `printer_last_confidence_score` | Gauge | Confidence score from the most recent frame |
| `printer_confidence_score_histogram` | Histogram | Distribution of confidence scores over time |
| `printer_monitoring_active` | Gauge | `1` while monitoring is running, `0` otherwise |

### Grafana

1. In Grafana, go to **Dashboards → Import**
2. Upload `grafana_dashboard.json` from this repo
3. Select your Prometheus data source when prompted

The dashboard includes:
- Confidence score over time with a 70% alert threshold line
- Failure rate per minute broken down by failure type
- Frame analysis rate
- Failure type pie chart
- Confidence score percentile trends (p50 / p90 / p95)

---

## Project Structure

```
Ender3Monitor/
├── monitor.py           # Entry point — terminal UI and monitoring loop
├── camera.py            # USB webcam management and frame capture
├── analyzer.py          # Claude vision API integration and response parsing
├── notifier.py          # SMTP email alerts with attached failure snapshot
├── metrics.py           # Prometheus metrics definitions and server
├── timelapse.py         # Frame saving and MP4 compilation
├── config.py            # Configuration loaded from .env
├── requirements.txt     # Python dependencies
├── grafana_dashboard.json  # Pre-built Grafana dashboard
└── .env.example         # Environment variable template
```

---

## How It Works

1. `monitor.py` starts a background thread that captures frames from the webcam
2. Every **30 seconds**, a frame is encoded as JPEG and sent to `claude-sonnet-4-6` with a structured prompt asking it to identify failure types and return a confidence score as JSON
3. Every **60 seconds**, a frame is saved to disk for the timelapse
4. If the returned confidence score is at or above `CONFIDENCE_THRESHOLD` (default 70%), an email alert is sent with the frame attached
5. All results are recorded to Prometheus metrics in real time
6. When `stop` is called (or the session ends), the timelapse frames can be compiled into an MP4 with the `timelapse` command

---

## License

MIT
