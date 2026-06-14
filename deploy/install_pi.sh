#!/usr/bin/env bash
# Ender3Monitor — Raspberry Pi installer.
#
# Sets up a virtualenv, installs deps (headless OpenCV + ffmpeg), grants the
# service user serial + camera access, and installs a systemd service so the
# monitor starts on boot and restarts on crash.
#
# Usage (from the repo root on the Pi):
#   bash deploy/install_pi.sh
#
# Re-runnable — safe to run again after a `git pull`.
set -euo pipefail

# Resolve the repo root (this script lives in deploy/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_USER="${SUDO_USER:-$USER}"
VENV="$REPO_DIR/.venv"
SERVICE="/etc/systemd/system/ender3monitor.service"

echo "==> Ender3Monitor Pi install"
echo "    repo : $REPO_DIR"
echo "    user : $RUN_USER"

# 1. System packages: venv tooling, ffmpeg (H.264 timelapse), OpenCV runtime libs.
echo "==> Installing system packages (sudo)…"
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip ffmpeg libatlas-base-dev libgl1 libglib2.0-0

# 2. Serial (printer) + video (camera) access for the service user.
echo "==> Adding $RUN_USER to dialout + video groups…"
sudo usermod -aG dialout,video "$RUN_USER" || true

# 3. Python venv + dependencies. Use the HEADLESS OpenCV build (no GUI libs)
#    and install everything else from requirements.txt.
echo "==> Creating virtualenv + installing Python deps…"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel
"$VENV/bin/pip" install "opencv-python-headless>=4.9.0"
grep -v '^opencv-python' "$REPO_DIR/requirements.txt" | "$VENV/bin/pip" install -r /dev/stdin

# 4. .env check.
if [ ! -f "$REPO_DIR/.env" ]; then
    echo "==> No .env found — creating from .env.example. EDIT IT before first run:"
    echo "    nano $REPO_DIR/.env   (set ANTHROPIC_API_KEY, WEB_USERNAME/WEB_PASSWORD, etc.)"
    cp "$REPO_DIR/.env.example" "$REPO_DIR/.env"
fi

# 5. Install the systemd service with real paths/user substituted.
echo "==> Installing systemd service…"
sudo tee "$SERVICE" >/dev/null <<UNIT
[Unit]
Description=Ender3Monitor — 3D print failure detection
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$REPO_DIR
ExecStart=$VENV/bin/python web.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable ender3monitor

cat <<DONE

==> Done.

Next:
  1. Edit your secrets/config:   nano $REPO_DIR/.env
  2. The dialout/video group change needs a re-login (or reboot) to take effect.
  3. Start it:                   sudo systemctl start ender3monitor
  4. Watch logs:                 journalctl -u ender3monitor -f
  5. Open the dashboard:         http://$(hostname -I | awk '{print $1}'):8080

Pi tuning (optional, in .env): STREAM_FPS=6  STREAM_WIDTH=960  STREAM_HEIGHT=540
Point timelapse at a USB stick to spare the SD card: TIMELAPSE_DIR=/mnt/usb/timelapse
Use the Anthropic backend on a Pi (Ollama/local vision models are too heavy).
DONE
