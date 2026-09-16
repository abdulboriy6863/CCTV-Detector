#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if command -v systemctl >/dev/null 2>&1 && [ -f /etc/systemd/system/cctv-detector.service ]; then
    echo "Starting CCTV-Detector via systemd..."
    sudo systemctl restart cctv-detector
    sudo systemctl status cctv-detector --no-pager
else
    # Standalone mode
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    fuser -k 8000/tcp 2>/dev/null || true
    sleep 1
    nohup "$DIR/venv/bin/python" "$DIR/run.py" > "$DIR/nohup.out" 2>&1 &
    echo "CCTV-Detector started in standalone mode (PID: $!). Logs: $DIR/nohup.out"
fi
