#!/bin/bash
if command -v systemctl >/dev/null 2>&1 && [ -f /etc/systemd/system/cctv-detector.service ]; then
    echo "Stopping CCTV-Detector via systemd..."
    sudo systemctl stop cctv-detector
fi
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
fuser -k 8000/tcp 2>/dev/null || true
echo "CCTV-Detector stopped."
