#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
# Kill any existing process on port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
fuser -k 8000/tcp 2>/dev/null || true
sleep 1
nohup "$DIR/venv/bin/python" "$DIR/run.py" > "$DIR/nohup.out" 2>&1 &
echo "CCTV-Detector started (PID: $!). Logs: $DIR/nohup.out"
