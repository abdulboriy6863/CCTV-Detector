#!/bin/bash
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
fuser -k 8000/tcp 2>/dev/null || true
echo "CCTV-Detector stopped."
