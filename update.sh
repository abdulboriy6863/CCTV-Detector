#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
echo "Pulling latest changes from GitHub..."
git pull origin main
echo "Restarting service..."
./run.sh
