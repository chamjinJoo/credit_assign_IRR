#!/bin/bash

echo "=== SMACv2 & Training Process Killer ==="

# 1. StarCraft II Clients
if pgrep -f "SC2_x64" > /dev/null; then
    echo "[INFO] Found StarCraft II processes. Killing..."
    pkill -9 -f "SC2_x64"
    echo "[OK] StarCraft II processes killed."
else
    echo "[INFO] No StarCraft II processes found."
fi

# 2. Main Training Scripts (main.py)
if pgrep -f "python main.py" > /dev/null; then
    echo "[INFO] Found training processes (main.py). Killing..."
    pkill -9 -f "python main.py"
    echo "[OK] Training processes killed."
else
    echo "[INFO] No training processes found."
fi

# 3. Virtual Display (xvfb)
if pgrep -f "xvfb-run" > /dev/null; then
    echo "[INFO] Found xvfb-run processes. Killing..."
    pkill -9 -f "xvfb-run"
    echo "[OK] xvfb-run processes killed."
else
    echo "[INFO] No xvfb-run processes found."
fi

echo "----------------------------------------"
echo "Checking for any remaining processes..."
ps aux | grep -E "SC2_x64|python main.py|xvfb" | grep -v grep

echo "========================================"
echo "Cleanup Complete."
