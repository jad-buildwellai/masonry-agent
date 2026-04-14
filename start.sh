#!/bin/bash
set -e

# ── SSH setup from PUBLIC_KEY env ─────────────────────────────────────────────
mkdir -p /root/.ssh && chmod 700 /root/.ssh
if [ -n "$PUBLIC_KEY" ]; then
  echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

# Generate host keys if missing
ssh-keygen -A 2>/dev/null || true

# Start sshd in background
/usr/sbin/sshd -D &

# ── Set root password if provided ────────────────────────────────────────────
if [ -n "$ROOT_PASSWORD" ]; then
  echo "root:${ROOT_PASSWORD}" | chpasswd
fi

# ── Model: use volume cache, pull from LFS on first boot ────────────────────
mkdir -p /runpod-volume/models
if [ ! -f /runpod-volume/models/masonry.pt ]; then
  echo "[start.sh] First boot — pulling masonry.pt from git LFS..."
  cd /app && git lfs pull --include="masonry.pt"
  cp /app/masonry.pt /runpod-volume/models/masonry.pt
  echo "[start.sh] Model cached to /runpod-volume/models/masonry.pt"
else
  echo "[start.sh] Model found in volume cache."
fi

# ── Start FastAPI server ─────────────────────────────────────────────────────
echo "[start.sh] Starting brick counter API on port ${PORT:-8000}..."
exec python /app/pod_server.py
