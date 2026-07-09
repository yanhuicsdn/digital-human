#!/usr/bin/env bash
cd /root/workspace/digital-human
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export HOST="0.0.0.0"
export PORT="5000"
export VIDEO_BACKEND="flashhead"
export FLASHHEAD_MODEL_TYPE="pro"

exec /opt/miniconda3/envs/torch28/bin/python -m app.main \
  > /tmp/digital-human-server.log 2>&1
