#!/usr/bin/env bash

set -euo pipefail

OUTPUT_PREFIX=${1:-bags/brain_run}
shift || true

if [ $# -gt 0 ]; then
  TOPICS=("$@")
else
  TOPICS=(/odom /snn_action /cmd_vel /brain_markers)
fi

mkdir -p "$(dirname "${OUTPUT_PREFIX}")"

echo "Recording topics: ${TOPICS[*]}"
echo "Bag output prefix: ${OUTPUT_PREFIX}"

exec ros2 bag record -o "${OUTPUT_PREFIX}" "${TOPICS[@]}"


