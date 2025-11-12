#!/usr/bin/env bash

set -euo pipefail

if [ $# -eq 0 ]; then
  echo "Usage: $0 <bag_directory> [ros2 bag play args...]"
  echo "Example: $0 bags/brain_run --loop"
  exit 1
fi

BAG_PATH=$1
shift

if [ ! -d "${BAG_PATH}" ]; then
  echo "Bag directory '${BAG_PATH}' not found."
  exit 2
fi

echo "Replaying bag: ${BAG_PATH}"
exec ros2 bag play "${BAG_PATH}" "$@"

