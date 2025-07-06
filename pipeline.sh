#!/usr/bin/env bash

if [[ -z "$1" ]]; then
    echo "Run again providing running option (0/1):"
    echo "0: Does not run TensorBoard (assuming you are already running it"
    echo "1: Runs TensorBoard"
    echo "2: Debug mode, doesn't run anything, just the training script"
    exit
else
  RUNNING_MODE="$1"
  if [[ $RUNNING_MODE == "0" ]]; then
    tensorboard_dir=($(uv run -m utils.model_handler))
    uv run -m scripts.train ${tensorboard_dir[1]}
  elif [[ $RUNNING_MODE == "1" ]]; then
    tensorboard_dir=($(uv run -m utils.model_handler))
    tensorboard --logdir=${tensorboard_dir[0]} &
    uv run -m scripts.train ${tensorboard_dir[1]} &
  elif [[ $RUNNING_MODE == "2" ]]; then
    echo "[PIPELINE] Running pipeline script on debug mode (2)"
    uv run -m scripts.train
  else
    echo "Unknown command."
    exit
  fi
fi
