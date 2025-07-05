#!/usr/bin/env bash

if [[ -z "$1" ]]; then
    echo "Run again providing running option (0/1):"
    echo "0: Does not run TensorBoard (assuming you are already running it"
    echo "1: Runs TensorBoard"
    exit
else
  tensorboard_dir=($(uv run -m utils.model_handler))
  RUNNING_MODE="$1"
  if [[ $RUNNING_MODE == "0" ]]; then
    uv run -m scripts.train ${tensorboard_dir[1]}

  elif [[ $RUNNING_MODE == "1" ]]; then
    tensorboard --logdir=${tensorboard_dir[0]} &
    uv run -m scripts.train ${tensorboard_dir[1]} &
  else
    echo "Unknown command."
    exit
  fi
fi
