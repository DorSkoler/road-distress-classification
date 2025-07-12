#!/bin/bash

# Run training in a screen session
# This allows you to detach/reattach to monitor progress

SESSION_NAME="model_e_training"

echo "Starting Model E training in screen session: $SESSION_NAME"
echo ""
echo "Commands to use:"
echo "  screen -r $SESSION_NAME    # Reattach to session"
echo "  Ctrl+A, D                  # Detach from session"
echo "  screen -ls                 # List all sessions"
echo "  screen -X -S $SESSION_NAME quit  # Kill session"
echo ""

# Check if screen is installed
if ! command -v screen &> /dev/null; then
    echo "Screen is not installed. Installing..."
    sudo apt-get update && sudo apt-get install -y screen
fi

# Start training in screen session
screen -dmS $SESSION_NAME python run_training.py

echo "Training started in screen session: $SESSION_NAME"
echo "Use 'screen -r $SESSION_NAME' to attach and monitor progress"
echo "You can now safely close this terminal." 