#!/bin/bash

# Run training in a tmux session
# This allows you to detach/reattach to monitor progress

SESSION_NAME="model_e_training"

echo "Starting Model E training in tmux session: $SESSION_NAME"
echo ""
echo "Commands to use:"
echo "  tmux attach -t $SESSION_NAME    # Reattach to session"
echo "  Ctrl+B, D                       # Detach from session"
echo "  tmux ls                         # List all sessions"
echo "  tmux kill-session -t $SESSION_NAME  # Kill session"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Tmux is not installed. Installing..."
    sudo apt-get update && sudo apt-get install -y tmux
fi

# Start training in tmux session
tmux new-session -d -s $SESSION_NAME 'python run_training.py'

echo "Training started in tmux session: $SESSION_NAME"
echo "Use 'tmux attach -t $SESSION_NAME' to attach and monitor progress"
echo "You can now safely close this terminal." 