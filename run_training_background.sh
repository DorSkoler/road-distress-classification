#!/bin/bash

# Run training in background with nohup
# This allows the process to continue even after closing the terminal

echo "Starting Model E training in background..."
echo "Logs will be saved to training.log"
echo "Use 'tail -f training.log' to monitor progress"
echo "Use 'ps aux | grep train_model_e' to check if still running"
echo "Use 'kill PID' to stop the process if needed"
echo ""

# Run with nohup and redirect output to log file
nohup python run_training.py > training.log 2>&1 &

# Get the process ID
PID=$!
echo "Training started with PID: $PID"
echo "Process ID saved to training.pid"
echo $PID > training.pid

echo ""
echo "Commands to monitor:"
echo "  tail -f training.log    # Watch live output"
echo "  ps -p $PID             # Check if process is running"
echo "  kill $PID              # Stop training if needed"
echo ""
echo "You can now safely close this terminal." 