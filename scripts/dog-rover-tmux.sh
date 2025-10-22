#!/usr/bin/bash

# Name the tmux session
SESSION_NAME="anytraverse-streamlit"

# Start a new tmux session
tmux new-session -d -s "$SESSION_NAME"

# Split the window into two panes (side-by-side)
tmux split-window -h

# Run the first script in the left pane
tmux send-keys -t "$SESSION_NAME:0.0" "bash ./scripts/run-dog-rover-streamlit.sh" Enter

# Run the second script in the right pane
tmux send-keys -t "$SESSION_NAME:0.1" "bash ./scripts/expose-streamlit.sh" Enter

# Attach to the tmux session
tmux attach-session -t "$SESSION_NAME"
