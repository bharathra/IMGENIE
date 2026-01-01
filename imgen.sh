#!/bin/bash
# Container lifecycle management

set -e

SERVICE="imgenie_app"

run_server() {
    echo "Starting IMGEN Server in background..."
    docker compose up -d $SERVICE
    echo "✓ Server started. Wait for models to load..."
}

if [ -n "$(docker compose ps -q $SERVICE)" ]; then
    echo "Server is already running."
else
    run_server
    echo "Server is running at http://localhost:8000"
    echo "You can now run the GUI client: python imgen/gui_client.py"
fi
