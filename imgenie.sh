#!/bin/bash

# Container lifecycle management
set -e

SERVICE="imgenie_app"
COMPOSE_FILE=".devcontainer/docker-compose.yml"

# Validate compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "Error: docker-compose.yml not found!"
    exit 1
fi

run_server() {
    echo "Starting IMGEN Server in background..."
    docker compose -f "$COMPOSE_FILE" up -d $SERVICE
}

if [ -n "$(docker compose -f "$COMPOSE_FILE" ps -q $SERVICE)" ]; then
    echo "Server is already running."
else
    run_server
    echo "Server is running at http://localhost:8000"
    echo "You can now run the GUI client: python imgen/gui_client.py"
fi
