#!/bin/bash

# IMGENIE UI Server Launcher
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
# echo -e "${BLUE}║         🎨 IMGENIE Web UI Server       ║${NC}"
# echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
# echo ""

# Get port and config file from command line arguments
PORT=${1:-5000}
# Check if port is available
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null; then
    echo -e "${RED}❌ Port $PORT is already in use${NC}"
    echo -e "${YELLOW}Try using a different port: ./run.sh 5000 <config_file>${NC}"
    exit 1
fi

CONFIG_FILE=${2}
# Check if config file is provided and exists
if [ -z "$CONFIG_FILE" ]; then
    # echo -e "${YELLOW}⚠️  No config file provided, using default${NC}"
    CONFIG_FILE="$SCRIPT_DIR/config/imgenie.config.default.yaml"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Config file not found: $CONFIG_FILE${NC}"
    echo -e "${YELLOW}Usage: ./run.sh [PORT] [CONFIG_FILE]${NC}"
    echo -e "${YELLOW}Example: ./run.sh 5000 ../imgenie/config/imgenie.config.yaml${NC}"
    exit 1
fi

# Translate path for container
HOST_ROOT="/home/bharath/1river/IMGENIE"
CONTAINER_ROOT="/IMGENIE"

# Ensure absolute path for config
CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"

if [[ "$CONFIG_FILE" == "$HOST_ROOT"* ]]; then
    CONTAINER_CONFIG_FILE="${CONFIG_FILE/$HOST_ROOT/$CONTAINER_ROOT}"
else
    # Fallback: assume relative path works if not absolute, or just pass it 
    # But since we resolved to absolute, this case means it is outside the project.
    # We'll try to use the container path if it matches the structure, otherwise warning.
    echo -e "${YELLOW}⚠️  Config file seems outside project root. Passing as is, might fail if not mounted.${NC}"
    CONTAINER_CONFIG_FILE="$CONFIG_FILE" 
fi

# echo -e "${GREEN}✓ Config file (Host): $CONFIG_FILE${NC}"
# echo -e "${GREEN}✓ Config file (Container): $CONTAINER_CONFIG_FILE${NC}"
# echo ""
# echo -e "${GREEN}✓ All systems ready!${NC}"
# echo ""
# echo -e "${BLUE}════════════════════════════════════════${NC}"
# echo -e "${GREEN}🚀 Starting IMGENIE Web UI Server inside Docker...${NC}"
# echo -e "${BLUE}════════════════════════════════════════${NC}"
# echo ""
# echo -e "📍 ${YELLOW}Server URL:${NC} http://localhost:$PORT"
# echo -e "💻 ${YELLOW}API Base:${NC}  http://localhost:$PORT/api"
# echo -e "🐳 ${YELLOW}Container:${NC} imgenie"
# echo ""
# echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
# echo ""

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^imgenie$"; then
    # Check if container exists (created/exited)
    if docker ps -a --format '{{.Names}}' | grep -q "^imgenie$"; then
        echo -e "${YELLOW}Container 'imgenie' exists but is not running.${NC}"
        echo -e "${GREEN}Starting existing container 'imgenie'...${NC}"
        docker start imgenie
    else
        echo -e "${YELLOW}Container 'imgenie' does not exist.${NC}"
        read -p "Do you want to create and start it from imgenie:latest? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${GREEN}Creating and starting 'imgenie' container...${NC}"
            COMPOSE_FILE="$SCRIPT_DIR/../.devcontainer/docker-compose.yml"
            if [ -f "$COMPOSE_FILE" ]; then
                docker compose -f "$COMPOSE_FILE" up -d
            else
                echo -e "${RED}❌ Compose file not found at $COMPOSE_FILE${NC}"
                exit 1
            fi
        else
            echo -e "${RED}Aborted. Please start the container manually.${NC}"
            exit 1
        fi
    fi

    # Final check if running
    if ! docker ps --format '{{.Names}}' | grep -q "^imgenie$"; then
        echo -e "${RED}❌ Failed to start container 'imgenie'.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Container 'imgenie' is running.${NC}"
fi

# Run the server inside docker
# Determine docker flags
DOCKER_FLAGS="-i"
if [ -t 0 ]; then
    DOCKER_FLAGS="-it"
fi

# Run the server inside docker
docker exec -e PYTHONUNBUFFERED=1 $DOCKER_FLAGS imgenie python3 /IMGENIE/imgenie/imgenie_server.py --port=$PORT --config="$CONTAINER_CONFIG_FILE"
