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

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         🎨 IMGENIE Web UI Server       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "$SCRIPT_DIR/server.py" ]; then
    echo -e "${RED}❌ Error: server.py not found in $(pwd)${NC}"
    exit 1
fi

# Get port and config file from command line arguments
PORT=${1:-5000}
CONFIG_FILE=${2}

# Check if port is available
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null; then
    echo -e "${RED}❌ Port $PORT is already in use${NC}"
    echo -e "${YELLOW}Try using a different port: ./run.sh 5000 <config_file>${NC}"
    exit 1
fi

# Check if config file is provided and exists
if [ -z "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}⚠️  No config file provided, using default${NC}"
    CONFIG_FILE="$SCRIPT_DIR/config/imgenie.config.default.yaml"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Config file not found: $CONFIG_FILE${NC}"
    echo -e "${YELLOW}Usage: ./run.sh [PORT] [CONFIG_FILE]${NC}"
    echo -e "${YELLOW}Example: ./run.sh 5000 ../imgenie/config/imgenie.config.yaml${NC}"
    exit 1
fi

# Convert relative path to absolute path
CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"

echo -e "${GREEN}✓ Config file: $CONFIG_FILE${NC}"
echo ""
echo -e "${GREEN}✓ All systems ready!${NC}"
echo ""
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 Starting IMGENIE Web UI Server...${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""
echo -e "📍 ${YELLOW}Server URL:${NC} http://localhost:$PORT"
echo -e "💻 ${YELLOW}API Base:${NC}  http://localhost:$PORT/api"
echo -e "📄 ${YELLOW}Config:${NC}    $CONFIG_FILE"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Run the server
cd "$SCRIPT_DIR"
python3 server.py --port=$PORT --config="$CONFIG_FILE"
