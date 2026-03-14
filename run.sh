#!/bin/bash
# Meallion Voice AI - Run Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}  Meallion Voice AI - Elena    ${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo "Creating from env.example..."
    cp env.example .env
    echo -e "${YELLOW}Please edit .env with your credentials${NC}"
fi

# Parse arguments
case "$1" in
    "dev")
        echo -e "${GREEN}Starting full development stack...${NC}"
        docker-compose up --build
        ;;
    "frontend")
        echo -e "${GREEN}Starting frontend only...${NC}"
        cd frontend && npm run dev
        ;;
    "api")
        echo -e "${GREEN}Starting API server only...${NC}"
        python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
        ;;
    "agent")
        echo -e "${GREEN}Starting Elena agent...${NC}"
        python -m src.agents.elena
        ;;
    "test")
        echo -e "${GREEN}Running tests...${NC}"
        python -m pytest tests/ -v
        ;;
    "build")
        echo -e "${GREEN}Building Docker images...${NC}"
        docker-compose build
        ;;
    "stop")
        echo -e "${GREEN}Stopping services...${NC}"
        docker-compose down
        ;;
    "logs")
        echo -e "${GREEN}Showing logs...${NC}"
        docker-compose logs -f
        ;;
    "clean")
        echo -e "${YELLOW}Cleaning up...${NC}"
        docker-compose down -v
        rm -rf data/*.db
        ;;
    *)
        echo "Usage: $0 {dev|frontend|api|agent|test|build|stop|logs|clean}"
        echo ""
        echo "Commands:"
        echo "  dev      - Start full development environment (Docker)"
        echo "  frontend - Start React frontend only (Vite)"
        echo "  api      - Start API server only (local Python)"
        echo "  agent    - Start Elena voice agent (local Python)"
        echo "  test     - Run test suite"
        echo "  build    - Build Docker images"
        echo "  stop     - Stop all services"
        echo "  logs     - Show service logs"
        echo "  clean    - Clean up containers and data"
        exit 1
        ;;
esac
