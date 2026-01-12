#!/bin/bash
# ============================================================================
# KmiDi Linux Deployment Script
# ============================================================================
# Usage: ./deploy-linux.sh [options]
#
# Options:
#   --dev       Deploy in development mode (with hot reload)
#   --prod      Deploy in production mode (default)
#   --api-only  Deploy only the FastAPI service
#   --full      Deploy all services (API + Streamlit + LLM)
#   --gpu       Enable NVIDIA GPU support (requires nvidia-docker)
#   --stop      Stop all running containers
#   --clean     Stop and remove all containers, volumes
#   --logs      Show container logs
#   --status    Show deployment status
#   --help      Show this help message
#
# Prerequisites:
#   - Docker Engine installed
#   - Docker Compose v2+ installed
#   - (Optional) NVIDIA Container Toolkit for GPU support
#
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
DOCKER_DIR="$PROJECT_ROOT/deployment/docker"

# Default settings
MODE="prod"
SERVICES="api"
ACTION="deploy"
USE_GPU=false

# Functions
print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              KmiDi - Linux Deployment Script                ║"
    echo "║                    Music Intelligence API                   ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✖ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✔ $1${NC}"
}

check_prerequisites() {
    print_step "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed."
        echo "  Install: curl -fsSL https://get.docker.com | sh"
        exit 1
    fi

    # Check Docker daemon
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running or you don't have permissions."
        echo "  Try: sudo systemctl start docker"
        echo "  Or add user to docker group: sudo usermod -aG docker \$USER"
        exit 1
    fi

    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose v2 is not available."
        echo "  Install: sudo apt-get install docker-compose-plugin"
        exit 1
    fi

    # Check GPU support if requested
    if [ "$USE_GPU" = true ]; then
        if ! command -v nvidia-smi &> /dev/null; then
            print_error "nvidia-smi not found. NVIDIA drivers may not be installed."
            exit 1
        fi

        if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
            print_error "NVIDIA Container Toolkit not working."
            echo "  Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            exit 1
        fi
        print_success "NVIDIA GPU support verified"
    fi

    # Check available memory
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEM" -lt 4 ]; then
        print_warning "System has less than 4GB RAM. Some features may not work properly."
    fi

    print_success "Prerequisites check passed"
}

setup_environment() {
    print_step "Setting up environment..."

    cd "$PROJECT_ROOT"

    # Create .env if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f "env.example" ]; then
            cp env.example .env
            print_warning "Created .env from env.example. Please review and update settings."
        else
            print_error "No env.example found. Cannot create .env file."
            exit 1
        fi
    fi

    # Create necessary directories with proper permissions
    mkdir -p output models logs data
    chmod 755 output models logs data

    # Set Linux-specific defaults if not set
    if ! grep -q "KELLY_AUDIO_DATA_ROOT" .env 2>/dev/null; then
        echo "KELLY_AUDIO_DATA_ROOT=$HOME/kmidi-data" >> .env
    fi

    # Set training device based on GPU availability
    if [ "$USE_GPU" = true ]; then
        sed -i 's/TRAINING_DEVICE=.*/TRAINING_DEVICE=cuda/' .env 2>/dev/null || true
        sed -i 's/KELLY_DEVICE=.*/KELLY_DEVICE=cuda/' .env 2>/dev/null || true
    fi

    print_success "Environment configured"
}

build_images() {
    print_step "Building Docker images..."

    cd "$DOCKER_DIR"

    if [ "$MODE" = "dev" ]; then
        docker compose build --no-cache daiw-dev
    elif [ "$USE_GPU" = true ]; then
        docker compose -f docker-compose.yml build daiw-ml-training
    else
        docker compose -f docker-compose.yml build
    fi

    print_success "Docker images built"
}

deploy_services() {
    print_step "Deploying services..."

    cd "$DOCKER_DIR"

    GPU_FLAG=""
    if [ "$USE_GPU" = true ]; then
        GPU_FLAG="--gpus all"
    fi

    case "$SERVICES" in
        "api")
            if [ "$MODE" = "dev" ]; then
                docker compose up -d daiw-dev
            else
                # Use production Dockerfile
                docker build -t kmidi-api:prod -f Dockerfile.prod "$PROJECT_ROOT"
                docker run -d \
                    --name kmidi-api \
                    -p 8000:8000 \
                    -v "$PROJECT_ROOT/data:/data:ro" \
                    -v "$PROJECT_ROOT/models:/models:ro" \
                    -v "$PROJECT_ROOT/output:/output:rw" \
                    --env-file "$PROJECT_ROOT/.env" \
                    --restart unless-stopped \
                    $GPU_FLAG \
                    kmidi-api:prod
            fi
            ;;
        "full")
            docker compose up -d
            ;;
    esac

    print_success "Services deployed"
}

show_status() {
    print_step "Deployment Status"
    echo ""

    # Show running containers
    echo -e "${BLUE}Running Containers:${NC}"
    docker ps --filter "name=kmidi" --filter "name=daiw" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""

    # Health check
    echo -e "${BLUE}Health Checks:${NC}"

    # Check API
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        API_STATUS=$(curl -s http://localhost:8000/health | grep -o '"status":"[^"]*"' | head -1)
        print_success "API: $API_STATUS"
    else
        print_warning "API: Not responding on port 8000"
    fi

    # Check Streamlit
    if curl -sf http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        print_success "Streamlit: Running on port 8501"
    else
        print_warning "Streamlit: Not responding on port 8501"
    fi

    # Check GPU if enabled
    if [ "$USE_GPU" = true ]; then
        echo ""
        echo -e "${BLUE}GPU Status:${NC}"
        nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || print_warning "GPU info not available"
    fi

    echo ""
    echo -e "${BLUE}Endpoints:${NC}"
    echo "  API:        http://localhost:8000"
    echo "  API Docs:   http://localhost:8000/docs"
    echo "  Metrics:    http://localhost:8000/metrics"
    echo "  Streamlit:  http://localhost:8501"
}

stop_services() {
    print_step "Stopping services..."

    docker stop kmidi-api 2>/dev/null || true
    docker rm kmidi-api 2>/dev/null || true

    cd "$DOCKER_DIR"
    docker compose down 2>/dev/null || true

    print_success "Services stopped"
}

clean_all() {
    print_step "Cleaning up all containers and volumes..."

    stop_services

    cd "$DOCKER_DIR"
    docker compose down -v --remove-orphans 2>/dev/null || true

    # Remove images
    docker rmi kmidi-api:prod 2>/dev/null || true
    docker rmi daiw-music-brain:latest 2>/dev/null || true
    docker rmi daiw-music-brain:dev 2>/dev/null || true

    print_success "Cleanup complete"
}

show_logs() {
    cd "$DOCKER_DIR"
    docker compose logs -f
}

show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --dev       Deploy in development mode (with hot reload)"
    echo "  --prod      Deploy in production mode (default)"
    echo "  --api-only  Deploy only the FastAPI service"
    echo "  --full      Deploy all services"
    echo "  --gpu       Enable NVIDIA GPU support"
    echo "  --stop      Stop all running containers"
    echo "  --clean     Stop and remove all containers, volumes"
    echo "  --logs      Show container logs"
    echo "  --status    Show deployment status"
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Deploy API in production mode"
    echo "  $0 --dev              # Deploy in development mode"
    echo "  $0 --full --gpu       # Deploy all services with GPU"
    echo "  $0 --status           # Check deployment status"
    echo "  $0 --stop             # Stop all services"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            MODE="dev"
            shift
            ;;
        --prod)
            MODE="prod"
            shift
            ;;
        --api-only)
            SERVICES="api"
            shift
            ;;
        --full)
            SERVICES="full"
            shift
            ;;
        --gpu)
            USE_GPU=true
            shift
            ;;
        --stop)
            ACTION="stop"
            shift
            ;;
        --clean)
            ACTION="clean"
            shift
            ;;
        --logs)
            ACTION="logs"
            shift
            ;;
        --status)
            ACTION="status"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
print_banner

case "$ACTION" in
    "deploy")
        check_prerequisites
        setup_environment
        build_images
        deploy_services
        echo ""
        show_status
        ;;
    "stop")
        stop_services
        ;;
    "clean")
        clean_all
        ;;
    "logs")
        show_logs
        ;;
    "status")
        show_status
        ;;
esac

echo ""
print_success "Done!"
