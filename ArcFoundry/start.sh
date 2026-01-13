#!/bin/bash

# --- START OF FILE start.sh ---
# ArcFoundry Bootloader
# Level-based function architecture for stability and maintainability.

# Global Constants
SDK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SDK_ROOT}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"
LOG_DIR="${SDK_ROOT}/workspace/logs"
LOG_FILE="${LOG_DIR}/sys_boot.log"

# ==============================================================================
# Level 0: Helper Functions & Logging
# ==============================================================================

func_0_0_help() {
    echo "Usage: $(basename "$0") [command] [options]"
    echo ""
    echo "Commands:"
    echo "  init        Initialize environment (venv, dependencies)"
    echo "  convert     Run conversion pipeline (wraps core python logic)"
    echo "  clean       Remove workspace and temporary files"
    echo "  distclean   Remove venv and all generated files"
    echo "  help        Show this help message"
    echo ""
    echo "Options:"
    echo "  -c <config> Path to YAML config file (required for convert)"
    echo ""
}

func_0_1_log() {
    local msg="$1"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[${timestamp}] ${msg}" | tee -a "${LOG_FILE}"
}

func_0_2_error() {
    local msg="$1"
    func_0_1_log "[ERROR] ${msg}"
    exit 1
}

# ==============================================================================
# Level 1: Environment Initialization
# ==============================================================================

func_1_0_init_workspace() {
    if [ ! -d "${LOG_DIR}" ]; then
        mkdir -p "${LOG_DIR}"
    fi
    # Create log file if not exists
    touch "${LOG_FILE}"
}

# ==============================================================================
# Level 2: System & Dependency Checks
# ==============================================================================

func_2_0_check_sys_python() {
    func_0_1_log "Checking system Python 3.8..."
    if command -v python3.8 &> /dev/null; then
        func_0_1_log "Python 3.8 found: $(which python3.8)"
    else
        func_0_2_error "Python 3.8 not found. Please install it first (apt install python3.8 python3.8-venv)."
    fi
}

func_2_1_setup_venv() {
    if [ -d "${VENV_DIR}" ]; then
        func_0_1_log "Virtual environment detected at ${VENV_DIR}"
        return 0
    fi

    func_0_1_log "Creating virtual environment..."
    python3.8 -m venv "${VENV_DIR}" || func_0_2_error "Failed to create venv."
    
    # Upgrade pip immediately
    "${PIP_BIN}" install --upgrade pip >> "${LOG_FILE}" 2>&1
    func_0_1_log "Virtual environment created successfully."
}

func_2_2_install_requirements() {
    func_0_1_log "Installing/Updating dependencies from requirements.txt..."
    if [ -f "${SDK_ROOT}/requirements.txt" ]; then
        "${PIP_BIN}" install -r "${SDK_ROOT}/requirements.txt" >> "${LOG_FILE}" 2>&1 || func_0_2_error "Dependency installation failed. Check log."
    else
        func_0_1_log "[WARN] requirements.txt not found. Skipping pip install."
    fi
    
    # Note: RKNN Toolkit2 whl installation logic should be handled here or manually
    # For V1, we assume user might manually install the whl or via pip if available
    func_0_1_log "Dependencies checked."
}

# ==============================================================================
# Level 3: Core Execution Bridge
# ==============================================================================

func_3_0_run_init() {
    func_1_0_init_workspace
    func_2_0_check_sys_python
    func_2_1_setup_venv
    func_2_2_install_requirements
    func_0_1_log "Initialization complete. SDK is ready."
}

func_3_1_run_convert() {
    # Ensure environment is ready before running logic
    if [ ! -f "${PYTHON_BIN}" ]; then
        func_0_1_log "[WARN] Environment not found. Auto-initializing..."
        func_3_0_run_init
    fi

    local args="$@"
    func_0_1_log "Handing over to Python Kernel..."
    
    # --- HANDOVER: Replace Shell Process with Python Process ---
    # This keeps the PID clean and propagates signals correctly
    export PYTHONPATH="${SDK_ROOT}"
    exec "${PYTHON_BIN}" "${SDK_ROOT}/core/main.py" $args
}

func_3_2_clean() {
    func_0_1_log "Cleaning workspace..."
    rm -rf "${SDK_ROOT}/workspace"
    rm -rf "${SDK_ROOT}/output"
    func_0_1_log "Workspace cleaned."
}

func_3_3_distclean() {
    func_3_2_clean
    func_0_1_log "Removing virtual environment..."
    rm -rf "${VENV_DIR}"
    func_0_1_log "Full cleanup complete."
}

# ==============================================================================
# Main Entry Point
# ==============================================================================

main() {
    # Ensure workspace exists for logging
    func_1_0_init_workspace

    if [ $# -eq 0 ]; then
        func_0_0_help
        exit 0
    fi

    case "$1" in
        init)
            func_3_0_run_init
            ;;
        convert)
            shift # Remove 'convert' from args
            func_3_1_run_convert "$@"
            ;;
        clean)
            func_3_2_clean
            ;;
        distclean)
            func_3_3_distclean
            ;;
        help|-h|--help)
            func_0_0_help
            ;;
        *)
            func_0_2_error "Unknown command: $1. Use 'help' for usage."
            ;;
    esac
}

# Execute Main
main "$@"
