#!/bin/bash

# --- ArcFoundry Bootloader V1.2 (Bugfix Release) ---

SDK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SDK_ROOT}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"
CONFIG_DIR="${SDK_ROOT}/configs"
LOG_DIR="${SDK_ROOT}/workspace/logs"

# ==============================================================================
# Helpers
# ==============================================================================
func_log() { echo -e "\033[1;32m[ArcFoundry]\033[0m $1"; }
func_err() { echo -e "\033[1;31m[ERROR]\033[0m $1"; exit 1; }

# ==============================================================================
# Level 1: Environment Logic
# ==============================================================================
func_ensure_env() {
    # 1. Check/Create Venv
    if [ ! -f "${PYTHON_BIN}" ]; then
        func_log "Initializing virtual environment..."
        if ! command -v python3.8 &> /dev/null; then
            func_err "Python 3.8 not installed. Run: sudo apt install python3.8 python3.8-venv"
        fi
        python3.8 -m venv "${VENV_DIR}"
        "${PIP_BIN}" install --upgrade pip
    fi
    
    # 2. Check Dependencies (Force check for V1.1 new deps: requests)
    # If import requests fails, we run pip install
    if ! "${PYTHON_BIN}" -c "import requests, tqdm, yaml" &> /dev/null; then
        func_log "Installing/Updating dependencies..."
        "${PIP_BIN}" install -r "${SDK_ROOT}/requirements.txt"
    fi
}

# ==============================================================================
# Level 2: Smart Selection Logic (Fixed)
# ==============================================================================
# Returns result via Global Variable 'SELECTED_CONFIG' to avoid stdout pollution
func_resolve_config() {
    local input="$1"
    SELECTED_CONFIG=""

    # Case A: User gave nothing -> Show Menu (Interactive)
    if [ -z "$input" ]; then
        local files=(${CONFIG_DIR}/*.yaml)
        if [ ${#files[@]} -eq 0 ] || [ ! -e "${files[0]}" ]; then
             func_err "No config files found in ${CONFIG_DIR}/"
        fi
        
        func_log "Available Configurations:"
        # Use simple array iteration for cleaner output than 'select'
        local i=1
        for f in "${files[@]}"; do
            echo "  $i) $(basename "$f")"
            ((i++))
        done
        
        echo -n "Select config number: "
        read choice
        
        # Validate input
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -lt "$i" ]; then
            SELECTED_CONFIG="${files[$((choice-1))]}"
        else
            func_err "Invalid selection."
        fi

    # Case B: User gave a short name (e.g., "rv1126b_sherpa")
    elif [ -f "${CONFIG_DIR}/${input}.yaml" ]; then
        SELECTED_CONFIG="${CONFIG_DIR}/${input}.yaml"
        
    # Case C: User gave a specific file path
    elif [ -f "$input" ]; then
        SELECTED_CONFIG="$input"
        
    else
        func_err "Configuration not found: $input"
    fi
}

# ==============================================================================
# Main Entry Point
# ==============================================================================
main() {
    # 1. Handle maintenance
    if [ "$1" == "clean" ]; then
        rm -rf "${SDK_ROOT}/workspace" "${SDK_ROOT}/output"
        func_log "Cleaned workspace."
        exit 0
    elif [ "$1" == "distclean" ]; then
        rm -rf "${SDK_ROOT}/workspace" "${SDK_ROOT}/output" "${VENV_DIR}"
        func_log "Factory reset complete."
        exit 0
    elif [ "$1" == "help" ] || [ "$1" == "-h" ]; then
        echo "Usage: ./start.sh [config_name | config_path]"
        exit 0
    fi

    # 2. Prepare Environment
    mkdir -p "${LOG_DIR}"
    func_ensure_env

    # 3. Resolve Target (Handle legacy 'convert -c' args if present)
    TARGET_ARG="$1"
    if [ "$1" == "convert" ]; then
        if [ "$2" == "-c" ]; then TARGET_ARG="$3"; else TARGET_ARG="$2"; fi
    fi

    # 4. Resolve Config Path
    func_resolve_config "$TARGET_ARG"
    
    if [ -z "$SELECTED_CONFIG" ]; then
        func_err "No config selected."
    fi

    func_log "Target Config: $(basename ${SELECTED_CONFIG})"

    # 5. Launch Python Kernel
    export PYTHONPATH="${SDK_ROOT}"
    exec "${PYTHON_BIN}" "${SDK_ROOT}/core/main.py" -c "${SELECTED_CONFIG}"
}

main "$@"
