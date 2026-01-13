#!/bin/bash

# --- ArcFoundry Bootloader V1.2 (Bugfix Release) ---

SDK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SDK_ROOT}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"
CONFIG_DIR="${SDK_ROOT}/configs"
LOG_DIR="${SDK_ROOT}/workspace/logs"
MODELS_DIR="${SDK_ROOT}/models"
RK_REPOS_DIR="${SDK_ROOT}/rockchip-repos"

# ==============================================================================
# Helpers
# ==============================================================================
func_log() { echo -e "\033[1;32m[ArcFoundry]\033[0m $1"; }
func_err() { echo -e "\033[1;31m[ERROR]\033[0m $1"; exit 1; }

# ==============================================================================
# Level 1.5: RKNN Toolkit Management
# ==============================================================================
func_install_rknn() {
    # 1. Check if already installed
    if "${PYTHON_BIN}" -c "import rknn.api" &> /dev/null; then
        return 0
    fi

    func_log "RKNN Toolkit2 not found. Initiating auto-install..."

    # 2. Define Paths
    local repo_dir="${SDK_ROOT}/rockchip-repos/rknn-toolkit2"
    local repo_url="https://github.com/airockchip/rknn-toolkit2.git"

    # 3. Clone if missing
    if [ ! -d "${repo_dir}" ]; then
        func_log "Cloning rknn-toolkit2 repository (Depth 1)..."
        # Ensure parent dir exists
        mkdir -p "$(dirname "${repo_dir}")"
        git clone --depth 1 "${repo_url}" "${repo_dir}" || func_err "Failed to clone repo."
    fi

    # 4. Find the correct Wheel file for Python 3.8 (cp38) on x86_64
    # Pattern: rknn_toolkit2-*-cp38-cp38-manylinux*x86_64.whl
    func_log "Searching for wheel package..."
    local search_path="${repo_dir}/rknn-toolkit2/packages/x86_64"
    local whl_file=$(find "${search_path}" -name "rknn_toolkit2*-cp38-cp38-*x86_64.whl" | head -n 1)

    if [ -z "${whl_file}" ]; then
        func_err "Could not find compatible .whl file in: ${search_path}"
    fi

    # 5. Install
    func_log "Installing: $(basename "${whl_file}")"
    "${PIP_BIN}" install "${whl_file}" || func_err "Failed to install RKNN Toolkit2."

    # 6. Verify
    if ! "${PYTHON_BIN}" -c "import rknn.api" &> /dev/null; then
        func_err "Installation completed but import failed."
    fi

    func_log "RKNN Toolkit2 installed successfully."
}

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

    # 3. Check/Install RKNN Toolkit2 (The Auto-Magic Step)
    func_install_rknn
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
# Level 3: Execution Modes (The 3 Ways to Run)
# ==============================================================================

# Mode 1: Interactive Menu (Lazy Mode)
func_3_1_mode_menu() {
    func_ensure_env
    func_resolve_config ""  # Passing empty triggers the menu
    func_3_9_launch_kernel
}

# Mode 2 & 3: Short Name or Explicit Path
func_3_2_mode_direct() {
    local target="$1"
    func_ensure_env
    func_resolve_config "$target"
    func_3_9_launch_kernel
}

# The actual Python Kernel launcher (Called by Mode 1, 2, 3)
func_3_9_launch_kernel() {
    if [ -z "$SELECTED_CONFIG" ]; then
        func_err "No configuration selected. Aborting."
    fi
    func_log "Target Config: $(basename ${SELECTED_CONFIG})"
    export PYTHONPATH="${SDK_ROOT}"
    #exec "${PYTHON_BIN}" "${SDK_ROOT}/core/main.py" -c "${SELECTED_CONFIG}"
    "${PYTHON_BIN}" "${SDK_ROOT}/core/main.py" -c "${SELECTED_CONFIG}"

    # --- Post-Execution Cleanup (The Shield) ---
    # Move RKNN generated intermediate files to workspace instead of littering root
    local dump_dir="${SDK_ROOT}/workspace/rknn_dumps"
    mkdir -p "${dump_dir}"

    # Move files if they exist (suppress errors if not found)
    mv "${SDK_ROOT}"/check*.onnx "${dump_dir}/" 2>/dev/null
    mv "${SDK_ROOT}"/debug*.onnx "${dump_dir}/" 2>/dev/null

    if [ $py_ret -ne 0 ]; then
        func_err "Conversion failed with error code ${py_ret}."
    fi

    func_log "Done."
}

# ==============================================================================
# Level 4: Maintenance & Help
# ==============================================================================

func_4_0_show_help() {
    echo "Usage: $(basename "$0") [command | mode_argument]"
    echo ""
    echo "--- 3 Ways to Run (Modes) ---"
    echo "1. Interactive Menu (Lazy Mode):"
    echo "   $ ./start.sh"
    echo "   (Shows a list of all configs to select)"
    echo ""
    echo "2. Short Name Mode (Toolchain Style):"
    echo "   $ ./start.sh rv1126b_sherpa"
    echo "   (Automatically finds configs/rv1126b_sherpa.yaml)"
    echo ""
    echo "3. Explicit Path Mode:"
    echo "   $ ./start.sh configs/my_custom_config.yaml"
    echo ""
    echo "--- Maintenance Commands ---"
    echo "  clean       : Remove workspace/ and output/ directories"
    echo "  distclean   : Remove .venv/, workspace/ models/, rockchip-repos/ and output/ ALMOST EVERYTHING DANGER (Factory Reset)"
    echo "  init        : Force environment initialization"
    echo "  help, -h    : Show this help message"
}

func_4_1_clean() {
    rm -rf "${SDK_ROOT}/workspace" "${SDK_ROOT}/output"
    func_log "Workspace, Output cleaned."

    # 2. Clean Root Artifacts (The "check*.onnx" files)
    rm -f "${SDK_ROOT}"/check*.onnx \
          "${SDK_ROOT}"/debug*.onnx \
          "${SDK_ROOT}"/verify*.onnx \
          "${SDK_ROOT}"/*.rknn_util_*.log

    func_log "root artifacts cleaned."
}

func_4_2_distclean() {
    func_4_1_clean

    rm -rf "${VENV_DIR}"
    func_log "Virtual Environment removed. Factory reset complete."

    rm -rf "${MODELS_DIR}"
    func_log "Models directory removed."

    rm -rf "${RK_REPOS_DIR}"
    func_log "Rockchip repositories removed."
}

# ==============================================================================
# Main Entry Point (The Router)
# ==============================================================================
main() {
    # Pre-setup: Ensure log dir exists
    mkdir -p "${LOG_DIR}"

    # Route by Argument Count
    if [ $# -eq 0 ]; then
        # No arguments: Enter Mode 1 (Interactive)
        func_3_1_mode_menu
        exit 0
    fi

    if [ $# -eq 1 ]; then
        # One argument: Check if it's a command or a config
        case "$1" in
            help|-h|--help) func_4_0_show_help ;;
            clean)          func_4_1_clean ;;
            distclean)      func_4_2_distclean ;;
            init)           func_ensure_env ;;
            *)              func_3_2_mode_direct "$1" ;; # If not command, treat as Mode 2/3
        esac
        exit 0
    fi

    # Fallback for Legacy calls (e.g. `./start.sh convert -c ...`)
    if [ "$1" == "convert" ] && [ "$2" == "-c" ] && [ -n "$3" ]; then
        func_3_2_mode_direct "$3"
    elif [ "$1" == "convert" ] && [ -n "$2" ]; then
        func_3_2_mode_direct "$2"
    else
        func_err "Invalid arguments. Try './start.sh help'."
    fi
}


main "$@"
