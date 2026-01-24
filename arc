#!/bin/bash

# set -xe

# ==============================================================================
# Level 1: Helpers
# ==============================================================================
func_1_1_log() { echo -e "\033[1;32m[ArcFoundry]\033[0m $1"; }
func_1_2_err() { echo -e "\033[1;31m[ERROR]\033[0m $1"; exit 1; }

func_1_3_get_python_version() {
    # Description:
    #   1. Define a helper function to check if a given python command meets version requirements (3.8 <= v <= 3.12)
    # Returns:
    #   0 if valid, 1 if invalid
    _check_py_validity() {
        local py_cmd="$1"

        # 1. 检查命令是否存在
        if ! command -v "$py_cmd" &>/dev/null; then
            return 1
        fi

        # 2. 使用 python 自身来判断版本号
        # 逻辑：major必须是3，minor必须在8到12之间
        "$py_cmd" -c "import sys; v=sys.version_info; sys.exit(0 if (v.major == 3 and 8 <= v.minor <= 12) else 1)" 2>/dev/null
        return $?
    }

    # --- First Stage: Check default python3 ---
    # This addresses the pain point where 'python3' might be symlinked to 'python3.8'
    if _check_py_validity "python3"; then
        basename $(readlink -f $(which python3))
        return 0
    fi

    # --- Second Stage: Check default python ---
    if _check_py_validity "python"; then
        basename $(readlink -f $(which python))
        return 0
    fi

    # --- Third Stage: Check specific versions in order ---
    # The order here defines the fallback priority (newer stable versions first)
    for py in python3.11 python3.10 python3.12 python3.9 python3.8; do
        if _check_py_validity "$py"; then
            echo "$py"
            return 0
        fi
    done

    # --- Fourth Stage: No valid python found ---
    func_1_2_err "No suitable Python found (need 3.8~3.12)"
}

func_1_4_show_help() {
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

# RKNN Toolkit Management
func_1_5_install_rknn() {
    # 1. Check if already installed
    if "${PYTHON_BIN}" -c "import rknn.api" &> /dev/null; then
        return 0
    fi

    func_1_1_log "RKNN Toolkit2 not found. Initiating auto-install..."

    # 2. Define Paths
    local repo_dir="${SDK_ROOT}/rockchip-repos/rknn-toolkit2.git"
    local repo_url="https://github.com/airockchip/rknn-toolkit2.git"

    # 3. Clone if missing
    if [ ! -d "${repo_dir}" ]; then
        func_1_1_log "Cloning rknn-toolkit2 repository (Depth 1)..."
        # Ensure parent dir exists
        mkdir -p "$(dirname "${repo_dir}")"
        git clone --depth 1 "${repo_url}" "${repo_dir}" || func_1_2_err "Failed to clone repo."
    fi

    # 4. Find the correct Wheel file for Python 3.x (cp3x) on x86_64
    # Pattern: rknn_toolkit2-*-cp3x-cp3x-manylinux*x86_64.whl
    func_1_1_log "Searching for wheel package..."
    local search_path="${repo_dir}/rknn-toolkit2/packages/x86_64"
    local requirements_file=$(find "${search_path}" -name "requirements_${WHEEL_TAG}-*.txt" | head -n 1)
    local whl_file=$(find "${search_path}" -name "rknn_toolkit2*-${WHEEL_TAG}-${WHEEL_TAG}-*x86_64.whl" | head -n 1)

    if [ -z "${whl_file}" ]; then
        func_1_2_err "Could not find compatible .whl file in: ${search_path}"
    fi

    # 5. Install requirements.txt
    func_1_1_log "\nInstalling requirements.txt..."
    "${PIP_BIN}" install -r "${requirements_file}" || func_1_2_err "Failed to install requirements.txt."

    # 5. Install
    func_1_1_log "\nInstalling: $(basename "${whl_file}")"
    "${PIP_BIN}" install "${whl_file}" || func_1_2_err "Failed to install RKNN Toolkit2."

    # # 6. Verify
    # if ! "${PYTHON_BIN}" -c "import rknn.api" &> /dev/null; then
    #     func_1_2_err "Installation completed but import failed."
    # fi

    func_1_1_log "RKNN Toolkit2 installed successfully."
}

func_1_6_setup_environment_vars() {

    SDK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    VENV_DIR="${SDK_ROOT}/.venv"
    CONFIG_DIR="${SDK_ROOT}/configs"
    LOG_DIR="${SDK_ROOT}/workspace/logs"
    MODELS_DIR="${SDK_ROOT}/models"
    RK_REPOS_DIR="${SDK_ROOT}/rockchip-repos"

    # 1. Determine Venv Python Binary and Version
    PYTHON_BIN_NAME=$(func_1_3_get_python_version)
    PYTHON_BIN="${VENV_DIR}/bin/python"
    PIP_BIN="${VENV_DIR}/bin/pip"


    # 2. Determine Host Python Binary and Version
    HOST_PYTHON_BIN_NAME=$PYTHON_BIN_NAME
    HOST_PYTHON_BIN=$(which $HOST_PYTHON_BIN_NAME)

    # 2. Determine Wheel Tag based on Python Version
    case "$PYTHON_BIN_NAME" in
        python3.8)  WHEEL_TAG="cp38" ;;
        python3.9)  WHEEL_TAG="cp39" ;;
        python3.10) WHEEL_TAG="cp310" ;;
        python3.11) WHEEL_TAG="cp311" ;;
        python3.12) WHEEL_TAG="cp312" ;;
        *) func_1_2_err "Unsupported Python version: $PYTHON_BIN_NAME" ;;
    esac
}


# ==============================================================================
# Level 2: Environment Logic
# ==============================================================================
func_2_1_setup_venv() {

    # 1. Check/Create Venv
    if [ ! -f "${VENV_DIR}/bin/python" ]; then
        func_1_1_log "Initializing virtual environment..."
        # if ! command -v python3.8 &> /dev/null; then
        #     func_1_2_err "Python 3.8 not installed. Run: sudo apt install python3.8 python3.8-venv"
        # fi
        if ! ${HOST_PYTHON_BIN} -m venv "${VENV_DIR}"; then
            func_1_2_err "Please install the venv package as the hint above and Re-execute me again."
            return 1
        fi
        "${PIP_BIN}" install --upgrade pip
    fi

    # 2. Check/Install RKNN Toolkit2 (The Auto-Magic Step)
    func_1_5_install_rknn

    # 3. Check Dependencies (Force check for V1.1 new deps: requests)
    #    We should run after RKNN Toolkit2 is installed to avoid conflicts.
    if ! "${PYTHON_BIN}" -c "import requests, tqdm, yaml" &> /dev/null; then
        func_1_1_log "\nInstalling/Updating Our own dependencies..."
        "${PIP_BIN}" install -r "${SDK_ROOT}/envs/requirements.txt"
    fi
}


func_2_2_clean() {
    rm -rf "${SDK_ROOT}/workspace" "${SDK_ROOT}/output"
    func_1_1_log "Workspace, Output cleaned."

    # 2. Clean Root Artifacts (The "check*.onnx" files)
    rm -f "${SDK_ROOT}"/check*.onnx \
          "${SDK_ROOT}"/debug*.onnx \
          "${SDK_ROOT}"/verify*.onnx \
          "${SDK_ROOT}"/*.rknn_util_*.log

    func_1_1_log "root artifacts cleaned."
}

func_2_3_distclean() {
    func_2_2_clean

    rm -rf "${VENV_DIR}"
    func_1_1_log "Virtual Environment removed. Factory reset complete."

    rm -rf "${MODELS_DIR}"
    func_1_1_log "Models directory removed."

    rm -rf "${RK_REPOS_DIR}"
    func_1_1_log "Rockchip repositories removed."
}


# ==============================================================================
# Level 3:
# ==============================================================================
# Returns result via Global Variable 'SELECTED_CONFIG' to avoid stdout pollution
func_3_1_resolve_config() {
    local input="$1"
    SELECTED_CONFIG=""

    # Case A: User gave nothing -> Show Menu (Interactive)
    if [ -z "$input" ]; then
        local files=(${CONFIG_DIR}/*.yaml)
        if [ ${#files[@]} -eq 0 ] || [ ! -e "${files[0]}" ]; then
             func_1_2_err "No config files found in ${CONFIG_DIR}/"
        fi

        func_1_1_log "Available Configurations:"
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
            func_1_2_err "Invalid selection."
        fi

    # Case B: User gave a short name (e.g., "rv1126b_sherpa")
    elif [ -f "${CONFIG_DIR}/${input}.yaml" ]; then
        SELECTED_CONFIG="${CONFIG_DIR}/${input}.yaml"

    # Case C: User gave a specific file path
    elif [ -f "$input" ]; then
        SELECTED_CONFIG="$input"

    else
        func_1_2_err "Configuration not found: $input"
    fi
}

# The actual Python Kernel launcher (Called by Mode 1, 2, 3)
func_3_2_launch_kernel() {
    if [ -z "$SELECTED_CONFIG" ]; then
        func_1_2_err "No configuration selected. Aborting."
        return 1
    fi
    func_1_1_log "Target Config: $(basename ${SELECTED_CONFIG})"

    # [新增] 选好配置后，才开始检查环境（实现 Lazy Check）
    func_2_1_setup_venv || return 1

    export PYTHONPATH="${SDK_ROOT}"
    #exec "${PYTHON_BIN}" "${SDK_ROOT}/core/main.py" -c "${SELECTED_CONFIG}"
    "${PYTHON_BIN}" "${SDK_ROOT}/core/main.py" -c "${SELECTED_CONFIG}"
    local py_ret=$?  # <--- 这行是之前缺失的，必须紧跟在 python 命令后面

    # --- Post-Execution Cleanup (The Shield) ---
    # Move RKNN generated intermediate files to workspace instead of littering root
    local dump_dir="${SDK_ROOT}/workspace/rknn_dumps"
    mkdir -p "${dump_dir}"

    # Move files if they exist (suppress errors if not found)
    mv "${SDK_ROOT}"/check*.onnx "${dump_dir}/" 2>/dev/null
    mv "${SDK_ROOT}"/debug*.onnx "${dump_dir}/" 2>/dev/null

    if [ $py_ret -ne 0 ]; then
        func_1_2_err "Conversion failed with error code ${py_ret}."
    fi

    func_1_1_log "Done."
}

# ==============================================================================
# Level 4: Execution Modes (The 3 Ways to Run)
# ==============================================================================

# Mode 1: Interactive Menu (Lazy Mode)
func_4_1_mode_menu() {
    func_3_1_resolve_config ""  # Passing empty triggers the menu
    func_3_2_launch_kernel || return 1
}

# Mode 2 & 3: Short Name or Explicit Path
func_4_2_mode_direct() {
    local target="$1"

    func_3_1_resolve_config "$target"
    func_3_2_launch_kernel || return 1
}

# ==============================================================================
# Main Entry Point (The Router)
# ==============================================================================
main() {
    # 1. Setup Env Vars
    func_1_6_setup_environment_vars

    # Pre-setup: Ensure log dir exists
    mkdir -p "${LOG_DIR}"

    # Route by Argument Count
    if [ $# -eq 0 ]; then
        # No arguments: Enter Mode 1 (Interactive)
        func_4_1_mode_menu || exit
        exit 0
    fi

    if [ $# -eq 1 ]; then
        # One argument: Check if it's a command or a config
        case "$1" in
            help|-h|--help) func_1_4_show_help ;;
            clean)          func_2_2_clean ;;
            distclean)      func_2_3_distclean ;;
            init)           func_2_1_setup_venv || exit 1 ;;
            *)              func_4_2_mode_direct "$1" ;; # If not command, treat as Mode 2/3
        esac
        exit 0
    fi

    # Fallback for Legacy calls (e.g. `./start.sh convert -c ...`)
    if [ "$1" == "convert" ] && [ "$2" == "-c" ] && [ -n "$3" ]; then
        func_4_2_mode_direct "$3"
    elif [ "$1" == "convert" ] && [ -n "$2" ]; then
        func_4_2_mode_direct "$2"
    else
        func_1_2_err "Invalid arguments. Try './start.sh help'."
    fi
}


main "$@"
