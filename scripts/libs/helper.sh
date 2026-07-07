


# ==============================================================================
# Level 1: Helpers
# ==============================================================================
func_1_1_log() {
    echo -e "\033[1;32m[ArcFoundry]\033[0m $1";
}
func_1_2_err() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"; exit 1;
}
func_1_3_debug() {
    # echo "DEBUG_MODE=${DEBUG_MODE}"
    if [ "${DEBUG_MODE:-0}" -ne 1 ]; then
        return
    fi
    echo -e "\033[1;33m[Debug]\033[0m $1";
}

func_1_13_get_python_version() {
    # Description:
    #   1. Define a helper function to check if a given python command meets version requirements (3.8 <= v <= 3.12)
    # Returns:
    #   0 if valid, 1 if invalid

    _check_py_validity() {
        local py_cmd="$1"

        # 1. Check if command exists
        if ! command -v "$py_cmd" &>/dev/null; then
            return 1
        fi

        # 2. Utilize Python to check version
        # Logic: major must be 3, minor must be between 8 and 12 inclusive
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
    echo "   $ ./arc"
    echo "   (Shows a list of all configs to select)"
    echo ""
    echo "2. Short Name Mode (Toolchain Style):"
    echo "   $ ./arc rv1126b_sherpa"
    echo "   (Automatically finds configs/rv1126b_sherpa.yaml)"
    echo ""
    echo "3. Explicit Path Mode:"
    echo "   $ ./arc configs/my_custom_config.yaml"
    echo ""
    echo "--- Maintenance Commands ---"
    echo "  clean       : Remove workspace/ and output/ directories"
    echo "  distclean   : Remove .venv/, workspace/ models/, rockchip-repos/ and output/ ALMOST EVERYTHING DANGER (Factory Reset)"
    echo "  init        : Force env init (clone/pin Rockchip repos + install rknn-toolkit2 wheel; needs network)"
    echo "  help, -h    : Show this help message"
    echo ""
    echo "--- Use a local RKNN SDK tarball (rknpu2) ---"
    echo "  Default: clones rknn-toolkit2 from GitHub."
    echo "  To use a local .tgz (e.g. an SDK version not on GitHub):"
    echo "      cp rk-toolchain.env.example rk-toolchain.env"
    echo "      \$EDITOR rk-toolchain.env   # set RKNN_TARBALL_PATH (+ SHA256)"
    echo "  arc auto-detects rk-toolchain.env and installs from it."
}

func_1_6_setup_environment_vars() {
    if [ "$V" = "1" ]; then
        DEBUG_MODE="1"
        set -x
    fi

    # Preparation -- 1. Define Most Important Timers
    # __ELAPSED_TIME=
    __START_TIME=
    __SECOND_STAGE_BUILD_START_TIME=

    # ==============================================================================
    # Pinned Rockchip Repo SHAs
    # ------------------------------------------------------------------------------
    # WHY pin: every team member must run the SAME upstream snapshot. Without
    # pinning, a teammate who runs `./arc init` two months from now may pull a
    # newer rknn_model_zoo whose example outputs no longer match our golden
    # numerics, or a newer rknn-toolkit2 whose wheel is incompatible with the
    # one we tested. Pinning makes roll-back and bug-bisection deterministic.
    #
    # Both pins correspond to RKNN-Toolkit2 v2.3.2 (verified on 2026-07-04):
    #   * rknn_model_zoo: HEAD == tag v2.3.2 == bad6c73...
    #   * rknn-toolkit2 : HEAD has no tag, but wheel/requirements inside the repo
    #                     are stamped "2.3.2" (commit message "Update LICENSE")
    #
    # To bump these pins: change the values here, then commit. Do NOT edit
    # silently — every change must be reflected in a commit so git bisect works.
    # ==============================================================================
    RKNN_REPO_BASE_URL="https://github.com/airockchip"
    RKNN_TOOLKIT2_REPO_URL="${RKNN_REPO_BASE_URL}/rknn-toolkit2.git"
    RKNN_TOOLKIT2_PINNED_SHA="59a913d172e7f5ff03c9076e2ec7b1b1288ffd08"
    RKNN_MODEL_ZOO_REPO_URL="${RKNN_REPO_BASE_URL}/rknn_model_zoo.git"
    RKNN_MODEL_ZOO_PINNED_SHA="bad6c7334531becaf90a561988519b7bec34d0ab"
    RKNN_MODEL_ZOO_PINNED_TAG="v2.3.2"  # for human-readable log only; clone uses SHA

    # Processing -- 1. start timer
    func_1_9_start_time_count __START_TIME
    # func_1_1_log "Environment setup started at ${__START_TIME}."

    # Preparation -- 2. Ensure Finalization on Exit
    trap func_2_4_finalize EXIT

    # Preparation -- 3. Define Important Paths
    SDK_ROOT="${REPO_TOP_DIR}"
    VENV_DIR="${SDK_ROOT}/.venv"
    CONFIG_DIR="${SDK_ROOT}/configs"
    LOG_DIR="${SDK_ROOT}/workspace/logs"
    MODELS_DIR="${SDK_ROOT}/models"
    RK_REPOS_DIR="${SDK_ROOT}/rockchip-repos"

    # 1. Determine Venv Python Binary and Version
    PYTHON_BIN_NAME=$(func_1_13_get_python_version)
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
