#!/bin/bash

# set -xe

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
}

# RKNN Toolkit Management
# ------------------------------------------------------------------------------
# Toolchain override flow (optional):
#   - Bash side: func_3_3_load_toolchain_overrides reads the user config +
#     .config/common/rk-toolchain.yaml via tools/load_toolchain_overrides.py
#     and sets CFG_RKNN_TARBALL_PATH / CFG_RKNN_TARBALL_SHA256.
#   - This side: func_1_5_install_rknn ALWAYS git-clones the official repo
#     (giving us the directory structure that example scripts depend on).
#     If CFG_RKNN_TARBALL_PATH is set, func_1_5_1_1_overlay_with_tarball
#     overwrites the INNER rknn-toolkit2/ subdirectory with the tarball's
#     content BEFORE wheel search. .git/ stays intact so verify_pinned_sha
#     keeps working.
# ------------------------------------------------------------------------------

# Overlay rknn-toolkit2 source with a local tarball.
# Called from func_1_5_install_rknn AFTER git clone (which provides the
# directory structure that example scripts depend on, see realpath.index
# in arc) but BEFORE wheel search (so pip installs the tarball's wheel).
#
# Why overlay, not replace: example scripts under rockchip-repos/rknn-toolkit2
# do realpath.index('rknn-toolkit2') to extend sys.path. We must keep the
# OUTER rknn-toolkit2/ directory (where .git/ lives); only the INNER
# rknn-toolkit2/ subdirectory (containing packages/, doc/, examples/) gets
# overwritten from the tarball.
#
# Args:
#   $1 = absolute path to .tgz file (already validated to exist by Python)
#   $2 = expected SHA256 (empty = skip check, with warning)
func_1_5_1_1_overlay_with_tarball() {
    local tarball_path="$1"
    local expected_sha256="$2"
    local repo_dir="${SDK_ROOT}/rockchip-repos/rknn-toolkit2"

    func_1_1_log "Overlaying SDK source with tarball: $(basename "${tarball_path}")"

    # 1. Existence re-check (in case file was deleted between validate and overlay)
    if [ ! -f "${tarball_path}" ]; then
        func_1_2_err "Tarball disappeared between validate and overlay: ${tarball_path}"
    fi

    # 2. SHA256 check
    if [ -n "${expected_sha256}" ]; then
        local actual_sha256
        actual_sha256=$(sha256sum "${tarball_path}" | awk '{print $1}')
        if [ "${actual_sha256}" != "${expected_sha256}" ]; then
            func_1_2_err "SHA256 mismatch for tarball:
    File:      ${tarball_path}
    Expected:  ${expected_sha256}
    Actual:    ${actual_sha256}
  Fix: re-download the tarball from the Rockchip distribution channel and update rknn_toolkit2.tarball_sha256 in your config."
        fi
        func_1_1_log "  SHA256 verified: ${actual_sha256:0:16}..."
    else
        func_1_1_log "  WARNING: No SHA256 specified, skipping integrity check"
    fi

    # 3. Extract to temp dir
    local tmp_dir
    tmp_dir=$(mktemp -d -t rknn-tarball-XXXXXX) \
        || func_1_2_err "Failed to create temp dir for tarball extraction"
    # Ensure cleanup on any error path
    trap "rm -rf '${tmp_dir}'" RETURN

    func_1_1_log "  Extracting tarball..."
    tar -xzf "${tarball_path}" -C "${tmp_dir}" \
        || func_1_2_err "Failed to extract tarball"

    # 4. Locate the rknn-toolkit2/ subdirectory inside the tarball.
    # The tarball's TOP-LEVEL dir name is unstable across versions (e.g.
    # rknn-toolkit2-v2.4.0-2026-01-17-RV1126B/), but the SECOND-level
    # rknn-toolkit2/ dir (containing packages/, doc/, examples/) is consistent
    # in v2.3.2 and v2.4.0. We search for it.
    local tarball_sdk_dir
    tarball_sdk_dir=$(find "${tmp_dir}" -maxdepth 3 -type d -name rknn-toolkit2 | head -n 1)
    if [ -z "${tarball_sdk_dir}" ]; then
        func_1_2_err "Could not find rknn-toolkit2/ subdirectory inside tarball.
    Expected structure: <top>/rknn-toolkit2/packages/x86_64/*.whl
    Top-level contents: $(ls -1 "${tmp_dir}")"
    fi

    # 5. Verify the wheel path actually exists in the tarball
    if [ ! -d "${tarball_sdk_dir}/packages/x86_64" ]; then
        func_1_2_err "Found rknn-toolkit2/ in tarball, but no packages/x86_64/ inside.
    This tarball may not be an x86_64 RKNN SDK.
    Tarball sdk dir: ${tarball_sdk_dir}"
    fi

    # 6. Overwrite INNER rknn-toolkit2/ subdirectory
    #    (OUTER rknn-toolkit2/ with .git/ stays intact)
    func_1_1_log "  Overwriting ${repo_dir}/rknn-toolkit2/ with tarball content..."
    rm -rf "${repo_dir}/rknn-toolkit2"
    cp -r "${tarball_sdk_dir}" "${repo_dir}/rknn-toolkit2" \
        || func_1_2_err "Failed to copy tarball content into ${repo_dir}/"

    func_1_1_log "  Tarball overlay complete. Wheel source now from tarball."
}

# RKNN Toolkit Management
func_1_5_install_rknn() {
    # 1. Check if already installed
    if "${PYTHON_BIN}" -c "import rknn.api" &> /dev/null; then
        func_1_1_log "RKNN Toolkit2 already installed in the virtual environment."
        return 0
    fi

    func_1_1_log "RKNN Toolkit2 not found. Initiating auto-install..."

    # 2. Define Paths
    # NOTE: we deliberately drop the ".git" suffix so example scripts under
    # this repo (which do realpath.index('rknn-toolkit2') to extend sys.path)
    # resolve correctly. See commit message for the trade-off vs the default
    # `git clone` naming.
    local repo_dir="${SDK_ROOT}/rockchip-repos/rknn-toolkit2"

    # 3. Clone if missing (full clone + checkout pinned SHA, NOT --depth 1)
    #    Full clone is required because pinned-SHA checkout needs the SHA to
    #    be reachable from a ref; depth-1 clones only have the branch tip.
    if [ ! -d "${repo_dir}" ]; then
        func_1_1_log "Cloning rknn-toolkit2 repository (pinned to ${RKNN_TOOLKIT2_PINNED_SHA:0:7})..."
        # Ensure parent dir exists
        mkdir -p "$(dirname "${repo_dir}")"
        git clone "${RKNN_TOOLKIT2_REPO_URL}" "${repo_dir}" \
            || func_1_2_err "Failed to clone rknn-toolkit2 repo."
        (cd "${repo_dir}" && git checkout "${RKNN_TOOLKIT2_PINNED_SHA}") \
            || func_1_2_err "Failed to checkout pinned SHA ${RKNN_TOOLKIT2_PINNED_SHA}."
    fi

    # 3b. Optional: overlay wheel source with a local tarball.
    #     Only runs if the user (or .config/common/rk-toolchain.yaml) supplied
    #     a tarball_path. CFG_RKNN_TARBALL_PATH / CFG_RKNN_TARBALL_SHA256 are
    #     set by func_3_3_load_toolchain_overrides BEFORE this function runs.
    if [ -n "${CFG_RKNN_TARBALL_PATH:-}" ]; then
        func_1_5_1_1_overlay_with_tarball \
            "${CFG_RKNN_TARBALL_PATH}" \
            "${CFG_RKNN_TARBALL_SHA256:-}"
    fi

    # 4. Find the correct Wheel file for Python 3.x (cp3x) on x86_64
    # Pattern: rknn_toolkit2-*-cp3x-cp3x-manylinux*x86_64.whl
    func_1_1_log "Searching for RKNN Toolkit2 wheel package..."
    local search_path="${repo_dir}/rknn-toolkit2/packages/x86_64"
    local whl_file
    whl_file=$(find "${search_path}" -name "rknn_toolkit2*-${WHEEL_TAG}-${WHEEL_TAG}-*x86_64.whl" | head -n 1)

    if [ -z "${whl_file}" ]; then
        func_1_2_err "Could not find compatible .whl file in: ${search_path}"
    fi

    # 5. Install RKNN Toolkit2 wheel into the venv
    func_1_1_log "\nInstalling: $(basename "${whl_file}")"
    "${PIP_BIN}" install "${whl_file}" || func_1_2_err "Failed to install RKNN Toolkit2."

    # 6. Install official RKNN runtime requirements matching the current Python tag
    local requirements_file=$(find "${search_path}" -name "requirements_${WHEEL_TAG}-*.txt" | head -n 1)

    if [ -f "${requirements_file}" ]; then
        func_1_1_log "\nInstalling: $(basename "${requirements_file}")"
        "${PIP_BIN}" install -r "${requirements_file}" || func_1_2_err "Failed to install requirements.txt."
    else
        # func_1_2_err "Could not find compatible requirements.txt file in: ${search_path}"
        func_1_1_log "No RKNN requirements file found for WHEEL_TAG=${WHEEL_TAG}, skipping RKNN extra requirements."
    fi

    # 7. Force ONNX version compatible with RKNN Toolkit2
    func_1_1_log "\nEnsuring ONNX version is compatible with RKNN Toolkit2 (onnx>=1.16.1,<1.19.0)..."
    "${PIP_BIN}" install "onnx>=1.16.1,<1.19.0" || func_1_2_err "Failed to install a compatible ONNX version."

    func_1_1_log "RKNN Toolkit2 installed successfully."
}

# Verify (and optionally repair) the HEAD of an existing Rockchip repo clone.
# Called by func_2_1_setup_venv BEFORE func_1_5 / func_1_5_3 so we can decide
# what to do when the directory exists but is on the wrong commit.
#
# Args:
#   $1 = repo display name (e.g. "rknn-toolkit2")
#   $2 = absolute path to the repo dir (e.g. "${SDK_ROOT}/rockchip-repos/rknn-toolkit2")
#   $3 = pinned SHA expected for this repo
#
# Behavior:
#   - Repo absent   -> return 0; the caller (func_1_5 / func_1_5_3) will clone.
#   - Repo present, HEAD matches pinned -> log [OK], return 0.
#   - Repo present, HEAD differs        -> prompt user:
#         ENTER (default) -> reset hard to pinned SHA. Use this if you want
#                            a clean roll-back to the tested snapshot.
#         s               -> skip; keep the existing HEAD. Use this if you
#                            intentionally upgraded and accept the risk.
#
# Returns:
#   0 always (skip is allowed; reset is non-fatal). Only errors out if `git`
#   itself is broken or the directory isn't a git repo.
func_1_5_2_verify_pinned_sha() {
    local display_name="$1"
    local repo_dir="$2"
    local pinned_sha="$3"

    # Case 1: directory doesn't exist yet -> caller will clone
    if [ ! -d "${repo_dir}" ]; then
        return 0
    fi

    # Case 2: directory exists but isn't a git repo -> caller error
    if [ ! -d "${repo_dir}/.git" ] && [ ! -f "${repo_dir}/.git" ]; then
        func_1_2_err "${repo_dir} exists but is not a git repository. Remove it manually and re-run."
    fi

    local current_sha
    current_sha=$(cd "${repo_dir}" && git rev-parse HEAD 2>/dev/null) \
        || func_1_2_err "Failed to read HEAD in ${repo_dir}."

    # Case 3: HEAD already matches pinned SHA
    if [ "${current_sha}" = "${pinned_sha}" ]; then
        func_1_1_log "[OK] ${display_name} @ ${current_sha:0:7} (matches pinned)"
        return 0
    fi

    # Case 4: HEAD differs from pinned SHA -> ask the user
    echo ""
    echo -e "\033[1;33m[WARN]\033[0m ${display_name} HEAD differs from pinned commit:"
    echo "        current : ${current_sha}"
    echo "        pinned  : ${pinned_sha}"
    echo ""
    echo "  Press ENTER to FORCE-RESET to pinned SHA (recommended for roll-back)"
    echo "  Type 's' + ENTER to SKIP and keep the current HEAD (you take the risk)"

    local answer=""
    # read -r blocks until user hits Enter. Ctrl-C falls through to the
    # default (RESET) — we assume roll-back is the safer default in a
    # CI-like / shared-machine flow.
    read -r -p "  Your choice [Enter=reset / s=skip]: " answer

    case "${answer}" in
        s|S)
            func_1_1_log "[SKIP] ${display_name} kept at ${current_sha:0:7} (not pinned)"
            return 0
            ;;
        *)
            func_1_1_log "[RESET] ${display_name}: ${current_sha:0:7} -> ${pinned_sha:0:7}"
            (cd "${repo_dir}" && git reset --hard "${pinned_sha}") \
                || func_1_2_err "Failed to reset ${repo_dir} to pinned SHA."
            return 0
            ;;
    esac
}

# Clone rknn_model_zoo (examples / reference scripts only — no Python wheel).
# Mirrors func_1_5's structure: full clone + pinned SHA checkout.
#
# IMPORTANT: this function ASSUMES func_1_5_2_verify_pinned_sha has already
# been called for this repo. If the directory was missing, we clone; if it
# was kept by user choice during verify, we touch nothing here.
func_1_5_3_clone_rknn_model_zoo() {
    # NOTE: no ".git" suffix — same reason as func_1_5. example scripts under
    # this repo rely on realpath.index('rknn_model_zoo') matching literally.
    local repo_dir="${SDK_ROOT}/rockchip-repos/rknn_model_zoo"

    # 1. Already cloned (either by previous run, or by user's "skip" choice)?
    if [ -d "${repo_dir}" ]; then
        local current_sha
        current_sha=$(cd "${repo_dir}" && git rev-parse HEAD 2>/dev/null) || true
        if [ "${current_sha}" = "${RKNN_MODEL_ZOO_PINNED_SHA}" ]; then
            func_1_1_log "rknn_model_zoo already present and pinned. Skipping."
        else
            func_1_1_log "rknn_model_zoo present at ${current_sha:0:7} (not pinned, kept by user choice). Skipping clone."
        fi
        return 0
    fi

    # 2. Clone + checkout pinned SHA
    func_1_1_log "Cloning rknn_model_zoo repository (pinned to ${RKNN_MODEL_ZOO_PINNED_TAG} / ${RKNN_MODEL_ZOO_PINNED_SHA:0:7})..."
    mkdir -p "$(dirname "${repo_dir}")"
    git clone "${RKNN_MODEL_ZOO_REPO_URL}" "${repo_dir}" \
        || func_1_2_err "Failed to clone rknn_model_zoo repo."
    (cd "${repo_dir}" && git checkout "${RKNN_MODEL_ZOO_PINNED_SHA}") \
        || func_1_2_err "Failed to checkout pinned SHA ${RKNN_MODEL_ZOO_PINNED_SHA}."

    func_1_1_log "rknn_model_zoo cloned successfully at ${RKNN_MODEL_ZOO_PINNED_TAG}."
}

func_1_6_setup_environment_vars() {
    if [ "$V" = "1" ]; then
        DEBUG_MODE="1"
    fi

    # Processing -- 1. start timer
    func_1_9_start_time_count __START_TIME
    # func_1_1_log "Environment setup started at ${__START_TIME}."

    # Preparation -- 2. Ensure Finalization on Exit
    trap func_2_4_finalize EXIT

    # Preparation -- 3. Define Important Paths
    SDK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

func_1_7_check_rknn_and_cv2() {
    func_1_1_log "Verifying RKNN and OpenCV environment inside the venv..."

    # First try with whatever is currently installed (RKNN wheel + official requirements)
    if "${PYTHON_BIN}" - <<'EOF'
try:
    import rknn.api  # basic RKNN import
    import cv2       # OpenCV import
except Exception:
    raise SystemExit(1)
EOF
    then
        func_1_1_log "RKNN and OpenCV imports succeeded."
        return 0
    fi

    func_1_1_log "RKNN / OpenCV import failed, trying to switch to opencv-python-headless..."

    # Replace GUI OpenCV with headless OpenCV (works across Python 3.8–3.12)
    "${PIP_BIN}" uninstall -y opencv-python || true
    "${PIP_BIN}" install "opencv-python-headless==4.11.0.86" || func_1_2_err "Failed to install opencv-python-headless."

    # Re-check after switching to headless OpenCV
    if "${PYTHON_BIN}" - <<'EOF'
try:
    import rknn.api
    import cv2
except Exception:
    raise SystemExit(1)
EOF
    then
        func_1_1_log "RKNN and OpenCV(headless) imports succeeded."
        return 0
    fi

    func_1_2_err "RKNN / OpenCV environment check still failed even after installing opencv-python-headless."
}

func_1_8_get_current_milliseconds() {
    date +%s%3N
}

# start timer: $1 = variable name to store start time
func_1_9_start_time_count() {
    local -n _timer_ref=$1
    # _timer_ref=$(date +%s)
    _timer_ref="$(func_1_8_get_current_milliseconds)"
}

# format duration from milliseconds to human readable string
# $1 = milliseconds
# $2 = variable name to store formatted result
func_1_10_format_duration_ms() {
    local ms=$1
    local -n _out_ref=$2

    if (( ms < 1000 )); then
        _out_ref="${ms} ms"
        return
    fi

    if (( ms < 60000 )); then
        # seconds with 2 decimal places
        _out_ref=$(printf "%.2f seconds" "$(awk "BEGIN { print $ms / 1000 }")")
        return
    fi

    # minutes + seconds
    local total_seconds minutes remaining_ms
    total_seconds=$(( ms / 1000 ))
    minutes=$(( total_seconds / 60 ))
    remaining_ms=$(( ms % 60000 ))

    _out_ref=$(printf "%d minutes %.2f seconds" \
        "$minutes" \
        "$(awk "BEGIN { print $remaining_ms / 1000 }")")
}

# finalize timer:
# $1 = start time variable
# $2 = end time variable
# $3 = the variable name to store formatted elapsed time string
func_1_11_elapsed_time_calculation() {
    local start_time=$1
    local end_time=$2
    local -n elapsed_string_ref=$3

    # func_1_1_log "\$1=$1 ; \$2=$2; \$3=$3"

    local elapsed_time=$((end_time - start_time))

    func_1_10_format_duration_ms "${elapsed_time}" elapsed_string_ref
}


# ==============================================================================
# Level 2: Environment Logic
# ==============================================================================
func_2_1_setup_venv() {

    # 1. Check/Create Venv
    #
    # Two failure modes this guards against:
    #
    # (a) Half-broken venv: `python -m venv` was invoked once when
    #     python3.X-venv wasn't installed, leaving .venv/bin/python
    #     symlinks behind but no pip. Original code only checked for
    #     python, so it would silently keep this corpse and crash later
    #     at the RKNN wheel install with a misleading "pip: No such file".
    #
    # (b) Silent ensurepip failure: even after a successful venv create,
    #     pip can be missing if ensurepip itself failed. Original code
    #     ran `pip install --upgrade pip` without checking it worked,
    #     then crashed much later with the same misleading error.
    #
    # We now do THREE checks:
    #   1. On entry:  is the existing .venv healthy? (python AND pip AND importable pip module)
    #   2. On create: did `python -m venv` succeed AND did ensurepip produce a working pip?
    #   3. On upgrade: did `pip install --upgrade pip` succeed?
    # Each failed check exits with a message that tells the user exactly
    # what system package they need to install.

    # Step 1: existing venv health check
    if [ -f "${PYTHON_BIN}" ] && [ -x "${PIP_BIN}" ] && "${PYTHON_BIN}" -m pip --version &> /dev/null; then
        func_1_1_log "Virtual environment already exists and is healthy. Skipping creation."
    else
        # If a half-broken venv exists, explain WHY we blow it away — the
        # user shouldn't be surprised by rm -rf of their venv.
        if [ -d "${VENV_DIR}" ] && [ -f "${PYTHON_BIN}" ]; then
            func_1_1_log "Existing .venv is half-broken (python present, pip missing). Re-creating from scratch..."
            rm -rf "${VENV_DIR}"
        else
            func_1_1_log "Initializing virtual environment..."
        fi

        # Step 2: create venv. Detect host python minor version so the
        # error message names the exact apt package the user needs.
        local host_py_minor
        host_py_minor=$(${HOST_PYTHON_BIN} -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "?")
        local apt_pkg="python3.${host_py_minor}-venv"

        if ! ${HOST_PYTHON_BIN} -m venv "${VENV_DIR}" 2>&1 | tee /tmp/arc_venv_err.log; then
            # venv creation itself failed. Most common cause on Ubuntu:
            # `python3.X-venv` apt package is not installed. Tell the user.
            if grep -q "ensurepip is not available" /tmp/arc_venv_err.log 2>/dev/null; then
                rm -rf "${VENV_DIR}"
                func_1_2_err "Python's ensurepip module is unavailable. On Debian/Ubuntu, install the venv package for your Python version:    sudo apt install ${apt_pkg}    Then re-run './arc init'."
            else
                rm -rf "${VENV_DIR}"
                func_1_2_err "Failed to create venv at ${VENV_DIR}. See error above."
            fi
        fi
        rm -f /tmp/arc_venv_err.log

        # Step 2b: post-create verification. Even if `python -m venv`
        # returned 0, pip may still be missing (Ubuntu 22.04 bug where
        # python3.10-venv is installed but bundled pip wheel was somehow
        # dropped during venv creation).
        if ! "${PYTHON_BIN}" -m pip --version &> /dev/null; then
            rm -rf "${VENV_DIR}"
            func_1_2_err "Freshly-created .venv has no working pip. Likely cause: ${apt_pkg} is missing or broken. Run:    sudo apt install ${apt_pkg}    Then re-run './arc init'."
        fi

        # Step 3: upgrade pip. Treat failure as fatal — silently continuing
        # was the original arc's mistake that led to the "line 158: pip:
        # No such file" cryptic error two minutes later.
        "${PIP_BIN}" install --upgrade pip || func_1_2_err "Failed to upgrade pip inside the venv. Check network/proxy."
    fi

    # 2. Check/Install RKNN Toolkit2 (The Auto-Magic Step)
    #    Order matters:
    #      (a) verify both rockchip repos against their pinned SHA,
    #          giving user the chance to roll-back / skip BEFORE we touch them
    #      (b) install RKNN Toolkit2 (which clones rknn-toolkit2.git if absent)
    #      (c) clone rknn_model_zoo.git (verify-already-handled-or-skipped)
    func_1_5_2_verify_pinned_sha \
        "rknn-toolkit2" \
        "${SDK_ROOT}/rockchip-repos/rknn-toolkit2" \
        "${RKNN_TOOLKIT2_PINNED_SHA}"
    func_1_5_2_verify_pinned_sha \
        "rknn_model_zoo" \
        "${SDK_ROOT}/rockchip-repos/rknn_model_zoo" \
        "${RKNN_MODEL_ZOO_PINNED_SHA}"

    func_1_5_install_rknn
    func_1_5_3_clone_rknn_model_zoo

    # 3. Check Dependencies (Force check for V1.1 new deps: requests)
    #    We should run after RKNN Toolkit2 is installed to avoid conflicts.
    if ! "${PYTHON_BIN}" -c "import requests, tqdm, yaml" &> /dev/null; then
        func_1_1_log "\nInstalling/Updating project dependencies from envs/requirements.txt ..."
        "${PIP_BIN}" install -r "${SDK_ROOT}/envs/requirements.txt"
    else
        func_1_1_log "Project dependencies already satisfied."
    fi

    # 4. Validate RKNN + OpenCV stack, and auto-switch to headless OpenCV when needed
    func_1_7_check_rknn_and_cv2
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

func_2_4_finalize() {
    local exit_code=$?
    local final_time=$(func_1_8_get_current_milliseconds)

    if [[ -n "${__START_TIME:-}" ]]; then

        # Processing -- 1. second stage build time statistics
        if [[ ! -z "${__SECOND_STAGE_BUILD_START_TIME:-}" ]]; then
            local second_duration_human
            func_1_11_elapsed_time_calculation "${__SECOND_STAGE_BUILD_START_TIME}" "${final_time}" second_duration_human
            func_1_1_log "Build elapsed time: ${second_duration_human}."
        fi

        # Processing -- 2. total elapsed time statistics
        local whole_duration_human
        func_1_11_elapsed_time_calculation "${__START_TIME}" "${final_time}" whole_duration_human
        func_1_1_log "Total elapsed time: ${whole_duration_human}."
    else
        func_1_1_log "Elapsed time information not available."
    fi

    exit "$exit_code"
}

# ==============================================================================
# Level 3:
# ==============================================================================
# Returns result via Global Variable 'SELECTED_CONFIG' to avoid stdout pollution
func_3_1_resolve_config() {
    local input="$1"
    SELECTED_CONFIG=""

    # Case A: User gave nothing -> Show Two-Level Menu (Interactive)
    #
    # WHY two levels: configs/ now has model-based subdirectories (modnet/, rvm/,
    # asr/, etc.) to prevent misselection when the total config count grows.
    # Level 1 picks a category (subdirectory); Level 2 picks a config within it.
    # Loose .yaml files in configs/ root are grouped under "other" for compat.
    if [ -z "$input" ]; then

        # --- Level 1: Collect categories (subdirectories with .yaml files) ---
        local categories=()
        local category_counts=()

        # Check for subdirectories containing .yaml files
        for dir in "${CONFIG_DIR}"/*/; do
            [ -d "$dir" ] || continue
            local yamls=("${dir}"*.yaml)
            if [ -e "${yamls[0]}" ]; then
                categories+=("$(basename "$dir")")
                category_counts+=(${#yamls[@]})
            fi
        done

        # Check for loose .yaml files in the root of CONFIG_DIR
        local loose_files=(${CONFIG_DIR}/*.yaml)
        local has_loose=false
        if [ -e "${loose_files[0]}" ]; then
            has_loose=true
            categories+=("other")
            category_counts+=(${#loose_files[@]})
        fi

        # If no categories found at all, error out
        if [ ${#categories[@]} -eq 0 ]; then
            func_1_2_err "No config files found in ${CONFIG_DIR}/"
        fi

        # If only one category, skip level 1 and go straight to level 2
        local selected_category=""
        if [ ${#categories[@]} -eq 1 ]; then
            selected_category="${categories[0]}"
        else
            # Display category menu
            func_1_1_log "Available Model Categories:"
            local i=1
            for idx in "${!categories[@]}"; do
                echo "  $i) ${categories[$idx]}  (${category_counts[$idx]} configs)"
                ((i++))
            done

            echo -n "Select category: "
            read cat_choice

            if [[ "$cat_choice" =~ ^[0-9]+$ ]] && [ "$cat_choice" -ge 1 ] && [ "$cat_choice" -lt "$i" ]; then
                selected_category="${categories[$((cat_choice-1))]}"
            else
                func_1_2_err "Invalid selection."
            fi
        fi

        # --- Level 2: List configs within the selected category ---
        local config_files=()
        if [ "$selected_category" = "other" ]; then
            config_files=(${CONFIG_DIR}/*.yaml)
        else
            config_files=(${CONFIG_DIR}/${selected_category}/*.yaml)
        fi

        func_1_1_log "Available Configurations [${selected_category}]:"
        local j=1
        for f in "${config_files[@]}"; do
            local fullname="$(basename "$f")"
            local shrinked_name="${fullname%.yaml}"
            echo "  $j) ${shrinked_name}"
            ((j++))
        done

        echo -n "Select config: "
        read cfg_choice

        if [[ "$cfg_choice" =~ ^[0-9]+$ ]] && [ "$cfg_choice" -ge 1 ] && [ "$cfg_choice" -lt "$j" ]; then
            SELECTED_CONFIG="${config_files[$((cfg_choice-1))]}"
        else
            func_1_2_err "Invalid selection."
        fi

    # Case B: User gave a short name (e.g., "rk3588s_rvm_mobilenetv3_256x256_int8")
    #
    # WHY recursive search: configs are now in subdirectories, but we want
    # backward compatibility — users should not need to type the subdirectory.
    # We search CONFIG_DIR root first (legacy), then subdirectories.
    elif [ -f "${CONFIG_DIR}/${input}.yaml" ]; then
        SELECTED_CONFIG="${CONFIG_DIR}/${input}.yaml"
    else
        # Recursive search in subdirectories
        local found
        found=$(find "${CONFIG_DIR}" -maxdepth 2 -name "${input}.yaml" -type f 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            SELECTED_CONFIG="$found"

        # Case C: User gave a specific file path
        elif [ -f "$input" ]; then
            SELECTED_CONFIG="$input"

        else
            func_1_2_err "Configuration not found: $input"
        fi
    fi
}

# The actual Python Kernel launcher (Called by Mode 1, 2, 3)
func_3_2_launch_kernel() {
    if [ -z "$SELECTED_CONFIG" ]; then
        func_1_2_err "No configuration selected. Aborting."
        return 1
    fi
    func_1_1_log "Target Config: $(basename ${SELECTED_CONFIG})"

    # Load toolchain overrides from the user config (+ .config/common/rk-toolchain.yaml
    # if present). This sets CFG_RKNN_TARBALL_PATH and CFG_RKNN_TARBALL_SHA256, which
    # func_1_5_install_rknn consumes to decide whether to overlay the wheel source.
    # Must run BEFORE func_2_1_setup_venv.
    func_3_3_load_toolchain_overrides

    func_2_1_setup_venv || return 1

    func_1_9_start_time_count __SECOND_STAGE_BUILD_START_TIME
    export PYTHONPATH="${SDK_ROOT}"
    #exec "${PYTHON_BIN}" "${SDK_ROOT}/core/main.py" -c "${SELECTED_CONFIG}"
    "${PYTHON_BIN}" "${SDK_ROOT}/core/main.py" -c "${SELECTED_CONFIG}"
    local py_ret=$?

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

# Pre-flight toolchain config loader.
# Called from func_3_2_launch_kernel BEFORE func_2_1_setup_venv.
#
# Reads the user config (+ .config/common/rk-toolchain.yaml auto-merge),
# validates the rknn_toolkit2 fields, and exports CFG_RKNN_TARBALL_PATH +
# CFG_RKNN_TARBALL_SHA256 in this shell. The actual validation logic lives
# in tools/load_toolchain_overrides.py (so main.py can reuse it via
# load_merged_config).
#
# Python interpreter choice:
#   - Prefer the venv python if it already exists (always has PyYAML from
#     envs/requirements.txt).
#   - Fall back to the host python for first-run. PyYAML must be installed
#     there, OR we error out with a clear install hint.
func_3_3_load_toolchain_overrides() {
    local helper_script="${SDK_ROOT}/tools/load_toolchain_overrides.py"
    if [ ! -f "${helper_script}" ]; then
        func_1_2_err "Toolchain override helper not found: ${helper_script}
    Did you accidentally delete tools/load_toolchain_overrides.py?"
    fi

    # Choose python interpreter
    local py_bin
    if [ -x "${VENV_DIR}/bin/python" ]; then
        py_bin="${VENV_DIR}/bin/python"
    else
        py_bin="${HOST_PYTHON_BIN}"
        if ! "${py_bin}" -c "import yaml" &> /dev/null; then
            func_1_2_err "First-run requires PyYAML on system python (${py_bin}).
    Install with:    ${py_bin} -m pip install pyyaml
    Or run './arc init' once to create the venv (with PyYAML in requirements.txt), then re-run."
        fi
    fi

    func_1_1_log "Loading toolchain overrides from config: $(basename "${SELECTED_CONFIG}")"

    # Capture helper output (the helper prints `export VAR='value'` lines).
    local helper_output
    if ! helper_output=$("${py_bin}" "${helper_script}" "${SELECTED_CONFIG}" "${SDK_ROOT}" 2>&1); then
        func_1_2_err "Failed to load toolchain overrides:
${helper_output}"
    fi

    # Apply the export lines in this shell
    eval "${helper_output}"

    # Echo what we decided (so the user sees it in the log)
    if [ -n "${CFG_RKNN_TARBALL_PATH:-}" ]; then
        func_1_1_log "  Toolchain override active: tarball_path=${CFG_RKNN_TARBALL_PATH}"
        if [ -n "${CFG_RKNN_TARBALL_SHA256:-}" ]; then
            func_1_1_log "  sha256=${CFG_RKNN_TARBALL_SHA256:0:16}... (will be verified before install)"
        else
            func_1_1_log "  WARNING: no SHA256 specified, integrity will NOT be checked at install"
        fi
    else
        func_1_1_log "  No toolchain override; will use official repo (v2.3.2)"
    fi
}

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
    # Preparation -- 2. Setup Env Vars
    func_1_6_setup_environment_vars

    # Preparation -- 3. Ensure log dir exists
    mkdir -p "${LOG_DIR}"

    # Processing -- 1. Interactive Mode
    if [ $# -eq 0 ]; then
        # No arguments: Enter Mode 1 (Interactive)
        func_4_1_mode_menu || exit
        exit 0
    fi

    # Processing -- 2. Direct Mode
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

    # Processing -- 3. Fallback for Legacy calls (e.g. `./arc convert -c ...`)
    if [ "$1" == "convert" ] && [ "$2" == "-c" ] && [ -n "$3" ]; then
        func_4_2_mode_direct "$3"
    elif [ "$1" == "convert" ] && [ -n "$2" ]; then
        func_4_2_mode_direct "$2"
    else
        func_1_2_err "Invalid arguments. Try './arc help'."
    fi
}


main "$@"
