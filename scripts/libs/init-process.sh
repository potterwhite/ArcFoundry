
# RKNN Toolkit Management
# ------------------------------------------------------------------------------
# Two install paths for RKNN Toolkit2. The choice is made at the TOP of
# func_2_1_2_rockchip_repos_init based on CFG_RKNN_TARBALL_PATH:
#
#   - Set   -> func_1_5_b_install_rknn_from_tarball: rm -rf + tar -xzf
#              the local .tgz, then func_1_5_b_1_flatten_tarball_layout
#              removes any versioned wrapper dir so the SDK ends up at
#              the same path Path A produces. No .git/ preserved.
#
#   - Empty -> func_1_5_a_install_rknn_from_official_repo: git clone the
#              airockchip repo at the pinned SHA.
#
# Both paths converge on the same final layout
# (${repo_dir}/rknn-toolkit2/packages/x86_64/...) and feed into
# func_1_5_x_install_wheel_from_dir for wheel + requirements + onnx.
# ------------------------------------------------------------------------------

# Path A: install from official airockchip repo at pinned SHA
func_1_5_a_install_rknn_from_official_repo() {
    if "${PYTHON_BIN}" -c "import rknn.api" &> /dev/null; then
        func_1_1_log "RKNN Toolkit2 already installed in the virtual environment."
        return 0
    fi
    func_1_1_log "RKNN Toolkit2 not found. Cloning official repo (pinned to ${RKNN_TOOLKIT2_PINNED_SHA:0:7})..."

    local repo_dir="${SDK_ROOT}/rockchip-repos/rknn-toolkit2"
    mkdir -p "${repo_dir}"
    git clone "${RKNN_TOOLKIT2_REPO_URL}" "${repo_dir}" \
        || func_1_2_err "Failed to clone rknn-toolkit2 repo."
    (cd "${repo_dir}" && git checkout "${RKNN_TOOLKIT2_PINNED_SHA}") \
        || func_1_2_err "Failed to checkout pinned SHA ${RKNN_TOOLKIT2_PINNED_SHA}."

    func_1_5_x_install_wheel_from_dir "${repo_dir}/rknn-toolkit2"
}

# Path B: install from a local tarball (CFG_RKNN_TARBALL_PATH set)
#
# Rockchip's tarballs sometimes wrap the SDK in a versioned top-level dir
# (e.g. rknn-toolkit2-v2.4.0-2026-01-17-RV1126B/) and sometimes don't.
# To stay compatible with BOTH layouts, we extract as-is, then call
# func_1_5_b_1_flatten_tarball_layout to flatten any wrapper to the
# Path A (git-clone) layout:  ${repo_dir}/rknn-toolkit2/
func_1_5_b_install_rknn_from_tarball() {
    local tarball_path="${CFG_RKNN_TARBALL_PATH}"
    local expected_sha256="${CFG_RKNN_TARBALL_SHA256:-}"

    # 1. Validate
    if [ ! -f "${tarball_path}" ]; then
        func_1_2_err "Tarball not found: ${tarball_path}"
    fi
    if [ -n "${expected_sha256}" ]; then
        local actual_sha256
        actual_sha256=$(sha256sum "${tarball_path}" | awk '{print $1}')
        if [ "${actual_sha256}" != "${expected_sha256}" ]; then
            func_1_2_err "SHA256 mismatch for ${tarball_path}
    Expected: ${expected_sha256}
    Actual:   ${actual_sha256}
  Fix: re-download the tarball and update RKNN_TARBALL_SHA256 in rk-toolchain.env."
        fi
        func_1_1_log "  SHA256 verified: ${actual_sha256:0:16}..."
    else
        func_1_1_log "  WARNING: no SHA256 specified, skipping integrity check"
    fi

    if "${PYTHON_BIN}" -c "import rknn.api" &> /dev/null; then
        func_1_1_log "RKNN Toolkit2 already installed in the virtual environment."
        return 0
    fi
    func_1_1_log "RKNN Toolkit2 not found. Extracting local tarball: $(basename "${tarball_path}")..."

    # 2. Wipe and extract.
    local repo_dir="${SDK_ROOT}/rockchip-repos/rknn-toolkit2"
    rm -rf "${repo_dir}"
    mkdir -p "${repo_dir}"
    tar -xzf "${tarball_path}" -C "${repo_dir}" \
        || func_1_2_err "Failed to extract tarball."

    # 3. Flatten any wrapper dir; install. Same call shape as Path A.
    func_1_5_b_1_flatten_tarball_layout "${repo_dir}"
    if [ ! -d "${repo_dir}/rknn-toolkit2/packages/x86_64" ]; then
        func_1_2_err "After extraction, ${repo_dir}/rknn-toolkit2/packages/x86_64 is missing.
  Tarball does not contain the expected RKNN wheel layout."
    fi
    func_1_5_x_install_wheel_from_dir "${repo_dir}/rknn-toolkit2"
}

# Detect whether the tarball extracted with a versioned top-level
# wrapper dir (e.g. rknn-toolkit2-v2.4.0-2026-01-17-RV1126B/) and if so,
# move the inner rknn-toolkit2/ up one level so the final layout matches
# what 'git clone' produces.
#
# Accepted in-shapes at ${repo_dir}:
#   (a) rknn-toolkit2/packages/...           -> already correct, no-op
#   (b) <wrapper>/rknn-toolkit2/packages/... -> mv inner up, rmdir wrapper
# Anything else: error.
func_1_5_b_1_flatten_tarball_layout() {
    local repo_dir="$1"

    # Case (a): no wrapper.
    if [ -d "${repo_dir}/rknn-toolkit2/packages" ]; then
        return 0
    fi

    # Case (b): wrapped. Find the inner rknn-toolkit2/ one level deep.
    local inner
    inner=$(find "${repo_dir}" -mindepth 2 -maxdepth 2 -type d -name rknn-toolkit2 | head -n 1)
    if [ -z "${inner}" ] || [ ! -d "${inner}/packages" ]; then
        func_1_2_err "Extracted tarball but cannot locate rknn-toolkit2/packages/ in any expected shape.
  Looked for:    ${repo_dir}/rknn-toolkit2/packages/
  Also searched: ${repo_dir}/*/rknn-toolkit2/
  Tarball layout is not recognized. Extracted to: ${repo_dir}"
    fi

    local wrapper
    wrapper=$(basename "$(dirname "${inner}")")
    mv "${inner}" "${repo_dir}/rknn-toolkit2" \
        || func_1_2_err "Failed to rename ${inner} to ${repo_dir}/rknn-toolkit2"
    # rmdir (not rm -rf): if the wrapper dir has stray content, fail
    # loud — the tarball likely has files outside rknn-toolkit2/ we
    # should know about.
    rmdir "$(dirname "${inner}")" 2>/dev/null \
        || func_1_3_debug "Wrapper dir '${wrapper}/' had stray content; left in place."
    func_1_1_log "Detected wrapper dir '${wrapper}/'; flattened to ${repo_dir}/rknn-toolkit2/"
}

# Shared: find wheel + install + requirements + onnx constraint
func_1_5_x_install_wheel_from_dir() {
    local sdk_dir="$1"
    local search_path="${sdk_dir}/packages/x86_64"

    func_1_1_log "Searching for RKNN Toolkit2 wheel in ${search_path}..."
    local whl_file
    whl_file=$(find "${search_path}" -name "rknn_toolkit2*-${WHEEL_TAG}-${WHEEL_TAG}-*x86_64.whl" | head -n 1)
    if [ -z "${whl_file}" ]; then
        func_1_2_err "Could not find compatible .whl in: ${search_path}"
    fi

    func_1_1_log "Installing: $(basename "${whl_file}")"
    "${PIP_BIN}" install "${whl_file}" || func_1_2_err "Failed to install RKNN Toolkit2."

    local requirements_file
    requirements_file=$(find "${search_path}" -name "requirements_${WHEEL_TAG}-*.txt" | head -n 1)
    if [ -f "${requirements_file}" ]; then
        func_1_1_log "Installing: $(basename "${requirements_file}")"
        "${PIP_BIN}" install -r "${requirements_file}" || func_1_2_err "Failed to install requirements.txt."
    else
        func_1_1_log "No RKNN requirements file found for WHEEL_TAG=${WHEEL_TAG}, skipping."
    fi

    func_1_1_log "Ensuring ONNX version is compatible (onnx>=1.16.1,<1.19.0)..."
    "${PIP_BIN}" install "onnx>=1.16.1,<1.19.0" || func_1_2_err "Failed to install compatible ONNX."

    func_1_1_log "RKNN Toolkit2 installed successfully."
}

# Verify (and optionally repair) the HEAD of an existing Rockchip repo clone.
# Called by func_2_1_init BEFORE func_1_5 / func_1_5_3 so we can decide
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


func_2_1_1_setup_venv(){
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

}

func_2_1_2_rockchip_repos_init(){

    func_3_3_load_toolchain_env

    # 2. Check/Install RKNN Toolkit2 (The Auto-Magic Step)
    #    Order matters:
    #      (a) verify both rockchip repos against their pinned SHA,
    #          giving user the chance to roll-back / skip BEFORE we touch them
    #      (b) install RKNN Toolkit2: tarball path OR official repo, never both
    #      (c) clone rknn_model_zoo.git (verify-already-handled-or-skipped)
    #
    # Skip rknn-toolkit2 verify under tarball mode — tarball users don't
    # have a git checkout, so pinned-SHA comparison is meaningless.
    if [ -z "${CFG_RKNN_TARBALL_PATH:-}" ]; then
        func_1_5_2_verify_pinned_sha \
            "rknn-toolkit2" \
            "${SDK_ROOT}/rockchip-repos/rknn-toolkit2" \
            "${RKNN_TOOLKIT2_PINNED_SHA}"
    else
        func_1_1_log "Tarball mode: skipping pinned-SHA check for rknn-toolkit2."
    fi
    func_1_5_2_verify_pinned_sha \
        "rknn_model_zoo" \
        "${SDK_ROOT}/rockchip-repos/rknn_model_zoo" \
        "${RKNN_MODEL_ZOO_PINNED_SHA}"

    # 3. Pick install path: tarball OR official repo. Single dispatch.
    if [ -n "${CFG_RKNN_TARBALL_PATH:-}" ]; then
        func_1_5_b_install_rknn_from_tarball
    else
        func_1_5_a_install_rknn_from_official_repo
    fi
    func_1_5_3_clone_rknn_model_zoo
}

func_2_1_3_python_deps_init(){
    # 3. Check Dependencies (Force check for V1.1 new deps: requests)
    #    We should run after RKNN Toolkit2 is installed to avoid conflicts.
    if ! "${PYTHON_BIN}" -c "import requests, tqdm, yaml" &> /dev/null; then
        func_1_1_log "\nInstalling/Updating project dependencies from envs/requirements.txt ..."
        "${PIP_BIN}" install -r "${SDK_ROOT}/envs/requirements.txt"
    else
        func_1_1_log "Project dependencies already satisfied."
    fi

}

func_2_1_4_check_rknn_and_cv2() {
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

# Pre-flight: read rk-toolchain.env (if present) and set CFG_RKNN_TARBALL_PATH
# / CFG_RKNN_TARBALL_SHA256. Pure bash, no Python, no YAML, no helper script.
# No env file (the common case) = use official airockchip repo.
func_3_3_load_toolchain_env() {
    local env_file="${SDK_ROOT}/rk-toolchain.env"
    if [ ! -f "${env_file}" ]; then
        func_1_1_log "No rk-toolchain.env found. Will use official airockchip repo."
        export CFG_RKNN_TARBALL_PATH=""
        export CFG_RKNN_TARBALL_SHA256=""
        return 0
    fi
    # shellcheck disable=SC1090
    source "${env_file}"
    export CFG_RKNN_TARBALL_PATH="${RKNN_TARBALL_PATH:-}"
    export CFG_RKNN_TARBALL_SHA256="${RKNN_TARBALL_SHA256:-}"
    if [ -n "${CFG_RKNN_TARBALL_PATH}" ]; then
        func_1_1_log "Loaded rk-toolchain.env: tarball=${CFG_RKNN_TARBALL_PATH}"
    else
        func_1_1_log "rk-toolchain.env present but RKNN_TARBALL_PATH empty; using official repo."
    fi
}

func_2_1_init() {

    func_2_1_1_setup_venv

    func_2_1_2_rockchip_repos_init

    func_2_1_3_python_deps_init

    # 4. Validate RKNN + OpenCV stack, and auto-switch to headless OpenCV when needed
    func_2_1_4_check_rknn_and_cv2
}
