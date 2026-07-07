


func_2_2_1_clean() {
    rm -rf "${SDK_ROOT}/workspace" "${SDK_ROOT}/output"
    func_1_1_log "Workspace, Output cleaned."

    # 2. Clean Root Artifacts (The "check*.onnx" files)
    rm -f "${SDK_ROOT}"/check*.onnx \
          "${SDK_ROOT}"/debug*.onnx \
          "${SDK_ROOT}"/verify*.onnx \
          "${SDK_ROOT}"/*.rknn_util_*.log

    func_1_1_log "root artifacts cleaned."
}


func_2_2_distclean() {
    func_2_2_1_clean

    rm -rf "${VENV_DIR}"
    func_1_1_log "Virtual Environment removed. Factory reset complete."

    rm -rf "${MODELS_DIR}"
    func_1_1_log "Models directory removed."

    rm -rf "${RK_REPOS_DIR}"
    func_1_1_log "Rockchip repositories removed."
}
