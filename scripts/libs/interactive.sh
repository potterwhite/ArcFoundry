


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

    # Load toolchain overrides from the user config (+ configs/common/rk-toolchain.yaml
    # if present). This sets CFG_RKNN_TARBALL_PATH and CFG_RKNN_TARBALL_SHA256, which
    # func_1_5_install_rknn consumes to decide whether to overlay the wheel source.
    # Must run BEFORE func_2_1_init.
    func_3_3_load_toolchain_overrides

    func_2_1_init || return 1

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
