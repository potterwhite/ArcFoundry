#!/bin/bash

# set -xe

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_TOP_DIR="${SCRIPT_DIR}"
source "${SCRIPT_DIR}/scripts/libs/helper.sh"
source "${SCRIPT_DIR}/scripts/libs/interactive.sh"
source "${SCRIPT_DIR}/scripts/libs/init-process.sh"
source "${SCRIPT_DIR}/scripts/libs/time-statics.sh"
source "${SCRIPT_DIR}/scripts/libs/distclean-process.sh"


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
            clean)          func_2_2_1_clean ;;
            distclean)      func_2_2_distclean ;;
            init)           func_2_1_init || exit 1 ;;
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
        func_1_4_err "Invalid arguments. Try './arc help'."
    fi
}


main "$@"
