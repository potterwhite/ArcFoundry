#!/bin/bash

# 0_0_helper: Display usage information (Level 0)
func_0_0_helper(){
    echo "Usage: $(basename "${SCRIPT_PATH}") [command]" | tee -a "${INSTALL_LOG_FILE}"
    echo "Commands:" | tee -a "${INSTALL_LOG_FILE}"
    echo "  prepare         Install Python 3.8 and create virtual environment" | tee -a "${INSTALL_LOG_FILE}"
    echo "  install-rknn    Install rknn-toolkit2 in the virtual environment" | tee -a "${INSTALL_LOG_FILE}"
    echo "  clean           Remove specified directories (virtual environment, logs, or both)" | tee -a "${INSTALL_LOG_FILE}"
    echo "  -h              Display this help message" | tee -a "${INSTALL_LOG_FILE}"
    echo | tee -a "${INSTALL_LOG_FILE}"
}

# 1_0_load_env: Initialize environment variables and paths (Level 1)
func_1_0_load_env(){
    set -e

    if [ "$V" == "1" ];then
        set -x
    fi

    SCRIPT_PATH="$(realpath ${BASH_SOURCE})"
    SCRIPT_DIR="$(dirname ${SCRIPT_PATH})"
    REPO_TOP_DIR="${SCRIPT_DIR}"
	RKNN_TOOLKIT2_REPO_DIR="${REPO_TOP_DIR}/rockchip-repos/rknn-toolkit2.git"

    INSTALL_LOG_DIR="${REPO_TOP_DIR}/logs"
    INSTALL_LOG_FILE="${INSTALL_LOG_DIR}/python38_install.log"
    PYTHON_VENV_DIR="${REPO_TOP_DIR}/.venv"

    if [ ! -d "${INSTALL_LOG_DIR}" ]; then
        mkdir -p "${INSTALL_LOG_DIR}" || exit
        echo "Created log directory: ${INSTALL_LOG_DIR}" | tee -a "${INSTALL_LOG_FILE}"
    fi
}

# 2_0_prepare_environment: Prepare system, check and install Python 3.8 (Level 2)
func_2_0_prepare_environment(){
    # Check if Python 3.8 is already installed
    if command -v python3.8 &> /dev/null; then
        PYTHON38_VERSION=$(python3.8 --version 2>&1 | grep -oP '\d+\.\d+\.\d+')
        echo "Python 3.8 is already installed: ${PYTHON38_VERSION}" | tee -a "${INSTALL_LOG_FILE}"
    else
        # Update system and install dependencies
        echo "Updating system and installing dependencies..." | tee -a "${INSTALL_LOG_FILE}"
        sudo apt update && sudo apt upgrade -y || exit
        sudo apt install -y software-properties-common | tee -a "${INSTALL_LOG_FILE}"

        # Check if Python 3.8 is available in apt list
        echo "Checking if Python 3.8 is available in apt..." | tee -a "${INSTALL_LOG_FILE}"
        if apt list | grep -q "^python3.8/"; then
            echo "Python 3.8 is available in default apt repositories" | tee -a "${INSTALL_LOG_FILE}"
        else
            echo "Python 3.8 not found in default repositories, adding Deadsnakes PPA..." | tee -a "${INSTALL_LOG_FILE}"
            sudo add-apt-repository ppa:deadsnakes/ppa -y || exit
            sudo apt update | tee -a "${INSTALL_LOG_FILE}"
        fi

        # Install Python 3.8 and related tools
        echo "Installing Python 3.8..." | tee -a "${INSTALL_LOG_FILE}"
        sudo apt install -y python3.8 python3.8-dev python3.8-venv python3-pip | tee -a "${INSTALL_LOG_FILE}"

        # Verify installation
        echo "Verifying Python 3.8 installation..." | tee -a "${INSTALL_LOG_FILE}"
        python3.8 --version | tee -a "${INSTALL_LOG_FILE}"
        if [ $? -eq 0 ]; then
            echo "Python 3.8 installed successfully!" | tee -a "${INSTALL_LOG_FILE}"
        else
            echo "Python 3.8 installation failed!" | tee -a "${INSTALL_LOG_FILE}"
            exit 1
        fi
    fi

    # Install dependencies for rknn-toolkit2
    echo "Installing dependencies for rknn-toolkit2..." | tee -a "${INSTALL_LOG_FILE}"
    sudo apt-get install -y libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 libgl1-mesa-glx libprotobuf-dev gcc | tee -a "${INSTALL_LOG_FILE}"
}

# 2_1_create_venv: Create and activate Python 3.8 virtual environment (Level 2)
func_2_1_create_venv(){
    if [ -d "${PYTHON_VENV_DIR}" ]; then
        echo "Virtual environment already exists at ${PYTHON_VENV_DIR}" | tee -a "${INSTALL_LOG_FILE}"
    else
        echo "Creating Python 3.8 virtual environment at ${PYTHON_VENV_DIR}..." | tee -a "${INSTALL_LOG_FILE}"
        python3.8 -m venv "${PYTHON_VENV_DIR}" || exit
        if [ ! -f "${PYTHON_VENV_DIR}/bin/activate" ]; then
            echo "Failed to create virtual environment at ${PYTHON_VENV_DIR}" | tee -a "${INSTALL_LOG_FILE}"
            exit 1
        fi
        echo "Virtual environment created successfully at ${PYTHON_VENV_DIR}" | tee -a "${INSTALL_LOG_FILE}"
    fi

    # Activate the virtual environment
    echo "Activating virtual environment at ${PYTHON_VENV_DIR}..." | tee -a "${INSTALL_LOG_FILE}"
    source "${PYTHON_VENV_DIR}/bin/activate" || return 1
    echo "Virtual environment activated" | tee -a "${INSTALL_LOG_FILE}"
}

# 2_2_clean: Clean up specified directories based on user input (Level 2)
func_2_2_clean(){
    echo "Select what to delete:" | tee -a "${INSTALL_LOG_FILE}"
    echo "1) Virtual environment directory (${PYTHON_VENV_DIR})" | tee -a "${INSTALL_LOG_FILE}"
    echo "2) Log directory (${INSTALL_LOG_DIR})" | tee -a "${INSTALL_LOG_FILE}"
    echo "3) Both directories" | tee -a "${INSTALL_LOG_FILE}"
    echo -n "Enter your choice (1-3): " | tee -a "${INSTALL_LOG_FILE}"
    read choice

    case $choice in
        1)
            if [ -d "${PYTHON_VENV_DIR}" ]; then
                echo "Removing virtual environment directory: ${PYTHON_VENV_DIR}" | tee -a "${INSTALL_LOG_FILE}"
                rm -rf "${PYTHON_VENV_DIR}" || return 1
                echo "Virtual environment directory removed successfully" | tee -a "${INSTALL_LOG_FILE}"
            else
                echo "Virtual environment directory does not exist: ${PYTHON_VENV_DIR}" | tee -a "${INSTALL_LOG_FILE}"
            fi
            ;;
        2)
            if [ -d "${INSTALL_LOG_DIR}" ]; then
                echo "Removing log directory: ${INSTALL_LOG_DIR}" | tee -a "${INSTALL_LOG_FILE}"
                rm -rf "${INSTALL_LOG_DIR}" || return 1
                echo "Log directory removed successfully" | tee -a "${INSTALL_LOG_FILE}"
            else
                echo "Log directory does not exist: ${INSTALL_LOG_DIR}" | tee -a "${INSTALL_LOG_FILE}"
            fi
            ;;
        3)
            if [ -d "${PYTHON_VENV_DIR}" ]; then
                echo "Removing virtual environment directory: ${PYTHON_VENV_DIR}" | tee -a "${INSTALL_LOG_FILE}"
                rm -rf "${PYTHON_VENV_DIR}" || return 1
                echo "Virtual environment directory removed successfully" | tee -a "${INSTALL_LOG_FILE}"
            else
                echo "Virtual environment directory does not exist: ${PYTHON_VENV_DIR}" | tee -a "${INSTALL_LOG_FILE}"
            fi
            if [ -d "${INSTALL_LOG_DIR}" ]; then
                echo "Removing log directory: ${INSTALL_LOG_DIR}" | tee -a "${INSTALL_LOG_FILE}"
                rm -rf "${INSTALL_LOG_DIR}" || return 1
                echo "Log directory removed successfully" | tee -a "${INSTALL_LOG_FILE}"
            else
                echo "Log directory does not exist: ${INSTALL_LOG_DIR}" | tee -a "${INSTALL_LOG_FILE}"
            fi
            ;;
        *)
            echo "Invalid choice: ${choice}. No directories were removed." | tee -a "${INSTALL_LOG_FILE}"
            return 1
            ;;
    esac
}

# 2_3_install_rknn_toolkit2: Install rknn-toolkit2 in the virtual environment (Level 2)
func_2_3_install_rknn_toolkit2(){
    if [ ! -f "${PYTHON_VENV_DIR}/bin/activate" ]; then
        echo "Virtual environment does not exist at ${PYTHON_VENV_DIR}. Please run 'prepare' first." | tee -a "${INSTALL_LOG_FILE}"
        return 1
    fi

    echo "Upgrading pip in virtual environment..." | tee -a "${INSTALL_LOG_FILE}"
    pip install --upgrade pip | tee -a "${INSTALL_LOG_FILE}"

    # # Install dependencies required by rknn-toolkit2
    # echo "Installing Python dependencies for rknn-toolkit2..." | tee -a "${INSTALL_LOG_FILE}"
    # pip install numpy protobuf lxml | tee -a "${INSTALL_LOG_FILE}"

    # Install rknn-toolkit2
    # Note: Assuming rknn-toolkit2 is available via pip or a specific wheel file.
    # Adjust the command below based on the actual source (e.g., a local wheel file or git repository).
    echo "Installing rknn-toolkit2..." | tee -a "${INSTALL_LOG_FILE}"
    # pip install rknn-toolkit2 || exit
	pip install -r "${RKNN_TOOLKIT2_REPO_DIR}/rknn-toolkit2/packages/x86_64/requirements_cp38-2.3.2.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple || return
	pip install "${RKNN_TOOLKIT2_REPO_DIR}/rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" -i https://pypi.tuna.tsinghua.edu.cn/simple

    if [ $? -eq 0 ]; then
        echo "rknn-toolkit2 installed successfully in virtual environment!" | tee -a "${INSTALL_LOG_FILE}"
    else
        echo "Failed to install rknn-toolkit2!" | tee -a "${INSTALL_LOG_FILE}"
        return 1
    fi
}

# main: Orchestrate the installation or cleanup process using case statement
main(){
	func_1_0_load_env || exit

	case "$1" in
		-h)
			func_0_0_helper
			;;
		prepare)
			func_2_0_prepare_environment || exit
			func_2_1_create_venv || exit
			echo | tee -a "${INSTALL_LOG_FILE}"
			echo "Environment preparation completed!" | tee -a "${INSTALL_LOG_FILE}"
			echo | tee -a "${INSTALL_LOG_FILE}"
			;;
		install-rknn)
			func_2_0_prepare_environment || exit
			func_2_1_create_venv || exit
			func_2_3_install_rknn_toolkit2 || exit
			echo | tee -a "${INSTALL_LOG_FILE}"
			echo "rknn-toolkit2 installation completed successfully!" | tee -a "${INSTALL_LOG_FILE}"
			echo | tee -a "${INSTALL_LOG_FILE}"
			;;
		clean)
			func_2_2_clean || exit
			echo | tee -a "${INSTALL_LOG_FILE}"
			echo "Cleanup completed successfully!" | tee -a "${INSTALL_LOG_FILE}"
			echo | tee -a "${INSTALL_LOG_FILE}"
			;;
		*)
			func_0_0_helper
			return 1
			;;
	esac
}

main "$@"
