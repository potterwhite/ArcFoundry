#!/usr/bin/env python3
# Copyright (c) 2026 PotterWhite — MIT License
#
# load_toolchain_overrides.py
# ============================================================================
# Pre-flight toolchain config loader, called by `arc` BEFORE `func_2_1_setup_venv`.
#
# Reads the user-specified YAML config, auto-merges .config/common/rk-toolchain.yaml
# (if present), validates the rknn_toolkit2 fields, and emits bash-compatible
# `export VAR=value` lines on stdout for arc to `eval`.
#
# Why this exists separately from main.py's load_merged_config call:
#   - arc needs to know tarball_path / tarball_sha256 BEFORE git-cloning the
#     rockchip repo and pip-installing the wheel, so it can overlay the wheel
#     source first. main.py runs AFTER that. So we read the merged config
#     TWICE — once here in bash-via-Python, once in main.py — using the same
#     shared load_merged_config to keep semantics identical.
#
# Exit codes:
#   0 — success. Bash exports for CFG_RKNN_TARBALL_PATH and CFG_RKNN_TARBALL_SHA256
#       are on stdout (may be empty strings; arc treats empty as "use official").
#   1 — validation error. Message on stderr naming the file + field + fix.
#   2 — usage error (bad argv).
#
# Usage (from arc):
#   eval $(HOST_PYTHON_BIN tools/load_toolchain_overrides.py <config> <sdk_root>)
# ============================================================================

import os
import sys


def _resolve_sdk_root(provided, config_path):
    """
    Trust the explicit sdk_root if given; otherwise walk up from config_path
    looking for a `.config/` subdir (the SDK_ROOT marker).
    """
    if provided:
        return os.path.abspath(provided)
    cur = os.path.dirname(os.path.abspath(config_path))
    while cur and cur != "/":
        if os.path.isdir(os.path.join(cur, ".config")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.getcwd()


def _shell_quote(s):
    """
    Quote a string for safe use as a bash value: single-quote the whole thing,
    escape embedded single quotes as '\\'' (close, escape, reopen).
    """
    if not s:
        return ""
    return "'" + s.replace("'", "'\\''") + "'"


def _emit_bash_exports(cfg):
    """Print `export VAR=value` for the rknn_toolkit2 fields."""
    rt = cfg.get("rknn_toolkit2", {}) or {}
    tarball_path = rt.get("tarball_path", "") or ""
    tarball_sha256 = rt.get("tarball_sha256", "") or ""
    print(f"export CFG_RKNN_TARBALL_PATH={_shell_quote(tarball_path)}")
    print(f"export CFG_RKNN_TARBALL_SHA256={_shell_quote(tarball_sha256)}")


def main():
    if len(sys.argv) != 3:
        sys.stderr.write(
            f"Usage: {sys.argv[0]} <config_path> <sdk_root>\n"
        )
        sys.exit(2)

    config_path = sys.argv[1]
    sdk_root = _resolve_sdk_root(sys.argv[2], config_path)

    # Make SDK_ROOT/core importable so we can reuse load_merged_config
    sys.path.insert(0, sdk_root)
    try:
        from core.utils.file_utils import load_merged_config
    except ImportError as e:
        sys.stderr.write(
            f"ERROR: failed to import core.utils.file_utils: {e}\n"
            f"  SDK_ROOT: {sdk_root}\n"
            f"  Fix: ensure core/utils/file_utils.py exists and PyYAML is installed.\n"
        )
        sys.exit(1)

    # load_merged_config handles all validation; on error it raises ValueError
    # with a message that includes the source file, bad field, and fix hint.
    try:
        cfg = load_merged_config(config_path, sdk_root)
    except ValueError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(
            f"ERROR: unexpected error while loading config: {e}\n"
            f"  Config: {config_path}\n"
        )
        sys.exit(1)

    _emit_bash_exports(cfg)


if __name__ == "__main__":
    main()
