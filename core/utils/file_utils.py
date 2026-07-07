# Copyright (c) 2026 PotterWhite
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import yaml
import os
import glob

from .input_signature import InputSignature

# Global logger instance
logger = logging.getLogger("ArcFoundry")


def load_config_file(path):
    """
    Backward-compat thin wrapper. Prefer load_merged_config() in new code.

    Historically, this only loaded a single YAML file. As of the rknn_toolkit2
    override feature, configs auto-merge with .config/common/rk-toolchain.yaml
    if it exists — see load_merged_config for details.
    """
    return load_merged_config(path)


def _deep_merge(base, override):
    """
    Deep-merge two dicts. `override` wins on scalar collisions; both sides
    recurse into dicts. Lists are replaced, not concatenated.

    Returns a NEW dict — neither input is mutated.
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _validate_rknn_toolkit2(cfg, file_path):
    """
    Validate the rknn_toolkit2 section of a single config file.

    Raises ValueError with a message that includes:
      - The bad field path
      - The source file (so user knows where to fix it)
      - A concrete fix hint

    Validation rules (in order):
      1. rknn_toolkit2 absent                  -> OK, use official repo default
      2. rknn_toolkit2 not a dict              -> ERROR
      3. tarball_path absent or empty string   -> OK, use official repo default
      4. tarball_path not a string             -> ERROR
      5. tarball_path not absolute             -> ERROR (prevent relative-path confusion)
      6. tarball_path file does not exist      -> ERROR
      7. tarball_sha256 mismatch               -> DEFERRED to bash overlay (not validated here)
    """
    rt = cfg.get("rknn_toolkit2")
    if rt is None:
        return  # rule 1
    if not isinstance(rt, dict):
        raise ValueError(
            f"rknn_toolkit2 must be a YAML mapping, got: {type(rt).__name__}\n"
            f"  File: {file_path}\n"
            f"  Fix: see schema in .config/common/rk-toolchain.template.yaml"
        )

    tarball_path = rt.get("tarball_path", "")
    if not tarball_path:
        return  # rule 3
    if not isinstance(tarball_path, str):
        raise ValueError(
            f"rknn_toolkit2.tarball_path must be a string, got: {type(tarball_path).__name__}\n"
            f"  File: {file_path}"
        )

    if not os.path.isabs(tarball_path):
        raise ValueError(
            f"rknn_toolkit2.tarball_path must be an ABSOLUTE path, got: {tarball_path}\n"
            f"  File: {file_path}\n"
            f"  Fix: use an absolute path like /development/toolchains/xxx.tgz"
        )

    if not os.path.isfile(tarball_path):
        raise ValueError(
            f"rknn_toolkit2.tarball_path not found: {tarball_path}\n"
            f"  File: {file_path}\n"
            f"  Fix: check the path, or remove the field to fall back to official repo (v2.3.2)"
        )


def _locate_sdk_root(config_path):
    """
    Walk up from config_path looking for a directory that contains a `.config/`
    subdir — that's our SDK_ROOT. Falls back to the parent of the .git dir, then
    to cwd. Returns absolute path string.
    """
    cur = os.path.dirname(os.path.abspath(config_path))
    while cur and cur != "/":
        if os.path.isdir(os.path.join(cur, ".config")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # Fallback: try to find .git parent
    cur = os.path.dirname(os.path.abspath(config_path))
    while cur and cur != "/":
        if os.path.isdir(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.getcwd()


def load_merged_config(config_path, sdk_root=None):
    """
    Load a user YAML config, auto-merging .config/common/rk-toolchain.yaml if
    present. Validates the rknn_toolkit2 field (each file validated separately
    so error messages name the correct source file).

    Merge semantics:
      - User config takes precedence over common config.
      - Dict values are deep-merged; lists/scalars are replaced.

    Args:
        config_path: Path to the user's YAML config.
        sdk_root:    SDK root directory. If None, derived by walking up from
                     config_path looking for a `.config/` directory.

    Returns:
        Merged config dict.

    Raises:
        ValueError on validation errors (with file/field/fix in the message).
    """
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config root must be a YAML mapping, got: {type(cfg).__name__}\n"
            f"  File: {config_path}"
        )

    # Validate the user config alone (so we know which file any bad field came from)
    _validate_rknn_toolkit2(cfg, config_path)

    # Locate SDK_ROOT
    if sdk_root is None:
        sdk_root = _locate_sdk_root(config_path)
    sdk_root = os.path.abspath(sdk_root)

    # Auto-merge common config if it exists
    common_path = os.path.join(sdk_root, ".config", "common", "rk-toolchain.yaml")
    if os.path.isfile(common_path):
        with open(common_path, "r") as f:
            common_cfg = yaml.safe_load(f) or {}
        if not isinstance(common_cfg, dict):
            raise ValueError(
                f"Common config root must be a YAML mapping, got: {type(common_cfg).__name__}\n"
                f"  File: {common_path}"
            )
        # Validate the common config alone (each file validated independently)
        _validate_rknn_toolkit2(common_cfg, common_path)
        # Deep-merge: user config wins on conflicts
        cfg = _deep_merge(common_cfg, cfg)

    return cfg


def cleanup_garbage(target_dir=".", patterns=None):
    """
    Cleans up temporary/garbage files matching specific patterns.

    Args:
        target_dir (str): Directory to search in. Defaults to current CWD.
        patterns (list): List of glob patterns. Defaults to RKNN junk files.
    """
    if patterns is None:
        # Default junk generated by RKNN C++ backend
        patterns = ["check*.onnx", "encoder.processed.*", "*.rknn_util_Config"]

    if 'logger' in globals():
        logger.info(f"\n🧹 Cleaning up junk files in {target_dir}...")

    for pattern in patterns:
        search_path = os.path.join(target_dir, pattern)

        found_files = glob.glob(search_path)

        for f in found_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    if 'logger' in globals():
                        # Use global logger if available
                        # global means who invoked this function has already setup logging
                        logger.info(f"🧹 Cleaned up junk file: {f}")
            except OSError as e:
                if 'logger' in globals():
                    logger.error(f"Failed to delete file {f}: {str(e)}")

    if 'logger' in globals():
        logger.info(f"\n🧹 Cleanup complete.\n")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_sdk_version():
    """
    Simple parser to get version from pyproject.toml (SSOT).
    """
    try:
        # Locate project root (assuming core/utils.py is in core/)
        root_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        toml_path = os.path.join(root_dir, "pyproject.toml")

        if os.path.exists(toml_path):
            with open(toml_path, 'r') as f:
                for line in f:
                    if line.strip().startswith("version"):
                        return line.split('=')[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return "unknown"


def get_input_signature_from_yaml(cfg, model_name=None):
    """
    Extract input signature from YAML config.
    If model_name is None, use the first model.
    """

    models = cfg.get("models", [])

    if not models:
        raise ValueError("No models found in YAML.")

    # If model_name specified, filter it
    if model_name:
        models = [m for m in models if m.get("name") == model_name]
        if not models:
            raise ValueError(f"Model {model_name} not found in YAML.")

    model = models[0]

    input_shapes = model.get("input_shapes", [])
    if not input_shapes:
        raise ValueError(f"Model {model.get('name')} has no input_shapes.")

    if isinstance(input_shapes, dict):
        shape = list(input_shapes.values())[0]
    else:
        shape = input_shapes[0]

    return InputSignature(shape)
