# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.1](https://github.com/potterwhite/ArcFoundry/compare/v0.9.0...v0.9.1) (2026-01-30)


### 🐛 Fixed

* **import:** resolve 'core.utils is not a package' error ([#19](https://github.com/potterwhite/ArcFoundry/issues/19)) ([3b21f2a](https://github.com/potterwhite/ArcFoundry/commit/3b21f2a941cab9aadf777fb18fe5ad728b640949))

## [0.9.0](https://github.com/potterwhite/ArcFoundry/compare/v0.8.0...v0.9.0) (2026-01-30)

### ✨ Added

* Added support for **multiple Python versions (3.8–3.12)** with automatic detection of the host Python interpreter.
* Introduced **automatic RKNN toolkit setup**, including ABI-aware wheel selection to match the active Python version.
* Added **RK3588_sherpa.yaml** for zipformer build configuration, verified on evaluation hardware.

---

### 🔧 Changed

* Refactored the bash launcher into a **layered execution workflow** with automatic recovery from common setup failures, including:

  * Python interpreter fallback when the default `python3` is unavailable.
  * Automatic installation of the RKNN toolkit when missing.
  * Dynamic selection of RKNN wheels based on Python ABI.
  * Graceful handling of OpenCV GUI dependencies by falling back to headless builds.
* Made the feature extractor **time-frame configuration explicit and YAML-driven**, ensuring deterministic alignment with model input shapes.
* Reorganized the **core workflow and utility module structure** to improve maintainability and separation of concerns.

---

### ⚙️ Improved

* Improved robustness of the build and conversion pipeline on heterogeneous host environments (desktop, server, and containerized setups).
* Reduced manual environment setup steps by embedding common recovery logic directly into the launcher scripts.

---

### 🧪 Conditional Behavior

* Skipped automated accuracy verification when **quantization is disabled**, avoiding unnecessary validation steps in non-quantized builds.

---

### 📝 Notes

* This release introduces significant internal refactoring and build-system enhancements while remaining backward compatible with existing workflows.
* No user-facing API changes are required.


* **build:** add multi-python (3.8–3.12) support and RKNN auto-setup ([4709e9f](https://github.com/potterwhite/ArcFoundry/commit/4709e9fdcbf512df1559a85cc9c702374a81cf95))

---

## [0.8.0](https://github.com/potterwhite/ArcFoundry/compare/v0.7.0...v0.8.0) (2026-01-23)

### 🚀 New Features 

*   **Modular Architecture**: Completely refactored the core engine. The monolithic pipeline is now split into three specialized workers:
    *   `QuantizationConfigurator`: Manages dataset preparation and precision decisions.
    *   `StandardConverter`: Handles the standard ONNX to RKNN conversion and verification.
    *   `PrecisionRecoverer`: Manages the interactive hybrid quantization recovery workflow.
*   **Smart FP16 Fallback**: If the calibration dataset is missing or invalid, the pipeline now intelligently asks to fallback to FP16 mode instead of crashing.
*   **Timed Interaction**: Introduced `timed_input` utility. Interactive prompts (e.g., asking to enable hybrid quantization) now have a timeout and will auto-select a default action if the user is away.
*   **Enhanced Hybrid Patching**:
    *   Added a **Whitelist Mechanism** (Conv, Gemm, MatMul, Linear) to prevent modifying unsafe internal nodes during auto-patching.
    *   Improved Regex parsing to support scientific notation in RKNN accuracy analysis reports.
*   **Auto Cleanup**: Added a `cleanup_garbage` utility to automatically remove intermediate ONNX and RKNN configuration files after a successful build.
*   **Visualized Verification**: The verification logs now include visual indicators (✅, ⚠️, ❌) and explicit checks for `NaN` or `Inf` values in inference outputs.

### 🛠 Refactoring 

*   **DSP Module**: Renamed `core/dsp/audio_features.py` to `core/dsp/sherpa_features_extractor.py` to explicitly indicate its purpose for Sherpa-Onnx compatibility.
*   **Streaming Strategy**: Optimized `core/quantization/strategies/streaming.py` to strictly handle dynamic shapes by forcing dimension parameters to `1`.
*   **Config Structure**: Moved complex build logic out of `engine.py` into dedicated workflow classes.

### 📚 Documentation 

*   **Complete Overhaul**: Redesigned `README.md` with new architectural diagrams (Mermaid) and clearer usage instructions.
*   **i18n Support**: Added Simplified Chinese documentation (`docs/README_ZH_CN.md`).
*   **Assets**: Added project banner and architecture illustrations.

### 🔧 Configuration

*   Updated `configs/rv1126b_sherpa.yaml`:
    *   Changed default `optimization_level` to `3`.
    *   Clarified `sampling_interval` comments for calibration.
*   Updated `.gitignore` to exclude RKNN generated temp files (`*.quantization.cfg`, `*.rknn_util_Config`, etc.).


* **quantization:** enhance hybrid quantization strategy and patching ([8ee741a](https://github.com/potterwhite/ArcFoundry/commit/8ee741a7bbd4c7d41c91e9a360ff6b2f5a0532a7))
* **utils:** add interactive utilities and garbage cleanup ([8ee741a](https://github.com/potterwhite/ArcFoundry/commit/8ee741a7bbd4c7d41c91e9a360ff6b2f5a0532a7))

## [0.7.0](https://github.com/potterwhite/ArcFoundry/compare/v0.6.0...v0.7.0) (2026-01-19)


### ✨ Added implement hybrid quantization workflow and refactor calibration architecture

**Core Engine (`core/engine.py`):**
- Refactor `run()` pipeline into modular stages (`_prepare_build_from_json`, `_convert_and_evaluate`).
- Implement `_recover_precision` to handle Hybrid Quantization (Steps 1 & 2) when model accuracy is below threshold.
- Add "Fast-Forward" logic to reuse existing analysis reports and skip redundant builds.

**Calibration (`core/quantization/`):**
- Refactor `CalibrationGenerator` to use a Strategy/Registry pattern.
- Extract streaming-specific logic to `StreamingAudioStrategy`.
- Implement caching in `generate()` to reuse existing dataset lists.

**RKNN Adapter (`core/rknn_adapter.py`):**
- Add wrappers for `hybrid_step1` and `hybrid_step2` to support mixed-precision building.
- Implement `apply_hybrid_patch` to parse accuracy reports and modify quantization config files (auto-switching sensitive layers to float16).
- Enhance regex parsing for analysis reports to support scientific notation and whitelist specific operators.

**Preprocessor & Utilities:**
- Add cache detection to `Preprocessor` to load existing processed ONNX models.
- Implement `SmartNewlineFormatter` in `utils.py` for better log formatting.
- Update `pyproject.toml` YAPF settings and add optimization level documentation to `rv1126b_sherpa.yaml`.

* implement hybrid quantization workflow and refactor calibration architecture ([#12](https://github.com/potterwhite/ArcFoundry/issues/12)) ([d15a43b](https://github.com/potterwhite/ArcFoundry/commit/d15a43bdb728631b461c689e01245e552c92bd57))

## [0.6.0](https://github.com/potterwhite/ArcFoundry/compare/v0.5.0...v0.6.0) (2026-01-15)


### ✨ Added

• Implement QuantizationAnalyzer using rknn.accuracy_analysis
• Auto deep analysis when verification score < 0.99
• Fix dataset_list.txt parsing issue in accuracy_analysis
• Improve log structure & add total execution time
• Refactor verification to return min cosine similarity

internal change: PipelineEngine._verify_model signature changed

* **quant:** add automatic layer-wise quantization accuracy analysis & trigger on low cosine score ([#10](https://github.com/potterwhite/ArcFoundry/issues/10)) ([36bf403](https://github.com/potterwhite/ArcFoundry/commit/36bf403c932177ca068668f97026e3112bbb060d))

## [0.5.0](https://github.com/potterwhite/ArcFoundry/compare/v0.4.0...v0.5.0) (2026-01-14)

### ✨ Added

- Change release-type from "simple" to "python" for better Python project support
- Replace simple extra-files string with jsonpath object: "$['project']['version']"
- This ensures reliable version updates in pyproject.toml without depending on heuristic detection

Fixes the previous issue where pyproject.toml version was not being bumped automatically.

* **release:** migrate release-please to python strategy ([#8](https://github.com/potterwhite/ArcFoundry/issues/8)) ([dbac995](https://github.com/potterwhite/ArcFoundry/commit/dbac9950d6ad097db7e561ff4bcf1279016278c2))

## [0.4.0](https://github.com/potterwhite/ArcFoundry/compare/v0.3.0...v0.4.0) (2026-01-14)

### ✨ Added
#### Summary
This version completes the **Quantization & Verification** roadmap. It integrates the streaming calibration logic (`09_generate...py`) into the core architecture and establishes a defensive pipeline to ensure stability during the INT8 conversion process.

#### Key Changes

##### 1. Quantization Engine (`core/quantization/`)
*   **Streaming Simulator**: Implemented `CalibrationGenerator` to perform streaming inference on-the-fly, capturing real hidden states (`h`, `c`) for RNN-T models.
*   **Smart Caching**: Added detection logic to skip expensive calibration (20+ mins) if valid `.npy` artifacts exist.
*   **Absolute Paths**: Enforced absolute path resolution to prevent RKNN Toolkit dataset parsing errors.

##### 2. Robust Pipeline (`core/engine.py`)
*   **Defensive "Firewall"**: Refactored `run()` to explicitly scrub `dataset` fields (setting to `None`) before passing config to RKNN, preventing raw FLAC file leakage that caused crashes.
*   **Config Isolation**: Decoupled `build_config` passed to the Validator from the global config, ensuring the "Shadow Build" verification uses the correct `.npy` dataset.

##### 3. Infrastructure
*   **SSOT Versioning**: Updated `core/utils.py` to read version strictly from `pyproject.toml`.
*   **CI/CD**: Configured `release-please` to manage `pyproject.toml` version bumping via `extra-files`.

##### 4. License-Check
* Add license comment block at the top of every files
* Add license-check.yml in .github to guard the license block in every file in the future

#### Current Status
*   **Engineering**: ✅ PASSED. The pipeline (Audio -> Streaming Inference -> NPY -> RKNN INT8 -> Verification) is fully operational.
*   **Accuracy**: ⚠️ WARNING. Baseline INT8 conversion for Sherpa-Zipformer shows ~0.91 Cosine Similarity on `encoder_out`. This confirms the validator is working and highlights the necessity for **Hybrid Quantization **.


* **core:** implement quantization calibration & robust pipeline ([#5](https://github.com/potterwhite/ArcFoundry/issues/5)) ([5e76c99](https://github.com/potterwhite/ArcFoundry/commit/5e76c99058f181560e9fc9c31c3c4da3c4fa6e57))

## [0.3.0](https://github.com/potterwhite/ArcFoundry/compare/v0.2.0...v0.3.0) (2026-01-14)

### ✨ Features (Verification & DSP)

* **Verification System**: Implemented automated "Zero-Hardware" accuracy validation.
  * Introduced **Shadow Build** mechanism to enable PC-based simulation for RV1126B/RV1106 targets (bypassing export limitations).
  * Added automatic **Cosine Similarity** checks between ONNX (FP32) and RKNN output.
  * Implemented `core/dsp` module to align Python feature extraction with Sherpa-Onnx C++ runtime (Pre-emphasis, DC Removal).

* **Core Engine**:
  * **Type-Aware Inputs**: Automatically detects `int64`/`bool` vs `float32` ONNX inputs to generate correct dummy data for state tensors.
  * **Hybrid Validation**: Supports validating models using real audio (via `build.test_input`) for features while simulating random states.

* **core:** implement automated accuracy verification pipeline (V1.1) ([#3](https://github.com/potterwhite/ArcFoundry/issues/3)) ([4c0d377](https://github.com/potterwhite/ArcFoundry/commit/4c0d377ee85fa4980d37c0b4197af989b3c0335a))

## [0.2.0](https://github.com/potterwhite/ArcFoundry/compare/v0.1.0...v0.2.0) (2026-01-14)

### ✨ Features (The Architecture Release)

* **System & CLI**: Introduced `arc` (formerly start.sh) as the unified entry point.
  * Implemented "Two-Stage Boot" architecture.
  * Added smart interactive menu and dependency auto-detection.
  * Added auto-installation for RKNN Toolkit2.

* **Core Engine**:
  * **Pipeline**: Implemented `PipelineEngine` to orchestrate Download -> Preprocess -> Convert flows.
  * **Adapter**: Created `RKNNAdapter` to decouple RKNN-Toolkit2 API calls.
  * **Assets**: Added `ModelDownloader` with URL fallback and progress bars.

* **Domain Logic (Sherpa-Zipformer)**:
  * Implemented specialized graph surgery strategies:
    * `fix_dynamic_shape`: Automatically sets dynamic params to static values.
    * `extract_metadata`: Preserves critical ASR metadata during conversion.
    * `fix_int64_type`: Repairs data type mismatches for Decoder inputs.

* **Configuration**:
  * Standardized YAML config schema (e.g., `configs/rv1126b_sherpa.yaml`).
  * Centralized project metadata in `pyproject.toml` and dependencies in `envs/requirements.txt`.

### 🐛 Bug Fixes

* Resolved dependency conflicts between `onnx` (1.14 vs 1.17) and `rknn-toolkit2`.
* Fixed interaction latency in the bootloader menu (moved environment check to post-selection).

---

## [0.1.0] - 2026-01-13

### 🎉 Initial Setup

* **Repository**: Initialized Git repository and directory structure.
* **CI/CD**: Configured Google Release Please workflow.
* **License**: Added MIT License.
