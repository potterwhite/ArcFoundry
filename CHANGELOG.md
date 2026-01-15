# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0](https://github.com/potterwhite/ArcFoundry/compare/v0.5.0...v0.6.0) (2026-01-15)


### ✨ Added

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
