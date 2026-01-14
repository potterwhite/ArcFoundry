# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0](https://github.com/potterwhite/ArcFoundry/compare/v0.2.0...v0.3.0) (2026-01-14)


### ✨ Added

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
