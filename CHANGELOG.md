# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-14

### Added
- **Core**: Introduced `PipelineEngine` and `RKNNAdapter` for standardized conversion flow.
- **Bootloader**: Added `arc` (formerly start.sh) with smart interactive menu and auto-dependency check.
- **Sherpa Support**: Added specialized preprocessor for Sherpa-Zipformer (dynamic shape fixing, metadata extraction, int64 type fix).
- **Config**: Introduced YAML-based configuration system.
- **Assets**: Added `ModelDownloader` for automatic model retrieval.

### Fixed
- Resolved dependency conflicts between `onnx` versions and `rknn-toolkit2`.
- Fixed interaction latency in the bootloader menu.
