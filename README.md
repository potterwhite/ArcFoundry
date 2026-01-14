# ArcFoundry: 标准化 RKNN 转换 SDK

**ArcFoundry** 是一个专为 Rockchip NPU 设计的工程化模型转换 SDK。它采用“两阶段引导”架构，将复杂的 Python 依赖环境与用户操作隔离，通过 YAML 配置文件管理模型参数，提供稳定、可复现的 **ONNX 到 RKNN** 转换流程。

本 SDK 旨在解决 `rknn-toolkit2` 脚本碎片化问题，支持 Sherpa-Zipformer 等复杂动态图模型的无损转换，并为未来扩展（如 YOLO 系列）预留了标准接口。

## 1. 支持平台 (Supported Platforms)

ArcFoundry 基于 `rknn-toolkit2` 内核，全面支持以下 Rockchip NPU 平台：

*   **RK3588 Series**
*   **RK3576 Series**
*   **RK3566 / RK3568 Series**
*   **RK3562 Series**
*   **RV1103 / RV1106**
*   **RV1103B / RV1106B**
*   **RV1126B** (注意：不适用于旧版 RV1126)
*   **RK2118**

## 2. 目录结构 (Directory Structure)

```text
ArcFoundry/
├── start.sh                  # [入口] SDK 引导脚本 (Stage 1 Bootloader)
├── configs/                  # [配置] 用户定义的转换配置文件
│   ├── templates/            # 配置模板
│   └── rv1126b_sherpa.yaml   # 示例配置文件
├── core/                     # [内核] Python 核心代码 (Stage 2 Kernel)
│   ├── main.py               # 核心入口
│   ├── engine.py             # 流水线引擎
│   └── ...
├── workspace/                # [工作区] 存放生成的中间文件 (如 fixed.onnx)
├── output/                   # [产出] 存放最终生成的 .rknn 模型
├── requirements.txt          # 依赖列表
└── README.md                 # 说明文档
```

## 3. 快速上手 (Quick Start)

ArcFoundry 的设计逻辑是：**只需一个脚本，搞定环境与运行。**

### 3.1 初始化环境 (Stage 1)

首次使用时，`start.sh` 会自动检测是否缺少 Python 虚拟环境。如果有缺失，它会自动执行初始化（创建 venv、安装依赖）。

```bash
# 显式初始化 (可选，第一次运行 convert 时也会自动触发)
./start.sh init
```

### 3.2 执行模型转换 (Stage 2)

编写好配置文件后，直接使用 `convert` 指令。`start.sh` 会自动挂载虚拟环境并启动核心转换引擎。

```bash
# 语法: ./start.sh convert -c <config_path>
./start.sh convert -c configs/rv1126b_sherpa.yaml
```

转换完成后：
*   **日志**：会在终端实时显示转换进度、内存评估和性能预估。
*   **产物**：最终的 `.rknn` 文件将保存在配置文件指定的 `output_dir` 中。

## 4. 配置文件规范 (Configuration)

ArcFoundry 采用声明式 YAML 配置。所有的模型参数、预处理策略、编译选项均在此定义。

### 示例：Sherpa-Zipformer 转 RV1126B

```yaml
# ==========================================
# 1. 项目定义
# ==========================================
project:
  name: "sherpa-zipformer-rv1126b"
  version: "1.0"
  output_dir: "./output/rv1126b_release"
  workspace_dir: "./workspace/tmp"

# ==========================================
# 2. 目标平台
# ==========================================
target:
  platform: "rv1126b"      # 必须严格匹配 rknn-toolkit2 支持列表
  # device_id: null        # 连板调试时填写设备 ID

# ==========================================
# 3. 编译参数
# ==========================================
build:
  optimization_level: 3    # 0 (关闭) - 3 (最高), 默认为 3
  verbose: true            # 打印详细日志
  pruning: true            # 开启模型剪枝优化
  eval_memory: true        # 转换后评估内存占用

  # 量化配置 (V1.0 暂建议关闭，用于跑通流程)
  quantization:
    enabled: false
    dtype: "asymmetric_quantized-8"
    dataset: ""

# ==========================================
# 4. 模型流水线
# ==========================================
models:
  # --- 示例：Encoder 模型 ---
  - name: "encoder"
    path: "models/onnx/encoder-epoch-99-avg-1.onnx"

    # [预处理] 在送入 RKNN 之前对计算图的操作
    preprocess:
      fix_dynamic_shape: true    # 将 ONNX 动态维度固定为 1
      simplify: true             # 使用 onnxsim 简化计算图
      extract_metadata: true     # (Sherpa专用) 提取 vocab_size 等元数据

    # [输入定义] 明确指定静态 Shape，防止自动推断错误
    input_shapes:
      - [1, 80, 50]              # [Batch, Feature, Time]

  # --- 示例：Decoder 模型 ---
  - name: "decoder"
    path: "models/onnx/decoder-epoch-99-avg-1.onnx"
    preprocess:
      fix_dynamic_shape: true
      simplify: true
      extract_metadata: true
      fix_int64_type: true       # (Sherpa专用) 修复 decoder 输入类型丢失问题
    input_shapes:
      - [1, 512]
```

## 5. 功能特性说明

### 5.1 自动预处理 (Auto Preprocessing)
为了解决原始 ONNX 模型无法直接转换的问题，SDK 内置了以下策略开关：

*   **`fix_dynamic_shape`**: 自动识别 ONNX 中的 `dim_param` (动态维度) 并将其固定为静态值（通常为 1），这是 NPU 推理的必要条件。
*   **`extract_metadata`**: 针对 Sherpa 等将配置参数（如 Token 数量）隐藏在 ONNX Metadata 中的模型，SDK 会自动提取并通过 `custom_string` 注入 RKNN，确保推理库能正确加载模型。
*   **`fix_int64_type`**: 针对某些导出工具导致的输入节点数据类型错误（例如本应是 INT64 却变成了未定义），在转换前强制修复。

## 6. 开发路线图 (Roadmap)

### V1.1 Quantization & Verification (In Progress)
本版本致力于解决模型量化与精度验证的闭环问题，引入无需开发板（SoC）即可验证精度的机制。

*   **PC 端模拟器验证 (Simulator Verification)**:
    *   集成 RKNN Simulator，在 x86 PC 上直接运行推理。
    *   通过对比 ONNX (FP32) 与 RKNN (Quantized) 的推理结果，计算余弦相似度 (Cosine Similarity)，自动判定量化精度是否达标。
*   **量化校准数据集抽象 (Calibration Abstraction)**:
    *   支持配置化生成校准数据集（Audio/Image）。
    *   针对 ASR 模型（如 Sherpa），提供专门的 Streaming Audio 切片逻辑。