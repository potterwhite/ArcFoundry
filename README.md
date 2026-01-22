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

*   **`extract_metadata`**: 针对 Sherpa 等将配置参数（如 Token 数量）隐藏在 ONNX Metadata 中的模型，SDK 会自动提取并通过 `custom_string` 注入 RKNN，确保推理库能正确加载模型。
*   **`fix_dynamic_shape`**: 自动识别 ONNX 中的 `dim_param` (动态维度) 并将其固定为静态值（通常为 1），这是 NPU 推理的必要条件。
*   **`fix_int64_type`**: 针对某些导出工具导致的输入节点数据类型错误（例如本应是 INT64 却变成了未定义），在转换前强制修复。
*   **`onnxsim`**: 图优化。

5.2 自动量化
需要准备数据集-（例如音频模型就是准备高质量的音频）
只接受encoder进行量化（其他模型fp16的推理速度不慢，即量化的收益较低，因此继续使用fp16可保证推理效果）



## 6. 自动化精度验证 (Auto-Verification)

ArcFoundry V1.1 引入了**“零硬件”验证机制**。无需连接开发板，SDK 会在转换结束后自动启动 PC 端模拟器，对比 ONNX (FP32) 与 RKNN (FP16/INT8) 的推理结果。

### 6.1 结果解读 (How to read)
系统会根据输出张量的**余弦相似度 (Cosine Similarity)** 自动给出评级：

*   ✅ **[PASSED] 验证通过** (Score > 0.98)
    *   意味着模型精度几乎无损，可直接部署上线。
*   ⚠️ **[WARNING] 精度警告** (Score < 0.98)
    *   意味着量化过程造成了明显的精度损失。
    *   **建议操作**：检查量化校准集 (Calibration Dataset) 的覆盖范围，或尝试开启混合量化。
*   ❌ **[FAILED] 验证失败**
    *   通常由输入维度 (NCHW/NHWC) 或数据类型 (INT64/Float32) 不匹配引起，需检查 `input_shapes` 配置或 ONNX 模型结构。

### 6.2 进阶配置 (Optional)
默认情况下，SDK 使用随机噪声进行“冒烟测试”以验证连通性。为了获得最真实的精度评估，建议在 `yaml` 配置中指定一个真实的测试样本：

```yaml
build:
  # [可选] 指定一个本地音频文件
  # 验证器会自动提取 Mel 特征并送入模型，计算真实的推理精度
  test_input: "./data/test_01.wav"
```
