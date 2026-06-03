# ArcFoundry 配置系统参考

> 本文档是 ArcFoundry YAML 配置系统的完整参考手册。
> 所有模型转换均通过声明式 YAML 配置驱动，无需修改 Python 代码。

---

## 1. 概述

ArcFoundry 使用单个 YAML 文件定义整个转换流水线。配置文件通过 `yaml.safe_load` 加载为 Python dict，各模块自行校验所需的 key。

**配置文件位置**：`configs/<model_family>/<platform>_<model>_<precision>_<HxW>[_dsr].yaml`

**运行方式**：
```bash
./arc <config_name>        # 快捷模式，自动匹配 configs/ 下的文件
./arc configs/custom.yaml  # 显式路径模式
```

---

## 2. 顶层 Schema

YAML 配置有四个顶层 section：

```yaml
project:    # 必需 — 项目元数据
target:     # 必需 — 目标平台
build:      # 必需 — 构建参数
models:     # 必需 — 模型列表（支持多模型）
```

### 2.1 `project`

| Key | Type | Description |
|-----|------|-------------|
| `name` | string | 构建标识符，用于日志和输出目录名 |
| `output_dir` | string | 最终 `.rknn` 文件的输出路径 |
| `workspace_dir` | string | 中间文件（处理后的 ONNX、校准数据集）路径 |

### 2.2 `target`

| Key | Type | Description |
|-----|------|-------------|
| `platform` | string | RKNN 目标平台，如 `"rk3588"`, `"rv1126b"` |

### 2.3 `build`

| Key | Type | Description |
|-----|------|-------------|
| `optimization_level` | int (0-3) | 图融合激进程度。0=关闭, 1=安全, 2=平衡, 3=激进 |
| `verbose` | bool | 启用 RKNN toolkit 详细日志 |
| `pruning` | bool | 启用图剪枝（通常与 optimization_level=3 配合） |
| `eval_memory` | bool | 启用内存评估模式 |
| `quantization` | dict | 量化子配置（见 §4） |

---

## 3. 预处理策略（`models[].preprocess`）

预处理流水线按固定顺序执行 6 个策略。其中 5 个是**必需的**（必须在每个模型的 `preprocess` 块中显式声明），1 个是**可选的**（不写则跳过）。

### 必需策略（`_REQUIRED_STRATEGY_KEYS`）

以下 5 个 key 必须存在于每个模型的 `preprocess` 块中，缺失任何一个都会导致 `KeyError`：

```yaml
preprocess:
  fix_dynamic_shape: { enabled: true, strict_override: true }
  fold_constant_inputs: { enabled: false, inputs: {} }
  fix_int64_type: false
  simplify: true
  extract_metadata: true
```

#### 3.1 `fix_dynamic_shape`

将 ONNX 模型中的动态维度（symbolic dim）固定为 `input_shapes` 中指定的值。

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | 启用/禁用 |
| `strict_override` | bool | `true` = 即使静态维度也强制覆盖；`false` = 只修复动态维度 |

#### 3.2 `fold_constant_inputs`

将指定的模型 input tensor 折叠为常量（从 graph.input 移到 graph.initializer）。

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | 启用/禁用 |
| `inputs` | dict | tensor_name → value_list 映射，如 `{downsample_ratio: [0.5]}` |

#### 3.3 `fix_int64_type`

将名为 `'y'` 的 input tensor 强制设为 INT64 类型（Sherpa decoder 专用）。

Type: `bool`

#### 3.4 `simplify`

运行 `onnxsim.simplify()` 简化 ONNX 图。

Type: `bool`

#### 3.5 `extract_metadata`

从 ONNX 模型中提取 `custom_metadata_map`（Sherpa 用于注入 vocab_size 等元数据）。

Type: `bool`

### 可选策略

#### 3.6 `graph_surgery`（可选）

基于 onnx-graphsurgeon 的模块化图手术。不加入 `_REQUIRED_STRATEGY_KEYS`，不写则跳过，不影响其他模型。

**模块化架构**：按 OGS 的四大操作对象组织子操作：

```
graph_surgery
├── outputs    (graph.outputs — 出口 tensor)     # 已实现
├── inputs     (graph.inputs — 入口 tensor)       # future
├── nodes      (graph.nodes — 算子节点)            # future
└── tensors    (tensor 属性)                       # future
```

每个子对象下有操作动词（modify / add / remove），操作动词下是参数列表。

**执行顺序**：在 `fold_constant_inputs` (3/6) 之后、`fix_int64_type` (5/6) 之前执行。

**配置格式**：

```yaml
graph_surgery:
  enabled: true
  outputs:                    # graph.outputs 操作
    modify:                   # 替换现有输出 tensor
      - existing: "fgr"       # 当前 graph.outputs 中的 tensor 名
        replacement: "777"    # 替换为同图中的 tensor 名
      - existing: "pha"
        replacement: "779"
  # inputs:                  # graph.inputs 操作（future）
  #   modify: [...]
  # nodes:                   # graph.nodes 操作（future）
  #   remove: [...]
  # tensors:                 # tensor 属性操作（future）
  #   modify: [...]
```

**`outputs.modify` 语义**：

- `existing`：ONNX graph output 的 tensor 名（**不是** RKNN profiler 的 `"OutputOperator:xxx"` 格式）
- `replacement`：同一 ONNX 图中已有的 tensor 名（用 onnx / Netron 查找）
- 操作：在 `graph.outputs` 列表中找到 `existing` tensor，替换为 `replacement` tensor
- `graph.cleanup().toposort()` 自动删除所有不贡献输出的死节点

**RVM 示例**：移除 DeepGuidedFilterRefiner 的 Resize×2 + element-wise 尾巴，保留 refiner Conv 网络在 NPU 上计算 A 和 b。详见 PKB `s5_8_22_12` / `s5_8_22_14`。

---

## 4. 量化配置（`build.quantization`）

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | INT8 量化总开关 |
| `dtype` | string | 量化类型，固定为 `"asymmetric_quantized-8"` |
| `dataset` | string | 校准图片列表文件路径 |
| `sampling_interval` | int | 从数据集中每隔 N 个样本取一个用于校准 |

---

## 5. 输入形状（`models[].input_shapes`）

字典，key 为 tensor 名，value 为 shape 列表。不同模型架构有不同的 tensor 名：

| 模型 | Tensor 名 | 示例 Shape |
|------|-----------|-----------|
| RVM | `src`, `r1i`, `r2i`, `r3i`, `r4i` | `[1, 3, 544, 960]` |
| MODNet | `input` | `[1, 3, 576, 1024]` |
| Sherpa encoder | `x` | `[1, 71, 80]` |
| Sherpa decoder | `y` | `[1, 512]` |
| Sherpa joiner | `encoder_out`, `decoder_out` | `[1, 512]` |

---

## 6. 归一化（`models[].normalization`）

可选。定义 RKNN 运行时的均值/标准差归一化参数。

| Key | Type | Description |
|-----|------|-------------|
| `mean_values` | list[list[float]] | 逐通道均值，如 `[[0, 0, 0]]` 或 `[[127.5, 127.5, 127.5]]` |
| `std_values` | list[list[float]] | 逐通道标准差，如 `[[255, 255, 255]]` 或 `[[127.5, 127.5, 127.5]]` |

**常见组合**：

| 模型 | mean | std | 含义 |
|------|------|-----|------|
| RVM | `[[0,0,0]]` | `[[255,255,255]]` | 像素值 /255 归一化到 [0,1] |
| MODNet | `[[127.5,127.5,127.5]]` | `[[127.5,127.5,127.5]]` | ImageNet 标准归一化 |
| ASR | 不设置 | 不设置 | 无归一化 |

---

## 7. 配置示例

### 7.1 RVM FP16（标准版）

```yaml
models:
  - name: "rvm_mobilenetv3_fp16_544x960_0.5-dsr_rk3588"
    path: "models/downloads/rvm_mobilenetv3.onnx"
    preprocess:
      fix_dynamic_shape: { enabled: true, strict_override: true }
      fold_constant_inputs: { enabled: true, inputs: { downsample_ratio: [0.5] } }
      fix_int64_type: false
      simplify: true
      extract_metadata: true
    input_shapes:
      src: [1, 3, 544, 960]
      r1i: [1, 16, 136, 240]
      r2i: [1, 20, 68, 120]
      r3i: [1, 40, 34, 60]
      r4i: [1, 64, 17, 30]
```

### 7.2 RVM FP16（no-resize，使用 graph_surgery）

```yaml
models:
  - name: "rvm_mobilenetv3_fp16_544x960_0.5-dsr-no-resize_rk3588"
    path: "models/downloads/rvm_mobilenetv3.onnx"
    preprocess:
      fix_dynamic_shape: { enabled: true, strict_override: true }
      fold_constant_inputs: { enabled: true, inputs: { downsample_ratio: [0.5] } }
      graph_surgery:
        enabled: true
        outputs:
          modify:
            - existing: "fgr"
              replacement: "777"
            - existing: "pha"
              replacement: "779"
      fix_int64_type: false
      simplify: true
      extract_metadata: true
    input_shapes:
      src: [1, 3, 544, 960]
      r1i: [1, 16, 136, 240]
      r2i: [1, 20, 68, 120]
      r3i: [1, 40, 34, 60]
      r4i: [1, 64, 17, 30]
```

### 7.3 MODNet FP16（无 graph_surgery）

```yaml
models:
  - name: "modnet"
    path: "models/downloads/modnet_photographic_portrait_matting.onnx"
    preprocess:
      fix_dynamic_shape: { enabled: true, strict_override: true }
      fold_constant_inputs: { enabled: false, inputs: {} }
      fix_int64_type: false
      simplify: true
      extract_metadata: true
    input_shapes:
      input: [1, 3, 576, 1024]
```
