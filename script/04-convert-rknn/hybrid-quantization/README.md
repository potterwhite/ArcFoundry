

# export the quant config
```bash
python3 ../00_02_onnx_to_rknn_converter_general.py encoder -o ../../../models/rknn/sherpa-zipformer/int8/12th-hybrid-encoder/ --export-quant-config
```

# determine the op`s name via script

## 1st. execute the py
```bash
python3 inspect_quant_config.py encoder_simplified.quantization.cfg --onnx-model ./check3_fuse_ops.onnx --find-inputs-for exSoftmax13
```
## 2nd. you will get:
```bash
--- Finding CFG layers for ONNX operator type: exSoftmax13 ---
Found 20 'exSoftmax13' nodes in the ONNX model.

# --- Matched Layers (for OP Type: exSoftmax13) ---
# Copy and paste these into 'custom_quantize_layers' in your .cfg file.
    '1514_conv': float16,
    '3319_conv': float16,
    '4705_conv': float16,
    '5677_conv': float16,
    '7482_conv': float16,

```

## 3rd. add in to encoder_simplified.quantization.cfg
only the `custom_quantize_layers` field
```bash
custom_quantize_layers: {
    '1514_conv': float16,
    '3319_conv': float16,
    '4705_conv': float16,
    '5677_conv': float16,
    '7482_conv': float16,
}
```

# import the config
```bash
python3 ../00_02_onnx_to_rknn_converter_general.py encoder -o ../../../models/rknn/sherpa-zipformer/int8/12th-hybrid-encoder/ --import-quant-config
```