import os
import yaml
from core.utils import logger, ensure_dir
from core.preprocessor import Preprocessor
from core.rknn_adapter import RKNNAdapter
from core.downloader import ModelDownloader  # <--- 新增引用
import numpy as np
import onnxruntime as ort
from core.dsp.audio_features import SherpaFeatureExtractor
from core.verification.comparator import ModelComparator
from core.quantization.calibrator import CalibrationGenerator


class PipelineEngine:
    """
    Orchestrates the conversion pipeline:
    Config -> Download(Optional) -> Preprocess -> Convert -> Output
    """

    def __init__(self, config_path):
        self.config_path = config_path
        self.cfg = self._load_config(config_path)

        # Paths
        self.workspace = self.cfg.get("project",
                                      {}).get("workspace_dir", "./workspace")
        self.output_dir = self.cfg.get("project",
                                       {}).get("output_dir", "./output")

        ensure_dir(self.workspace)
        ensure_dir(self.output_dir)

    def _load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def run(self):
        project_name = self.cfg.get("project", {}).get("name")
        target_plat = self.cfg.get("target", {}).get("platform")

        logger.info(
            f"=== Starting ArcFoundry Pipeline: {project_name} on {target_plat} ==="
        )

        # Initialize Helper Modules
        downloader = ModelDownloader()  # <--- 实例化下载器
        preprocessor = Preprocessor(self.cfg)

        models = self.cfg.get("models", [])
        success_count = 0

        for model_cfg in models:
            model_name = model_cfg["name"]
            target_path = model_cfg["path"]  # YAML里指定的目标本地路径
            model_url = model_cfg.get("url", None)  # 既然是可选的，就用 get

            logger.info(f"\n>>> Processing Model: {model_name}")

            # --- Stage 0: Asset Management ---
            # 检查文件是否存在，不存在则下载，下载不了则报错
            if not downloader.ensure_model(target_path, model_url):
                logger.error(
                    f"Skipping {model_name} due to missing input file.")
                continue

            # --- Stage 1: Preprocessing ---
            processed_onnx_name = f"{model_name}.processed.onnx"
            processed_onnx_path = os.path.join(self.workspace,
                                               processed_onnx_name)

            strategies = model_cfg.get("preprocess", {})

            final_onnx_path, custom_string = preprocessor.process(
                target_path,  # 这里已经是确认存在的路径了
                processed_onnx_path,
                strategies,
            )

            if not final_onnx_path:
                logger.error(f"Preprocessing failed for {model_name}")
                continue

            # --- Stage 2: RKNN Conversion ---
            rknn_out_path = os.path.join(self.output_dir, f"{model_name}.rknn")
            input_shapes = model_cfg.get('input_shapes', None)

            # [Defensive Logic] 1. 深度拷贝配置，避免污染全局
            import copy
            build_config = copy.deepcopy(self.cfg.get('build', {}))

            # [Defensive Logic] 2. 建立防火墙：默认先把 dataset 设为 None
            # 无论 YAML 里写了什么 FLAC 路径，这里先全部屏蔽，防止 RKNN 读取报错
            build_config['quantization']['dataset'] = None

            # 3. 尝试运行量化校准
            is_quant_enabled = self.cfg.get('build', {}).get('quantization', {}).get('enabled', False)

            if is_quant_enabled:
                # 目前只支持 Encoder 进行流式校准
                if "encoder" in model_name.lower():
                    logger.info(f"⚖️  Running Calibration for {model_name}...")
                    try:
                        calibrator = CalibrationGenerator(self.cfg)
                        # 生成 dataset_list.txt (包含 .npy 路径)
                        # 只有这里成功返回了路径，我们才把它填回 build_config
                        new_dataset_path = calibrator.generate(final_onnx_path, self.workspace)

                        if new_dataset_path and os.path.exists(new_dataset_path):
                            build_config['quantization']['dataset'] = new_dataset_path
                            logger.info(f"   Calibration dataset ready at: {new_dataset_path}")
                        else:
                            logger.warning("   Calibration generation returned invalid path.")
                            # 回退机制：强制关闭量化
                            build_config['quantization']['enabled'] = False
                    except Exception as e:
                        logger.error(f"   Calibration generation failed: {e}")
                        logger.warning("   Falling back to FP16.")
                        build_config['quantization']['enabled'] = False
                else:
                    # Decoder/Joiner 暂不支持，强制 FP16
                    logger.info(f"ℹ️  Skipping quantization for non-audio model: {model_name} (FP16 mode)")
                    build_config['quantization']['enabled'] = False

            adapter = RKNNAdapter(
                target_platform=target_plat,
                verbose=build_config.get('verbose', False)
            )

            ret = adapter.convert(
                onnx_path=final_onnx_path,
                output_path=rknn_out_path,
                input_shapes=input_shapes,
                config_dict=build_config,
                custom_string=custom_string
            )

            if ret:
                logger.info(f"SUCCESS: Model saved to {rknn_out_path}")

                # === [新增代码在这里] ===
                # 传入当前模型配置、处理后的ONNX路径、最终RKNN路径
                self._verify_model(model_cfg, final_onnx_path, rknn_out_path, build_config)
                # ======================

                success_count += 1
            else:
                logger.error(
                    f"FAILURE: RKNN Conversion failed for {model_name}")

        logger.info(
            f"\n=== Pipeline Completed: {success_count}/{len(models)} models successful ==="
        )

    def _verify_model(self, model_cfg, onnx_path, rknn_path, build_config):
        """V1.1 Feature: 自动验证转换后的 RKNN 模型精度"""
        logger.info(f"🔎 Starting Verification for {model_cfg['name']}...")

        try:
            # 1. 初始化对比器
            target_platform = self.cfg.get("target", {}).get("platform")
            comparator = ModelComparator(target_platform)

            # --- CHANGE START ---
            # 旧代码: comparator.load_rknn(rknn_path)
            # 新代码: 传入 onnx路径, input_shapes, 和 build配置 进行影子编译
            input_shapes = model_cfg.get("input_shapes", None)
            #build_config = self.cfg.get("build", {})

            comparator.prepare_simulator(onnx_path, input_shapes, build_config)
            # --- CHANGE END ---

            # 2. 准备输入数据 (保持不变)
            sess = ort.InferenceSession(onnx_path)
            input_feed = {}
            extractor = SherpaFeatureExtractor()

            test_audio_path = self.cfg.get("build", {}).get("test_input", None)

            for i, inp in enumerate(sess.get_inputs()):
                # 1. Handle Dynamic Shape (Replace string/None with 1)
                static_shape = [
                    1 if isinstance(d, str) or d is None else d
                    for d in inp.shape
                ]

                # 2. Detect NumPy Data Type
                onnx_type = inp.type
                np_dtype = np.float32  # Default fallback
                if "int64" in onnx_type:
                    np_dtype = np.int64
                elif "int32" in onnx_type:
                    np_dtype = np.int32
                elif "bool" in onnx_type:
                    np_dtype = bool
                elif "float16" in onnx_type:
                    np_dtype = np.float16

                # 处理动态 Shape
                static_shape = [
                    1 if isinstance(d, str) or d is None else d
                    for d in inp.shape
                ]

                # 3. Generate Input Data
                # Condition: Index 0 + Configured Path + File Exists + Is Float Type
                if (i == 0 and test_audio_path
                        and os.path.exists(test_audio_path)
                        and np.issubdtype(np_dtype, np.floating)):
                    logger.info(
                        f"   Using real audio for input '{inp.name}': {test_audio_path}"
                    )
                    feats = extractor.compute(test_audio_path)

                    # Crop to target length
                    target_len = static_shape[1]
                    if feats.shape[0] > target_len:
                        feats = feats[:target_len, :]

                    input_feed[inp.name] = np.expand_dims(
                        feats, axis=0).astype(np_dtype)

                else:
                    # Fallback: Random Data based on Type
                    if np.issubdtype(np_dtype, np.integer):
                        # Generate random integers (e.g. sequence lengths)
                        input_feed[inp.name] = np.random.randint(
                            0, 10, size=static_shape).astype(np_dtype)
                    elif np_dtype == bool:
                        input_feed[inp.name] = np.random.choice(
                            [True, False], size=static_shape)
                    else:
                        # Generate random floats
                        input_feed[inp.name] = np.random.rand(
                            *static_shape).astype(np_dtype)

            # 3. 执行对比
            metrics = comparator.compare_with_onnx(onnx_path, input_feed)

            # 4. 判定结果
            if comparator.validate_metric(metrics, threshold=0.98):
                logger.info(
                    f"✅ Verification PASSED: {model_cfg['name']} matches ONNX baseline."
                )
            else:
                logger.warning(
                    f"⚠️ Verification WARNING: {model_cfg['name']} accuracy might be low."
                )

        except Exception as e:
            logger.error(f"❌ Verification Failed: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())  # 打印详细堆栈方便调试
