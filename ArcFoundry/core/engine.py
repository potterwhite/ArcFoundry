import os
import yaml
from core.utils import logger, ensure_dir
from core.preprocessor import Preprocessor
from core.rknn_adapter import RKNNAdapter
from core.downloader import ModelDownloader  # <--- 新增引用

class PipelineEngine:
    """
    Orchestrates the conversion pipeline:
    Config -> Download(Optional) -> Preprocess -> Convert -> Output
    """
    def __init__(self, config_path):
        self.config_path = config_path
        self.cfg = self._load_config(config_path)
        
        # Paths
        self.workspace = self.cfg.get('project', {}).get('workspace_dir', './workspace')
        self.output_dir = self.cfg.get('project', {}).get('output_dir', './output')
        
        ensure_dir(self.workspace)
        ensure_dir(self.output_dir)

    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def run(self):
        project_name = self.cfg.get('project', {}).get('name')
        target_plat = self.cfg.get('target', {}).get('platform')
        
        logger.info(f"=== Starting ArcFoundry Pipeline: {project_name} on {target_plat} ===")

        # Initialize Helper Modules
        downloader = ModelDownloader() # <--- 实例化下载器
        preprocessor = Preprocessor(self.cfg)
        
        models = self.cfg.get('models', [])
        success_count = 0
        
        for model_cfg in models:
            model_name = model_cfg['name']
            target_path = model_cfg['path'] # YAML里指定的目标本地路径
            model_url = model_cfg.get('url', None) # 既然是可选的，就用 get

            logger.info(f"\n>>> Processing Model: {model_name}")

            # --- Stage 0: Asset Management ---
            # 检查文件是否存在，不存在则下载，下载不了则报错
            if not downloader.ensure_model(target_path, model_url):
                logger.error(f"Skipping {model_name} due to missing input file.")
                continue

            # --- Stage 1: Preprocessing ---
            processed_onnx_name = f"{model_name}.processed.onnx"
            processed_onnx_path = os.path.join(self.workspace, processed_onnx_name)
            
            strategies = model_cfg.get('preprocess', {})
            
            final_onnx_path, custom_string = preprocessor.process(
                target_path,  # 这里已经是确认存在的路径了
                processed_onnx_path, 
                strategies
            )

            if not final_onnx_path:
                logger.error(f"Preprocessing failed for {model_name}")
                continue

            # --- Stage 2: RKNN Conversion ---
            rknn_out_path = os.path.join(self.output_dir, f"{model_name}.rknn")
            input_shapes = model_cfg.get('input_shapes', None)
            
            adapter = RKNNAdapter(
                target_platform=target_plat, 
                verbose=self.cfg.get('build', {}).get('verbose', False)
            )
            
            ret = adapter.convert(
                onnx_path=final_onnx_path,
                output_path=rknn_out_path,
                input_shapes=input_shapes,
                config_dict=self.cfg.get('build', {}),
                custom_string=custom_string
            )

            if ret:
                logger.info(f"SUCCESS: Model saved to {rknn_out_path}")
                success_count += 1
            else:
                logger.error(f"FAILURE: RKNN Conversion failed for {model_name}")

        logger.info(f"\n=== Pipeline Completed: {success_count}/{len(models)} models successful ===")
