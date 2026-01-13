import os
import yaml
from core.utils import logger, ensure_dir
from core.preprocessor import Preprocessor
from core.rknn_adapter import RKNNAdapter

class PipelineEngine:
    """
    Orchestrates the conversion pipeline:
    Config -> Preprocess -> Convert -> Output
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
        preprocessor = Preprocessor(self.cfg)
        
        models = self.cfg.get('models', [])
        
        success_count = 0
        
        for model_cfg in models:
            model_name = model_cfg['name']
            raw_onnx_path = model_cfg['path']
            logger.info(f"\n>>> Processing Model: {model_name}")

            if not os.path.exists(raw_onnx_path):
                logger.error(f"Input file not found: {raw_onnx_path}")
                continue

            # --- Stage 1: Preprocessing ---
            # Define intermediate path
            processed_onnx_name = f"{model_name}.processed.onnx"
            processed_onnx_path = os.path.join(self.workspace, processed_onnx_name)
            
            strategies = model_cfg.get('preprocess', {})
            
            final_onnx_path, custom_string = preprocessor.process(
                raw_onnx_path, 
                processed_onnx_path, 
                strategies
            )

            if not final_onnx_path:
                logger.error(f"Preprocessing failed for {model_name}")
                continue

            # --- Stage 2: RKNN Conversion ---
            rknn_out_path = os.path.join(self.output_dir, f"{model_name}.rknn")
            input_shapes = model_cfg.get('input_shapes', None)
            
            # Initialize Adapter (New instance per model to ensure clean state)
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
