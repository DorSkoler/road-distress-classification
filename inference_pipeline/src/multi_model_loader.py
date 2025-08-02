"""
Multi-Model Loader for Ensemble Inference
Handles loading and managing both Model B and Model H for ensemble predictions.
"""

import torch
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import yaml

from .model_loader import ModelLoader

logger = logging.getLogger(__name__)

class MultiModelLoader:
    """Loads and manages multiple models for ensemble inference."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize multi-model loader.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.models = {}
        self.model_configs = {}
        self.device = None
        
        self._load_config()
        self._setup_device()
        
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.model_configs = self.config['models']
            logger.info(f"Loaded configuration for {len(self.model_configs) - 2} models")  # -2 for ensemble and class_names
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise
    
    def _setup_device(self):
        """Setup computation device."""
        device_config = self.model_configs.get('device', 'auto')
        
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
            
        logger.info(f"Using device: {self.device}")
    
    def load_models(self, model_names: Optional[List[str]] = None) -> Dict[str, torch.nn.Module]:
        """
        Load specified models or all available models.
        
        Args:
            model_names: List of model names to load. If None, loads all models.
            
        Returns:
            Dictionary of loaded models
        """
        if model_names is None:
            # Load all models except ensemble and class_names entries
            model_names = [k for k in self.model_configs.keys() 
                          if k not in ['ensemble', 'class_names', 'device']]
        
        logger.info(f"Loading models: {model_names}")
        
        for model_name in model_names:
            if model_name in self.models:
                logger.info(f"Model {model_name} already loaded, skipping")
                continue
                
            try:
                model_config = self.model_configs[model_name]
                checkpoint_path = model_config['checkpoint_path']
                
                # Create model loader for this specific model
                model_loader = ModelLoader(
                    checkpoint_path=checkpoint_path,
                    device=self.device
                )
                
                # Load the model
                model = model_loader.load_model()
                self.models[model_name] = {
                    'model': model,
                    'loader': model_loader,
                    'config': model_config
                }
                
                logger.info(f"Successfully loaded {model_name}")
                logger.info(f"  - Architecture: {model_config['architecture']}")
                logger.info(f"  - Macro F1: {model_config.get('macro_f1', 'N/A')}")
                logger.info(f"  - Description: {model_config.get('description', 'N/A')}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
        
        return {name: info['model'] for name, info in self.models.items()}
    
    def get_model(self, model_name: str) -> torch.nn.Module:
        """Get a specific loaded model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded. Available models: {list(self.models.keys())}")
        return self.models[model_name]['model']
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded. Available models: {list(self.models.keys())}")
        return self.models[model_name]['config']
    
    def get_all_model_info(self) -> Dict[str, Dict]:
        """Get information about all loaded models."""
        return {name: info['config'] for name, info in self.models.items()}
    
    def get_ensemble_weights(self) -> List[float]:
        """Get ensemble weights from config."""
        return self.model_configs.get('ensemble', {}).get('default_weights', [0.5, 0.5])
    
    def get_class_names(self) -> List[str]:
        """Get class names."""
        return self.model_configs.get('class_names', ['damage', 'occlusion', 'crop'])
    
    def is_ensemble_enabled(self) -> bool:
        """Check if ensemble is enabled."""
        return self.model_configs.get('ensemble', {}).get('enabled', False)
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self.models.keys())
    
    def unload_model(self, model_name: str):
        """Unload a specific model to free memory."""
        if model_name in self.models:
            del self.models[model_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model {model_name}")
    
    def unload_all_models(self):
        """Unload all models to free memory."""
        self.models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded all models")