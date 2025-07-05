#!/usr/bin/env python3
"""
Cross-platform utility functions for the 2025-07-05 Hybrid Training Experiment
Ensures compatibility between Mac and Windows systems
"""

import os
import sys
import platform
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import torch
import yaml

logger = logging.getLogger(__name__)

class PlatformManager:
    """Manages platform-specific configurations and operations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize platform manager with configuration.
        
        Args:
            config: Configuration dictionary, if None will auto-detect
        """
        self.config = config or {}
        self.platform_info = self._detect_platform()
        self.setup_environment()
    
    def _detect_platform(self) -> Dict[str, Any]:
        """Detect current platform and return platform information.
        
        Returns:
            Dictionary with platform information
        """
        system = platform.system().lower()
        
        # Map platform names
        platform_map = {
            'windows': 'windows',
            'darwin': 'mac',
            'linux': 'linux'
        }
        
        detected_platform = platform_map.get(system, 'linux')
        
        info = {
            'os': detected_platform,
            'system': system,
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'is_windows': system == 'windows',
            'is_mac': system == 'darwin',
            'is_linux': system == 'linux'
        }
        
        logger.info(f"Detected platform: {detected_platform} ({system})")
        return info
    
    def setup_environment(self):
        """Setup environment variables based on platform."""
        env_vars = self.config.get('system', {}).get('env_vars', {})
        
        for var, value in env_vars.items():
            os.environ[var] = str(value)
            logger.debug(f"Set environment variable: {var}={value}")
    
    def get_platform_config(self) -> Dict[str, Any]:
        """Get platform-specific configuration.
        
        Returns:
            Platform-specific configuration dictionary
        """
        platform_configs = self.config.get('platform', {})
        current_platform = self.platform_info['os']
        
        return platform_configs.get(current_platform, {})
    
    def get_device(self) -> str:
        """Get the best available device for the current platform.
        
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        hardware_config = self.config.get('hardware', {})
        device_config = hardware_config.get('device', 'auto')
        
        if device_config != 'auto':
            return device_config
        
        # Get platform-specific device preferences
        device_preferences = hardware_config.get('device_preferences', {})
        current_platform = self.platform_info['os']
        preferred_devices = device_preferences.get(current_platform, ['cuda', 'cpu'])
        
        # Check each preferred device in order
        for device in preferred_devices:
            if device == 'cuda' and torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"Using CUDA device: {device_name}")
                return 'cuda'
            elif device == 'mps' and torch.backends.mps.is_available():
                logger.info("Using Apple Silicon MPS device")
                return 'mps'
            elif device == 'cpu':
                logger.info("Using CPU device")
                return 'cpu'
        
        # Fallback to CPU
        logger.warning("No preferred device available, falling back to CPU")
        return 'cpu'
    
    def get_num_workers(self) -> int:
        """Get the optimal number of workers for the current platform.
        
        Returns:
            Number of workers
        """
        dataset_config = self.config.get('dataset', {})
        num_workers = dataset_config.get('num_workers')
        
        if num_workers is not None:
            return num_workers
        
        # Get platform-specific default
        platform_config = self.get_platform_config()
        default_workers = platform_config.get('default_workers')
        
        if default_workers is None:
            raise ValueError(f"No default_workers found for platform '{self.platform_info['os']}' in config. "
                           f"Available platform configs: {list(self.config.get('platform', {}).keys())}")
        
        # Adjust based on CPU count
        cpu_count = os.cpu_count() or 4
        optimal_workers = min(default_workers, cpu_count)
        
        logger.info(f"Using {optimal_workers} workers for data loading")
        return optimal_workers
    
    def normalize_path(self, path: Union[str, Path]) -> Path:
        """Normalize path to be platform-appropriate.
        
        Args:
            path: Path to normalize
            
        Returns:
            Normalized Path object
        """
        if isinstance(path, str):
            # Convert forward slashes to platform-appropriate separators
            path = Path(path)
        
        # Resolve to absolute path if relative
        if not path.is_absolute():
            path = Path.cwd() / path
        
        return path.resolve()
    
    def create_directory(self, path: Union[str, Path], parents: bool = True, exist_ok: bool = True) -> Path:
        """Create directory with platform-appropriate permissions.
        
        Args:
            path: Directory path to create
            parents: Create parent directories if they don't exist
            exist_ok: Don't raise error if directory already exists
            
        Returns:
            Created directory Path object
        """
        path = self.normalize_path(path)
        
        try:
            path.mkdir(parents=parents, exist_ok=exist_ok)
            
            # Set permissions on Unix-like systems
            if not self.platform_info['is_windows']:
                permissions = self.config.get('system', {}).get('file_permissions', 0o755)
                path.chmod(permissions)
                
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            raise
        
        return path
    
    def get_file_extensions(self, file_type: str = 'image') -> List[str]:
        """Get file extensions for the specified file type.
        
        Args:
            file_type: Type of file ('image' or 'annotation')
            
        Returns:
            List of file extensions
        """
        dataset_config = self.config.get('dataset', {})
        
        if file_type == 'image':
            return dataset_config.get('image_extensions', ['.png', '.jpg', '.jpeg'])
        elif file_type == 'annotation':
            return dataset_config.get('annotation_extensions', ['.json'])
        else:
            raise ValueError(f"Unknown file type: {file_type}")
    
    def find_files(self, directory: Union[str, Path], file_type: str = 'image', 
                   recursive: bool = False) -> List[Path]:
        """Find files of specified type in directory.
        
        Args:
            directory: Directory to search
            file_type: Type of files to find ('image' or 'annotation')
            recursive: Search recursively in subdirectories
            
        Returns:
            List of found file paths
        """
        directory = self.normalize_path(directory)
        extensions = self.get_file_extensions(file_type)
        
        files = []
        search_pattern = "**/*" if recursive else "*"
        
        for ext in extensions:
            files.extend(directory.glob(f"{search_pattern}{ext}"))
        
        return sorted(files)
    
    def get_line_ending(self) -> str:
        """Get platform-appropriate line ending.
        
        Returns:
            Line ending string
        """
        platform_config = self.get_platform_config()
        return platform_config.get('line_ending', '\n')
    
    def get_data_path(self, data_type: str) -> Path:
        """Get path to data directory for the specified type.
        
        Args:
            data_type: Type of data directory ('splits', 'masks', 'augmented')
            
        Returns:
            Path to the data directory
        """
        # Base data directory (relative to experiment root)
        base_data_dir = Path("data")
        
        # Create full path for the data type
        data_path = base_data_dir / data_type
        
        # Normalize and ensure directory exists
        normalized_path = self.normalize_path(data_path)
        self.create_directory(normalized_path, parents=True, exist_ok=True)
        
        return normalized_path
    
    def save_text_file(self, content: str, path: Union[str, Path], encoding: str = 'utf-8'):
        """Save text file with platform-appropriate line endings.
        
        Args:
            content: Content to save
            path: File path
            encoding: File encoding
        """
        path = self.normalize_path(path)
        line_ending = self.get_line_ending()
        
        # Normalize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        if line_ending != '\n':
            content = content.replace('\n', line_ending)
        
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration file with platform-appropriate handling.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = self.normalize_path(config_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Update paths to be platform-appropriate
        self._update_config_paths(config)
        
        return config
    
    def _update_config_paths(self, config: Dict[str, Any]):
        """Update paths in configuration to be platform-appropriate.
        
        Args:
            config: Configuration dictionary to update
        """
        # Update dataset paths
        if 'dataset' in config and 'coryell_path' in config['dataset']:
            config['dataset']['coryell_path'] = str(self.normalize_path(config['dataset']['coryell_path']))
        
        # Update other path configurations
        path_keys = ['output_dir', 'log_dir', 'checkpoint_dir', 'model_checkpoint']
        
        def update_paths_recursive(obj, parent_key=''):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in path_keys or key.endswith('_path') or key.endswith('_dir'):
                        if isinstance(value, str):
                            obj[key] = str(self.normalize_path(value))
                    else:
                        update_paths_recursive(value, key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    update_paths_recursive(item, f"{parent_key}[{i}]")
        
        update_paths_recursive(config)


def get_platform_manager(config_path: Optional[str] = None) -> PlatformManager:
    """Get platform manager instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        PlatformManager instance
    """
    config = {}
    
    if config_path:
        manager = PlatformManager()
        config = manager.load_config(config_path)
    
    return PlatformManager(config)


def setup_cross_platform_logging(config: Dict[str, Any]):
    """Setup cross-platform logging configuration.
    
    Args:
        config: Configuration dictionary
    """
    logging_config = config.get('logging', {})
    
    # Get platform-appropriate log format
    log_format = logging_config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    date_format = logging_config.get('date_format', '%Y-%m-%d %H:%M:%S')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('experiment.log', encoding='utf-8')
        ]
    )


# Convenience functions for common operations
def get_device(config: Dict[str, Any] = None) -> str:
    """Get the best available device for current platform."""
    manager = PlatformManager(config)
    return manager.get_device()


def get_num_workers(config: Dict[str, Any] = None) -> int:
    """Get optimal number of workers for current platform."""
    manager = PlatformManager(config)
    return manager.get_num_workers()


def normalize_path(path: Union[str, Path]) -> Path:
    """Normalize path to be platform-appropriate."""
    manager = PlatformManager()
    return manager.normalize_path(path)


def create_directory(path: Union[str, Path], parents: bool = True, exist_ok: bool = True) -> Path:
    """Create directory with platform-appropriate permissions."""
    manager = PlatformManager()
    return manager.create_directory(path, parents, exist_ok)


if __name__ == "__main__":
    # Test the platform manager
    manager = PlatformManager()
    
    print(f"Platform: {manager.platform_info['os']}")
    print(f"Device: {manager.get_device()}")
    print(f"Workers: {manager.get_num_workers()}")
    print(f"Line ending: repr({manager.get_line_ending()})")
    
    # Test path normalization
    test_path = "../../data/test"
    normalized = manager.normalize_path(test_path)
    print(f"Normalized path: {normalized}") 