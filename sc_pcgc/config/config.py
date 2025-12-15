"""Configuration management for Sc-PCGC."""

import yaml
from typing import Dict, Any


class Config:
    """Configuration class for point cloud compression."""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """Initialize configuration.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        if config_dict is None:
            config_dict = self.get_default_config()
        
        self.config = config_dict
        self._set_attributes()
    
    def _set_attributes(self):
        """Set configuration attributes."""
        for key, value in self.config.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            # Model parameters
            "model": {
                "d_model": 256,
                "n_heads": 8,
                "n_layers": 6,
                "d_ff": 1024,
                "dropout": 0.1,
            },
            
            # Octree parameters
            "octree": {
                "max_depth": 10,
                "voxel_size": 0.1,
                "min_points": 1,
            },
            
            # Priority parameters
            "priority": {
                "semantic_weights": {
                    "car": 1.0,
                    "pedestrian": 1.0,
                    "cyclist": 0.9,
                    "truck": 0.8,
                    "building": 0.5,
                    "vegetation": 0.3,
                    "road": 0.4,
                    "sidewalk": 0.4,
                    "other": 0.2,
                },
            },
            
            # Compression parameters
            "compression": {
                "use_entropy_coding": True,
                "quantization_bits": 10,
            },
            
            # Training parameters
            "training": {
                "batch_size": 4,
                "learning_rate": 1e-4,
                "num_epochs": 100,
                "weight_decay": 1e-5,
            },
            
            # Data parameters
            "data": {
                "dataset": "semantickitti",
                "data_root": "./data/SemanticKITTI",
                "num_workers": 4,
            },
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        result = {}
        for key, value in self.__dict__.items():
            if key == "config":
                continue
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def save(self, filepath: str):
        """Save configuration to YAML file.
        
        Args:
            filepath: Path to save configuration
        """
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_config(filepath: str) -> Config:
    """Load configuration from YAML file.
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        Configuration object
    """
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)
