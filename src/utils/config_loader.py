import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Utility class to load YAML configuration files for SEPA agents.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the ConfigLoader.
        
        Args:
            config_dir (str): Relative path to the configuration directory.
        """
        # Resolve absolute path relative to project root (assuming this script is in src/utils)
        # Getting to project root from src/utils: ../../
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.config_path = self.project_root / config_dir
        
        if not self.config_path.exists():
            logger.warning(f"Configuration directory not found at {self.config_path}")

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a specific configuration file.
        
        Args:
            config_name (str): Name of the config file (e.g., 'trend_template_criteria.yaml')
            
        Returns:
            Dict[str, Any]: The configuration dictionary.
        """
        file_path = self.config_path / config_name
        
        if not file_path.exists():
            logger.error(f"Config file {config_name} not found at {file_path}")
            return {}
            
        try:
            with open(file_path, 'r') as file:
                config = yaml.safe_load(file)
                logger.info(f"Successfully loaded configuration: {config_name}")
                return config or {}
        except Exception as e:
            logger.error(f"Error loading config {config_name}: {e}")
            return {}

    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all YAML files in the config directory.
        
        Returns:
            Dict[str, Dict]: Dictionary of all configurations, keyed by filename.
        """
        configs = {}
        if not self.config_path.exists():
            return configs

        for file_path in self.config_path.glob("*.yaml"):
            config_name = file_path.name
            configs[config_name] = self.load_config(config_name)
            
        return configs

# Example usage
if __name__ == "__main__":
    loader = ConfigLoader()
    trend_config = loader.load_config("trend_template_criteria.yaml")
    print(f"Trend Config: {trend_config}")
