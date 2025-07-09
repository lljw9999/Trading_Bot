"""
Configuration Manager for the Trading System

Handles loading and managing configuration from YAML files and environment variables.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Manages configuration loading and access for the trading system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the main configuration file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        current_dir = Path(__file__).parent
        return str(current_dir.parent / "config" / "base_config.yaml")
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            # Resolve environment variables
            self._resolve_env_vars(self.config)
            
            self.logger.info(f"Configuration loaded from {self.config_path}")
            
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def _resolve_env_vars(self, obj: Any) -> None:
        """Recursively resolve environment variables in configuration."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    # Parse environment variable with optional default
                    env_expr = value[2:-1]  # Remove ${ and }
                    
                    if ":" in env_expr:
                        env_var, default_value = env_expr.split(":", 1)
                    else:
                        env_var, default_value = env_expr, None
                    
                    obj[key] = os.getenv(env_var, default_value)
                    
                    # Convert to appropriate type
                    if obj[key] is not None:
                        obj[key] = self._convert_type(obj[key])
                        
                elif isinstance(value, (dict, list)):
                    self._resolve_env_vars(value)
                    
        elif isinstance(obj, list):
            for item in obj:
                self._resolve_env_vars(item)
    
    def _convert_type(self, value: str) -> Any:
        """Convert string values to appropriate types."""
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        elif value.isdigit():
            return int(value)
        elif self._is_float(value):
            return float(value)
        else:
            return value
    
    def _is_float(self, value: str) -> bool:
        """Check if string represents a float."""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to set
            value: Value to set
        """
        keys = key_path.split('.')
        current = self.config
        
        # Navigate to parent dictionary
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section dictionary
        """
        return self.config.get(section, {})
    
    def load_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        Load strategy-specific configuration.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy configuration dictionary
        """
        strategy_path = Path(self.config_path).parent / "strategies" / f"{strategy_name}.yaml"
        
        if strategy_path.exists():
            try:
                with open(strategy_path, 'r') as file:
                    strategy_config = yaml.safe_load(file)
                
                self._resolve_env_vars(strategy_config)
                return strategy_config
                
            except yaml.YAMLError as e:
                self.logger.error(f"Error loading strategy config {strategy_name}: {e}")
                return {}
        else:
            self.logger.warning(f"Strategy config file not found: {strategy_path}")
            return {}
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
        self.logger.info("Configuration reloaded")


# Global configuration instance
config = ConfigManager() 