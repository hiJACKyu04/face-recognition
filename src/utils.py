"""
Utility functions for the face recognition system
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        # Try example config
        example_config = Path("config/config.example.yaml")
        if example_config.exists():
            config_file = example_config
            logging.warning(f"Config file not found, using example config: {config_file}")
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables if present
    config = _override_with_env(config)
    
    return config


def _override_with_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """Override config values with environment variables"""
    # API settings
    if os.getenv("API_HOST"):
        config.setdefault("api", {})["host"] = os.getenv("API_HOST")
    if os.getenv("API_PORT"):
        config.setdefault("api", {})["port"] = int(os.getenv("API_PORT"))
    if os.getenv("API_KEY"):
        config.setdefault("api", {}).setdefault("auth", {})["api_key"] = os.getenv("API_KEY")
    
    # Database settings
    if os.getenv("DATABASE_TYPE"):
        config.setdefault("database", {})["type"] = os.getenv("DATABASE_TYPE")
    if os.getenv("DATABASE_PATH"):
        config.setdefault("database", {})["path"] = os.getenv("DATABASE_PATH")
    
    # Model settings
    if os.getenv("MODEL_NAME"):
        config.setdefault("model", {})["name"] = os.getenv("MODEL_NAME")
    
    return config


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if it doesn't"""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_model_path(model_name: str = "buffalo_l") -> Path:
    """Get path to InsightFace model directory"""
    home = Path.home()
    model_path = home / ".insightface" / "models" / model_name
    return model_path


def validate_image_file(file_path: str) -> bool:
    """Validate if file is a valid image"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_ext = Path(file_path).suffix.lower()
    return file_ext in valid_extensions


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    return Path(file_path).stat().st_size / (1024 * 1024)

