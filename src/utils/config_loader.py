import yaml
from pathlib import Path
import os

def find_project_root():
    """Find the project root"""
    current = Path(os.getcwd())
    while current != current.parent:
        if (current / "config").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root containing config/")

def convert_paths(config_dict):
    """Convert strings to Path objects"""
    for key, value in config_dict.items():
        if isinstance(value, dict):
            config_dict[key] = convert_paths(value)
        elif isinstance(value, str):
            config_dict[key] = Path(value)
    return config_dict

def load_config(config_file = "config/paths.yaml"):
    """Load  config file"""
    project_root = find_project_root()
    config_path = project_root / config_file
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    for category in ['data', 'graphs', 'output']:
        config[category] = convert_paths(config[category])
    return config
