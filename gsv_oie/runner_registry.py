# This file is used to register various callback functions for use in the application. Mostly, these callbacks are related to model inference functions.
# The registry allows for easy retrieval and management of these functions by name.

def get_models_dir() -> str:
    """获取模型存储目录"""
    import os
    from pathlib import Path
    DEFAULT_MODELS_DIR = Path(__file__).parent / "pretrained_models"
    MODELS_DIR = os.getenv("GSV_OIE_MODELS", str(DEFAULT_MODELS_DIR))
    return MODELS_DIR