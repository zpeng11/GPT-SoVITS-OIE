# This file is used to register various callback functions for use in the application. Mostly, these callbacks are related to model inference functions.
# The registry allows for easy retrieval and management of these functions by name.
import os
from typing import Callable, List, Dict
import numpy as np

CALLBACK_REGISTRY = {}

def register_callback(name, func):
    """注册一个回调函数"""
    # print(f"Registry: Registering callback '{name}'")
    CALLBACK_REGISTRY[name] = func

def get_callback(name):
    """获取一个已注册的回调函数"""
    # print(f"Registry: Getting callback '{name}'")
    if name not in CALLBACK_REGISTRY:
        raise ValueError(f"Callback '{name}' is not registered.")
    return CALLBACK_REGISTRY.get(name)

def get_models_dir() -> str:
    """获取模型存储目录"""
    import os
    from pathlib import Path
    DEFAULT_MODELS_DIR = Path(__file__).parent / "pretrained_models"
    MODELS_DIR = os.getenv("GSV_OIE_MODELS", str(DEFAULT_MODELS_DIR))
    return MODELS_DIR

def set_g2pw_predict(func:Callable[[str, Dict[str, np.ndarray]], List[np.ndarray]]):
    G2PW_MODEL_PATH = os.path.join(get_models_dir(),"G2PWModel", "g2pW.mnn")
    def wrapper(inputs):
        return func(G2PW_MODEL_PATH, inputs)
    register_callback('g2pw_predict', wrapper)

def set_roberta_predict(func:Callable[[str, Dict[str, np.ndarray]], List[np.ndarray]]):
    ROBERTA_MODEL_PATH = os.path.join(get_models_dir(),"chinese-roberta-wwm-ext-large","chinese-roberta-wwm-ext-large.mnn")
    def wrapper(inputs):
        return func(ROBERTA_MODEL_PATH, inputs)
    register_callback('roberta_predict', wrapper)

def set_audio_preprocess_predict(func:Callable[[str, Dict[str, np.ndarray]], List[np.ndarray]]):
    AUDIO_PREPROCESS_MODEL_PATH = os.path.join(get_models_dir(),"audio-preprocess", "audio-preprocess.mnn")
    def wrapper(inputs):
        return func(AUDIO_PREPROCESS_MODEL_PATH, inputs)
    register_callback('audio_preprocess_predict', wrapper)