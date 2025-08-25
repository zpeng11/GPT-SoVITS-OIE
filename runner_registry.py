from abc import ABC, abstractmethod
from typing import Dict, List


# def g2pw_predict(self, inputs: Dict[str, List[List]]) -> List[List[float]]:

# def roberta_predict(self, inputs: Dict[str, List[List[int]]]) -> List[List[float]]:

CALLBACK_REGISTRY = {}

def register_callback(name, func):
    """注册一个回调函数"""
    print(f"Registry: Registering callback '{name}'")
    CALLBACK_REGISTRY[name] = func

def get_callback(name):
    """获取一个已注册的回调函数"""
    # print(f"Registry: Getting callback '{name}'")
    if name not in CALLBACK_REGISTRY:
        raise ValueError(f"Callback '{name}' is not registered.")
    return CALLBACK_REGISTRY.get(name)