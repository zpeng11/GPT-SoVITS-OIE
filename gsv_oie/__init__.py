#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
GPT-SoVITS-OIE Text Processing Library

A comprehensive text preprocessing library for multilingual speech synthesis.
"""

__version__ = '1.0.0'
__author__ = 'GPT-SoVITS-OIE Contributors'
__email__ = ''

# Import main classes and functions
from .text_preprocess.text_preprocessor import TextPreprocessor
from .runner_registry import set_g2pw_predict, set_roberta_predict

# Expose only the public API
__all__ = [
    'TextPreprocessor',
    'set_g2pw_predict',
    'set_roberta_predict',
]