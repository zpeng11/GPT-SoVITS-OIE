"""
GSV Runtime Python Wrapper

This module provides a Python interface for the GSV (Speech Synthesis) engine
implemented in C++ using ONNX Runtime and MNN.
"""

import os
import sys
from typing import Set, Optional, Union, List, Dict, Any
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json
import bsdiff4
from tqdm import tqdm
from ctypes import CDLL

CDLL(os.path.join(os.path.dirname(__file__), '..','..','extern','onnxruntime','lib','libonnxruntime.so'), mode=2)

# Try to import the compiled C++ module
from .gsv_engine import GSVEngine
from .gsv_engine import MNNInferenceEngineInterpreter
from .reference import ReferenceSet

class GSVRuntime:
    """
    Python wrapper for the GSV (Speech Synthesis) engine.

    This class provides a high-level interface for interacting with the GSV engine
    for speech synthesis tasks using ONNX Runtime and MNN backends.
    """

    def __init__(self, gsv_file: str, use_gpu: bool = True, use_npu: bool = False):
        """
        Initialize the GSV Runtime engine.

        Args:
            gsv_file: Path to the GSV model file (.gsv)
            use_gpu: Whether to use GPU acceleration
            use_npu: Whether to use NPU acceleration
        """
        if not os.path.isfile(gsv_file):
            raise ValueError(f"GSV file '{gsv_file}' does not exist.")

        base_name = os.path.splitext(os.path.basename(gsv_file))[0]
        temp_dir = tempfile.mkdtemp(prefix=f"gsv_{base_name}_")
        shutil.unpack_archive(gsv_file, temp_dir, "zip")
        self.config: dict = json.load(open(os.path.join(temp_dir, "config.json"), "r", encoding="utf-8"))
        self.is_v2pro = self.config.get('version', 'v2proplus').lower() in ['v2pro', 'v2proplus']
        self.is_quantized = self.config.get('quantized', False)
        self.project_name = self.config.get('project_name', 'UnknownProject')

        if self.is_quantized:
            bsdiff4.file_patch(
                os.path.join(temp_dir, 't2s', 't2s_sdec_quant.onnx'),
                os.path.join(temp_dir, 't2s', 't2s_fsdec_quant.onnx'),
                os.path.join(temp_dir, 't2s', 't2s_fsdec_quant.diff4')
            )
        text_ref =[
            np.load(os.path.join(temp_dir, "reference", "ref_text_seq.npy")),
            np.load(os.path.join(temp_dir, "reference", "ref_text_bert.npy"))
        ]
        audio_ref = [
            np.load(os.path.join(temp_dir, "reference", "ref_ssl_content.npy")),
            np.load(os.path.join(temp_dir, "reference", "ref_spectrum.npy")),
        ]
        if self.is_v2pro:
            audio_ref.append(np.load(os.path.join(temp_dir, "reference", "ref_sv_emb.npy")))
        audio_file_name = os.path.basename(self.config.get('audio_file_name', 'UnknownAudio.wav'))
        ref_text = self.config.get('ref_text', 'Unknown Text')
        self.reference_set = ReferenceSet(audio_ref, text_ref, audio_file_name, ref_text)
        self.use_gpu = use_gpu
        self.use_npu = use_npu

        self.fsdec_path = os.path.join(temp_dir, 't2s', 't2s_fsdec_quant.onnx') if self.is_quantized else os.path.join(temp_dir, 't2s', 't2s_fsdec.mnn')
        self.sdec_path = os.path.join(temp_dir, 't2s', 't2s_sdec_quant.onnx') if self.is_quantized else os.path.join(temp_dir, 't2s', 't2s_sdec.onnx')
        self.sovits_path = os.path.join(temp_dir, 'sovits', 'sovits_v1v2.mnn')

        # Initialize the C++ engine
        self.engine = GSVEngine(
            self.fsdec_path,
            self.sdec_path,
            self.sovits_path,
            self.use_gpu,
            self.use_npu,
            self.is_quantized
        )

        from gsv_oie import TextPreprocessor
        self.text_preprocessor = TextPreprocessor()

    def get_reference_set(self) -> ReferenceSet:
        """
        Get the reference set used by the GSV engine.

        Returns:
            ReferenceSet: The reference audio and text data
        """
        return self.reference_set

    def set_reference_set(self, reference_set: ReferenceSet) -> None:
        """
        Set a new reference set for the GSV engine.

        Args:
            reference_set: The new ReferenceSet to use
        """
        self.reference_set = reference_set

    def set_use_gpu(self, use_gpu: bool) -> None:
        """
        Enable or disable GPU acceleration.

        Args:
            use_gpu: Whether to use GPU acceleration
        """
        if self.use_gpu != use_gpu:
            self.use_gpu = use_gpu
            self.engine = GSVEngine(
                self.fsdec_path,
                self.sdec_path,
                self.sovits_path,
                self.use_gpu,
                self.use_npu,
                self.is_quantized
            )

    def set_use_npu(self, use_npu: bool) -> None:
        """
        Enable or disable NPU acceleration.

        Args:
            use_npu: Whether to use NPU acceleration
        """
        if self.use_npu != use_npu:
            self.use_npu = use_npu
            self.engine = GSVEngine(
                self.fsdec_path,
                self.sdec_path,
                self.sovits_path,
                self.use_gpu,
                self.use_npu,
                self.is_quantized
            )

    def infer(self, text_input:str,
              language: str = 'auto',
              text_split_method: str = None,
              output_audio_interval: float = 0.0,
              top_k: int = 15,
              temperature: float = 1.0,
              repeat_penalty: float = 1.35) -> np.ndarray:
        """
        Run inference with the GSV engine.

        Args:
            text_input: The text input for the inference
            language: Language of the input text
            text_split_method: Method for splitting text
            output_audio_interval: Interval for audio output
            top_k: Top-k sampling parameter
            temperature: Temperature for sampling
            repeat_penalty: Repeat penalty for sampling

        Returns:
            numpy.ndarray: The inference result as a numpy array
        """
        sampling_params = {
            'top_k': top_k,
            'temperature': temperature,
            'repeat_penalty': repeat_penalty
        }
        result_audios = []
        processed_texts = []
        if text_split_method is None:
            processed_texts = [self.text_preprocessor(text_input, language=language)]
        else:
            processed_texts = self.text_preprocessor.preprocess(text_input, lang=language, text_split_method=text_split_method)
        for processed_text in tqdm(processed_texts):
            if isinstance(processed_text, list):
                text_input_dict = {
                    'phones': processed_text[0],
                    'bert_features': processed_text[1],
                    'norm_text': processed_text[2]
                }
            else:
                text_input_dict = processed_text
            ref_set_dict = self.reference_set.to_dict()
            for key in ref_set_dict:
                ref_set_dict[key] = np.ascontiguousarray(ref_set_dict[key])
            for key in text_input_dict:
                if isinstance(text_input_dict[key], np.ndarray):
                    text_input_dict[key] = np.ascontiguousarray(text_input_dict[key])
            result_audios.append(self.engine.infer(ref_set_dict, text_input_dict, sampling_params))
            if output_audio_interval > 0.0:
                interval_samples = int(output_audio_interval * 32000)
                result_audios.append(np.zeros((interval_samples,), dtype=np.float32))

        if len(result_audios) != 0:
            result = np.concatenate(result_audios, axis=0)
        else:
            result = np.array([], dtype=np.float32)

        return result

    def __repr__(self) -> str:
        """String representation of the GSV Runtime."""
        return f"GSVRuntime({self.project_name}), reference_set={self.reference_set.ref_text}"



# Export public API
__all__ = [
    'GSVRuntime',
    'MNNInferenceEngineInterpreter'
]