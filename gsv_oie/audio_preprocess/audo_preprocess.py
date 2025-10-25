import soundfile as sf
import numpy as np
import os
from typing import Dict, List

def resample_fft(data: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    """
    使用 FFT 进行音频重采样的函数。
    
    参数:
    - data: np.ndarray, 音频数据。形状为 (samples,) 对于单声道，或 (samples, channels) 对于多声道。
    - original_sr: int, 原采样率 (Hz)。
    - target_sr: int, 目标采样率 (Hz)。
    
    返回:
    - np.ndarray, 重采样后的音频数据，形状与输入相同。
    
    注意:
    - 这是一个基本实现：上采样通过零填充频谱，下采样通过截断高频（简单低通）。
    - 对于高质量下采样，建议添加更复杂的滤波器。
    - 假设数据是 float 类型，范围 [-1, 1] 或类似。
    """
    if original_sr == target_sr:
        return data.copy()  # 无需重采样
    
    ratio = target_sr / original_sr
    num_samples = int(len(data) * ratio)
    
    # 确保数据是 float
    data = data.astype(np.float64)
    
    def _resample_channel(channel: np.ndarray) -> np.ndarray:
        """处理单通道的重采样"""
        # 计算 FFT (实数 FFT，使用 rfft 以节省空间)
        fft_len = len(channel)
        fft_data = np.fft.rfft(channel)
        
        # 新 FFT 长度
        new_fft_len = num_samples // 2 + 1
        
        # 新频谱
        new_fft = np.zeros(new_fft_len, dtype=np.complex128)
        
        if ratio > 1.0:  # 上采样：零填充
            # 复制原有频谱，剩余零填充
            copy_len = min(len(fft_data), new_fft_len)
            new_fft[:copy_len] = fft_data[:copy_len]
        else:  # 下采样：截断高频（简单低通，避免混叠）
            # 截取低频部分
            cut_len = min(len(fft_data), new_fft_len)
            new_fft[:cut_len] = fft_data[:cut_len]
        
        # IFFT 回波形
        resampled = np.fft.irfft(new_fft, n=num_samples)
        
        # 归一化（防止幅度变化）
        if np.max(np.abs(resampled)) > 0:
            resampled /= np.max(np.abs(resampled))
        
        return resampled
    
    if data.ndim == 1:  # 单声道
        return _resample_channel(data)
    else:  # 多声道：逐通道处理
        channels = data.shape[1]
        resampled_channels = np.zeros((num_samples, channels), dtype=np.float64)
        for ch in range(channels):
            resampled_channels[:, ch] = _resample_channel(data[:, ch])
        return resampled_channels
    
def load_audio(file_path: str) -> np.ndarray:
    audio, sample_rate = sf.read(file_path)
    if audio.ndim > 1:
        audio = audio[:, 0]  # 取单声道简化
    if sample_rate != 32000:
        audio = resample_fft(audio, sample_rate, 32000)
    return audio

def save_audio(file_path: str, audio: np.ndarray, sample_rate: int = 32000):
    sf.write(file_path, audio, sample_rate)

from gsv_oie.gsv_runtime import MNNInferenceEngineInterpreter
from gsv_oie.runner_registry import get_models_dir
def audio_preprocess_predict(inputs:Dict[str, np.ndarray]) -> List[np.ndarray]:
    model_path:str = os.path.join(get_models_dir(),"audio-preprocess","audio-preprocess.mnn")
    output_names = ['hubert_ssl_output', 'spectrum', 'sv_emb']
    if not hasattr(audio_preprocess_predict, 'mnn_engine'):
        audio_preprocess_predict.mnn_engine = MNNInferenceEngineInterpreter(model_path)
        print("Initialized audio preprocess MNN engine.")
    for name in inputs:
        value = inputs[name]
        if value.dtype == np.int64:
            value = value.astype(np.int32)
    mnn_outputs = audio_preprocess_predict.mnn_engine.infer(inputs)
    output = [mnn_outputs[name] for name in output_names]
    return output

class AudioPreprocessor:
    def __init__(self):
        pass
    def __call__(self, audio: np.ndarray | str, sample_rate: int = 32000) -> np.ndarray:
        if isinstance(audio, str):
            if os.path.exists(audio):
                return self.preprocess(audio)
            else:
                raise ValueError(f"Audio file '{audio}' does not exist.")
        return self.preprocess(audio, sample_rate)

    def preprocess(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if audio.ndim > 1:
            audio = audio[:, 0]  # 取单声道简化
        if sample_rate != 32000:
            audio = resample_fft(audio, sample_rate, 32000)
        return audio_preprocess_predict({"audio32k": np.expand_dims(audio, axis=0)})
    def preprocess(self, file_path: str) -> np.ndarray:
        audio, sample_rate = sf.read(file_path)
        if audio.ndim > 1:
            audio = audio[:, 0]  # 取单声道简化
        if sample_rate != 32000:
            audio = resample_fft(audio, sample_rate, 32000)
        return audio_preprocess_predict({"audio32k": np.expand_dims(audio, axis=0)})

if __name__ == "__main__":
    audio_preprocessor = AudioPreprocessor()
    outputs = audio_preprocessor("/home/eleven/GPT-SoVITS/playground/(A)あなたと空を見上げるのは、いつも夏でしたわね.wav")
    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}")