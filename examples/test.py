from gsv_oie import TextPreprocessor
import numpy as np
import wave
from gsv_oie import GSVRuntime

def audio_postprocess(
    audios,
    output_path: str,
    fragment_interval: float = 0.3,
):
    zero_wav = np.zeros((int(32000 * fragment_interval),)).astype(np.float32)
    for i, audio in enumerate(audios):
        max_audio = np.abs(audio).max()  # 简单防止16bit爆音
        if max_audio > 1:
            audio /= max_audio
        audio = audio.astype(np.float32)
        audio = np.concatenate([audio, zero_wav], axis=0)
        audios[i] = audio

    audio = np.concatenate(audios, axis=0)
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(32000)
        wav_file.writeframes(audio.tobytes())

text_preprocessor = TextPreprocessor()
result = text_preprocessor.get_phones_and_bert('인공지능: 인간 지혜의 찬란한 빛', 'ko','v2')
print(result)
result = text_preprocessor.get_phones_and_bert('人工智能：人类智慧的璀璨之光', 'zh','v2')
print(result)
result = text_preprocessor.get_phones_and_bert('人工知能：人類の知恵の輝かしい光', 'ja','v2')
print(result)
result = text_preprocessor.get_phones_and_bert('Artificial Intelligence: The Brilliant Light of Human Wisdom', 'en','v2')
print(result)

# exit()


gsv_runtime = GSVRuntime('/home/eleven/GPT-SoVITS-export/onnx/v2.gsv')

print(gsv_runtime.config)
print(gsv_runtime.get_reference_set())

result = gsv_runtime.infer("""
やがて来る世界を見渡せば、必ず赤い旗の世界となるだろう。人工知能：人類の知恵の輝かしい光
""", language='ja', text_split_method='cut5')

print(result.min(), result.max(), result.mean())
audio_postprocess([result], 'gsv_output.wav')

