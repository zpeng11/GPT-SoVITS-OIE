from text_preprocess.g2pw import G2PWPinyin
from pypinyin import Style, lazy_pinyin
from typing import Any, Dict, List
from runner_registry import register_callback
import onnxruntime
import numpy as np


g2pw_session = onnxruntime.InferenceSession('pretrained_models/G2PWModel/g2pW.onnx', providers=["CPUExecutionProvider"])

def g2pw_predict(inputs:Dict[str, List[List]]) -> List[List[float]]:
    for k, v in inputs.items():
        if k != "phoneme_mask":
            inputs[k] = np.array(v).astype(np.int64)
        else:
            inputs[k] = np.array(v).astype(np.float32)
    outputs = g2pw_session.run(None, inputs)
    return outputs[0].tolist()

register_callback('g2pw_predict', g2pw_predict)

roberta_session = onnxruntime.InferenceSession('pretrained_models/chinese-roberta-wwm-ext-large/chinese-roberta-wwm-ext-large.onnx', providers=["CPUExecutionProvider"])

def roberta_predict(inputs:Dict[str, List[List[int]]]) -> List[List[float]]:
    for k, v in inputs.items():
        inputs[k] = np.array(v).astype(np.int64)
    outputs = roberta_session.run(None, inputs)
    return outputs[0].tolist()

register_callback('roberta_predict', roberta_predict)

g2pw = G2PWPinyin(
    model_dir="pretrained_models/G2PWModel",
    tokenizer_source="pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json",
    enable_non_tradional_chinese=True,
    v_to_u=False,
    neutral_tone_with_five=False,
    tone_sandhi=False,
)

from text_preprocess.cleaner import clean_text, cleaned_text_to_sequence
phones, word2ph, norm_text = clean_text("你好", language='zh', version= 'v2')
phones = cleaned_text_to_sequence(phones, 'v2')

from text_preprocess.LangSegmenter import LangSegmenter

print(LangSegmenter.getTexts("你好","zh"))

from text_preprocess.text_preprocessor import TextPreprocessor
processor = TextPreprocessor("pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json")

phones, bert_features, norm_text = processor.get_phones_and_bert("卖狗?,你也喜欢まいご吗？","auto","v2")

print(len(phones), len(bert_features), len(bert_features[0]), norm_text)