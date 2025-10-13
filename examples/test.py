from text_preprocess.g2pw import G2PWPinyin
from pypinyin import Style, lazy_pinyin
from typing import Any, Dict, List
from runner_registry import register_callback
import onnxruntime
import numpy as np
import MNN
import MNN.numpy as mnp

def g2pw_predict(inputs:Dict[str, np.ndarray]) -> List[np.ndarray]:
    input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'phoneme_mask', 'char_ids', 'position_ids']
    output_names = ['probs']
    # Initialize on first call
    if not hasattr(g2pw_predict, 'mnn_model'):
        g2pw_predict.mnn_model = MNN.nn.load_module_from_file('pretrained_models/G2PWModel/g2pW.mnn', input_names, output_names)
        print("Initialized g2pw MNN model.")

    mnn_inputs = []
    for name in input_names:
        value = inputs[name]
        if value.dtype == np.int64:
            value = value.astype(np.int32)
        mnn_inputs.append(mnp.array(value))
    mnn_outputs = g2pw_predict.mnn_model(mnn_inputs)

    return [mnn_output.read() for mnn_output in mnn_outputs]

register_callback('g2pw_predict', g2pw_predict)

def roberta_predict(inputs:Dict[str, np.ndarray]) -> List[np.ndarray]:
    input_names = ['input_ids']
    output_names = ['logits']
    if not hasattr(roberta_predict, 'mnn_model'):
        roberta_predict.mnn_model = MNN.nn.load_module_from_file('pretrained_models/chinese-roberta-wwm-ext-large/chinese-roberta-wwm-ext-large.mnn',
                                             input_names, output_names)
        print("Initialized roberta MNN model.")
    mnn_inputs = []
    for name in input_names:
        value = inputs[name]
        if value.dtype == np.int64:
            value = value.astype(np.int32)
        mnn_inputs.append(mnp.array(value))
    mnn_outputs = roberta_predict.mnn_model(mnn_inputs)

    return [mnn_output.read() for mnn_output in mnn_outputs]

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

phones, bert_features, norm_text = processor.get_phones_and_bert("量子纠缠现象在超低温环境下表现出非局域性关联特征。","auto","v2")

print(phones.shape, bert_features.shape, norm_text)