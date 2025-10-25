import os
import sys

from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(now_dir)

import re
from .LangSegmenter import LangSegmenter
from typing import Dict, List, Tuple, Any
from .cleaner import clean_text
from ..runner_registry import get_callback
from .tokenizers_cpp import TokenizerCPP
from pathlib import Path
from .text_segmentation_method import split_big_text, splits, get_method as get_seg_method
import numpy as np
punctuation = set(["!", "?", "…", ",", ".", "-"])

from . import symbols as symbols_v1
from . import symbols2 as symbols_v2

_symbol_to_id_v1 = {s: i for i, s in enumerate(symbols_v1.symbols)}
_symbol_to_id_v2 = {s: i for i, s in enumerate(symbols_v2.symbols)}


def cleaned_text_to_sequence(cleaned_text, version):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text_preprocess.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    if version == "v1":
        phones = [_symbol_to_id_v1[symbol] for symbol in cleaned_text]
    else:
        phones = [_symbol_to_id_v2[symbol] for symbol in cleaned_text]

    return phones


def get_first(text: str) -> str:
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def merge_short_text_in_array(texts: str, threshold: int) -> list:
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

class TextPreprocessor:
    def __init__(self):
        from ..runner_registry import get_models_dir
        content = Path(os.path.join(get_models_dir(),"chinese-roberta-wwm-ext-large","tokenizer.json")).read_text()
        self.tokenizer = TokenizerCPP.FromBlobJSON(content)

    def __call__(self, text: str, language: str = 'auto', version: str = "v2", text_split_method: str = None) -> List[Any]:
        if text_split_method is not None:
            return self.preprocess(text, language, text_split_method, version)
        return self.get_phones_and_bert(text, language, version)

    def preprocess(self, text: str, lang: str, text_split_method: str, version: str = "v2") -> List[Dict]:
        print(f"############ 切分文本 ############")
        text = self.replace_consecutive_punctuation(text)
        texts = self.pre_seg_text(text, lang, text_split_method)
        result = []
        print(f"############ 提取文本Bert特征 ############")
        for text in tqdm(texts):
            phones, bert_features, norm_text = self.segment_and_extract_feature_for_text(text, lang, version)
            if phones is None or norm_text == "":
                continue
            res = {
                "phones": phones,
                "bert_features": bert_features,
                "norm_text": norm_text,
            }
            result.append(res)
        return result

    def pre_seg_text(self, text: str, lang: str, text_split_method: str):
        text = text.strip("\n")
        if len(text) == 0:
            return []
        if text[0] not in splits and len(get_first(text)) < 4:
            text = "。" + text if lang != "en" else "." + text
        print("实际输入的目标文本:")
        print(text)

        seg_method = get_seg_method(text_split_method)
        text = seg_method(text)

        while "\n\n" in text:
            text = text.replace("\n\n", "\n")

        _texts = text.split("\n")
        _texts = self.filter_text(_texts)
        _texts = merge_short_text_in_array(_texts, 5)
        texts = []

        for text in _texts:
            # 解决输入目标文本的空行导致报错的问题
            if len(text.strip()) == 0:
                continue
            if not re.sub("\W+", "", text):
                # 检测一下，如果是纯符号，就跳过。
                continue
            if text[-1] not in splits:
                text += "。" if lang != "en" else "."

            # 解决句子过长导致Bert报错的问题
            if len(text) > 510:
                texts.extend(split_big_text(text))
            else:
                texts.append(text)

        print("实际输入的目标文本(切句后):")
        print(texts)
        return texts

    def segment_and_extract_feature_for_text(
        self, text: str, language: str, version: str = "v1"
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        return self.get_phones_and_bert(text, language, version)

    def get_phones_and_bert(self, text: str, language: str, version: str, final: bool = False):
        text = re.sub(r' {2,}', ' ', text)
        textlist = []
        langlist = []
        if language == "all_zh":
            for tmp in LangSegmenter.getTexts(text,"zh"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_yue":
            for tmp in LangSegmenter.getTexts(text,"zh"):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_ja":
            for tmp in LangSegmenter.getTexts(text,"ja"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "all_ko":
            for tmp in LangSegmenter.getTexts(text,"ko"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "en":
            langlist.append("en")
            textlist.append(text)
        elif language == "auto":
            for tmp in LangSegmenter.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegmenter.getTexts(text):
                if langlist:
                    if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                        textlist[-1] += tmp["text"]
                        continue
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日韩文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        # print(textlist)
        # print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang, version)
            bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = np.concatenate(bert_list, axis=0).T # shape:[N, 1024] -> [1024, N]
        phones = np.concatenate(phones_list, axis=0).astype(np.int64)
        norm_text = "".join(norm_text_list)

        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text, language, version, final=True)

        return np.expand_dims(phones, axis=0), bert.T, norm_text

    def get_bert_feature(self, text: str, word2ph: list) -> list:
        inputs = np.array([[self.tokenizer.TokenToId("[CLS]")] + self.tokenizer.Encode(text) + [self.tokenizer.TokenToId("[SEP]")]]).astype(np.int64)
        res = get_callback("roberta_predict")({"input_ids":inputs})[0]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            num_repeats = word2ph[i]
            for _ in range(num_repeats):
                phone_level_feature.append(res[i:i+1, :])

        return np.concatenate(phone_level_feature, axis=0) # shape:[N, 1024]

    def clean_text_inf(self, text: str, language: str, version: str = "v2"):
        language = language.replace("all_", "")
        phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    def get_bert_inf(self, phones: list, word2ph: list, norm_text: str, language: str):
        language = language.replace("all_", "")
        if language == "zh":
            feature = self.get_bert_feature(norm_text, word2ph)
        else:
            feature = np.zeros((len(phones), 1024), dtype=np.float32)

        return feature

    def filter_text(self, texts):
        _text = []
        if all(text in [None, " ", "\n", ""] for text in texts):
            raise ValueError("请输入有效文本")
        for text in texts:
            if text in [None, " ", ""]:
                pass
            else:
                _text.append(text)
        return _text

    def replace_consecutive_punctuation(self, text):
        punctuations = "".join(re.escape(p) for p in punctuation)
        pattern = f"([{punctuations}])([{punctuations}])+"
        result = re.sub(pattern, r"\1", text)
        return result