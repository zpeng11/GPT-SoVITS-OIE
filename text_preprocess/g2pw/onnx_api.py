# This code is modified from https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw
# This code is modified from https://github.com/GitYCC/g2pW

import json
import os
import warnings

from typing import Any, Dict, List, Tuple
from opencc import OpenCC
from pypinyin import Style, pinyin
from tokenizers import Tokenizer

from ..zh_normalization.char_convert import tranditional_to_simplified
from .dataset import get_char_phoneme_labels, get_phoneme_labels, prepare_onnx_input
from .utils import load_config


warnings.filterwarnings("ignore")

model_version = "1.1"

from runner_registry import get_callback

from typing import List
from typing import Tuple




class G2PWOnnxConverter:
    def __init__(
        self,
        model_dir: str = "pretrained_models/G2PWModel/",
        tokenizer_source: str = "pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json",
        style: str = "bopomofo",
        enable_non_tradional_chinese: bool = False,
    ):
        self.config = load_config(config_path=os.path.join(model_dir, "config.py"), use_default=True)

        self.enable_opencc = enable_non_tradional_chinese

        self.tokenizer = Tokenizer.from_file(tokenizer_source)

        polyphonic_chars_path = os.path.join(model_dir, "POLYPHONIC_CHARS.txt")
        monophonic_chars_path = os.path.join(model_dir, "MONOPHONIC_CHARS.txt")
        self.polyphonic_chars = [
            line.split("\t") for line in open(polyphonic_chars_path, encoding="utf-8").read().strip().split("\n")
        ]
        self.non_polyphonic = {
            "一",
            "不",
            "和",
            "咋",
            "嗲",
            "剖",
            "差",
            "攢",
            "倒",
            "難",
            "奔",
            "勁",
            "拗",
            "肖",
            "瘙",
            "誒",
            "泊",
            "听",
            "噢",
        }
        self.non_monophonic = {"似", "攢"}
        self.monophonic_chars = [
            line.split("\t") for line in open(monophonic_chars_path, encoding="utf-8").read().strip().split("\n")
        ]
        self.labels, self.char2phonemes = (
            get_char_phoneme_labels(polyphonic_chars=self.polyphonic_chars)
            if self.config.use_char_phoneme
            else get_phoneme_labels(polyphonic_chars=self.polyphonic_chars)
        )

        self.chars = sorted(list(self.char2phonemes.keys()))

        self.polyphonic_chars_new = set(self.chars)
        for char in self.non_polyphonic:
            if char in self.polyphonic_chars_new:
                self.polyphonic_chars_new.remove(char)

        self.monophonic_chars_dict = {char: phoneme for char, phoneme in self.monophonic_chars}
        for char in self.non_monophonic:
            if char in self.monophonic_chars_dict:
                self.monophonic_chars_dict.pop(char)

        self.pos_tags = ["UNK", "A", "C", "D", "I", "N", "P", "T", "V", "DE", "SHI"]

        with open(os.path.join(model_dir, "bopomofo_to_pinyin_wo_tune_dict.json"), "r", encoding="utf-8") as fr:
            self.bopomofo_convert_dict = json.load(fr)
        self.style_convert_func = {
            "bopomofo": lambda x: x,
            "pinyin": self._convert_bopomofo_to_pinyin,
        }[style]

        with open(os.path.join(model_dir, "char_bopomofo_dict.json"), "r", encoding="utf-8") as fr:
            self.char_bopomofo_dict = json.load(fr)

        if self.enable_opencc:
            self.cc = OpenCC("s2tw")

    def _convert_bopomofo_to_pinyin(self, bopomofo: str) -> str:
        tone = bopomofo[-1]
        assert tone in "12345"
        component = self.bopomofo_convert_dict.get(bopomofo[:-1])
        if component:
            return component + tone
        else:
            print(f'Warning: "{bopomofo}" cannot convert to pinyin')
            return None

    def __call__(self, sentences: List[str]) -> List[List[str]]:
        if isinstance(sentences, str):
            sentences = [sentences]

        if self.enable_opencc:
            translated_sentences = []
            for sent in sentences:
                translated_sent = self.cc.convert(sent)
                assert len(translated_sent) == len(sent)
                translated_sentences.append(translated_sent)
            sentences = translated_sentences

        texts, query_ids, sent_ids, partial_results = self._prepare_data(sentences=sentences)
        if len(texts) == 0:
            # sentences no polyphonic words
            return partial_results

        onnx_input = prepare_onnx_input(
            tokenizer=self.tokenizer,
            labels=self.labels,
            char2phonemes=self.char2phonemes,
            chars=self.chars,
            texts=texts,
            query_ids=query_ids,
            use_mask=self.config.use_mask,
            window_size=None,
        )

        all_preds = []
        g2pw_predict_func = get_callback('g2pw_predict')  # 获取注册的回调函数
        probs = g2pw_predict_func(onnx_input)  # get probability distribution
        preds = [prob.index(max(prob)) for prob in probs]  # prediction index
        all_preds += [self.labels[pred] for pred in preds]  # prediction values

        if self.config.use_char_phoneme:
            all_preds = [pred.split(" ")[1] for pred in all_preds]

        results = partial_results
        for sent_id, query_id, pred in zip(sent_ids, query_ids, all_preds):
            results[sent_id][query_id] = self.style_convert_func(pred)

        return results

    def _prepare_data(self, sentences: List[str]) -> Tuple[List[str], List[int], List[int], List[List[str]]]:
        texts, query_ids, sent_ids, partial_results = [], [], [], []
        for sent_id, sent in enumerate(sentences):
            # pypinyin works well for Simplified Chinese than Traditional Chinese
            sent_s = tranditional_to_simplified(sent)
            pypinyin_result = pinyin(sent_s, neutral_tone_with_five=True, style=Style.TONE3)
            partial_result = [None] * len(sent)
            for i, char in enumerate(sent):
                if char in self.polyphonic_chars_new:
                    texts.append(sent)
                    query_ids.append(i)
                    sent_ids.append(sent_id)
                elif char in self.monophonic_chars_dict:
                    partial_result[i] = self.style_convert_func(self.monophonic_chars_dict[char])
                elif char in self.char_bopomofo_dict:
                    partial_result[i] = pypinyin_result[i][0]
                    # partial_result[i] =  self.style_convert_func(self.char_bopomofo_dict[char][0])
                else:
                    partial_result[i] = pypinyin_result[i][0]

            partial_results.append(partial_result)
        return texts, query_ids, sent_ids, partial_results
