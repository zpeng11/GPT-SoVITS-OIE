# This code is modified from https://github.com/mozillazg/pypinyin-g2pW

import pickle
import os

from pypinyin.constants import RE_HANS
from pypinyin.core import Pinyin, Style
from pypinyin.seg.simpleseg import simple_seg
from pypinyin.converter import UltimateConverter
from pypinyin.contrib.tone_convert import to_tone
from text_preprocess.g2pw.onnx_api import G2PWOnnxConverter
import zipfile
import requests

current_file_path = os.path.dirname(__file__)
CACHE_PATH = os.path.join(current_file_path, "polyphonic.pickle")
PP_DICT_PATH = os.path.join(current_file_path, "polyphonic.rep")
PP_FIX_DICT_PATH = os.path.join(current_file_path, "polyphonic-fix.rep")


class G2PWPinyin(Pinyin):
    def __init__(
        self,
        model_dir="pretrained_models/G2PWModel/",
        tokenizer_source="pretrained_models/G2PWModel/tokenizer.json",
        enable_non_tradional_chinese=True,
        v_to_u=False,
        neutral_tone_with_five=False,
        tone_sandhi=False,
        **kwargs,
    ):
        uncompress_path = download_and_decompress(model_dir)
        self._g2pw = G2PWOnnxConverter(
            model_dir=uncompress_path,
            tokenizer_source=tokenizer_source,
            style="pinyin",
            enable_non_tradional_chinese=enable_non_tradional_chinese,
        )
        self._converter = Converter(
            self._g2pw,
            v_to_u=v_to_u,
            neutral_tone_with_five=neutral_tone_with_five,
            tone_sandhi=tone_sandhi,
        )

    def get_seg(self, **kwargs):
        return simple_seg


class Converter(UltimateConverter):
    def __init__(self, g2pw_instance, v_to_u=False, neutral_tone_with_five=False, tone_sandhi=False, **kwargs):
        super(Converter, self).__init__(
            v_to_u=v_to_u, neutral_tone_with_five=neutral_tone_with_five, tone_sandhi=tone_sandhi, **kwargs
        )

        self._g2pw = g2pw_instance

    def convert(self, words, style, heteronym, errors, strict, **kwargs):
        pys = []
        if RE_HANS.match(words):
            pys = self._to_pinyin(words, style=style, heteronym=heteronym, errors=errors, strict=strict)
            post_data = self.post_pinyin(words, heteronym, pys)
            if post_data is not None:
                pys = post_data

            pys = self.convert_styles(pys, words, style, heteronym, errors, strict)

        else:
            py = self.handle_nopinyin(words, style=style, errors=errors, heteronym=heteronym, strict=strict)
            if py:
                pys.extend(py)

        return _remove_dup_and_empty(pys)

    def _to_pinyin(self, han, style, heteronym, errors, strict, **kwargs):
        pinyins = []

        g2pw_pinyin = self._g2pw(han)

        if not g2pw_pinyin:  # g2pw 不支持的汉字改为使用 pypinyin 原有逻辑
            return super(Converter, self).convert(han, Style.TONE, heteronym, errors, strict, **kwargs)

        for i, item in enumerate(g2pw_pinyin[0]):
            if item is None:  # g2pw 不支持的汉字改为使用 pypinyin 原有逻辑
                py = super(Converter, self).convert(han[i], Style.TONE, heteronym, errors, strict, **kwargs)
                pinyins.extend(py)
            else:
                pinyins.append([to_tone(item)])

        return pinyins

def download_and_decompress(model_dir: str = "pretrained_models/G2PWModel/"):
    if not os.path.exists(model_dir):
        parent_directory = os.path.dirname(model_dir)
        zip_dir = os.path.join(parent_directory, "G2PWModel_1.1.zip")
        extract_dir = os.path.join(parent_directory, "G2PWModel_1.1")
        extract_dir_new = os.path.join(parent_directory, "G2PWModel")
        print("Downloading g2pw model...")
        modelscope_url = "https://www.modelscope.cn/models/kamiorinn/g2pw/resolve/master/G2PWModel_1.1.zip"  # "https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip"
        with requests.get(modelscope_url, stream=True) as r:
            r.raise_for_status()
            with open(zip_dir, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print("Extracting g2pw model...")
        with zipfile.ZipFile(zip_dir, "r") as zip_ref:
            zip_ref.extractall(parent_directory)

        os.rename(extract_dir, extract_dir_new)

        print("Downloading tokenizer.json...")
        tokenizer_json_url = 'https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/tokenizer.json'
        with requests.get(tokenizer_json_url, stream=True) as r:
            r.raise_for_status()
            with open(os.path.join(extract_dir_new, "tokenizer.json"), "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    return model_dir

def _remove_dup_items(lst, remove_empty=False):
    new_lst = []
    for item in lst:
        if remove_empty and not item:
            continue
        if item not in new_lst:
            new_lst.append(item)
    return new_lst


def _remove_dup_and_empty(lst_list):
    new_lst_list = []
    for lst in lst_list:
        lst = _remove_dup_items(lst, remove_empty=True)
        if lst:
            new_lst_list.append(lst)
        else:
            new_lst_list.append([""])

    return new_lst_list


def cache_dict(polyphonic_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(polyphonic_dict, pickle_file)


def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            polyphonic_dict = pickle.load(pickle_file)
    else:
        polyphonic_dict = read_dict()
        cache_dict(polyphonic_dict, CACHE_PATH)

    return polyphonic_dict


def read_dict():
    polyphonic_dict = {}
    with open(PP_DICT_PATH, encoding="utf-8") as f:
        line = f.readline()
        while line:
            key, value_str = line.split(":")
            value = eval(value_str.strip())
            polyphonic_dict[key.strip()] = value
            line = f.readline()
    with open(PP_FIX_DICT_PATH, encoding="utf-8") as f:
        line = f.readline()
        while line:
            key, value_str = line.split(":")
            value = eval(value_str.strip())
            polyphonic_dict[key.strip()] = value
            line = f.readline()
    return polyphonic_dict


def correct_pronunciation(word, word_pinyins):
    new_pinyins = pp_dict.get(word, "")
    if new_pinyins == "":
        for idx, w in enumerate(word):
            w_pinyin = pp_dict.get(w, "")
            if w_pinyin != "":
                word_pinyins[idx] = w_pinyin[0]
        return word_pinyins
    else:
        return new_pinyins


pp_dict = get_dict()

