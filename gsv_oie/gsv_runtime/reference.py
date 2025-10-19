import numpy as np
from typing import List
import os

class ReferenceSet:
    def __init__(self, *args):
        """
        Initialize ReferenceSet with one of three patterns:
        1. ReferenceSet(audio_references, text_references, audio_file_name, text_normalized)
        2. ReferenceSet(audio_path, text)
        3. ReferenceSet(audio_array, sample_rate, audio_file_name, text)
        """

        # Pattern 1: Pre-processed data
        if (len(args) == 4 and
            isinstance(args[0], list) and
            isinstance(args[1], list) and
            isinstance(args[2], str) and
            isinstance(args[3], str)):
            self.audio_references = args[0]
            self.text_references = args[1]
            self.audio_file_name = args[2]
            self.text_normalized = args[3]
            return

        # Pattern 2: Audio file path and text
        elif (len(args) == 2 and
              isinstance(args[0], str) and
              isinstance(args[1], str)):
            from gsv_oie import AudioPreprocessor
            from gsv_oie import TextPreprocessor
            audio_preprocessor = AudioPreprocessor()
            text_preprocessor = TextPreprocessor()
            audio_path, text = args
            if not os.path.exists(audio_path):
                raise ValueError(f"Audio file '{audio_path}' does not exist.")
            self.audio_references = audio_preprocessor(audio_path)
            text_outputs = text_preprocessor(text)
            self.text_references = text_outputs[:-1]
            self.audio_file_name = os.path.basename(audio_path)
            self.text_normalized = text_outputs[-1]
            return

        # Pattern 3: Audio array, sample rate, file name and text
        elif (len(args) == 4 and
              isinstance(args[0], np.ndarray) and
              isinstance(args[1], int) and
              isinstance(args[2], str) and
              isinstance(args[3], str)):
            from gsv_oie import AudioPreprocessor
            from gsv_oie import TextPreprocessor
            audio_preprocessor = AudioPreprocessor()
            text_preprocessor = TextPreprocessor()
            audio_array, sample_rate, audio_file_name, text = args
            self.audio_references = audio_preprocessor.preprocess(audio_array, sample_rate)
            text_outputs = text_preprocessor(text)
            self.text_references = text_outputs[:-1]
            self.audio_file_name = os.path.basename(audio_file_name)
            self.text_normalized = text_outputs[-1]
            return

        else:
            raise ValueError("Invalid arguments. Expected one of:\n"
                           "1. ReferenceSet(audio_references, text_references, audio_file_name, text_normalized)\n"
                           "2. ReferenceSet(audio_path, text)\n"
                           "3. ReferenceSet(audio_array, sample_rate, audio_file_name, text)")