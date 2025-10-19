from typing import Any, Dict, List
from ..runner_registry import  set_audio_preprocess_predict
import numpy as np

def audio_preprocess_predict(model_path:str, inputs:Dict[str, np.ndarray]) -> List[np.ndarray]:
    import MNN
    import MNN.numpy as mnp
    input_names = ['audio32k']
    output_names = ['hubert_ssl_output', 'spectrum', 'sv_emb']
    # Initialize on first call
    if not hasattr(audio_preprocess_predict, 'mnn_model'):
        audio_preprocess_predict.mnn_model = MNN.nn.load_module_from_file(model_path, input_names, output_names)
        print("Initialized audio preprocess MNN model.")

    mnn_inputs = []
    for name in input_names:
        value = inputs[name]
        if value.dtype == np.int64:
            value = value.astype(np.int32)
        mnn_inputs.append(mnp.array(value))
    mnn_outputs = audio_preprocess_predict.mnn_model(mnn_inputs)

    return [mnn_output.read() for mnn_output in mnn_outputs]

set_audio_preprocess_predict(audio_preprocess_predict)