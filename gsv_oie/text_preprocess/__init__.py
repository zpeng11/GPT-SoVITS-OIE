from typing import Any, Dict, List
from ..runner_registry import  set_g2pw_predict, set_roberta_predict
import numpy as np

def g2pw_predict(model_path:str, inputs:Dict[str, np.ndarray]) -> List[np.ndarray]:
    import MNN
    import MNN.numpy as mnp
    input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'phoneme_mask', 'char_ids', 'position_ids']
    output_names = ['probs']
    # Initialize on first call
    if not hasattr(g2pw_predict, 'mnn_model'):
        g2pw_predict.mnn_model = MNN.nn.load_module_from_file(model_path, input_names, output_names)
        print("Initialized g2pw MNN model.")

    mnn_inputs = []
    for name in input_names:
        value = inputs[name]
        if value.dtype == np.int64:
            value = value.astype(np.int32)
        mnn_inputs.append(mnp.array(value))
    mnn_outputs = g2pw_predict.mnn_model(mnn_inputs)

    return [mnn_output.read() for mnn_output in mnn_outputs]

set_g2pw_predict(g2pw_predict)

def roberta_predict(model_path:str, inputs:Dict[str, np.ndarray]) -> List[np.ndarray]:
    import MNN
    import MNN.numpy as mnp
    input_names = ['input_ids']
    output_names = ['logits']
    if not hasattr(roberta_predict, 'mnn_model'):
        roberta_predict.mnn_model = MNN.nn.load_module_from_file(model_path, input_names, output_names)
        print("Initialized roberta MNN model.")
    mnn_inputs = []
    for name in input_names:
        value = inputs[name]
        if value.dtype == np.int64:
            value = value.astype(np.int32)
        mnn_inputs.append(mnp.array(value))
    mnn_outputs = roberta_predict.mnn_model(mnn_inputs)

    return [mnn_output.read() for mnn_output in mnn_outputs]

set_roberta_predict(roberta_predict)