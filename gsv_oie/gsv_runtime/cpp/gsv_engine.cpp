#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <filesystem>
#include <vector>
#include <iostream>
#include <cstdint>

#include "utils.hpp"
#include "MNNInferenceEngine.h"
#include "MNNInferenceEngineInterpreter.hpp"

namespace py = pybind11;
constexpr int PREFILL_THREAD_NUM = 0; // 0 means using default thread number
constexpr int STEP_DECODE_THREAD_NUM = 0; // 1 is usually enough for step decode
constexpr int HEAD_NUM = 24;
constexpr int KV_CACHE_PREPARED_LENGTH = 512;
constexpr int MNN_CPU_NUM_THREAD = 8;
#define CPP_PRINT(msg) py::print("[C++] " + std::string(msg))
using OrtValueShapeType = std::vector<int64_t>;


class GSVEngine : NonCopyable {
    public:
    GSVEngine(const std::string fsdec_path,
              const std::string sdec_path,
              const std::string sovits_path,
              bool use_gpu = false,
              bool use_npu = false,
              bool quantized = false){
        CPP_PRINT("GSVEngine initialized with directories:");
        CPP_PRINT("  FSDec path: " + fsdec_path);
        CPP_PRINT("  SDec path: " + sdec_path);
        CPP_PRINT("  SoVITS path: " + sovits_path);
        // fsdec_engine_ = std::make_shared<MNNInferenceEngine>(fsdec_path, {});
        use_gpu_ = use_gpu;
        use_npu_ = use_npu;
        quantized_ = quantized;
        fsdec_path_ = fsdec_path;
        sdec_path_ = sdec_path;
        sovits_path_ = sovits_path;
    }
    ~GSVEngine(){
        CPP_PRINT("GSVEngine destroyed");
    }

    py::array_t<float> infer(const py::dict& ref_dict, const py::dict& text_input, const py::dict& sampling_params) {
        check_dict_key(text_input, "norm_text");
        auto norm_text = text_input["norm_text"].cast<std::string>();
        CPP_PRINT("Received norm_text: " + norm_text);

        check_dict_key(ref_dict, "phones");
        py::array encoder_ref_seq;
        if(!quantized_){
            encoder_ref_seq = ref_dict["phones"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
        }
        else{
            encoder_ref_seq = ref_dict["phones"].cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
        }
        check_dict_key(ref_dict, "bert_features");
        py::array encoder_ref_bert = ref_dict["bert_features"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        check_dict_key(ref_dict, "hubert_ssl_output");
        py::array encoder_ssl_content = ref_dict["hubert_ssl_output"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();

        check_dict_key(text_input, "phones");
        py::array encoder_text_seq;
        if(!quantized_){
            encoder_text_seq = text_input["phones"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
        }
        else{
            encoder_text_seq = text_input["phones"].cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
        }
        check_dict_key(text_input, "bert_features");
        py::array encoder_text_bert = text_input["bert_features"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();

        check_dict_key(sampling_params, "top_k");
        int top_k = sampling_params["top_k"].cast<int>();
        check_dict_key(sampling_params, "temperature");
        float temperature = sampling_params["temperature"].cast<float>();
        check_dict_key(sampling_params, "repeat_penalty");
        float repeat_penalty = sampling_params["repeat_penalty"].cast<float>();

        if(!fsdec_engine_){
            fsdec_engine_ = std::make_shared<MNNInferenceEngineInterpreter>(fsdec_path_);
        }
        fsdec_infer(encoder_ref_seq, encoder_ref_bert, encoder_ssl_content, encoder_text_seq, encoder_text_bert);
        // Dummy output for illustration
        std::vector<float> output_data = {0.0f, 1.0f, 2.0f};
        return py::array_t<float>(output_data.size(), output_data.data());
    }

    void fsdec_infer(const py::array&encoder_ref_seq, 
                     const py::array&encoder_ref_bert, 
                     const py::array&encoder_ssl_content,
                     const py::array&encoder_text_seq,
                     const py::array&encoder_text_bert) {
        // Implementation of fsdec_infer would go here
        auto input_map = std::map<std::string, const py::array *>{
            {"encoder_ref_seq", &encoder_ref_seq},
            {"encoder_ref_bert", &encoder_ref_bert},
            {"encoder_ssl_content", &encoder_ssl_content},
            {"encoder_text_seq", &encoder_text_seq},
            {"encoder_text_bert", &encoder_text_bert}
        };
        auto output_map = fsdec_engine_->infer_tensor(input_map);
        // for(auto item : output_map){
        //     const auto name = item.first;
        //     auto tensor_guard = std::move(item.second);
        //     CPP_PRINT("FSDec Output Tensor Name: " + name);
        //     CPP_PRINT("FSDec Output Tensor Shape: " + shape_vector_to_string(tensor_guard->tensor()->shape()));
        // }
    }

    private:
    std::shared_ptr<MNNInferenceEngineInterpreter> fsdec_engine_;
    bool use_gpu_ = false;
    bool use_npu_ = false;
    bool quantized_ = false;
    std::string fsdec_path_;
    std::string sdec_path_;
    std::string sovits_path_;
};




PYBIND11_MODULE(gsv_engine, m) {
// 绑定类
    py::class_<MNNInferenceEngine>(m, "MNNInferenceEngine")
        .def(py::init<const std::string&, const std::vector<std::string>&, const std::vector<std::string>&>(),
             py::arg("model_path"),  // 构造函数参数命名（可选，但推荐用于 Python 侧文档）
             py::arg("input_list"),
             py::arg("output_list"),
             "初始化 MNN 推理引擎")

        .def("infer", &MNNInferenceEngine::infer,
             py::arg("input_arrays"),
             "执行推理：输入 NumPy 数组列表，返回输出 NumPy 数组列表");
    py::class_<MNNInferenceEngineInterpreter>(m, "MNNInferenceEngineInterpreter")
        .def(py::init<const std::string&>(),
             py::arg("model_path"),
             "Initialize MNN Inference Engine Interpreter with model path")
        .def("infer", &MNNInferenceEngineInterpreter::infer,
             py::arg("input_dict"),
             "Execute inference with input dictionary, returns output dictionary");
    
    py::class_<GSVEngine>(m, "GSVEngine")
        .def(py::init<const std::string&, const std::string&, const std::string&, bool, bool, bool>(),
             py::arg("fsdec_path"),
             py::arg("sdec_path"),
             py::arg("sovits_path"),
             py::arg("use_gpu") = false,
             py::arg("use_npu") = false,
             py::arg("quantized") = false)
        .def("infer", &GSVEngine::infer, py::arg("ref_dict"), py::arg("text_input"), py::arg("sampling_params"));
}