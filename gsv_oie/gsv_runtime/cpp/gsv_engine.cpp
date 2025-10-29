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
constexpr int KV_CACHE_PREPARED_LENGTH = 640;
constexpr int DECODE_DIMENSION = 512;
constexpr int MNN_CPU_NUM_THREAD = 8;
#define CPP_PRINT(msg) py::print("[C++] " + std::string(msg))
using OrtValueShapeType = std::vector<int64_t>;
static Ort::AllocatorWithDefaultOptions default_allocator;
static Ort::MemoryInfo pre_allocated_memory_info("Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault);
static Ort::Env global_env(ORT_LOGGING_LEVEL_WARNING, "Global");
static Ort::RunOptions global_run_options;
static Ort::SessionOptions global_session_options;

class GSVEngine : NonCopyable {
    public:
    GSVEngine(const std::string fsdec_path,
              const std::string sdec_path,
              const std::string sovits_path,
              bool sv_emb = false,
              bool use_gpu = false,
              bool use_npu = false,
              bool quantized = false){
        CPP_PRINT("GSVEngine initialized with directories:");
        CPP_PRINT("  FSDec path: " + fsdec_path);
        CPP_PRINT("  SDec path: " + sdec_path);
        CPP_PRINT("  SoVITS path: " + sovits_path);
        sv_emb_ = sv_emb;
        use_gpu_ = use_gpu;
        use_npu_ = use_npu;
        quantized_ = quantized;
        fsdec_path_ = fsdec_path;
        sdec_path_ = sdec_path;
        sovits_path_ = sovits_path;

        if(!quantized_){ //kv cache using fp16 only when not quantized
            cache_element_size_ = sizeof(uint16_t);
            argument_element_size_ = sizeof(uint16_t);
            fsdec_engine_ = std::make_shared<MNNInferenceEngineInterpreter>(fsdec_path_);
        }
        else{
            cache_element_size_ = sizeof(uint8_t);
            argument_element_size_ = sizeof(float);
        }
        for(int i = 0; i < HEAD_NUM; ++i){
            k_cache_.emplace_back(AlignedVec(KV_CACHE_PREPARED_LENGTH * 1 * DECODE_DIMENSION * cache_element_size_));
            v_cache_.emplace_back(AlignedVec(KV_CACHE_PREPARED_LENGTH * 1 * DECODE_DIMENSION * cache_element_size_));
        }
        y_emb_cache_ = AlignedVec(1 * KV_CACHE_PREPARED_LENGTH * DECODE_DIMENSION * cache_element_size_);
        
        global_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        global_session_options.SetIntraOpNumThreads(STEP_DECODE_THREAD_NUM);
        global_session_options.SetInterOpNumThreads(STEP_DECODE_THREAD_NUM);
        sdec_session_ = std::make_shared<Ort::Session>(global_env, sdec_path.c_str(), global_session_options);
        for(const auto& name : sdec_session_->GetInputNames()) sdec_input_names_.push_back(name);
        for(const auto& name : sdec_session_->GetOutputNames()) sdec_output_names_.push_back(name);

        sovits_engine_ = std::make_shared<MNNInferenceEngineInterpreter>(sovits_path_);
    }
    ~GSVEngine(){
        CPP_PRINT("GSVEngine destroyed");
    }

    py::array_t<float> infer(const py::dict& ref_dict, const py::dict& text_input, const py::dict& sampling_params) {
        check_dict_key(text_input, "norm_text");
        auto norm_text = text_input["norm_text"].cast<std::string>();
        // CPP_PRINT("Received norm_text: " + norm_text);

        check_dict_key(ref_dict, "phones");
        const py::array encoder_ref_seq = ref_dict["phones"].cast<py::array>();

        check_dict_key(ref_dict, "bert_features");
        const py::array encoder_ref_bert = ref_dict["bert_features"].cast<py::array>();
        check_dict_key(ref_dict, "hubert_ssl_output");
        const py::array encoder_ssl_content = ref_dict["hubert_ssl_output"].cast<py::array>();

        check_dict_key(text_input, "phones");
        const py::array encoder_text_seq = text_input["phones"].cast<py::array>();
        check_dict_key(text_input, "bert_features");
        const py::array encoder_text_bert = text_input["bert_features"].cast<py::array>();

        check_dict_key(ref_dict, "spectrum");
        const py::array spectrum = ref_dict["spectrum"].cast<py::array>();

        if(sv_emb_) check_dict_key(ref_dict, "sv_emb");
        const py::array sv_emb = sv_emb_ ? ref_dict["sv_emb"].cast<py::array>() : py::array();

        check_dict_key(sampling_params, "top_k");
        top_k_ = sampling_params["top_k"].cast<int64_t>();
        check_dict_key(sampling_params, "temperature");
        temperature_ = sampling_params["temperature"].cast<float>();
        check_dict_key(sampling_params, "repeat_penalty");
        repeat_penalty_ = sampling_params["repeat_penalty"].cast<float>();

        iteration_ = 0;
        kv_cache_seq_init_len_ = static_cast<int>(encoder_ref_seq.shape(1) + encoder_text_seq.shape(1) + (encoder_ssl_content.shape(2)/2));
        y_init_len_ = static_cast<int>(encoder_ssl_content.shape(2) / 2);

        if(!quantized_)
            fsdec_infer(encoder_ref_seq, encoder_ref_bert, encoder_ssl_content, encoder_text_seq, encoder_text_bert);

        while (true){
            if(sdec_infer() || iteration_ >= (KV_CACHE_PREPARED_LENGTH - kv_cache_seq_init_len_ -1)){
                break;
            }
        }
        iteration_ -= 1; //last iteration is not used
        int result_starting_idx = y_.GetTensorTypeAndShapeInfo().GetShape()[1] - 1 - iteration_;
        py::array y_cropped_array = py::array_t<int64_t>(
            {1, 1, iteration_},
            y_.GetTensorMutableData<int64_t>() + result_starting_idx);

        return infer_sovits(encoder_text_seq, y_cropped_array, spectrum, sv_emb);
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
        // Call the inference engine
        auto output_map = fsdec_engine_->infer_tensor(input_map);

        // Process output tensors y
        auto y_shape = to_ort_shape_vector(output_map["y"]->shape());
        y_ = Ort::Value::CreateTensor<int64_t>(default_allocator, y_shape.data(), y_shape.size());

        convert_vector_int32_to_int64(
            y_.GetTensorMutableData<int64_t>(),
            reinterpret_cast<const int32_t*>(output_map["y"]->ptr()),
            y_shape[0] * y_shape[1]);

        // Process output tensor y_emb
        auto y_emb_shape = to_ort_shape_vector(output_map["y_emb"]->shape());
        convert_vector_fp32_to_fp16(
            reinterpret_cast<uint16_t*>(y_emb_cache_.data()),
            reinterpret_cast<const float*>(output_map["y_emb"]->ptr()),
            y_emb_shape[0] * y_emb_shape[1] * y_emb_shape[2]);

        for(int i = 0; i < HEAD_NUM; ++i){
            // Process k_cache
            std::string k_cache_name = "present_k_layer_" + std::to_string(i);
            auto k_cache_shape = to_ort_shape_vector(output_map[k_cache_name]->shape());
            convert_vector_fp32_to_fp16(
                reinterpret_cast<uint16_t*>(k_cache_[i].data()),
                reinterpret_cast<const float*>(output_map[k_cache_name]->ptr()),
                k_cache_shape[0] * k_cache_shape[1] * k_cache_shape[2]);
            // Process v_cache
            std::string v_cache_name = "present_v_layer_" + std::to_string(i);
            auto v_cache_shape = to_ort_shape_vector(output_map[v_cache_name]->shape());
            convert_vector_fp32_to_fp16(
                reinterpret_cast<uint16_t*>(v_cache_[i].data()),
                reinterpret_cast<const float*>(output_map[v_cache_name]->ptr()),
                v_cache_shape[0] * v_cache_shape[1] * v_cache_shape[2]);
        }
        // Process output tensor y_emb, needs conversion from fp32 to fp16
    }
    
    bool sdec_infer(){
        assert(k_cache_.size() == HEAD_NUM && v_cache_.size() == HEAD_NUM);
        assert(y_.IsTensor());
        assert(sdec_session_ != nullptr);
        int current_y_len = y_init_len_ + iteration_ + 1;
        int current_y_emb_len = y_init_len_ + iteration_;
        int current_kv_cache_len = kv_cache_seq_init_len_ + iteration_;
        int kv_cache_current_size = current_kv_cache_len * 1 * 512;
        int y_emb_current_size = 1 * current_y_emb_len * 512;

         // CPU memory info
        Ort::IoBinding sdec_iobinding(*sdec_session_);

        // Bind inputs
        sdec_iobinding.BindInput("iy", y_);
        OrtValueShapeType y_emb_shape = {1, current_y_emb_len, 512};

        Ort::Value y_emb = Ort::Value::CreateTensor(pre_allocated_memory_info, 
                                                    static_cast<void*>(y_emb_cache_.data()), 
                                                    y_emb_current_size * argument_element_size_, 
                                                    y_emb_shape.data(), 
                                                    y_emb_shape.size(),
                                                    quantized_ ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        sdec_iobinding.BindInput("iy_emb", y_emb);

        uint16_t temperature_data_fp16 = fp16_ieee_from_fp32_value(temperature_);
        OrtValueShapeType temperature_shape = {1};
        Ort::Value temperature_tensor = Ort::Value::CreateTensor(pre_allocated_memory_info, 
                                                                reinterpret_cast<void*>(&temperature_data_fp16), 
                                                                argument_element_size_, 
                                                                temperature_shape.data(), 
                                                                temperature_shape.size(),
                                                                quantized_ ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        sdec_iobinding.BindInput("temperature", temperature_tensor);

        OrtValueShapeType top_k_shape = {1};
        Ort::Value top_k_tensor = Ort::Value::CreateTensor<int64_t>(pre_allocated_memory_info, 
                                                                    &top_k_, 
                                                                    1, 
                                                                    top_k_shape.data(), 
                                                                    top_k_shape.size());
        sdec_iobinding.BindInput("top_k", top_k_tensor);

        uint16_t repeat_penalty_data_fp16 = fp16_ieee_from_fp32_value(repeat_penalty_);
        OrtValueShapeType repeat_penalty_shape = {1};
        Ort::Value repeat_penalty_tensor = Ort::Value::CreateTensor(pre_allocated_memory_info, 
                                                                    reinterpret_cast<void*>(&repeat_penalty_data_fp16), 
                                                                    argument_element_size_, 
                                                                    repeat_penalty_shape.data(), 
                                                                    repeat_penalty_shape.size(),
                                                                    quantized_ ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        sdec_iobinding.BindInput("repeat_penalty", repeat_penalty_tensor);

        OrtValueShapeType kv_cache_shape = {current_kv_cache_len, 1, 512};
        for(int i = 0; i < HEAD_NUM; ++i){
            Ort::Value k_tensor = Ort::Value::CreateTensor(pre_allocated_memory_info, 
                                                        reinterpret_cast<void*>(k_cache_[i].data()), 
                                                        kv_cache_current_size * cache_element_size_, 
                                                        kv_cache_shape.data(), 
                                                        kv_cache_shape.size(),
                                                        quantized_ ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
            Ort::Value v_tensor = Ort::Value::CreateTensor(pre_allocated_memory_info, 
                                                        reinterpret_cast<void*>(v_cache_[i].data()), 
                                                        kv_cache_current_size * cache_element_size_, 
                                                        kv_cache_shape.data(), 
                                                        kv_cache_shape.size(),
                                                        quantized_ ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
            sdec_iobinding.BindInput(sdec_input_names_[5 + i * 2].c_str(), k_tensor);
            sdec_iobinding.BindInput(sdec_input_names_[5 + i * 2 + 1].c_str(), v_tensor);
            // CPP_PRINT("Debug data:"+ std::to_string(k_cache_[i][kv_cache_current_size * cache_element_size_ -1])+","+ std::to_string(v_cache_[i][kv_cache_current_size * cache_element_size_ -1]));
            // CPP_PRINT("Bound k_cache and v_cache for head " + std::to_string(i)+ ":"+sdec_input_names_[5 + i * 2]+" , "+sdec_input_names_[5 + i * 2 + 1]);
        }

        // Prepare and bind outputs
        OrtValueShapeType y_new_shape = {1, current_y_len + 1};
        Ort::Value y_new = Ort::Value::CreateTensor<int64_t>(default_allocator, y_new_shape.data(), y_new_shape.size());
        sdec_iobinding.BindOutput("y", y_new);

        bool stop_condition_data = false;
        Ort::Value stop_condition_tensor = Ort::Value::CreateTensor<bool>(pre_allocated_memory_info, &stop_condition_data, 1, nullptr, 0);
        sdec_iobinding.BindOutput("stop_condition_tensor", stop_condition_tensor);

        OrtValueShapeType y_emb_new_shape = {1, 1, 512};
        int y_emb_increased_size = 1 * 1 * 512;
        void* y_emb_out_ptr = reinterpret_cast<void*>(y_emb_cache_.data() + y_emb_current_size * cache_element_size_);
        Ort::Value y_emb_new = Ort::Value::CreateTensor(pre_allocated_memory_info, 
                                                        y_emb_out_ptr, 
                                                        y_emb_increased_size * argument_element_size_,
                                                        y_emb_new_shape.data(),
                                                        y_emb_new_shape.size(),
                                                        quantized_ ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        sdec_iobinding.BindOutput("increased_y_emb", y_emb_new);

        OrtValueShapeType kv_cache_shape_new = {1, 1, 512};
        int kv_cache_increased_size = 1 * 1 * 512;
        for(int i = 0; i < HEAD_NUM; ++i){
            void* k_cache_out_ptr = reinterpret_cast<void*>(k_cache_[i].data() + kv_cache_current_size * cache_element_size_);
            void* v_cache_out_ptr = reinterpret_cast<void*>(v_cache_[i].data() + kv_cache_current_size * cache_element_size_);
            Ort::Value k_tensor = Ort::Value::CreateTensor(pre_allocated_memory_info, 
                                                        k_cache_out_ptr, 
                                                        kv_cache_increased_size * cache_element_size_, 
                                                        kv_cache_shape_new.data(), 
                                                        kv_cache_shape_new.size(),
                                                        quantized_ ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
            Ort::Value v_tensor = Ort::Value::CreateTensor(pre_allocated_memory_info, 
                                                        v_cache_out_ptr, 
                                                        kv_cache_increased_size * cache_element_size_, 
                                                        kv_cache_shape_new.data(), 
                                                        kv_cache_shape_new.size(), 
                                                        quantized_ ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
            sdec_iobinding.BindOutput(sdec_output_names_[3 + i * 2].c_str(), k_tensor);
            sdec_iobinding.BindOutput(sdec_output_names_[3 + i * 2 + 1].c_str(), v_tensor);
        }
        
        // CPP_PRINT("Running stage decoder...");
        sdec_session_->Run(global_run_options, sdec_iobinding);
        // CPP_PRINT("Stage decoder run completed.");
        sdec_iobinding.SynchronizeOutputs();

        std::swap(y_, y_new);
        iteration_ += 1;

        return stop_condition_data;
    }

    py::array_t<float> infer_sovits(const py::array& input_text_phones,
                                    const py::array& y_array,
                                    const py::array& spectrum,
                                    const py::array& sv_emb) {
        auto input_map = std::map<std::string, const py::array *>{
            {"input_text_phones", &input_text_phones},
            {"pred_semantic", &y_array},
            {"spectrum", &spectrum},
            {"sv_emb", &sv_emb}
        };
        auto output_map = sovits_engine_->infer_tensor(input_map);
        return create_numpy_array_from_tensor(output_map["audio32k"]->tensor());
    }

    private:
    std::shared_ptr<MNNInferenceEngineInterpreter> fsdec_engine_;
    std::shared_ptr<Ort::Session> sdec_session_;
    std::shared_ptr<MNNInferenceEngineInterpreter> sovits_engine_;
    bool sv_emb_ = false;
    bool use_gpu_ = false;
    bool use_npu_ = false;
    bool quantized_ = false;
    int cache_element_size_ = 0;
    int argument_element_size_ = 0;
    std::string fsdec_path_;
    std::string sdec_path_;
    std::string sovits_path_;

    std::vector<AlignedVec> k_cache_;
    std::vector<AlignedVec> v_cache_;
    AlignedVec y_emb_cache_;
    Ort::Value y_;

    int64_t top_k_ = 0;
    float temperature_ = 0.0f;
    float repeat_penalty_ = 0.0f;

    int iteration_ = 0;
    int kv_cache_seq_init_len_ = 0;
    int y_init_len_ = 0;

    std::vector<std::string> sdec_input_names_;
    std::vector<std::string> sdec_output_names_;
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
        .def(py::init<const std::string&, const std::string&, const std::string&, bool, bool, bool, bool>(),
             py::arg("fsdec_path"),
             py::arg("sdec_path"),
             py::arg("sovits_path"),
             py::arg("sv_emb") = false,
             py::arg("use_gpu") = false,
             py::arg("use_npu") = false,
             py::arg("quantized") = false)
        .def("infer", &GSVEngine::infer, py::arg("ref_dict"), py::arg("text_input"), py::arg("sampling_params"));
}