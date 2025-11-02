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
static Ort::MemoryInfo pre_allocated_memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
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
            fsdec_quant_session_ = std::make_shared<Ort::Session>(global_env, fsdec_path.c_str(), global_session_options);
        }
        for(int i = 0; i < HEAD_NUM; ++i){
            k_cache_.emplace_back(AlignedVec(KV_CACHE_PREPARED_LENGTH * 1 * DECODE_DIMENSION * cache_element_size_));
            v_cache_.emplace_back(AlignedVec(KV_CACHE_PREPARED_LENGTH * 1 * DECODE_DIMENSION * cache_element_size_));
        }
        y_emb_cache_ = AlignedVec(1 * KV_CACHE_PREPARED_LENGTH * DECODE_DIMENSION * argument_element_size_);
        
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
        else
            fsdec_quant_infer(encoder_ref_seq, encoder_ref_bert, encoder_ssl_content, encoder_text_seq, encoder_text_bert);

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
    }

    void fsdec_quant_infer(const py::array&encoder_ref_seq, 
                     const py::array&encoder_ref_bert, 
                     const py::array&encoder_ssl_content,
                     const py::array&encoder_text_seq,
                     const py::array&encoder_text_bert) {
        // Implementation of fsdec_quant_infer would go here
        std::vector<const char*> input_names;
        input_names.push_back("encoder_ref_seq");
        input_names.push_back("encoder_text_seq");
        input_names.push_back("encoder_ref_bert");
        input_names.push_back("encoder_text_bert");
        input_names.push_back("encoder_ssl_content");
        std::vector<const char*> output_names;
        output_names.push_back("y");
        output_names.push_back("y_emb");
        std::vector<std::string> kv_name_strings;
        for(int i = 0; i < HEAD_NUM; ++i){
            kv_name_strings.push_back(std::string("present_k_layer_") + std::to_string(i) + std::string("_quantized"));
            output_names.push_back(kv_name_strings.back().c_str());
            kv_name_strings.push_back(std::string("present_v_layer_") + std::to_string(i) + std::string("_quantized"));
            output_names.push_back(kv_name_strings.back().c_str());
        }

        std::vector<Ort::Value> input_tensors;
        std::vector<Ort::Value> output_tensors;

        OrtValueShapeType encoder_ref_seq_shape = {static_cast<int64_t>(encoder_ref_seq.shape(0)), static_cast<int64_t>(encoder_ref_seq.shape(1))};
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(pre_allocated_memory_info, 
                                                                            reinterpret_cast<int64_t*>(encoder_ref_seq.request().ptr),
                                                                            encoder_ref_seq_shape[0] * encoder_ref_seq_shape[1], 
                                                                            encoder_ref_seq_shape.data(), 
                                                                            encoder_ref_seq_shape.size()));
        
        OrtValueShapeType encoder_text_seq_shape = {static_cast<int64_t>(encoder_text_seq.shape(0)), static_cast<int64_t>(encoder_text_seq.shape(1))};
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(pre_allocated_memory_info, 
                                                                            reinterpret_cast<int64_t*>(encoder_text_seq.request().ptr),
                                                                            encoder_text_seq_shape[0] * encoder_text_seq_shape[1], 
                                                                            encoder_text_seq_shape.data(), 
                                                                            encoder_text_seq_shape.size()));

        OrtValueShapeType encoder_ref_bert_shape = {static_cast<int64_t>(encoder_ref_bert.shape(0)), static_cast<int64_t>(encoder_ref_bert.shape(1))};
        input_tensors.push_back(Ort::Value::CreateTensor<float>(pre_allocated_memory_info, 
                                                                            reinterpret_cast<float*>(encoder_ref_bert.request().ptr),
                                                                            encoder_ref_bert_shape[0] * encoder_ref_bert_shape[1], 
                                                                            encoder_ref_bert_shape.data(), 
                                                                            encoder_ref_bert_shape.size()));

        OrtValueShapeType encoder_text_bert_shape = {static_cast<int64_t>(encoder_text_bert.shape(0)), static_cast<int64_t>(encoder_text_bert.shape(1))};
        input_tensors.push_back(Ort::Value::CreateTensor<float>(pre_allocated_memory_info, 
                                                                            reinterpret_cast<float*>(encoder_text_bert.request().ptr),
                                                                            encoder_text_bert_shape[0] * encoder_text_bert_shape[1], 
                                                                            encoder_text_bert_shape.data(), 
                                                                            encoder_text_bert_shape.size()));

        OrtValueShapeType encoder_ssl_content_shape = {static_cast<int64_t>(encoder_ssl_content.shape(0)), static_cast<int64_t>(encoder_ssl_content.shape(1)), static_cast<int64_t>(encoder_ssl_content.shape(2))};
        input_tensors.push_back(Ort::Value::CreateTensor<float>(pre_allocated_memory_info, 
                                                                            reinterpret_cast<float*>(encoder_ssl_content.request().ptr),
                                                                            encoder_ssl_content_shape[0] * encoder_ssl_content_shape[1] * encoder_ssl_content_shape[2], 
                                                                            encoder_ssl_content_shape.data(), 
                                                                            encoder_ssl_content_shape.size()));

        OrtValueShapeType y_new_shape = {1, y_init_len_ + 1};
        output_tensors.push_back(Ort::Value::CreateTensor<int64_t>(default_allocator, y_new_shape.data(), y_new_shape.size()));

        OrtValueShapeType y_emb_shape = {1, y_init_len_, 512};
        output_tensors.push_back(Ort::Value::CreateTensor<float>(pre_allocated_memory_info, 
                                                    reinterpret_cast<float*>(y_emb_cache_.data()), 
                                                    y_init_len_ * 512, 
                                                    y_emb_shape.data(), 
                                                    y_emb_shape.size()));
        
        OrtValueShapeType kv_cache_shape = {kv_cache_seq_init_len_, 1, 512};
        for(int i = 0; i < HEAD_NUM; ++i){
            output_tensors.push_back(Ort::Value::CreateTensor<uint8_t>(pre_allocated_memory_info, 
                                                                k_cache_[i].data(), 
                                                                kv_cache_seq_init_len_ * 1 * 512,
                                                                kv_cache_shape.data(), 
                                                                kv_cache_shape.size()));
            output_tensors.push_back(Ort::Value::CreateTensor<uint8_t>(pre_allocated_memory_info, 
                                                                v_cache_[i].data(), 
                                                                kv_cache_seq_init_len_ * 1 * 512,
                                                                kv_cache_shape.data(), 
                                                                kv_cache_shape.size()));
        }
        CPP_PRINT("Running fsdec quantized...");
        try{
            fsdec_quant_session_->Run(global_run_options, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_tensors.data(), output_tensors.size());
        }
        catch (const Ort::Exception& e) {
            printf("ONNX Runtime exception: %s\n", e.what());
            // CPP_PRINT("Error during fsdec quantized inference: " + std::string(e.what()));
            throw;
        }
        CPP_PRINT("Finished running fsdec quantized.");
       // Process output tensors y
        y_ = std::move(output_tensors[0]); 
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
                                                                quantized_? reinterpret_cast<void*>(&temperature_) : reinterpret_cast<void*>(&temperature_data_fp16), 
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
                                                                    quantized_ ? reinterpret_cast<void*>(&repeat_penalty_) : reinterpret_cast<void*>(&repeat_penalty_data_fp16), 
                                                                    argument_element_size_, 
                                                                    repeat_penalty_shape.data(), 
                                                                    repeat_penalty_shape.size(),
                                                                    quantized_ ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        sdec_iobinding.BindInput("repeat_penalty", repeat_penalty_tensor);

        OrtValueShapeType kv_cache_shape = {current_kv_cache_len, 1, 512};
        std::vector<Ort::Value> k_tensors;
        std::vector<Ort::Value> v_tensors;
        for(int i = 0; i < HEAD_NUM; ++i){
            k_tensors.push_back(Ort::Value::CreateTensor(pre_allocated_memory_info, 
                                                        reinterpret_cast<void*>(k_cache_[i].data()), 
                                                        kv_cache_current_size * cache_element_size_, 
                                                        kv_cache_shape.data(), 
                                                        kv_cache_shape.size(),
                                                        quantized_ ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
            v_tensors.push_back(Ort::Value::CreateTensor(pre_allocated_memory_info, 
                                                        reinterpret_cast<void*>(v_cache_[i].data()), 
                                                        kv_cache_current_size * cache_element_size_, 
                                                        kv_cache_shape.data(), 
                                                        kv_cache_shape.size(),
                                                        quantized_ ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
            sdec_iobinding.BindInput(sdec_input_names_[5 + i * 2].c_str(), k_tensors.back());
            sdec_iobinding.BindInput(sdec_input_names_[5 + i * 2 + 1].c_str(), v_tensors.back());
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
            k_tensors.push_back(Ort::Value::CreateTensor(pre_allocated_memory_info, 
                                                        k_cache_out_ptr, 
                                                        kv_cache_increased_size * cache_element_size_, 
                                                        kv_cache_shape_new.data(), 
                                                        kv_cache_shape_new.size(),
                                                        quantized_ ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
            v_tensors.push_back(Ort::Value::CreateTensor(pre_allocated_memory_info, 
                                                        v_cache_out_ptr, 
                                                        kv_cache_increased_size * cache_element_size_, 
                                                        kv_cache_shape_new.data(), 
                                                        kv_cache_shape_new.size(), 
                                                        quantized_ ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
            sdec_iobinding.BindOutput(sdec_output_names_[3 + i * 2].c_str(), k_tensors.back());
            sdec_iobinding.BindOutput(sdec_output_names_[3 + i * 2 + 1].c_str(), v_tensors.back());
        }
        
        CPP_PRINT("Running stage decoder...");
        sdec_iobinding.SynchronizeInputs();
        sdec_session_->Run(global_run_options, sdec_iobinding);
        CPP_PRINT("Stage decoder run completed.");
        sdec_iobinding.SynchronizeOutputs();

        std::swap(y_, y_new);
        iteration_ += 1;
        sdec_iobinding.ClearBoundOutputs();
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
        CPP_PRINT("Running SoVITS inference...");
        auto output_map = sovits_engine_->infer_tensor(input_map);
        CPP_PRINT("SoVITS inference completed.");
        return create_numpy_array_from_tensor(output_map["audio32k"]->tensor());
    }

    private:
    std::shared_ptr<MNNInferenceEngineInterpreter> fsdec_engine_;
    std::shared_ptr<Ort::Session> fsdec_quant_session_;
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
    std::vector<int64_t> y_init_memory_;

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