#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include  <pybind11/complex.h>
#include <pybind11/functional.h>
#include <onnxruntime_cxx_api.h>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/MathOp.hpp>
#include <memory>
#include <string>
#include <filesystem>
#include <vector>
#include <iostream>
#include <cstdint>

namespace py = pybind11;
constexpr int PREFILL_THREAD_NUM = 0; // 0 means using default thread number
constexpr int STEP_DECODE_THREAD_NUM = 0; // 1 is usually enough for step decode
constexpr int HEAD_NUM = 24;
constexpr int KV_CACHE_PREPARED_LENGTH = 512;
constexpr int MNN_CPU_NUM_THREAD = 8;
#define CPP_PRINT(msg) py::print("[C++] " + std::string(msg))
using OrtValueShapeType = std::vector<int64_t>;

class NonCopyable {
protected:
    // Protected constructor - only derived classes can construct
    NonCopyable() = default;
    
    // Protected destructor - only derived classes can destruct
    ~NonCopyable() = default;

public:
    // Delete copy constructor
    NonCopyable(const NonCopyable&) = delete;
    
    // Delete copy assignment operator
    NonCopyable& operator=(const NonCopyable&) = delete;
    
    // Allow move constructor (optional)
    NonCopyable(NonCopyable&&) = default;
    
    // Allow move assignment operator (optional)
    NonCopyable& operator=(NonCopyable&&) = default;
};

static void check_dict_key(const py::dict& dict, const std::string& key) {
    if (!dict.contains(key)) {
        throw std::runtime_error("Input dictionary missing required key: " + key);
    }
}

static std::vector<int> get_shape_from_numpy_array(const py::array& array) {
    std::vector<int> shape;
    for (ssize_t i = 0; i < array.ndim(); ++i) {
        shape.push_back(static_cast<int>(array.shape(i)));
    }
    return shape;
}

static std::string shape_vector_to_string(const std::vector<int>& shape) {
    std::string shape_str = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);
        if (i != shape.size() - 1) {
            shape_str += ", ";
        }
    }
    shape_str += "]";
    return shape_str;
}

std::vector<ssize_t> to_ssize_t_vector(const std::vector<int>& vec) {
    std::vector<ssize_t> result;
    result.reserve(vec.size());
    for (const auto& v : vec) {
        result.push_back(static_cast<ssize_t>(v));
    }
    return result;
}

std::vector<int32_t> transform_int64_to_int32(const int64_t* src, size_t size) {
    std::vector<int32_t> dst(size);  // 预分配输出 vector
    
    constexpr int64_t MIN_VAL = std::numeric_limits<int32_t>::min();  // -2147483648
    constexpr int64_t MAX_VAL = std::numeric_limits<int32_t>::max();  // 2147483647
    
    // Lambda：转换逻辑 + 溢出饱和处理
    auto converter = [MIN_VAL, MAX_VAL](int64_t val) -> int32_t {
        if (val < MIN_VAL) return std::numeric_limits<int32_t>::min();
        if (val > MAX_VAL) return std::numeric_limits<int32_t>::max();
        return static_cast<int32_t>(val);
    };
    std::transform(src, src + size, dst.begin(), converter);
    return dst;
}

std::vector<int64_t> transform_int32_to_int64(const int32_t* src, size_t size) {
    std::vector<int64_t> dst(size);  // 预分配输出 vector
    
    // Lambda：简单转换（无溢出处理）
    auto converter = [](int32_t val) -> int64_t {
        return static_cast<int64_t>(val);
    };
    
    // 并行执行：输入范围 src 到 src + size，输出到 dst.begin()
    std::transform(src, src + size, dst.begin(), converter);
    
    return dst;
}

MNN::Express::VARP make_varp_from_numpy_array(const py::array& array) {
    auto shape = get_shape_from_numpy_array(array);
    auto dtype = array.dtype();
    halide_type_t type;
    if (dtype.is(py::dtype::of<float>())) {
        type = halide_type_of<float>();
    } else if (dtype.is(py::dtype::of<int32_t>())) {
        type = halide_type_of<int32_t>();
    } else if (dtype.is(py::dtype::of<int64_t>())) {
        type = halide_type_of<int64_t>();
    } else {
        throw std::runtime_error("Unsupported numpy array data type.");
    }
    auto input = MNN::Express::_Input(shape, MNN::Express::NHWC, type);
    if (dtype.is(py::dtype::of<float>())) {
        std::memcpy(input->writeMap<float>(), array.data(), array.size() * sizeof(float));
    } else if (dtype.is(py::dtype::of<int32_t>())) {
        std::memcpy(input->writeMap<int32_t>(), array.data(), array.size() * sizeof(int32_t));
    } else if (dtype.is(py::dtype::of<int64_t>())) {
        std::memcpy(input->writeMap<int64_t>(), array.data(), array.size() * sizeof(int64_t));
        input = MNN::Express::_Cast<int32_t>(input);
    } else {
        throw std::runtime_error("Unsupported numpy array data type.");
    }
    return input;
}

template<typename T>
py::array_t<T> create_python_managed_array(const std::vector<ssize_t>& size) {
    ssize_t total_size = 1;
    for (const auto& dim : size) {
        total_size *= dim;
    }
    // 步骤1: 分配连续内存
    T* data = new T[total_size];
    // 步骤2: 创建 capsule 封装释放回调
    auto capsule = py::capsule(data, [](void* ptr) {
        delete[] static_cast<T*>(ptr);
    });
    // 步骤3: 创建 py::array_t 时直接指定 capsule 作为 base
    auto arr = py::array_t<T>(
        size,           // shape
        data,           // data pointer
        capsule         // base object (handles cleanup)
    );
    return std::move(arr);
}

py::array make_numpy_array_from_varp(const MNN::Express::VARP varp) {
    auto shape = to_ssize_t_vector(varp->getInfo()->dim);
    halide_type_t type = varp->getInfo()->type;
    py::array array;
    if (type.code == halide_type_float && type.bits == 32) {
        array = create_python_managed_array<float>(shape);
    } else if (type.code == halide_type_int && type.bits == 32) {
        array = create_python_managed_array<int32_t>(shape);
    } else {
        throw std::runtime_error("Unsupported VARP data type.");
    }
    std::memcpy(array.mutable_data(), varp->readMap<void>(), array.size() * array.itemsize());
    return std::move(array);
}

class MNNInferenceEngine: NonCopyable {
    public:
    MNNInferenceEngine(const std::string model_path, const std::vector<std::string>& input_list, const std::vector<std::string>& output_list){
        CPP_PRINT("MNNInferenceEngine initialized with model: " + model_path);
        model_filename_ = std::filesystem::path(model_path).filename().string();
        MNN::Express::Module::Config mdconfig;
        mdconfig.rearrange = true;
        model_.reset(MNN::Express::Module::load(input_list, output_list, model_path.c_str(), &mdconfig));
    }
    std::vector<py::array> infer(const std::vector<py::array>& input_arrays){
        std::vector<MNN::Express::VARP> inputs_;
        for(const auto& array : input_arrays){
            auto input_var = make_varp_from_numpy_array(array);
            inputs_.push_back(input_var);
        }
        auto output_varps = model_->onForward(inputs_);
        std::vector<py::array> output_arrays;
        for(auto varp : output_varps){
            CPP_PRINT("Output VARP shape: " + shape_vector_to_string(varp->getInfo()->dim));
            py::array output_array = make_numpy_array_from_varp(varp);
            output_arrays.push_back(output_array);
        }
        return output_arrays;
    }
    ~MNNInferenceEngine(){
        CPP_PRINT("MNNInferenceEngine destroyed");
    }

    private:
    std::shared_ptr<MNN::Express::Module> model_;
    std::string model_filename_;
};

class TensorDataGuard : NonCopyable {
public:
    TensorDataGuard(MNN::Tensor* tensor, bool is_writing = false) : tensor_(tensor), is_writing_(is_writing) {
        if(tensor_){
            ptr_ = tensor_->map( is_writing_ ? MNN::Tensor::MAP_TENSOR_WRITE : MNN::Tensor::MAP_TENSOR_READ, tensor_->getDimensionType());
        }
    }
    TensorDataGuard(TensorDataGuard&& other) noexcept {
        if (this == &other) {
            return;
        }
        tensor_ = other.tensor_;
        ptr_ = other.ptr_;
        is_writing_ = other.is_writing_;
        other.tensor_ = nullptr;
    }
    TensorDataGuard& operator=(TensorDataGuard&& other) noexcept{
        if (this == &other) {
            return *this;
        }
        tensor_ = other.tensor_;
        ptr_ = other.ptr_;
        is_writing_ = other.is_writing_;
        other.tensor_ = nullptr;
        return *this;
    }
    void * ptr() {
        return ptr_;
    }
    MNN::Tensor* tensor() {
        return tensor_;
    }
    ~TensorDataGuard(){
        if(tensor_){
            tensor_->unmap(is_writing_ ? MNN::Tensor::MAP_TENSOR_WRITE : MNN::Tensor::MAP_TENSOR_READ, tensor_->getDimensionType(), ptr_);
        }
    }
private:
    MNN::Tensor* tensor_ = nullptr;
    bool is_writing_ = false;
    void * ptr_ = nullptr;
};

void copy_numpy_array_to_tensor(const py::array& array, MNN::Tensor* tensor){
    TensorDataGuard guard(tensor, true);
    if(array.dtype().is(py::dtype::of<int64_t>())){
        std::vector<int32_t> int32_data = transform_int64_to_int32(static_cast<const int64_t*>(array.data()), array.size());
        std::memcpy(guard.ptr(), int32_data.data(), array.size() * sizeof(int32_t));
    }
    else{
        std::memcpy(guard.ptr(), array.data(), array.size() * array.itemsize());
    }
}

py::array create_numpy_array_from_tensor(MNN::Tensor* tensor){
    TensorDataGuard guard(tensor, false);
    std::vector<ssize_t> shape = to_ssize_t_vector(tensor->shape());
    py::array array;
    if(tensor->getType().code == halide_type_float && tensor->getType().bits == 32){
        array = create_python_managed_array<float>(shape);
    }
    else if(tensor->getType().code == halide_type_int && tensor->getType().bits == 32){
        array = create_python_managed_array<int32_t>(shape);
    }
    else{
        throw std::runtime_error("Unsupported Tensor data type.");
    }
    std::memcpy(array.mutable_data(), guard.ptr(), array.size() * array.itemsize());
    return std::move(array);
}

class MNNInferenceEngineInterpreter: NonCopyable {
    public:
    MNNInferenceEngineInterpreter(const std::string model_path){
        CPP_PRINT("MNNInferenceEngineInterpreter initialized with model: " + model_path);
        model_filename_ = std::filesystem::path(model_path).filename().string();
        interpreter_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
        MNN::ScheduleConfig config;
        config.numThread = MNN_CPU_NUM_THREAD;
        session_ = interpreter_->createSession(config);
    }
    py::dict infer(const py::dict& input_dict){
        auto input_tensors = interpreter_->getSessionInputAll(session_);
        for(auto item : input_tensors){
            const auto name = item.first;
            if(!input_dict.contains(name)){
                throw std::runtime_error("Input dictionary missing required key: " + name);
            }
            py::array input_array = input_dict[py::str(name)].cast<py::array>();
            MNN::Tensor* tensor = item.second;
            interpreter_->resizeTensor(tensor, get_shape_from_numpy_array(input_array));
        }
        interpreter_->resizeSession(session_);
        for(auto item : input_tensors){
            const auto name = item.first;
            py::array input_array = input_dict[py::str(name)].cast<py::array>();
            MNN::Tensor* tensor = item.second;
            copy_numpy_array_to_tensor(input_array, tensor);
        }
        interpreter_->runSession(session_);
        py::dict output_dict;
        auto output_tensors = interpreter_->getSessionOutputAll(session_);
        for(auto& items : output_tensors){
            const auto name = items.first;
            MNN::Tensor* tensor = items.second;
            py::array output_array = create_numpy_array_from_tensor(tensor);
            output_dict[py::str(name)] = output_array;
        }
        return output_dict;
    }
    ~MNNInferenceEngineInterpreter(){
        CPP_PRINT("MNNInferenceEngineInterpreter destroyed");
    }
    private:
    std::shared_ptr<MNN::Interpreter> interpreter_;
    MNN::Session* session_ = nullptr;
    std::string model_filename_;
};

class GSVEngine {
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
    }
    ~GSVEngine(){
        CPP_PRINT("GSVEngine destroyed");
    }

    py::array_t<float> infer(const py::dict& ref_dict, const py::dict& text_input, const py::dict& sampling_params) {
        check_dict_key(text_input, "norm_text");
        auto norm_text = text_input["norm_text"].cast<std::string>();
        CPP_PRINT("Received norm_text: " + norm_text);

        check_dict_key(ref_dict, "phones");
        auto encoder_ref_seq = ref_dict["phones"].cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
        check_dict_key(ref_dict, "bert_features");
        auto encoder_ref_bert = ref_dict["bert_features"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        check_dict_key(ref_dict, "hubert_ssl_output");
        auto encoder_ssl_content = ref_dict["hubert_ssl_output"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();

        check_dict_key(text_input, "phones");
        auto encoder_text_seq = text_input["phones"].cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
        check_dict_key(text_input, "bert_features");
        auto encoder_text_bert = text_input["bert_features"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();

        check_dict_key(sampling_params, "top_k");
        int top_k = sampling_params["top_k"].cast<int>();
        check_dict_key(sampling_params, "temperature");
        float temperature = sampling_params["temperature"].cast<float>();
        check_dict_key(sampling_params, "repeat_penalty");
        float repeat_penalty = sampling_params["repeat_penalty"].cast<float>();

        // Dummy output for illustration
        std::vector<float> output_data = {0.0f, 1.0f, 2.0f};
        return py::array_t<float>(output_data.size(), output_data.data());
    }

    private:
    std::shared_ptr<MNNInferenceEngine> fsdec_engine_;
    bool use_gpu_ = false;
    bool use_npu_ = false;
    bool quantized_ = false;
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