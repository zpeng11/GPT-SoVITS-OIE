#include "MNNInferenceEngineInterpreter.hpp"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <limits>

#define CPP_PRINT(msg) py::print("[C++] " + std::string(msg))
constexpr int MNN_CPU_NUM_THREAD = 8;
static MNN::BackendConfig create_backend_config() {
    MNN::BackendConfig config;
    config.memory = MNN::BackendConfig::Memory_Low;
    config.power = MNN::BackendConfig::Power_High;
    config.precision = MNN::BackendConfig::Precision_Low;
    config.sharedContext = nullptr;
    return config;
}

static MNN::BackendConfig global_backend_config = create_backend_config();

static MNN::ScheduleConfig create_schedule_config() {
    MNN::ScheduleConfig config;
    config.saveTensors = {};
    config.type = MNN_FORWARD_AUTO;
    config.numThread = MNN_CPU_NUM_THREAD;
    config.path = {};
    config.backupType = MNN_FORWARD_CPU;
    config.backendConfig = &global_backend_config;
    return config;
}

static MNN::ScheduleConfig global_config = create_schedule_config();


static MNN::RuntimeInfo runtime_info = MNN::Interpreter::createRuntime({global_config});

MNNInferenceEngineInterpreter::MNNInferenceEngineInterpreter(const std::string model_path){
    CPP_PRINT("MNNInferenceEngineInterpreter initialized with model: " + model_path);
    model_filename_ = std::filesystem::path(model_path).filename().string();
    interpreter_ = MNN::Interpreter::createFromFile(model_path.c_str());
    session_ = interpreter_->createSession(global_config, runtime_info);
}

std::vector<std::string> MNNInferenceEngineInterpreter::get_input_names(){
    std::vector<std::string> input_names;
    auto input_tensors = interpreter_->getSessionInputAll(session_);
    for(auto item : input_tensors){
        input_names.push_back(item.first);
    }
    return input_names;
}

std::vector<std::string> MNNInferenceEngineInterpreter::get_output_names(){
    std::vector<std::string> output_names;
    auto output_tensors = interpreter_->getSessionOutputAll(session_);
    for(auto item : output_tensors){
        output_names.push_back(item.first);
    }
    return output_names;
}

py::dict MNNInferenceEngineInterpreter::infer(const py::dict& input_dict){
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

std::map<std::string, std::shared_ptr<TensorDataGuard>> MNNInferenceEngineInterpreter::infer_tensor(const std::map<std::string, const py::array *>& input_map){
    auto input_tensors = interpreter_->getSessionInputAll(session_);
    for(auto item : input_tensors){
        const auto name = item.first;
        if(input_map.find(name) == input_map.end()){
            throw std::runtime_error("Input map missing required key: " + name);
        }
        const py::array& input_array = *(input_map.at(name));
        MNN::Tensor* tensor = item.second;
        interpreter_->resizeTensor(tensor, get_shape_from_numpy_array(input_array));
    }
    interpreter_->resizeSession(session_);
    for(auto item : input_tensors){
        const auto name = item.first;
        const py::array& input_array = *(input_map.at(name));
        MNN::Tensor* tensor = item.second;
        copy_numpy_array_to_tensor(input_array, tensor);
    }
    interpreter_->runSession(session_);
    std::map<std::string, std::shared_ptr<TensorDataGuard>> output_map;
    auto output_tensors = interpreter_->getSessionOutputAll(session_);
    for(auto& items : output_tensors){
        const auto name = items.first;
        MNN::Tensor* tensor = items.second;
        output_map.emplace(name, std::make_shared<TensorDataGuard>(tensor, false));
    }
    return output_map;
}

MNNInferenceEngineInterpreter::~MNNInferenceEngineInterpreter(){
    MNN::Interpreter::destroy(interpreter_);
    CPP_PRINT("MNNInferenceEngineInterpreter destroyed");
}