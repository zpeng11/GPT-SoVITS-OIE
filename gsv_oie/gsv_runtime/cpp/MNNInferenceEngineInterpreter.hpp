#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <MNN/Interpreter.hpp>
#include <memory>
#include <string>
#include <filesystem>
#include <vector>
#include <map>
#include <memory>
#include "utils.hpp"

namespace py = pybind11;

class MNNInferenceEngineInterpreter : NonCopyable {
public:
    MNNInferenceEngineInterpreter(const std::string model_path);

    py::dict infer(const py::dict& input_dict);

    std::map<std::string, std::shared_ptr<TensorDataGuard>> infer_tensor(const std::map<std::string, const py::array *>& input_map);

    std::vector<std::string> get_input_names();
    std::vector<std::string> get_output_names();

    ~MNNInferenceEngineInterpreter();

private:
    MNN::Interpreter* interpreter_; //Do not use shared_ptr here because in doc: Windows 上建议使用 Interpreter::destroy , Tensor::destroy , Module::destroy 等方法进行 MNN 相关内存对象的析构，不要直接使用 delete （直接使用 delete 在 -DMNN_WIN_RUNTIME_MT=ON 时会出问题）
    MNN::Session* session_ = nullptr;
    std::string model_filename_;
};