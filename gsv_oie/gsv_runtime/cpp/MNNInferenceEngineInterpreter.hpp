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

    ~MNNInferenceEngineInterpreter();

private:
    std::shared_ptr<MNN::Interpreter> interpreter_;
    MNN::Session* session_ = nullptr;
    std::string model_filename_;
};