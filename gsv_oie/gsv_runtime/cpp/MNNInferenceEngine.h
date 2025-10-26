#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include <string>
#include <filesystem>
#include <vector>

#include "utils.hpp"

class MNNInferenceEngine : NonCopyable {
public:
    MNNInferenceEngine(const std::string model_path,
                      const std::vector<std::string>& input_list,
                      const std::vector<std::string>& output_list);

    std::vector<py::array> infer(const std::vector<py::array>& input_arrays);

    ~MNNInferenceEngine();

private:
    std::shared_ptr<MNN::Express::Module> model_;
    std::string model_filename_;
};