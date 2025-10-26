#include "MNNInferenceEngine.h"
#include <iostream>
#include <algorithm>
#include <limits>
#include <cstring>

#define CPP_PRINT(msg) py::print("[C++] " + std::string(msg))

MNNInferenceEngine::MNNInferenceEngine(const std::string model_path,
                                       const std::vector<std::string>& input_list,
                                       const std::vector<std::string>& output_list) {
    CPP_PRINT("MNNInferenceEngine initialized with model: " + model_path);
    model_filename_ = std::filesystem::path(model_path).filename().string();
    MNN::Express::Module::Config mdconfig;
    mdconfig.rearrange = true;
    model_.reset(MNN::Express::Module::load(input_list, output_list, model_path.c_str(), &mdconfig));
}

std::vector<py::array> MNNInferenceEngine::infer(const std::vector<py::array>& input_arrays) {
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

MNNInferenceEngine::~MNNInferenceEngine() {
    CPP_PRINT("MNNInferenceEngine destroyed");
}