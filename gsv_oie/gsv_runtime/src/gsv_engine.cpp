#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <onnxruntime_cxx_api.h>
#include <MNN/Interpreter.hpp>

#include <string>
#include <memory>
#include <vector>
#include <iostream>
#include <cstdint>

namespace py = pybind11;
constexpr int PREFILL_THREAD_NUM = 0; // 0 means using default thread number
constexpr int STEP_DECODE_THREAD_NUM = 0; // 1 is usually enough for step decode
constexpr int HEAD_NUM = 24;
constexpr int KV_CACHE_PREPARED_LENGTH = 512;
#define CPP_PRINT(msg) py::print("[C++] " + std::string(msg))
using OrtValueShapeType = std::vector<int64_t>;

class GSVEngine {
    public:
    GSVEngine(const std::string fsdec_path,
              const std::string sdec_path,
              const std::string sovits_path,
              bool use_gpu = false,
              bool use_npu = false,
              bool quantized = false){
        CPP_PRINT("GSVEngine initialized with directories:");
    }
    ~GSVEngine(){
        CPP_PRINT("GSVEngine destroyed");
    }

    py::array_t<float> infer(const py::dict ref_dict, const py::dict text_input, const py::dict sampling_params) {
        CPP_PRINT("Inference called with reference set size: " + std::to_string(ref_dict.size()));
        CPP_PRINT("Inference called with text set size: " + std::to_string(text_input.size()));
        // Dummy output for illustration
        std::vector<float> output_data = {0.0f, 1.0f, 2.0f};
        return py::array_t<float>(output_data.size(), output_data.data());
    }

    private:
    bool use_gpu_ = false;
    bool use_npu_ = false;
    bool quantized_ = false;
};


PYBIND11_MODULE(gsv_engine, m) {
    py::class_<GSVEngine>(m, "GSVEngine")
        .def(py::init<const py::set&>(), py::arg("gsv_settings"))
        .def("infer", &GSVEngine::infer, py::arg("ref_set"), py::arg("text_set"));
}