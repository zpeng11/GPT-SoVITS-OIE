#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/MathOp.hpp>
#include <memory>
#include <string>
#include <filesystem>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <cstring>

namespace py = pybind11;

// NonCopyable class for preventing object copying
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

// Common utility functions
std::vector<int> get_shape_from_numpy_array(const py::array& array);
std::vector<ssize_t> to_ssize_t_vector(const std::vector<int>& vec);
std::vector<int32_t> transform_int64_to_int32(const int64_t* src, size_t size);
std::vector<int64_t> transform_int32_to_int64(const int32_t* src, size_t size);
std::string shape_vector_to_string(const std::vector<int>& shape);
MNN::Express::VARP make_varp_from_numpy_array(const py::array& array);
py::array make_numpy_array_from_varp(const MNN::Express::VARP varp);

// Template function for creating Python-managed arrays
template<typename T>
py::array_t<T> create_python_managed_array(const std::vector<ssize_t>& size) {
    ssize_t total_size = 1;
    for (const auto& dim : size) {
        total_size *= dim;
    }
    T* data = new T[total_size];
    auto capsule = py::capsule(data, [](void* ptr) {
        delete[] static_cast<T*>(ptr);
    });
    auto arr = py::array_t<T>(
        size,
        data,
        capsule
    );
    return std::move(arr);
}

// Common validation function
inline void check_dict_key(const py::dict& dict, const std::string& key) {
    if (!dict.contains(key)) {
        throw std::runtime_error("Input dictionary missing required key: " + key);
    }
}

// TensorDataGuard class for managing MNN tensor memory
class TensorDataGuard : NonCopyable {
public:
    TensorDataGuard(MNN::Tensor* tensor, bool is_writing = false);
    TensorDataGuard(TensorDataGuard&& other) noexcept;
    TensorDataGuard& operator=(TensorDataGuard&& other) noexcept;
    void * ptr();
    MNN::Tensor* tensor();
    ~TensorDataGuard();

private:
    MNN::Tensor* tensor_ = nullptr;
    bool is_writing_ = false;
    void * ptr_ = nullptr;
};

// Tensor utility functions for MNNInferenceEngineInterpreter
void copy_numpy_array_to_tensor(const py::array& array, MNN::Tensor* tensor);
py::array create_numpy_array_from_tensor(MNN::Tensor* tensor);