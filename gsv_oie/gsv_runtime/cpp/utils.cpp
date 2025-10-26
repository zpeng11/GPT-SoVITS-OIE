#include "utils.hpp"
#include <iostream>

#define CPP_PRINT(msg) py::print("[C++] " + std::string(msg))

std::vector<int> get_shape_from_numpy_array(const py::array& array) {
    std::vector<int> shape;
    for (ssize_t i = 0; i < array.ndim(); ++i) {
        shape.push_back(static_cast<int>(array.shape(i)));
    }
    return shape;
}

std::vector<ssize_t> to_ssize_t_vector(const std::vector<int>& vec) {
    std::vector<ssize_t> result;
    result.reserve(vec.size());
    for (const auto& v : vec) {
        result.push_back(static_cast<ssize_t>(v));
    }
    return result;
}

std::string shape_vector_to_string(const std::vector<int>& shape) {
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

std::vector<int32_t> transform_int64_to_int32(const int64_t* src, size_t size) {
    std::vector<int32_t> dst(size);

    constexpr int64_t MIN_VAL = std::numeric_limits<int32_t>::min();
    constexpr int64_t MAX_VAL = std::numeric_limits<int32_t>::max();

    auto converter = [MIN_VAL, MAX_VAL](int64_t val) -> int32_t {
        if (val < MIN_VAL) return std::numeric_limits<int32_t>::min();
        if (val > MAX_VAL) return std::numeric_limits<int32_t>::max();
        return static_cast<int32_t>(val);
    };
    std::transform(src, src + size, dst.begin(), converter);
    return dst;
}

std::vector<int64_t> transform_int32_to_int64(const int32_t* src, size_t size) {
    std::vector<int64_t> dst(size);

    auto converter = [](int32_t val) -> int64_t {
        return static_cast<int64_t>(val);
    };

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

// TensorDataGuard implementation
TensorDataGuard::TensorDataGuard(MNN::Tensor* tensor, bool is_writing) : tensor_(tensor), is_writing_(is_writing) {
    if(tensor_){
        ptr_ = tensor_->map( is_writing_ ? MNN::Tensor::MAP_TENSOR_WRITE : MNN::Tensor::MAP_TENSOR_READ, tensor_->getDimensionType());
    }
}

TensorDataGuard::TensorDataGuard(TensorDataGuard&& other) noexcept {
    if (this == &other) {
        return;
    }
    tensor_ = other.tensor_;
    ptr_ = other.ptr_;
    is_writing_ = other.is_writing_;
    other.tensor_ = nullptr;
}

TensorDataGuard& TensorDataGuard::operator=(TensorDataGuard&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    tensor_ = other.tensor_;
    ptr_ = other.ptr_;
    is_writing_ = other.is_writing_;
    other.tensor_ = nullptr;
    return *this;
}

void * TensorDataGuard::ptr() {
    return ptr_;
}

MNN::Tensor* TensorDataGuard::tensor() {
    return tensor_;
}

TensorDataGuard::~TensorDataGuard(){
    if(tensor_){
        tensor_->unmap(is_writing_ ? MNN::Tensor::MAP_TENSOR_WRITE : MNN::Tensor::MAP_TENSOR_READ, tensor_->getDimensionType(), ptr_);
    }
}