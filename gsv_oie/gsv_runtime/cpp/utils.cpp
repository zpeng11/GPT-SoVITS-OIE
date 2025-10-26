#include "utils.hpp"
#include <iostream>
#include <unistd.h>      // getauxval
#include <sys/auxv.h>    // AT_HWCAP2
#include <fstream>       // /proc fallback
#include <string>
#include <cstdint>
#include <cstddef>  // for std::size_t, but using ssize_t

// Forward declaration for intrinsics; include appropriate headers based on compiler
#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>  // SSE/AVX intrinsics
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>   // NEON intrinsics
#endif

// Conditional OpenMP support (define _OPENMP_USE or compile with -fopenmp)
#ifdef _OPENMP
    #include <omp.h>
    #define USE_OPENMP 1
#else
    #define USE_OPENMP 0
#endif

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
    if(UNLIKELY(array.dtype().is(py::dtype::of<int64_t>()))){
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

#ifndef HWCAP_FP16
#define HWCAP_FP16 (1UL << 24)  // ARM FP16 位掩码
#endif

bool has_fp16_support() {
    // 方法1: getauxval (AArch64 推荐，快速)
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    if (hwcap2 & HWCAP_FP16) {
        return true;
    }

    // 方法2: /proc/cpuinfo fallback (兼容性好)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.rfind("Features :", 0) == 0) {  // 注意冒号
            return line.find("fp16") != std::string::npos;
        }
    }
    return false;
}

#define I32_TO_I64_BLOCK_SIZE 512  // 块大小，可调节以优化缓存利用率

void process_block_int32_to_int64(int64_t* dst, const int32_t* src, ssize_t block_size);
void convert_vector_int32_to_int64(int64_t* dst, const int32_t* src, ssize_t size) {
    if (size <= 0) return;
    // Assume pointers are 64-byte aligned as per requirement
#if USE_OPENMP
    // OpenMP multi-threading: Parallelize over blocks (schedule static for contiguous data)
    #pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (ssize_t block_start = 0; block_start < size; block_start += I32_TO_I64_BLOCK_SIZE) {  // Block size tunable, e.g., 1024 elements
        ssize_t block_end = std::min(block_start + I32_TO_I64_BLOCK_SIZE, size);
        process_block_int32_to_int64(dst + block_start, src + block_start, block_end - block_start);
    }
#else
    // Single-thread fallback: Process entire range
    process_block_int32_to_int64(dst, src, size);
#endif
}

// Helper function to process a contiguous block with SIMD + unroll + prefetch
void process_block_int32_to_int64(int64_t* dst, const int32_t* src, ssize_t block_size) {
#if defined(__SSE2__)
    const ssize_t step = 8; // process 8 int32 per loop
    ssize_t i = 0;

    for (; i + step <= block_size; i += step) {
        _mm_prefetch(reinterpret_cast<const char*>(src + i + 32), _MM_HINT_T0);

        __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m128i lo64_0 = _mm_unpacklo_epi32(v0, _mm_srai_epi32(v0, 31));
        __m128i hi64_0 = _mm_unpackhi_epi32(v0, _mm_srai_epi32(v0, 31));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), lo64_0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i + 2), hi64_0);

        __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i + 4));
        __m128i lo64_1 = _mm_unpacklo_epi32(v1, _mm_srai_epi32(v1, 31));
        __m128i hi64_1 = _mm_unpackhi_epi32(v1, _mm_srai_epi32(v1, 31));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i + 4), lo64_1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i + 6), hi64_1);
    }

    for (; i < block_size; ++i)
        dst[i] = static_cast<int64_t>(src[i]);

#elif defined(__ARM_NEON)
    const ssize_t step = 8;
    ssize_t i = 0;

    for (; i + step <= block_size; i += step) {
        __builtin_prefetch(src + i + 32, 0, 3);

        int32x4_t v0 = vld1q_s32(src + i);
        vst1q_s64(dst + i, vmovl_s32(vget_low_s32(v0)));
        vst1q_s64(dst + i + 2, vmovl_s32(vget_high_s32(v0)));

        int32x4_t v1 = vld1q_s32(src + i + 4);
        vst1q_s64(dst + i + 4, vmovl_s32(vget_low_s32(v1)));
        vst1q_s64(dst + i + 6, vmovl_s32(vget_high_s32(v1)));
    }

    for (; i < block_size; ++i)
        dst[i] = static_cast<int64_t>(src[i]);

#else
    #pragma unroll 8
    for (ssize_t i = 0; i < block_size; ++i)
        dst[i] = static_cast<int64_t>(src[i]);
#endif
}

#define I64_TO_I32_BLOCK_SIZE 256  // 块大小，可调节以优化缓存利用率

void process_block_int64_to_int32(int32_t* dst, const int64_t* src, ssize_t block_size);
void convert_vector_int64_to_int32(int32_t* dst, const int64_t* src, ssize_t size) {
    if (size <= 0) return;
    // Assume pointers are 64-byte aligned as per requirement
#if USE_OPENMP
    // OpenMP multi-threading: Parallelize over blocks (schedule static for contiguous data)
    #pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (ssize_t block_start = 0; block_start < size; block_start += I64_TO_I32_BLOCK_SIZE) {  // Block size tunable, e.g., 1024 elements
        ssize_t block_end = std::min(block_start + I64_TO_I32_BLOCK_SIZE, size);
        process_block_int64_to_int32(dst + block_start, src + block_start, block_end - block_start);
    }
#else
    // Single-thread fallback: Process entire range
    process_block_int64_to_int32(dst, src, size);
#endif
}


// SIMD + Unroll + Prefetch version of int64 -> int32 conversion
void process_block_int64_to_int32(int32_t* dst, const int64_t* src, ssize_t block_size) {
#if defined(__SSE2__)
    const ssize_t simd_width = 2; // each __m128i holds 2 int64 values
    ssize_t i = 0;

    for (; i + simd_width * 2 <= block_size; i += simd_width * 2) {
        _mm_prefetch(reinterpret_cast<const char*>(src + i + 16), _MM_HINT_T0);

        // load 2x2 int64 = 4 int64 total
        __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i + 2));

        // convert: pack from 64-bit to 32-bit (truncate)
        __m128i packed = _mm_unpacklo_epi64(
            _mm_shuffle_epi32(v0, _MM_SHUFFLE(2, 0, 2, 0)),  // shuffle low 32 bits
            _mm_shuffle_epi32(v1, _MM_SHUFFLE(2, 0, 2, 0))
        );
        // 这里上面的 shuffle 方案是 trick，只取 int64 的低 32 位。

        // 存储低32位
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), packed);
    }

    // Tail
    for (; i < block_size; ++i) {
        dst[i] = static_cast<int32_t>(src[i]);
    }

#elif defined(__aarch64__) && defined(__ARM_NEON)
    const ssize_t simd_width = 2; // int64x2_t holds 2 values
    ssize_t i = 0;

    for (; i + simd_width * 2 <= block_size; i += simd_width * 2) {
        __builtin_prefetch(src + i + 16, 0, 3);

        // load 4x int64
        int64x2_t v0 = vld1q_s64(src + i);
        int64x2_t v1 = vld1q_s64(src + i + 2);

        // narrow to int32 (truncate)
        int32x2_t n0 = vmovn_s64(v0);
        int32x2_t n1 = vmovn_s64(v1);

        // combine to one int32x4_t
        int32x4_t out = vcombine_s32(n0, n1);

        vst1q_s32(dst + i, out);
    }

    for (; i < block_size; ++i) {
        dst[i] = static_cast<int32_t>(src[i]);
    }

#else
    #pragma unroll 8
    for (ssize_t i = 0; i < block_size; ++i) {
        dst[i] = static_cast<int32_t>(src[i]);
    }
#endif
}

std::vector<int32_t> transform_int64_to_int32(const int64_t* src, size_t size) {
    std::vector<int32_t> dst(size);
    convert_vector_int64_to_int32(dst.data(), src, static_cast<ssize_t>(size));
    return dst;
}

std::vector<int64_t> transform_int32_to_int64(const int32_t* src, size_t size) {
    std::vector<int64_t> dst(size);
    convert_vector_int32_to_int64(dst.data(), src, static_cast<ssize_t>(size));
    return dst;
}