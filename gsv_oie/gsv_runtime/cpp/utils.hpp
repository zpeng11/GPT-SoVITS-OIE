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
#include <fp16.h>
#include <cstddef>
#ifdef _WIN32
using ssize_t = std::ptrdiff_t;
#endif

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
std::vector<int64_t> to_ort_shape_vector(const std::vector<int>& vec);
std::vector<int64_t> to_ort_shape_vector(const std::vector<ssize_t>& vec);
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
    std::vector<int> shape();
    ~TensorDataGuard();

private:
    MNN::Tensor* tensor_ = nullptr;
    bool is_writing_ = false;
    void * ptr_ = nullptr;
};

// Tensor utility functions for MNNInferenceEngineInterpreter
void copy_numpy_array_to_tensor(const py::array& array, MNN::Tensor* tensor);
py::array create_numpy_array_from_tensor(MNN::Tensor* tensor);

#if defined(__GNUC__) || defined(__clang__)
    // GCC/Clang: 使用 __builtin_expect
    #define LIKELY(expr) __builtin_expect(!!(expr), 1)
    #define UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#elif defined(_MSC_VER)
    // MSVC: 无内置提示，回退到原表达式（在 -O2 等优化下仍有效）
    #define LIKELY(expr) (expr)
    #define UNLIKELY(expr) (expr)
#else
    // 其他编译器：默认回退
    #define LIKELY(expr) (expr)
    #define UNLIKELY(expr) (expr)
#endif

void convert_vector_fp32_to_fp16(uint16_t* dst, const float* src, size_t size);
void convert_vector_fp16_to_fp32(float* dst, const uint16_t* src, size_t size);
void convert_vector_int64_to_int32(int32_t* dst, const int64_t* src, ssize_t size);
void convert_vector_int32_to_int64(int64_t* dst, const int32_t* src, ssize_t size);

#include <vector>
#include <cstdlib>  // For std::aligned_alloc and std::free
#include <memory>   // For std::assume_aligned (optional, for compiler hints)
#include <new>      // std::bad_alloc

template <typename T, std::size_t Alignment = 16>
struct aligned_allocator {
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };

    aligned_allocator() noexcept = default;

    template <typename U>
    constexpr aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

    // Required: Full allocate signature (hint ignored here, but present for compliance)
    pointer allocate(size_type n, const_pointer hint = pointer()) {
        if (n == 0) return nullptr;
        if (size_type(-1) / sizeof(T) < n) {
            throw std::bad_alloc{};
        }
        void* p = std::aligned_alloc(Alignment, n * sizeof(T));
        if (!p) throw std::bad_alloc{};
        return static_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type n) noexcept {
        std::free(p);
    }

    // Optional but recommended: Use allocator_traits defaults for construct/destroy
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        static_assert(sizeof(U) == sizeof(T), "Mismatched types in construct");
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template <typename U>
    void destroy(U* p) {
        p->~U();
    }

    // Required for move/copy propagation
    using propagate_on_container_move_assignment = std::true_type;
};

using AlignedVec = std::vector<uint8_t, aligned_allocator<uint8_t, 16>>;