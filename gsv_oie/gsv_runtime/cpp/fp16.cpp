#include <unistd.h>      // getauxval
#include <sys/auxv.h>    // AT_HWCAP2
#include <fstream>       // /proc fallback
#include <string>

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

#include "fp16.h"        // FP16 库头 (header-only)
// #include <arm_neon.h>    // NEON intrinsics
#include <vector>
#include <cstdint>

/*

// 假设您的矩阵是 float* input, size_t rows, size_t cols
void convert_matrix_fp32_to_fp16(const float* input, uint16_t* output,
                                 size_t rows, size_t cols) {
    static bool fp16_enabled = has_fp16_support();  // 静态，一次检测

    if (fp16_enabled) {
        // 硬件路径: 用 __fp16 + NEON FP16 (v8.2+)
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wpsabi"  // 忽略 ABI 警告
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; j += 8) {  // NEON 处理 8 个 FP16
                float32x4_t lo = vld1q_f32(&input[i * cols + j]);
                float32x4_t hi = vld1q_f32(&input[i * cols + j + 4]);
                // 转换为 FP16 (需 v8.2+)
                float16x4_t lo_h = vcvt_f16_f32(lo);
                float16x4_t hi_h = vcvt_f16_f32(hi);
                // 打包到 uint16x8_t
                uint16x8_t packed = vcombine_u16(vreinterpretq_u16_f16(lo_h), vreinterpretq_u16_f16(hi_h));
                vst1q_u16(&output[i * cols + j], packed);
            }
        }
        #pragma clang diagnostic pop
    } else {
        // 软件 fallback: FP16 库 (兼容 v8.0)
        for (size_t i = 0; i < rows * cols; ++i) {
            output[i] = fp16::float_to_half(input[i]);  // IEEE 格式，处理 NaN/Inf
        }
    }
}

*/