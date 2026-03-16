#pragma once
#include "tensor.hpp"
#include <vector>
#include <cstdint>

namespace quant {
extern bool g_qat_enabled;
extern int g_qat_bits;

void fake_quantize_inplace(Tensor& t);

// out_data entries are in [0, 2^g_qat_bits-1]
void post_training_quantize(const Tensor& t,
                            std::vector<uint8_t>& out_data,
                            float& scale_out);
} // namespace quant