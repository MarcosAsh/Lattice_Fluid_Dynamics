#include "quantization.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace quant {
bool g_qat_enabled = false;
int g_qat_bits = 8;

void fake_quantize_inplace(Tensor& t) {
    if (!g_qat_enabled) return;
    // compute min and max
    float mn = std::numeric_limits<float>::infinity();
    float mx = -mn;
    for (float v : t.data) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    int levels = (1 << g_qat_bits) - 1;
    float range = mx - mn;
    if (range < 1e-8f) return;  // uniform tensor, nothing to quantize
    float scale = levels / range;
    // quantize and dequantize
    for (auto& v : t.data) {
        float q = std::round((v - mn) * scale);
        q = std::min<float>(std::max<float>(q, 0), levels);
        v = q / scale + mn;
    }
}

void post_training_quantize(const Tensor& t,
                            std::vector<uint8_t>& out_data,
                            float& scale_out) {
    // compute min and max
    float mn = std::numeric_limits<float>::infinity();
    float mx = -mn;
    for (float v : t.data) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    int levels = (1 << g_qat_bits) - 1;
    float range = mx - mn;
    if (range < 1e-8f) range = 1e-8f;
    scale_out = levels / range;
    size_t N = t.data.size();
    out_data.resize(N);
    for (size_t i = 0; i < N; ++i) {
        float q = std::round((t.data[i] - mn) * scale_out);
        int qi = static_cast<int>(q);
        qi = std::min(std::max(qi, 0), levels);
        out_data[i] = static_cast<uint8_t>(qi);
    }
}
} // namespace quant