#include "layers/ad_swiglu.hpp"
#include <random>
#include <cmath>

// Custom autodiff op: swish(x) = x * sigmoid(x)
static std::shared_ptr<ADTensor> swish_ad(const std::shared_ptr<ADTensor>& a) {
    Tensor v(a->val.rows, a->val.cols);
    for (size_t i = 0; i < v.data.size(); ++i) {
        float x = a->val.data[i];
        float sig = 1.0f / (1.0f + std::exp(-x));
        v.data[i] = x * sig;
    }
    auto out = std::make_shared<ADTensor>(v);
    out->deps.emplace_back(a, [a, out]() {
        for (size_t i = 0; i < a->grad.data.size(); ++i) {
            float x = a->val.data[i];
            float sig = 1.0f / (1.0f + std::exp(-x));
            // d/dx [x * sig(x)] = sig(x) + x * sig(x) * (1 - sig(x))
            //                    = sig(x) * (1 + x * (1 - sig(x)))
            float grad = sig * (1.0f + x * (1.0f - sig));
            a->grad.data[i] += grad * out->grad.data[i];
        }
    });
    return out;
}

ADSwiGLU::ADSwiGLU(int embed_dim, int hidden_dim) {
    Tensor tWg(hidden_dim, embed_dim);
    Tensor tWu(hidden_dim, embed_dim);
    Tensor tWd(embed_dim, hidden_dim);

    std::mt19937 gen(std::random_device{}());
    float r1 = std::sqrt(6.0f / (embed_dim + hidden_dim));
    std::uniform_real_distribution<float> dist1(-r1, r1);
    for (auto& v : tWg.data) v = dist1(gen);
    for (auto& v : tWu.data) v = dist1(gen);

    float r2 = std::sqrt(6.0f / (hidden_dim + embed_dim));
    std::uniform_real_distribution<float> dist2(-r2, r2);
    for (auto& v : tWd.data) v = dist2(gen);

    W_gate = make_ad(tWg); register_parameter(W_gate);
    W_up   = make_ad(tWu); register_parameter(W_up);
    W_down = make_ad(tWd); register_parameter(W_down);
}

std::shared_ptr<ADTensor> ADSwiGLU::forward(const std::shared_ptr<ADTensor>& x) {
    // Gate path: swish(x @ W_gate^T)
    auto gate = matmul(W_gate, x);
    auto gate_act = swish_ad(gate);

    // Up path: x @ W_up^T
    auto up = matmul(W_up, x);

    // Element-wise product of gate and up
    auto hidden = mul(gate_act, up);

    // Down projection
    return matmul(W_down, hidden);
}
