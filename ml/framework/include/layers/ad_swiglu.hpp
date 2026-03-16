#pragma once
#include "autodiff.hpp"

// SwiGLU Feed-Forward Network as used in LLaMA / DeepSeek
// Replaces GELU FFN with gated linear unit: SwiGLU(x) = (Swish(xW1) * xV) W2
// where Swish(x) = x * sigmoid(x)
class ADSwiGLU {
public:
    ADSwiGLU(int embed_dim, int hidden_dim);
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& x);

private:
    std::shared_ptr<ADTensor> W_gate;  // [hidden_dim x embed_dim] - gate projection
    std::shared_ptr<ADTensor> W_up;    // [hidden_dim x embed_dim] - up projection
    std::shared_ptr<ADTensor> W_down;  // [embed_dim x hidden_dim] - down projection

    // Cached ones for bias-free broadcast
    mutable Tensor cached_ones{1, 1};
    mutable int cached_seq_len = -1;
};
