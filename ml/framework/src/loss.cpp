#include "loss.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cassert>

float softmax_cross_entropy(const std::vector<float>& logits,
                            int target,
                            std::vector<float>& grad) {
    int V = (int)logits.size();
    if (target < 0 || target >= V) {
        grad.assign(V, 0.0f);
        return 0.0f;
    }
    grad.resize(V);
    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> exps(V);
    float sum_exp = 0.0f;
    for (int i = 0; i < V; ++i) {
        exps[i] = std::exp(logits[i] - max_logit);
        sum_exp += exps[i];
    }
    float loss = -std::log(exps[target] / sum_exp);
    for (int i = 0; i < V; ++i) {
        float p = exps[i] / sum_exp;
        grad[i] = p - (i == target ? 1.0f : 0.0f);
    }
    return loss;
}

std::shared_ptr<ADTensor> mse_loss(const std::shared_ptr<ADTensor>& pred,
                                   const std::shared_ptr<ADTensor>& target) {
    assert(pred->val.data.size() == target->val.data.size());
    // diff = pred - target
    auto diff = sub(pred, target);
    // sq = diff * diff  (element-wise)
    auto sq = mul(diff, diff);
    // mean = sum(sq) / N
    int N = static_cast<int>(pred->val.data.size());
    auto total = sum(sq);
    return scalar_mul(total, 1.0f / N);
}