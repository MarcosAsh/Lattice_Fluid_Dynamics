#pragma once
#include <vector>
#include "autodiff.hpp"

// loss = -log(softmax(logits)[target])
float softmax_cross_entropy(const std::vector<float>& logits,
                            int target,
                            std::vector<float>& grad);

// MSE loss for regression: loss = mean((pred - target)^2)
// pred: [output_dim x batch], target: [output_dim x batch]
// Returns a scalar ADTensor so gradients flow through autodiff.
std::shared_ptr<ADTensor> mse_loss(const std::shared_ptr<ADTensor>& pred,
                                   const std::shared_ptr<ADTensor>& target);