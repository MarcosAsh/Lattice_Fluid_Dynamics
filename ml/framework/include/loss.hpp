#pragma once
#include <vector>
// loss = -log(softmax(logits)[target])
float softmax_cross_entropy(const std::vector<float>& logits,
                            int target,
                            std::vector<float>& grad);