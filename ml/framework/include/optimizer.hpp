#pragma once
#include <vector>
#include "autodiff.hpp"

class SGD {
public:
    explicit SGD(float lr);
    void step();
    void zero_grad();

private:
    float lr;
};
class AdamW {
public:
    AdamW(float lr, float beta1=0.9f, float beta2=0.999f, float eps=1e-8f,
          float weight_decay=0.01f, float clip_norm=1.0f);
    void step();
    void zero_grad();
    float lr;
private:
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float clip_norm;
    int t;
    std::vector<Tensor> m;
    std::vector<Tensor> v;
};