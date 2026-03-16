#pragma once
#include "tensor.hpp"
#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>


struct ADTensor {
    Tensor val;
    Tensor grad;
    std::vector<std::pair<std::shared_ptr<ADTensor>, std::function<void()>>> deps;
    ADTensor(int rows, int cols);
    ADTensor(const Tensor& t);
    ADTensor(const std::vector<int>& shape);
    void backward();
};

std::shared_ptr<ADTensor> make_ad(const Tensor& t);

std::shared_ptr<ADTensor> add(const std::shared_ptr<ADTensor>& a,
                              const std::shared_ptr<ADTensor>& b);
std::shared_ptr<ADTensor> mul(const std::shared_ptr<ADTensor>& a,
                              const std::shared_ptr<ADTensor>& b);
std::shared_ptr<ADTensor> scalar_mul(const std::shared_ptr<ADTensor>& a,
                                     float s);
std::shared_ptr<ADTensor> matmul(const std::shared_ptr<ADTensor>& a,
                                 const std::shared_ptr<ADTensor>& b);
std::shared_ptr<ADTensor> tanh_ad(const std::shared_ptr<ADTensor>& a);
std::shared_ptr<ADTensor> exp_ad(const std::shared_ptr<ADTensor>& a);
std::shared_ptr<ADTensor> log_ad(const std::shared_ptr<ADTensor>& a);
std::shared_ptr<ADTensor> sqrt_ad(const std::shared_ptr<ADTensor>& a);
std::shared_ptr<ADTensor> reciprocal(const std::shared_ptr<ADTensor>& a);
std::shared_ptr<ADTensor> sub(const std::shared_ptr<ADTensor>& a,
                              const std::shared_ptr<ADTensor>& b);
std::shared_ptr<ADTensor> sum(const std::shared_ptr<ADTensor>& a);
void register_parameter(const std::shared_ptr<ADTensor>& p);
std::vector<std::shared_ptr<ADTensor>>& get_parameters();
void clear_parameters();
std::shared_ptr<ADTensor> transpose(const std::shared_ptr<ADTensor>& a);
std::shared_ptr<ADTensor> slice(const std::shared_ptr<ADTensor>& a,
                                 int row_offset, int row_count);
std::shared_ptr<ADTensor> concat(const std::vector<std::shared_ptr<ADTensor>>& parts);

// N-dim operations
std::shared_ptr<ADTensor> relu_ad(const std::shared_ptr<ADTensor>& a);
std::shared_ptr<ADTensor> sigmoid_ad(const std::shared_ptr<ADTensor>& a);
std::shared_ptr<ADTensor> reshape_ad(const std::shared_ptr<ADTensor>& a,
                                      const std::vector<int>& new_shape);
std::shared_ptr<ADTensor> flatten_ad(const std::shared_ptr<ADTensor>& a,
                                      int start_dim = 0, int end_dim = -1);
