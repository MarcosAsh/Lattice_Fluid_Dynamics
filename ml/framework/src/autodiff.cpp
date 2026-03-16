#include "autodiff.hpp"
#include <functional>
#include <vector>
#include <unordered_set>
#include <mutex>

ADTensor::ADTensor(int rows, int cols)
    : val(rows, cols), grad(rows, cols) {
    grad.fill(0.0f);
}

ADTensor::ADTensor(const Tensor& t)
    : val(t), grad(t.shape) {
    grad.fill(0.0f);
}

ADTensor::ADTensor(const std::vector<int>& shape)
    : val(shape), grad(shape) {
    grad.fill(0.0f);
}

void ADTensor::backward() {
    // Initialize gradient of the root node
    grad.fill(1.0f);
    // Build topological order
    std::vector<ADTensor*> topo;
    std::unordered_set<ADTensor*> visited;
    std::function<void(ADTensor*)> dfs = [&](ADTensor* node) {
        if (!visited.insert(node).second) return;
        for (auto& dep : node->deps) {
            dfs(dep.first.get());
        }
        topo.push_back(node);
    };
    dfs(this);
    // Backpropagate in reverse topological order
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        ADTensor* node = *it;
        for (auto& dep : node->deps) {
            // dep.second applies local gradient to dep.first->grad
            dep.second();
        }
    }
    // Clear dependencies to release the computation graph and free memory
    for (ADTensor* node : topo) {
        node->deps.clear();
    }
}

std::shared_ptr<ADTensor> make_ad(const Tensor& t) {
    return std::make_shared<ADTensor>(t);
}
// Parameter registry implementation
namespace {
    std::vector<std::shared_ptr<ADTensor>> param_list;
    std::mutex param_mutex;
}
void register_parameter(const std::shared_ptr<ADTensor>& p) {
    std::lock_guard<std::mutex> lock(param_mutex);
    param_list.push_back(p);
}
std::vector<std::shared_ptr<ADTensor>>& get_parameters() {
    return param_list;
}
void clear_parameters() {
    std::lock_guard<std::mutex> lock(param_mutex);
    param_list.clear();
}

std::shared_ptr<ADTensor> add(const std::shared_ptr<ADTensor>& a,
                              const std::shared_ptr<ADTensor>& b) {
    // elementwise addition
    Tensor v = a->val + b->val;
    auto out = std::make_shared<ADTensor>(v);
    // grad_a += grad_out
    out->deps.emplace_back(a, [a, out]() {
        for (size_t i = 0; i < a->grad.data.size(); ++i) {
            a->grad.data[i] += out->grad.data[i];
        }
    });
    // grad_b += grad_out
    out->deps.emplace_back(b, [b, out]() {
        for (size_t i = 0; i < b->grad.data.size(); ++i) {
            b->grad.data[i] += out->grad.data[i];
        }
    });
    return out;
}
// natural logarithm
std::shared_ptr<ADTensor> log_ad(const std::shared_ptr<ADTensor>& a) {
    Tensor v(a->val.rows, a->val.cols);
    for (size_t i = 0; i < v.data.size(); ++i) {
        v.data[i] = std::log(a->val.data[i]);
    }
    auto out = std::make_shared<ADTensor>(v);
    out->deps.emplace_back(a, [a, out]() {
        for (size_t i = 0; i < a->grad.data.size(); ++i) {
            // d/dx log(x) = 1/x
            a->grad.data[i] += out->grad.data[i] / a->val.data[i];
        }
    });
    return out;
}
// Transpose an ADTensor
std::shared_ptr<ADTensor> transpose(const std::shared_ptr<ADTensor>& a) {
    Tensor v = a->val.transpose();
    auto out = std::make_shared<ADTensor>(v);
    out->deps.emplace_back(a, [a, out]() {
        // a->grad += out->grad^T
        Tensor go = out->grad;
        Tensor goT = go.transpose();
        for (size_t i = 0; i < a->grad.data.size(); ++i) {
            a->grad.data[i] += goT.data[i];
        }
    });
    return out;
}
// Slice rows [row_offset .. row_offset+row_count) of a
std::shared_ptr<ADTensor> slice(const std::shared_ptr<ADTensor>& a,
                                 int row_offset, int row_count) {
    int cols = a->val.cols;
    Tensor v(row_count, cols);
    for (int i = 0; i < row_count; ++i) {
        for (int j = 0; j < cols; ++j) {
            v.data[i * cols + j] = a->val.data[(row_offset + i) * cols + j];
        }
    }
    auto out = std::make_shared<ADTensor>(v);
    out->deps.emplace_back(a, [a, out, row_offset, row_count]() {
        int cols = a->val.cols;
        for (int i = 0; i < row_count; ++i) {
            for (int j = 0; j < cols; ++j) {
                a->grad.data[(row_offset + i) * cols + j] +=
                    out->grad.data[i * cols + j];
            }
        }
    });
    return out;
}
// Concatenate parts vertically (row-wise)
std::shared_ptr<ADTensor> concat(const std::vector<std::shared_ptr<ADTensor>>& parts) {
    if (parts.empty()) throw std::runtime_error("concat: no parts");
    int total_rows = 0;
    int cols = parts[0]->val.cols;
    for (auto& p : parts) {
        if (p->val.cols != cols) throw std::runtime_error("concat: mismatched cols");
        total_rows += p->val.rows;
    }
    Tensor v(total_rows, cols);
    int row_off = 0;
    for (auto& p : parts) {
        for (int i = 0; i < p->val.rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                v.data[(row_off + i) * cols + j] = p->val.data[i * cols + j];
            }
        }
        row_off += p->val.rows;
    }
    auto out = std::make_shared<ADTensor>(v);
    for (auto& p : parts) {
        int row_off_p = 0;
        for (auto& prev : parts) {
            if (prev.get() == p.get()) break;
            row_off_p += prev->val.rows;
        }
        out->deps.emplace_back(p, [p, out, row_off_p]() {
            int cols = p->val.cols;
            for (int i = 0; i < p->val.rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    p->grad.data[i * cols + j] +=
                        out->grad.data[(row_off_p + i) * cols + j];
                }
            }
        });
    }
    return out;
}

std::shared_ptr<ADTensor> mul(const std::shared_ptr<ADTensor>& a,
                              const std::shared_ptr<ADTensor>& b) {
    // elementwise multiply
    Tensor v(a->val.rows, a->val.cols);
    for (size_t i = 0; i < a->val.data.size(); ++i) {
        v.data[i] = a->val.data[i] * b->val.data[i];
    }
    auto out = std::make_shared<ADTensor>(v);
    // grad_a += grad_out * b
    out->deps.emplace_back(a, [a, b, out]() {
        for (size_t i = 0; i < a->grad.data.size(); ++i) {
            a->grad.data[i] += b->val.data[i] * out->grad.data[i];
        }
    });
    // grad_b += grad_out * a
    out->deps.emplace_back(b, [a, b, out]() {
        for (size_t i = 0; i < b->grad.data.size(); ++i) {
            b->grad.data[i] += a->val.data[i] * out->grad.data[i];
        }
    });
    return out;
}

std::shared_ptr<ADTensor> scalar_mul(const std::shared_ptr<ADTensor>& a,
                                     float s) {
    Tensor v = a->val;
    for (auto& x : v.data) x *= s;
    auto out = std::make_shared<ADTensor>(v);
    // grad_a += s * grad_out
    out->deps.emplace_back(a, [a, out, s]() {
        for (size_t i = 0; i < a->grad.data.size(); ++i) {
            a->grad.data[i] += s * out->grad.data[i];
        }
    });
    return out;
}

std::shared_ptr<ADTensor> matmul(const std::shared_ptr<ADTensor>& a,
                                 const std::shared_ptr<ADTensor>& b) {
    Tensor v = a->val.matmul(b->val);
    auto out = std::make_shared<ADTensor>(v);
    // grad w.r.t a: grad_out.matmul(b^T)
    out->deps.emplace_back(a, [a, b, out]() {
        Tensor grad_out = out->grad;
        Tensor bT = b->val.transpose();
        Tensor ga = grad_out.matmul(bT);
        for (size_t i = 0; i < a->grad.data.size(); ++i) {
            a->grad.data[i] += ga.data[i];
        }
    });
    // grad w.r.t b: a^T.matmul(grad_out)
    out->deps.emplace_back(b, [a, b, out]() {
        Tensor grad_out = out->grad;
        Tensor aT = a->val.transpose();
        Tensor gb = aT.matmul(grad_out);
        for (size_t i = 0; i < b->grad.data.size(); ++i) {
            b->grad.data[i] += gb.data[i];
        }
    });
    return out;
}

std::shared_ptr<ADTensor> tanh_ad(const std::shared_ptr<ADTensor>& a) {
    Tensor v(a->val.rows, a->val.cols);
    for (size_t i = 0; i < v.data.size(); ++i) {
        v.data[i] = std::tanh(a->val.data[i]);
    }
    auto out = std::make_shared<ADTensor>(v);
    // grad_a += (1 - tanh^2(x)) * grad_out
    out->deps.emplace_back(a, [a, out]() {
        for (size_t i = 0; i < a->grad.data.size(); ++i) {
            float y = out->val.data[i];
            a->grad.data[i] += (1.0f - y * y) * out->grad.data[i];
        }
    });
    return out;
}

std::shared_ptr<ADTensor> exp_ad(const std::shared_ptr<ADTensor>& a) {
    Tensor v(a->val.rows, a->val.cols);
    for (size_t i = 0; i < v.data.size(); ++i) {
        v.data[i] = std::exp(a->val.data[i]);
    }
    auto out = std::make_shared<ADTensor>(v);
    // grad_a += exp(x) * grad_out
    out->deps.emplace_back(a, [a, out]() {
        for (size_t i = 0; i < a->grad.data.size(); ++i) {
            a->grad.data[i] += out->val.data[i] * out->grad.data[i];
        }
    });
    return out;
}
// sqrt(x)
std::shared_ptr<ADTensor> sqrt_ad(const std::shared_ptr<ADTensor>& a) {
    Tensor v(a->val.rows, a->val.cols);
    for (size_t i = 0; i < v.data.size(); ++i) {
        v.data[i] = std::sqrt(a->val.data[i]);
    }
    auto out = std::make_shared<ADTensor>(v);
    out->deps.emplace_back(a, [a, out]() {
        for (size_t i = 0; i < a->grad.data.size(); ++i) {
            float y = out->val.data[i];
            a->grad.data[i] += (out->grad.data[i] * 0.5f) / y;
        }
    });
    return out;
}
// 1/x
std::shared_ptr<ADTensor> reciprocal(const std::shared_ptr<ADTensor>& a) {
    Tensor v(a->val.rows, a->val.cols);
    for (size_t i = 0; i < v.data.size(); ++i) {
        v.data[i] = 1.0f / a->val.data[i];
    }
    auto out = std::make_shared<ADTensor>(v);
    out->deps.emplace_back(a, [a, out]() {
        for (size_t i = 0; i < a->grad.data.size(); ++i) {
            float ai = a->val.data[i];
            a->grad.data[i] -= out->grad.data[i] / (ai * ai);
        }
    });
    return out;
}
// a - b
std::shared_ptr<ADTensor> sub(const std::shared_ptr<ADTensor>& a,
                              const std::shared_ptr<ADTensor>& b) {
    Tensor v(a->val.rows, a->val.cols);
    for (size_t i = 0; i < v.data.size(); ++i)
        v.data[i] = a->val.data[i] - b->val.data[i];
    auto out = std::make_shared<ADTensor>(v);
    // grad a += grad_out
    out->deps.emplace_back(a, [a, out]() {
        for (size_t i = 0; i < a->grad.data.size(); ++i)
            a->grad.data[i] += out->grad.data[i];
    });
    // grad b -= grad_out
    out->deps.emplace_back(b, [b, out]() {
        for (size_t i = 0; i < b->grad.data.size(); ++i)
            b->grad.data[i] -= out->grad.data[i];
    });
    return out;
}
// Sum all elements to scalar
std::shared_ptr<ADTensor> sum(const std::shared_ptr<ADTensor>& a) {
    float s = 0.0f;
    for (float v : a->val.data) s += v;
    Tensor v(1, 1);
    v.data[0] = s;
    auto out = std::make_shared<ADTensor>(v);
    out->deps.emplace_back(a, [a, out]() {
        float d = out->grad.data[0];
        for (float &g : a->grad.data) g += d;
    });
    return out;
}

// ReLU: max(0, x)
std::shared_ptr<ADTensor> relu_ad(const std::shared_ptr<ADTensor>& a) {
    Tensor v(a->val.shape);
    for (size_t i = 0; i < v.data.size(); ++i) {
        v.data[i] = a->val.data[i] > 0.0f ? a->val.data[i] : 0.0f;
    }
    auto out = std::make_shared<ADTensor>(v);
    out->deps.emplace_back(a, [a, out]() {
        for (size_t i = 0; i < a->grad.data.size(); ++i) {
            if (a->val.data[i] > 0.0f) {
                a->grad.data[i] += out->grad.data[i];
            }
        }
    });
    return out;
}

// Sigmoid: 1 / (1 + exp(-x))
std::shared_ptr<ADTensor> sigmoid_ad(const std::shared_ptr<ADTensor>& a) {
    Tensor v(a->val.shape);
    for (size_t i = 0; i < v.data.size(); ++i) {
        v.data[i] = 1.0f / (1.0f + std::exp(-a->val.data[i]));
    }
    auto out = std::make_shared<ADTensor>(v);
    out->deps.emplace_back(a, [a, out]() {
        for (size_t i = 0; i < a->grad.data.size(); ++i) {
            float s = out->val.data[i];
            a->grad.data[i] += out->grad.data[i] * s * (1.0f - s);
        }
    });
    return out;
}

// Reshape (view)
std::shared_ptr<ADTensor> reshape_ad(const std::shared_ptr<ADTensor>& a,
                                      const std::vector<int>& new_shape) {
    Tensor v = a->val.reshape(new_shape);
    auto out = std::make_shared<ADTensor>(v);
    auto orig_shape = a->val.shape;
    out->deps.emplace_back(a, [a, out, orig_shape]() {
        // Gradient flows through unchanged - just reshape back
        for (size_t i = 0; i < a->grad.data.size(); ++i) {
            a->grad.data[i] += out->grad.data[i];
        }
    });
    return out;
}

// Flatten
std::shared_ptr<ADTensor> flatten_ad(const std::shared_ptr<ADTensor>& a,
                                      int start_dim, int end_dim) {
    Tensor v = a->val.flatten(start_dim, end_dim);
    auto out = std::make_shared<ADTensor>(v);
    out->deps.emplace_back(a, [a, out]() {
        for (size_t i = 0; i < a->grad.data.size(); ++i) {
            a->grad.data[i] += out->grad.data[i];
        }
    });
    return out;
}