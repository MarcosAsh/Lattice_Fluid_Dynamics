#pragma once
#include "autodiff.hpp"
#include <string>

// Save all registered parameters to a binary file.
// Format: magic (4B) | version (u32) | param_count (u32)
// Per param: ndim (u32) | shape (u32 x ndim) | data (f32 x numel)
bool save_weights(const std::string& path);

// Load weights from a binary file into the registered parameters.
// Parameters must already be created (same architecture) so that
// the count and shapes match what was saved.
bool load_weights(const std::string& path);
