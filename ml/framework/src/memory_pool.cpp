#include "memory_pool.hpp"
#include <cstdlib>
#include <stdexcept>

// Retrieve singleton instance (intentionally leaked to avoid static destruction order issues:
// other statics like the AD parameter registry may outlive this singleton and still
// need to deallocate through it during program exit)
UnifiedMemoryManager& UnifiedMemoryManager::instance() {
    static auto* inst = new UnifiedMemoryManager();
    return *inst;
}

UnifiedMemoryManager::UnifiedMemoryManager() = default;

UnifiedMemoryManager::~UnifiedMemoryManager() {
    std::lock_guard<std::mutex> lock(mu_);
    // Free any remaining fallback (heap) allocations before freeing the pool
    for (auto& kv : allocations_) {
        void* ptr = kv.first;
        bool is_pool = pool_ && (ptr >= pool_ && ptr < pool_ + max_on_chip_);
        if (!is_pool) {
            std::free(ptr);
        }
    }
    allocations_.clear();
    // Now free the pool itself
    if (pool_) {
        std::free(pool_);
        pool_ = nullptr;
    }
}

void UnifiedMemoryManager::init(std::size_t max_on_chip_bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    if (pool_) {
        // Already initialized
        return;
    }
    pool_ = static_cast<char*>(std::malloc(max_on_chip_bytes));
    if (!pool_) throw std::bad_alloc();
    max_on_chip_ = max_on_chip_bytes;
    allocated_on_chip_ = 0;
}

void* UnifiedMemoryManager::allocate(std::size_t bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    if (pool_ && allocated_on_chip_ + bytes <= max_on_chip_) {
        void* ptr = pool_ + allocated_on_chip_;
        allocations_[ptr] = bytes;
        allocated_on_chip_ += bytes;
        return ptr;
    } else {
        void* ptr = std::malloc(bytes);
        if (!ptr) throw std::bad_alloc();
        allocations_[ptr] = bytes;
        return ptr;
    }
}

void UnifiedMemoryManager::deallocate(void* ptr, std::size_t /*bytes*/) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        bool is_pool = pool_ && (ptr >= pool_ && ptr < pool_ + max_on_chip_);
        if (!is_pool) {
            // Only free heap (fallback) allocations; pool memory uses a bump
            // allocator and cannot reclaim interior blocks -- it is reclaimed
            // only when the entire pool is freed.
            std::free(ptr);
        }
        allocations_.erase(it);
    }
}