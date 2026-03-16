#pragma once
#include <cstddef>
#include <mutex>
#include <unordered_map>

class UnifiedMemoryManager {
public:
    static UnifiedMemoryManager& instance();

    void init(std::size_t max_on_chip_bytes);

    // Falls back to heap if on-chip pool is exhausted
    void* allocate(std::size_t bytes);

    void deallocate(void* ptr, std::size_t bytes);

    UnifiedMemoryManager(const UnifiedMemoryManager&) = delete;
    UnifiedMemoryManager& operator=(const UnifiedMemoryManager&) = delete;

private:
    UnifiedMemoryManager();
    ~UnifiedMemoryManager();

    std::mutex mu_;
    std::size_t max_on_chip_ = 0;
    std::size_t allocated_on_chip_ = 0;
    char* pool_ = nullptr;
    std::unordered_map<void*, std::size_t> allocations_;
};