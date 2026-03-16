#pragma once
#include <cstddef>
#include "memory_pool.hpp"

template <typename T>
struct UnifiedMemoryAllocator {
    using value_type = T;
    UnifiedMemoryAllocator() noexcept {}
    template <typename U>
    UnifiedMemoryAllocator(const UnifiedMemoryAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        void* ptr = UnifiedMemoryManager::instance().allocate(n * sizeof(T));
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t n) noexcept {
        UnifiedMemoryManager::instance().deallocate(static_cast<void*>(p), n * sizeof(T));
    }
};

// Comparisons are stateless; all allocators are equal
template <typename T, typename U>
bool operator==(const UnifiedMemoryAllocator<T>&, const UnifiedMemoryAllocator<U>&) noexcept {
    return true;
}
template <typename T, typename U>
bool operator!=(const UnifiedMemoryAllocator<T>& a, const UnifiedMemoryAllocator<U>& b) noexcept {
    return !(a == b);
}