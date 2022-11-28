
#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <type_traits>

#include <cstdint>
#include <cstring>
#include <cassert>
#include <cstdlib>

namespace niggly::trie::detail {

constexpr std::size_t MaxTrieDepth{13}; // maximum branch nodes... leaf nodes don't count here

constexpr uint32_t NotAnIndex{static_cast<uint32_t>(-1)};

constexpr uint8_t branch_free_popcount(uint32_t x) {
  x = x - ((x >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  return static_cast<uint8_t>(((x + (x >> 4) & 0x0f0f0f0fu) * 0x01010101u) >> 24);
}

constexpr uintptr_t cast_pointer(const void* ptr) {
  uintptr_t value;
  std::memcpy(&value, &ptr, sizeof(const void*));
  return value;
}

/**
 * @return The number of bits in 'x' with value 1
 */
constexpr uint8_t popcount(uint32_t x) {
#if __has_builtin(__builtin_popcount)
  return __builtin_popcount(x); // some versions of gcc/clang only
#else
  return branch_free_popcount(x);
#endif
}

constexpr uint32_t to_dense_index(uint32_t index, uint32_t bitmap) {
  assert(index < sizeof(uint32_t) * 8);
  const auto mask = (1u << index) - 1; // index=4  ==>  mask=0x0111b
  return popcount(bitmap & mask);      // count the position branchlessly
}

constexpr bool is_valid_index(uint32_t index, uint32_t bitmap) {
  const auto clamped_index = (index & 0x0000001fu); // clamp required so that
  const auto mask = (1u << clamped_index);          // this left-shift is well-defined
  return (bitmap & mask);                           // on the abstract c++ machine
}

constexpr uint32_t hash_chunk(std::size_t hash, uint32_t chunk_number) {
  assert(chunk_number < MaxTrieDepth);
  // 0x00011111b, shifted by chunk offset
  const auto mask = static_cast<std::size_t>(0x1fu) << (chunk_number * 5);
  return (hash & mask) >> (chunk_number * 5);
}

enum class NodeType : int { Branch = 0, Leaf = 1 };

/**
 * The bytes between successive objects in an array of T
 */
constexpr std::size_t calculate_logical_size_(std::size_t align, std::size_t size) {
  if (size <= align)
    return align;
  if (align <= 1)
    return size;
  const auto remainder = size % align;
  const auto chunks = size / align;
  return (remainder == 0) ? size : align * (chunks + 1);
}

template <typename T> constexpr std::size_t calculate_logical_size() {
  return calculate_logical_size_(alignof(T), sizeof(T));
}

// ---------------------------------------------------------------------------------------- NodeData

template <bool IsThreadSafe = true> struct NodeData {
  using ref_count_type = uint32_t; // Okay, so 64k copies will explode
  using node_size_type = uint32_t; // Need all 32 bits
  using counter_type =
      std::conditional_t<IsThreadSafe, std::atomic<ref_count_type>, ref_count_type>;

  static constexpr ref_count_type HighBitOffset{sizeof(ref_count_type) * 8 - 1}; // 31
  static constexpr ref_count_type HighBit{static_cast<ref_count_type>(1) << HighBitOffset};
  static constexpr ref_count_type HighMask{HighBit};
  static constexpr ref_count_type RefMask{HighBit - 1};
  static constexpr ref_count_type MaxRef{HighBit - 1};

  // @{ members
  mutable counter_type ref_count_; // The high bit is fixed at contruction
  node_size_type payload_;
  // @}

  constexpr NodeData(NodeType type, node_size_type payload)
      : ref_count_{HighBit * static_cast<ref_count_type>(type) + 1}, payload_{payload} {}

  constexpr ref_count_type add_ref() const {
    ref_count_type previous_count;
    if constexpr (IsThreadSafe) {
      previous_count = ref_count_.fetch_add(1, std::memory_order_acq_rel) & RefMask;
    } else {
      previous_count = ref_count_++ & RefMask;
    }
    assert(previous_count < MaxRef);
    return previous_count + 1;
  }

  constexpr ref_count_type dec_ref() const {
    ref_count_type previous_count;
    if constexpr (IsThreadSafe) {
      previous_count = ref_count_.fetch_sub(1, std::memory_order_acq_rel) & RefMask;
    } else {
      previous_count = ref_count_-- & RefMask;
    }
    assert(previous_count > 0);
    return previous_count - 1;
  }

  constexpr ref_count_type ref_count() const {
    if constexpr (IsThreadSafe) {
      return ref_count_.load(std::memory_order_acquire) & RefMask;
    } else {
      return ref_count_ & RefMask;
    }
  }

  constexpr NodeType type() const {
    const ref_count_type* ptr = std::bit_cast<const ref_count_type*>(&ref_count_);
    const auto bit = (*ptr & HighMask) >> HighBitOffset;
    return static_cast<NodeType>(bit);
  }
};

} // namespace niggly::trie::detail
