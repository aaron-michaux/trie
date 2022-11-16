
#pragma once

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <iostream>
#include <type_traits>

#include <cstdint>
#include <cstring>
#include <cassert>
#include <cstdlib>

constexpr bool DebugMessages{true};

std::string label_(const void* ptr) {
  return fmt::format("0x{:08x}", reinterpret_cast<uintptr_t>(ptr));
}

namespace niggly::trie {

/**
 * @TODO
 * - hash functions for integers (no penalty for monotonically increasing inserts)
 * - bulk insert... no reference counting
 * - performance: insert, delete, lookup,
 */

/**
 * Node { size, ref-count, is-leaf? | <storage> }
 *        everything at a left-node has the same hash
 *
 * Find (level): if branch node, then lookup branch and go
 *               if leaf node, then ibterate to find eq= match
 *
 * Insert (level): branch: lookup and go, or insert leaf if nullptr
 *                 leaf: hash= and insert, or make branch with two leaves
 *
 * Delete (level): could end up with an empty leaf node
 *
 * Iterate (level): easy
 */

namespace detail {

constexpr std::size_t MaxTrieDepth{13}; // maximum branch nodes... leaf nodes don't count here
constexpr uint32_t SparseIndexMax{31};
constexpr uint32_t InvalidSparseIndex{SparseIndexMax + 1};

constexpr uint8_t branch_free_popcount(uint32_t x) {
  x = x - ((x >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  return static_cast<uint8_t>(((x + (x >> 4) & 0x0f0f0f0fu) * 0x01010101u) >> 24);
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

  NodeData(NodeType type, node_size_type payload)
      : ref_count_{HighBit * static_cast<ref_count_type>(type) + 1}, payload_{payload} {}

  ref_count_type add_ref() const {
    ref_count_type previous_count;
    if constexpr (IsThreadSafe) {
      previous_count = ref_count_.fetch_add(1, std::memory_order_acq_rel) & RefMask;
    } else {
      previous_count = ref_count_++ & RefMask;
    }
    assert(previous_count < MaxRef);

    if constexpr (DebugMessages) {
      std::cout << fmt::format("inc_REF({}) = {}\n", label_(this), previous_count + 1);
    }

    return previous_count + 1;
  }

  ref_count_type dec_ref() const {
    ref_count_type previous_count;
    if constexpr (IsThreadSafe) {
      previous_count = ref_count_.fetch_sub(1, std::memory_order_acq_rel) & RefMask;
    } else {
      previous_count = ref_count_-- & RefMask;
    }
    assert(previous_count > 0);

    if constexpr (DebugMessages) {
      std::cout << fmt::format("DEC_REF({}) = {}\n", label_(this), previous_count - 1);
    }

    return previous_count - 1;
  }

  ref_count_type ref_count() const {
    if constexpr (IsThreadSafe) {
      return ref_count_.load(std::memory_order_acquire) & RefMask;
    } else {
      return ref_count_ & RefMask;
    }
  }

  NodeType type() const {
    auto bit = (reinterpret_cast<ref_count_type&>(ref_count_) & HighMask) >> HighBitOffset;
    return static_cast<NodeType>(bit);
  }
};

// ----------------------------------------------------------------------------------------- NodeOps

template <typename T, bool IsThreadSafe = true, bool IsSparseIndex = false> struct BaseNodeOps {

  using node_type = NodeData<IsThreadSafe>;
  using value_type = T;
  using hash_type = std::size_t;
  using node_size_type = typename node_type::node_size_type;
  using ref_count_type = typename node_type::ref_count_type;

  static constexpr NodeType DefaultType{IsSparseIndex ? NodeType::Branch : NodeType::Leaf};
  static constexpr std::size_t LogicalSize{calculate_logical_size<value_type>()};
  static constexpr std::size_t AlignOf{std::max(alignof(value_type), alignof(node_type))};
  static constexpr std::size_t MinStorageSize{std::max(sizeof(node_type), AlignOf)};
  static constexpr node_size_type MaxSize{(1 << (8 * sizeof(node_size_type) - 1)) - 1};

  // The start of a compact array (BranchNode), or array of values (LeafNode)
  static constexpr std::size_t offset() {
    if (alignof(value_type) <= sizeof(node_type)) {
      return sizeof(node_type); // align=[1, 2, 4, 8] => data starts at node_type edge
    }
    return LogicalSize;
  }

  static constexpr std::size_t offset_at(node_size_type index) {
    return offset() + LogicalSize * index;
  }

  static constexpr std::size_t storage_size(node_size_type size) {
    return (size == 0) ? MinStorageSize : offset() + LogicalSize * size;
  }

  static NodeType type(const node_type* node) { return node->type(); }

  static std::size_t size(const node_type* node) {
    return IsSparseIndex ? popcount(node->payload_) : node->payload_;
  }

  //@{ Member access
  static bool is_valid_index(const node_type* node, node_size_type index) {
    if constexpr (IsSparseIndex) {
      return ::niggly::trie::detail::is_valid_index(index, node->payload_);
    } else {
      return index < node->payload_;
    }
  }

  static value_type* ptr_at(const node_type* node, node_size_type index) {
    if constexpr (IsSparseIndex) {
      assert(index < 32);
      return dense_ptr_at(node, to_dense_index(index, node->payload_));
    } else {
      return dense_ptr_at(node, index);
    }
  }

  static value_type* dense_ptr_at(const node_type* node, node_size_type index) {
    auto ptr_idx = reinterpret_cast<uintptr_t>(node) + offset_at(index);
    assert(ptr_idx % alignof(value_type) == 0); // never unaligned access
    return reinterpret_cast<value_type*>(ptr_idx);
  }

  static value_type* begin(const node_type* node) { return ptr_at(node, 0); }
  static value_type* end(const node_type* node) { return ptr_at(node, 0) + size(node); }
  //@}

  //@{ Utility
  static node_type* make_uninitialized(node_size_type size, node_size_type payload) {
    auto ptr = static_cast<node_type*>(std::aligned_alloc(AlignOf, storage_size(size)));
    new (ptr) node_type{DefaultType, payload};
    return ptr;
  }

  static void copy_one(const value_type& src, value_type* dst) {
    assert(!IsSparseIndex); // Not useful for sparse indices
    if constexpr (std::is_trivial<value_type>::value) {
      std::memcpy(dst, &src, sizeof(value_type));
    } else {
      static_assert(std::is_copy_constructible<value_type>::value);
      new (dst) value_type{src};
    }
  }

  static void initialize_one(const value_type& src, value_type* dst) { copy_one(src, dst); }

  static void initialize_one(value_type&& src, value_type* dst) {
    assert(!IsSparseIndex); // Not useful for sparse indices
    if constexpr (std::is_move_constructible<value_type>::value) {
      new (dst) value_type{std::move(src)};
    } else if constexpr (std::is_default_constructible<value_type>::value &&
                         std::is_move_assignable<value_type>::value) {
      new (dst) value_type{};
      *dst = std::move(src);
    } else {
      copy_one(src, dst);
    }
  }

  static void copy_payload_to(const node_type* src, node_type* dst) {
    assert(!IsSparseIndex); // Not useful for sparse indices
    assert(src != nullptr);
    if constexpr (std::is_trivially_copyable<value_type>::value) {
      std::memcpy(ptr_at(dst, 0), ptr_at(src, 0), size(src) * LogicalSize);
    } else {
      static_assert(std::is_copy_constructible<value_type>::value);
      for (auto i = 0u; i < size(src); ++i) {
        copy_one(*ptr_at(src, i), ptr_at(dst, i));
      }
    }
  }
  //@}

  //@{ Factory/Destruct
  static node_type* make_empty() { return make_uninitialized(0, 0); }

  static node_type* make(const value_type& value) {
    assert(!IsSparseIndex); // Have to supply an index for such
    auto* ptr = make_uninitialized(1, 1);
    initialize_one(value, ptr_at(ptr, 0));
    return ptr;
  }

  static node_type* make(value_type&& value) {
    assert(!IsSparseIndex); // Have to supply an index for such
    auto* ptr = make_uninitialized(1, 1);
    initialize_one(std::move(value), ptr_at(ptr, 0));
    return ptr;
  }

  static node_type* duplicate(node_type* node, uint32_t dense_index_to_skip) {
    assert(IsSparseIndex);                    // only makes sense for sparse-index
    assert(node->type() == NodeType::Branch); // and this too
    const auto sz = size(node);
    node_type* ptr = make_uninitialized(sz, node->payload_);

    // Copy the pointers
    value_type* dst = dense_ptr_at(ptr, 0);
    const value_type* src = dense_ptr_at(node, 0);
    std::memcpy(dst, src, sz * sizeof(value_type));

    // Must bump up all references
    for (auto i = 0u; i < sz; ++i) {
      if (i == dense_index_to_skip)
        continue;
      dst[i]->add_ref();
    }
    // for (auto iterator = dst; iterator != dst + sz; ++iterator) {
    //   (*iterator)->add_ref();
    // }
    return ptr;
  }

  /**
   * Creates a new branch node, with `value` inserted at `index`
   */
  static node_type* insert_into_branch_node(const node_type* src, value_type value,
                                            uint32_t index) {
    assert(IsSparseIndex);
    assert(src->type() == NodeType::Branch);
    assert(index <= SparseIndexMax);
    assert(!is_valid_index(src, index)); // Cannot overwrite existing value

    const auto src_bitmap = src->payload_;
    const auto src_size = size(src);
    const auto dst_bitmap = (1u << index) | src_bitmap;
    const auto dst_size = src_size + 1;

    node_type* dst = make_uninitialized(dst_size, dst_bitmap);

    // Copy across the (densely stored) pointers
    auto* dst_array = dense_ptr_at(dst, 0);
    const auto* src_array = dense_ptr_at(src, 0);
    auto insert_pos = to_dense_index(index, dst_bitmap);

    for (auto index = 0u; index < insert_pos; ++index) {
      dst_array[index] = src_array[index];
      dst_array[index]->add_ref();
    }
    *dense_ptr_at(dst, insert_pos) = value; // insert the value
    for (auto index = insert_pos + 1; index < dst_size; ++index) {
      dst_array[index] = src_array[index - 1];
      dst_array[index]->add_ref();
    }

    return dst;
  }

  /**
   * Creates a new leaf node, with values copied, and `value` at the end
   */
  template <typename Value> static node_type* copy_append(const node_type* src, Value&& value) {
    assert(!IsSparseIndex);
    const auto sz = size(src);
    auto new_node = make_uninitialized(sz + 1, sz + 1);
    copy_payload_to(src, new_node);
    initialize_one(std::forward<Value>(value), ptr_at(new_node, sz));
    return new_node;
  }
  //@}
};

// ----------------------------------------------------------------------------------------- NodeOps

template <typename T, typename Hash = std::hash<T>, typename KeyEqual = std::equal_to<T>,
          typename Allocator = std::allocator<T>, bool IsThreadSafe = true>
struct NodeOps {
  using node_type = NodeData<IsThreadSafe>;
  using node_type_ptr = node_type*;
  using hash_type = std::size_t;
  using node_size_type = typename node_type::node_size_type;
  using ref_count_type = typename node_type::ref_count_type;
  using hasher = Hash;
  using key_equal = KeyEqual;

  using Branch = BaseNodeOps<node_type*, IsThreadSafe, true>;
  using Leaf = BaseNodeOps<T, IsThreadSafe, false>;

  //@{ Factory/Destroy
  static node_type* make_empty(NodeType type) {
    return (type == NodeType::Branch) ? Branch::make_empty() : Leaf::make_empty();
  }

  static void destroy(node_type* node_ptr) {
    if (node_ptr == nullptr) {
      return;
    }

    if constexpr (DebugMessages) {
      std::cout << fmt::format("ON DELETE({})\n", label_(node_ptr));
    }

    if (node_ptr->type() == NodeType::Branch) {
      node_type_ptr* iterator = Branch::dense_ptr_at(node_ptr, 0); // i.e., node_type**
      node_type_ptr* end = iterator + Branch::size(node_ptr);
      while (iterator != end) {
        node_type_ptr node_ptr = *iterator++;
        dec_ref(node_ptr);
      }

    } else {
      // Destroy payload only if its of "class" type
      if constexpr (std::is_class<T>::value) {
        auto* iterator = Leaf::ptr_at(node_ptr, 0);
        auto* end = iterator + Leaf::size(node_ptr);
        while (iterator != end) {
          std::cout << fmt::format("   DESTROY({})\n", label_(iterator));
          std::destroy_at(iterator++);
        }
      }
    }

    if constexpr (DebugMessages) {
      std::cout << fmt::format("done del ({})\n", label_(node_ptr));
    }

    node_ptr->~node_type();
    std::free(node_ptr);
  }
  //@}

  //@{ Getters
  static NodeType type(const node_type* node) { return node->type(); }

  static std::size_t size(const node_type* node) {
    return (node->type() == NodeType::Branch) ? Branch::size(node) : Leaf::size(node);
  }

  static bool is_valid_index(const node_type* node) {
    return (node->type() == NodeType::Branch) ? Branch::is_valid_index(node)
                                              : Leaf::is_valid_index(node);
  }

  static std::size_t hash(const node_type* node) {
    assert(node != nullptr);
    assert(node->type() == NodeType::Leaf);
    assert(Leaf::size(node) > 0);
    Hash hasher;
    return hasher(*Leaf::ptr_at(node, 0));
  }
  //@}

  //@{ Reference counting
  static void add_ref(const node_type* node) {
    if (node != nullptr)
      node->add_ref();
  }
  static void dec_ref(const node_type* node) {
    if (node != nullptr && node->dec_ref() == 0)
      destroy(const_cast<node_type*>(node));
  }
  static ref_count_type ref_count(const node_type* node) {
    return (node == nullptr) ? 0 : node->ref_count();
  }
  //@}

  //@{ Path
  struct TreePath {
    std::array<node_type*, MaxTrieDepth> nodes; //!< the path is {Branch, Branch, Branch}
    node_type* leaf_end = nullptr;              //!< set if the path ends in a leaf
    uint32_t size = 0;                          //!< number of branch elements in path
    void push(node_type* node) {
      assert(size + 1 < nodes.size());
      nodes[size++] = node;
    }
  };

  static TreePath make_path(node_type* root, std::size_t hash) {
    TreePath path;
    auto node = root;
    while (node != nullptr && type(node) == NodeType::Branch) {
      auto sparse_index = hash_chunk(hash, path.size);
      path.push(node);
      assert(!Branch::is_valid_index(node, sparse_index) ||
             Branch::ptr_at(node, sparse_index) != nullptr);
      node = Branch::is_valid_index(node, sparse_index) ? *Branch::ptr_at(node, sparse_index)
                                                        : nullptr;
    }
    path.leaf_end = node;
    assert(path.leaf_end == nullptr || type(path.leaf_end) == NodeType::Leaf);
    return path;
  }

  /**
   * Rewrites the branch nodes of the path, adding in the new branch at the end
   * @param path The path to rewrite
   * @param hash The hash associated with the value inserted (into leaf_end)
   * @param new_node The new node to append -- could be a branch node
   * @param leaf_end The node guaranteed to contain the newly inserted value
   *
   * Must know `leaf_end` to prevent a resource leak. Normally when appending
   * a leaf node, it is going to be reused by some other node, or it will have
   * been moved somewhere else in the tree. However, in the case of a hash
   * collision, the new leaf will contain all the collided values, and the
   * previous leaf should not be duplicated, because it will no longer be used
   * anywhere in this tree.
   *
   * @return The new root of the tree
   */
  template <bool is_bulk_insert>
  static node_type* rewrite_branch_path(const TreePath& path, std::size_t hash, node_type* new_node,
                                        node_type* leaf_end) {
    // When editing the tree, we need to make a copy of all the branch nodes,
    // Returning the head+tail, i.e, the new root, and the new branch node at the tip
    if (path.size == 0)
      return new_node; // The trivial case

    // Working backwards, so start with the tail
    const uint32_t last_level = path.size - 1;
    auto last_index = hash_chunk(hash, last_level);
    node_type* iterator = path.nodes[last_level];
    if (Branch::is_valid_index(iterator, last_index)) {
      const auto dense_index = to_dense_index(last_index, iterator->payload_);
      const auto old_leaf = *Branch::dense_ptr_at(iterator, dense_index);
      const auto skip_index =
          (old_leaf->payload_ == leaf_end->payload_) // means overwriting old_leaf
              ? InvalidSparseIndex                   // because it has been extended
              : dense_index;                         // so skip this index
      if constexpr (is_bulk_insert) {
        if (skip_index != InvalidSparseIndex)
          dec_ref(old_leaf);
      } else {
        iterator = Branch::duplicate(iterator, skip_index); // duplicate the branch node and
      }
      *Branch::dense_ptr_at(iterator, dense_index) = new_node; // overwrite

    } else {
      // insert into the branch node -- expanding it
      std::cout << fmt::format("{}:{}, iterator={}\n", "rewrite_branch_path", __LINE__,
                               label_(iterator));
      auto* node = Branch::insert_into_branch_node(iterator, new_node, last_index);
      std::cout << fmt::format("{}:{}, node    ={}\n", "rewrite_branch_path", __LINE__,
                               label_(node));
      if constexpr (is_bulk_insert) {
        std::cout << fmt::format("{}:{}, path.size={}\n", "rewrite_branch_path", __LINE__,
                                 path.size);
        if (path.size == 1) {
          // std::free(path.nodes[0]);
          return node;

        } else if (path.size > 1) {
          auto* penultimate = path.nodes[last_level - 1]; // Connect penultimate to node
          std::cout << fmt::format("{}:{}, penultimate={}\n", "rewrite_branch_path", __LINE__,
                                   label_(penultimate));
          const auto sparse_index = hash_chunk(hash, last_level - 1); // At this index
          *Branch::ptr_at(penultimate, sparse_index) = node;          // overwrite
          std::cout << fmt::format("{}:{}, about-to-decref={}\n", "rewrite_branch_path", __LINE__,
                                   label_(iterator));
          dec_ref(iterator); // otherwise will be orphan
        } else {
          iterator = node;
        }
      } else {
        iterator = node;
      }
    }

    if constexpr (is_bulk_insert) {
      iterator = path.nodes[0];

    } else {
      for (auto i = last_level; i > 0; --i) {
        auto* node = path.nodes[i - 1];
        const auto sparse_index = hash_chunk(hash, i - 1); // The overwrite position
        const auto dense_index = to_dense_index(sparse_index, node->payload_);
        auto* new_branch_node = Branch::duplicate(node, dense_index);   // Private copy
        *Branch::dense_ptr_at(new_branch_node, dense_index) = iterator; // Do the overwite
        iterator = new_branch_node;                                     // Update the current tail
      }
    }

    return iterator;
  }

  /**
   * It may be necessary to create a sequence of "common" branch nodes when
   * inserting a new leaf. This method handles creating the new leaf, and
   * creating however many common branch nodes required to insert leaf from the
   * existing position (level).
   *
   *    [...] -> old-leaf
   *
   *        goes to
   *
   *    [...] -> branch-head -> branch -> branch -> old-leaf
   *                                             -> new-leaf
   *
   * Returns {branch-head, new-leaf}
   */
  template <bool is_bulk_insert, typename Value>
  static std::pair<node_type*, node_type*>
  branch_to_leaves(const std::size_t value_hash, uint32_t level, node_type* existing_leaf,
                   Value&& value) {
    assert(type(existing_leaf) == NodeType::Leaf);
    assert(Leaf::size(existing_leaf) > 0);
    assert(level < MaxTrieDepth); // otherwise the hashes would have to have been equal

    Hash hasher;
    const auto existing_hash = hasher(*Leaf::ptr_at(existing_leaf, 0));

    // Trivial cases: Hash collision => create new leaf with new value appended
    if (value_hash == existing_hash) { // Trivial case: hash collision
      auto* new_leaf = Leaf::copy_append(existing_leaf, std::forward<Value>(value));
      return {new_leaf, new_leaf};
    }

    // Otherwise, make {branch, branch, branch, ...} until the indices diverge
    node_type_ptr branch = nullptr;
    node_type_ptr tail = nullptr;
    uint32_t last_index = 0u;

    auto append_to_tail = [&](node_type_ptr new_branch, uint32_t index) {
      if (branch == nullptr) {
        branch = new_branch;                            // nothing to connect
      } else {                                          //
        *Branch::ptr_at(tail, last_index) = new_branch; // connect [tail => new_branch]
      }                                                 //
      tail = new_branch;                                // update the tail
      last_index = index;                               // store the index for next insert into tail
    };

    node_type_ptr new_leaf = nullptr;
    for (auto i = level; new_leaf == nullptr; ++i) {
      assert(i < MaxTrieDepth); // otherwise there'd have to be a hash collission
      const auto index_lhs = hash_chunk(existing_hash, i);
      const auto index_rhs = hash_chunk(value_hash, i);
      if (index_lhs == index_rhs) {
        append_to_tail(Branch::make_uninitialized(1, 1u << index_lhs), index_lhs);
      } else { // divergence
        new_leaf = Leaf::make(std::forward<Value>(value));
        const auto pattern = (1u << index_lhs) | (1u << index_rhs);
        append_to_tail(Branch::make_uninitialized(2, pattern), 0 /* irrelevant */);
        *Branch::ptr_at(tail, index_lhs) = existing_leaf;
        *Branch::ptr_at(tail, index_rhs) = new_leaf;
        assert(Branch::is_valid_index(tail, index_lhs));
        assert(Branch::is_valid_index(tail, index_rhs));
        assert(*Branch::ptr_at(tail, index_lhs) == existing_leaf);
        assert(type(existing_leaf) == NodeType::Leaf);
        assert(type(*Branch::ptr_at(tail, index_rhs)) == NodeType::Leaf);
      }
    }

    return {branch, new_leaf};
  }
  //@}
};

} // namespace detail

// ----------------------------------------------------------------------------------- PersistentSet

template <typename ItemType,                             // Type of item to store
          typename Hash = std::hash<ItemType>,           // Hash function for item
          typename KeyEqual = std::equal_to<ItemType>,   // Equality comparision for Item
          typename Allocator = std::allocator<ItemType>, // Allocator for Item
          bool IsThreadSafe = true                       // True if Set is threadsafe
          >
class PersistentSet {
public:
  //@{
  using key_type = ItemType;
  using value_type = ItemType;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using hasher = Hash;
  using key_equal = KeyEqual;
  using allocator_type = Allocator;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = std::allocator_traits<Allocator>::pointer;
  using const_pointer = std::allocator_traits<Allocator>::const_pointer;
  static constexpr bool is_thread_safe = IsThreadSafe;
  // using iterator = ;
  // using const_iterator = ;
  //@}

private:
  using Ops = detail::NodeOps<ItemType, Hash, KeyEqual, Allocator, IsThreadSafe>;
  using node_type = typename Ops::node_type;
  using node_type_ptr = node_type*;
  using NodeType = detail::NodeType;

  node_type* root_{nullptr}; //!< Root of the tree could be branch of leaf
  std::size_t size_{0};      //!< Current size of the Set

public:
  //@{ Construction/Destruction
  PersistentSet() = default;
  PersistentSet(const PersistentSet& other) { *this = other; }
  PersistentSet(PersistentSet&& other) noexcept { swap(*this, other); }
  ~PersistentSet() { Ops::dec_ref(root_); }
  //@}

  //@{
  // allocator_type get_allocator() const noexcept;
  //@}

  //@{ Assignment
  PersistentSet& operator=(const PersistentSet& other) {
    Ops::dec_ref(root_);
    root_ = other.root_;
    size_ = other.size_;
    Ops::add_ref(root_);
    return *this;
  }

  PersistentSet& operator=(PersistentSet&& other) noexcept {
    swap(*this, other);
    return *this;
  }
  //@}

  //@{ Iterators
  // begin/begin/cbegin
  // end/end/cend
  //@}

  //@{ Capacity
  bool empty() const { return size() == 0; }
  std::size_t size() const { return size_; }
  std::size_t max_size() const { return std::numeric_limits<std::size_t>::max(); }
  //@}

  //@{ Modifiers
  void clear() {
    Ops::dec_ref(root_);
    root_ = nullptr;
    size_ = 0;
  }

  void insert(const value_type& value) { insert_<false>(value); }
  void insert(value_type&& value) { insert_<false>(std::move(value)); }

  void bulk_insert(const value_type& value) { insert_<true>(value); }
  void bulk_insert(value_type&& value) { insert_<true>(std::move(value)); }

  // emplace()...;
  // emplace_hint()...;
  // erase()...;
  void swap(PersistentSet& other) noexcept { // Should not be able
    std::swap(root_, other.root_);
    std::swap(size_, other.size_);
  }
  //@}

  //@{ Lookup
  // std::size_t count() const;
  // find();
  // contains();
  // equal_range();
  //@}

  //@{ Compatibility
  // rehash();
  // reserve();
  // load_factor();
  // max_load_factor();
  //@}

  //@{ Observers
  // hash_function
  // key_eq
  //@}

  //@{ Friends
  friend bool operator==(const PersistentSet& lhs, const PersistentSet& rhs) noexcept {
    return lhs.root_ == rhs.root_;
  }

  friend bool operator!=(const PersistentSet& lhs, const PersistentSet& rhs) noexcept {
    return !(lhs == rhs);
  }

  // std::erase_if

  friend void swap(PersistentSet& lhs, PersistentSet& rhs) noexcept { lhs.swap(rhs); }
  //@}

private:
  node_type* get_root_() { return root_; }

  static size_t calculate_hash(const value_type& value) {
    hasher hash_fun;
    return hash_fun(value);
  }

  static bool calculate_equals(const value_type& lhs, const value_type& rhs) {
    key_equal fun;
    return fun(lhs, rhs);
  }

  static bool has_eq(const value_type& value, node_type_ptr leaf) {
    assert(Ops::type(leaf) == NodeType::Leaf);
    auto* start = Ops::Leaf::ptr_at(leaf, 0);
    for (auto* iterator = start; iterator != start + Ops::Leaf::size(leaf); ++iterator) {
      assert(calculate_hash(*start) == calculate_hash(*iterator));
      if (calculate_equals(value, *iterator))
        return true;
    }
    return false;
  }

  template <bool is_bulk_insert, typename Value>
  static std::pair<node_type_ptr, bool> do_insert_(node_type_ptr root, Value&& value) {
    if (root == nullptr) { // inserting into an empty tree: trivial case
      return {Ops::Leaf::make(std::forward<Value>(value)), true};
    }

    // TO ADD:
    //  1. If there's no leaf node at the end, then
    //     (a) rewrite the branch
    //     (b) insert the new leaf into the tail of the rewrite
    //  2. If there is an existing leaf
    //     (a) if:   eq, then insert as failed
    //     (b) else: rewrite the branch
    //     (c)       create a new path to discriminate between the existing and new leaf
    //     (d)       join the two paths together
    //  Finally: update the root
    const auto hash = calculate_hash(value);
    const auto path = Ops::make_path(root, hash);
    node_type_ptr new_root = nullptr;
    bool success = false;

    if (path.leaf_end == nullptr) {
      auto new_leaf = Ops::Leaf::make(std::forward<Value>(value));
      new_root = Ops::template rewrite_branch_path<is_bulk_insert>(path, hash, new_leaf, new_leaf);
      success = true;

    } else if (has_eq(value, path.leaf_end)) {
      // (2.a) Duplicate!
      new_root = root;

    } else {
      auto [new_branch, leaf_end] = Ops::template branch_to_leaves<is_bulk_insert>(
          hash,                        // Hash of the value
          path.size,                   // The path starts here
          path.leaf_end,               // Includes this node
          std::forward<Value>(value)); // Must include this new value
      new_root =
          Ops::template rewrite_branch_path<is_bulk_insert>(path, hash, new_branch, leaf_end);
      success = true;
    }

    return {new_root, success};
  }

  template <bool is_bulk_insert, typename Value> bool insert_(Value&& value) {
    auto [new_root, success] = do_insert_<is_bulk_insert>(root_, std::forward<Value>(value));

    if (success) {
      if constexpr (!is_bulk_insert) {
        if (root_ != nullptr && Ops::type(root_) == NodeType::Leaf &&
            Ops::type(new_root) == NodeType::Branch) {
          // Never `dec_ref` a root_ "leaf node", when new_root is a Branch
          // because that leaf will have become part of the tree.
          //
          // If new_root is a branch then there was a collision at the root
        } else {
          Ops::dec_ref(root_);
        }
      }
      root_ = new_root;
      ++size_;
    }

    return success;
  }
};

// template <typename ItemType, typename Hash, typename KeyEqual, typename Allocator,
//           bool IsThreadSafe>
// friend void swap(PersistentSet<ItemType, Hash, KeyEqual, Allocator, IsThreadSafe>& lhs,
//                  PersistentSet<ItemType, Hash, KeyEqual, Allocator, IsThreadSafe>& rhs) {
//   lhs.swap(rhs);
// }

} // namespace niggly::trie
