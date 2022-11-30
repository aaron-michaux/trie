
#pragma once

#include <array>
#include <algorithm>
#include <atomic>
#include <bit>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include <cstdint>
#include <cstring>
#include <cassert>
#include <cstdlib>

namespace niggly::trie::detail {

// ---------------------------------------------------------------------------------- Base Functions

constexpr std::size_t MaxTrieDepth{13}; // maximum branch nodes... leaf nodes don't count here

constexpr uint32_t NotAnIndex{static_cast<uint32_t>(-1)};

constexpr uint8_t branch_free_popcount(uint32_t x) {
  x = x - ((x >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  return static_cast<uint8_t>((((x + (x >> 4)) & 0x0f0f0f0fu) * 0x01010101u) >> 24);
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

// ---------------------------------------------------------------------------------------- NodeType

enum class NodeType : int { Branch = 0, Leaf = 1 };

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

// ------------------------------------------------------------------------------------- BaseNodeOps

template <typename T, bool IsThreadSafe = true, bool IsBranchNode = false> struct BaseNodeOps {

  using node_type = NodeData<IsThreadSafe>;
  using node_ptr_type = node_type*;
  using node_const_ptr_type = const node_type*;
  using item_type = T;
  using hash_type = std::size_t;
  using node_size_type = typename node_type::node_size_type;
  using ref_count_type = typename node_type::ref_count_type;

  static constexpr bool is_thread_safe = IsThreadSafe;

  static constexpr NodeType DefaultType{IsBranchNode ? NodeType::Branch : NodeType::Leaf};
  static constexpr std::size_t LogicalSize{calculate_logical_size<item_type>()};
  static constexpr std::size_t AlignOf{std::max(alignof(item_type), alignof(node_type))};
  static constexpr std::size_t MinStorageSize{std::max(sizeof(node_type), AlignOf)};
  static constexpr node_size_type MaxSize{(1 << (8 * sizeof(node_size_type) - 1)) - 1};

  // The start of a compact array (BranchNode), or array of values (LeafNode)
  static constexpr std::size_t offset() {
    if (alignof(item_type) <= sizeof(node_type)) {
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

  static constexpr std::size_t size(node_const_ptr_type node) {
    if constexpr (IsBranchNode) {
      return popcount(node->payload_);
    } else {
      return node->payload_;
    }
  }

  //@{ Member access
  static constexpr bool is_valid_index(node_const_ptr_type node, node_size_type index) {
    if constexpr (IsBranchNode) {
      return ::niggly::trie::detail::is_valid_index(index, node->payload_);
    } else {
      return index < node->payload_;
    }
  }

  static constexpr item_type* ptr_at(node_const_ptr_type node, node_size_type index) {
    if constexpr (IsBranchNode) {
      assert(index < 32);
      return dense_ptr_at(node, to_dense_index(index, node->payload_));
    } else {
      return dense_ptr_at(node, index);
    }
  }

  static constexpr item_type* dense_ptr_at(node_const_ptr_type node, node_size_type index) {
    auto ptr_idx = reinterpret_cast<uintptr_t>(node) + offset_at(index);
    assert(ptr_idx % alignof(item_type) == 0); // never unaligned access
    return reinterpret_cast<item_type*>(ptr_idx);
  }

  static constexpr item_type* begin(node_const_ptr_type node) { return ptr_at(node, 0); }

  static constexpr item_type* end(node_const_ptr_type node) { return ptr_at(node, 0) + size(node); }
  //@}

  //@{ Utility
  static constexpr node_ptr_type make_uninitialized(node_size_type size, node_size_type payload) {
    auto ptr = static_cast<node_ptr_type>(std::aligned_alloc(AlignOf, storage_size(size)));
    new (ptr) node_type{DefaultType, payload};
    return ptr;
  }
  //@}
};

// ------------------------------------------------------------------------------------- LeafNodeOps

template <typename T, bool IsThreadSafe = true>
struct LeafNodeOps : public BaseNodeOps<T, IsThreadSafe, false> {

  using Base = BaseNodeOps<T, IsThreadSafe, false>;

  using node_type = typename Base::node_type;
  using node_ptr_type = typename Base::node_ptr_type;
  using node_const_ptr_type = typename Base::node_const_ptr_type;
  using item_type = typename Base::item_type;
  using hash_type = typename Base::hash_type;
  using node_size_type = typename Base::node_size_type;
  using ref_count_type = typename Base::ref_count_type;

  template <typename P> static constexpr void copy_one(P&& src, item_type* dst) {
    if constexpr (std::is_trivial<item_type>::value) {
      std::memcpy(dst, &src, sizeof(item_type));
    } else {
      static_assert(std::is_copy_constructible<item_type>::value);
      new (dst) item_type{std::forward<P>(src)};
    }
  }

  static constexpr void initialize_one(const item_type& src, item_type* dst) { copy_one(src, dst); }

  template <typename P> static constexpr void initialize_one(P&& src, item_type* dst) {
    if constexpr (std::is_move_constructible<item_type>::value) {
      new (dst) item_type{std::move(src)};
    } else if constexpr (std::is_default_constructible<item_type>::value &&
                         std::is_move_assignable<item_type>::value) {
      new (dst) item_type{};
      *dst = std::move(src);
    } else {
      copy_one(src, dst);
    }
  }

  static constexpr void copy_payload_to(node_const_ptr_type src, node_ptr_type dst) {
    assert(src != nullptr);
    if constexpr (std::is_trivially_copyable<item_type>::value) {
      std::memcpy(Base::ptr_at(dst, 0), Base::ptr_at(src, 0), Base::size(src) * Base::LogicalSize);
    } else {
      static_assert(std::is_copy_constructible<item_type>::value);
      for (auto i = 0u; i < Base::size(src); ++i) {
        copy_one(*Base::ptr_at(src, i), Base::ptr_at(dst, i));
      }
    }
  }

  static constexpr node_ptr_type make(const item_type& value) {
    auto* ptr = Base::make_uninitialized(1, 1);
    initialize_one(value, Base::ptr_at(ptr, 0));
    return ptr;
  }

  template <typename P> static constexpr node_ptr_type make(P&& value) {
    auto* ptr = Base::make_uninitialized(1, 1);
    initialize_one(std::forward<P>(value), Base::ptr_at(ptr, 0));
    return ptr;
  }

  /**
   * Duplicates a leaf node, optionally omitting the value at `index_to_skip`
   */
  static constexpr node_ptr_type duplicate_leaf(node_const_ptr_type node, uint32_t index_to_skip) {
    const auto sz = Base::size(node);
    node_ptr_type new_node = nullptr;
    if (index_to_skip < sz) {
      new_node = Base::make_uninitialized(sz - 1, sz - 1);
      uint32_t write_index = 0;
      for (auto index = 0u; index != sz; ++index) {
        if (index != index_to_skip)
          copy_one(*Base::ptr_at(node, index), Base::ptr_at(new_node, write_index++));
      }
    } else {
      new_node = Base::make_uninitialized(sz, sz);
      copy_payload_to(node, new_node);
    }
    return new_node;
  }

  /**
   * Duplicates a leaf node, overwrite a value
   */
  template <typename Item>
  static constexpr node_ptr_type duplicate_leaf_with_overwrite(node_const_ptr_type node,
                                                               uint32_t index_to_overwrite,
                                                               Item&& item) {
    const auto sz = Base::size(node);
    assert(index_to_overwrite < sz);
    node_ptr_type new_node = Base::make_uninitialized(sz, sz);
    for (auto index = 0u; index != sz; ++index) {
      if (index == index_to_overwrite) {
        copy_one(std::forward<Item>(item), Base::ptr_at(new_node, index));
      } else {
        copy_one(*Base::ptr_at(node, index), Base::ptr_at(new_node, index));
      }
    }
    return new_node;
  }

  /**
   * Creates a new leaf node, with values copied, and `value` at the end
   */
  template <typename Item>
  static constexpr node_ptr_type copy_append(node_const_ptr_type src, Item&& item) {
    const auto sz = Base::size(src);
    auto new_node = Base::make_uninitialized(sz + 1, sz + 1);
    copy_payload_to(src, new_node);
    initialize_one(std::forward<Item>(item), Base::ptr_at(new_node, sz));
    return new_node;
  }
};

// ----------------------------------------------------------------------------------- BranchNodeOps

template <bool IsThreadSafe = true>
struct BranchNodeOps : public BaseNodeOps<NodeData<IsThreadSafe>*, IsThreadSafe, true> {

  using Base = BaseNodeOps<NodeData<IsThreadSafe>*, IsThreadSafe, true>;

  using node_type = typename Base::node_type;
  using node_ptr_type = typename Base::node_ptr_type;
  using node_const_ptr_type = typename Base::node_const_ptr_type;
  using item_type = typename Base::item_type;
  using hash_type = typename Base::hash_type;
  using node_size_type = typename Base::node_size_type;
  using ref_count_type = typename Base::ref_count_type;

  /**
   * Duplicate a Branch node, copying the values, perhaps skipping an index that is being
   * overwritten
   */
  static constexpr node_ptr_type duplicate(node_ptr_type node, uint32_t dense_index_to_skip) {
    assert(node->type() == NodeType::Branch); // and this too
    const auto sz = Base::size(node);
    node_ptr_type ptr = Base::make_uninitialized(sz, node->payload_);

    // Copy the pointers
    item_type* dst = Base::dense_ptr_at(ptr, 0);
    const item_type* src = Base::dense_ptr_at(node, 0);
    std::memcpy(dst, src, sz * sizeof(item_type));

    // Must bump up all references
    for (auto i = 0u; i < sz; ++i) {
      if (i == dense_index_to_skip)
        continue;
      dst[i]->add_ref();
    }
    return ptr;
  }

  /**
   *
   */
  static constexpr node_ptr_type remove_from_branch_node(node_ptr_type node,
                                                         uint32_t sparse_index_to_remove) {
    assert(node->type() == NodeType::Branch); // and this too
    assert(Base::size(node) > 1);             // otherwise the branch node would become empty
    assert(Base::is_valid_index(node, sparse_index_to_remove)); // must remove something!

    const auto sz = Base::size(node);
    const auto dense_index = to_dense_index(sparse_index_to_remove, node->payload_);

    node_ptr_type ptr =
        Base::make_uninitialized(sz - 1, node->payload_ & ~(1u << sparse_index_to_remove));

    // Copy the pointers
    item_type* dst = Base::dense_ptr_at(ptr, 0);
    const item_type* src = Base::dense_ptr_at(node, 0);

    // Must bump up all references
    auto write_pos = 0u;
    for (auto i = 0u; i < sz; ++i) {
      if (i == dense_index)
        continue;
      src[i]->add_ref();
      dst[write_pos++] = src[i];
    }
    return ptr;
  }

  /**
   * Creates a new branch node, with `value` inserted at `index`
   */
  static constexpr node_ptr_type insert_into_branch_node(node_const_ptr_type src, item_type value,
                                                         uint32_t index) {
    assert(src->type() == NodeType::Branch);
    assert(index < 32);
    assert(!Base::is_valid_index(src, index)); // Cannot overwrite existing value

    const auto src_bitmap = src->payload_;
    const auto src_size = Base::size(src);
    const auto dst_bitmap = (1u << index) | src_bitmap;
    const auto dst_size = src_size + 1;

    auto dst = Base::make_uninitialized(dst_size, dst_bitmap);
    assert(Base::size(dst) == dst_size);
    assert(dst->type() == NodeType::Branch);

    // Copy across the (densely stored) pointers
    auto* dst_array = Base::dense_ptr_at(dst, 0);
    const auto* src_array = Base::dense_ptr_at(src, 0);
    auto insert_pos = to_dense_index(index, dst_bitmap);

    for (auto index = 0u; index < insert_pos; ++index) {
      dst_array[index] = src_array[index];
      dst_array[index]->add_ref();
    }
    *Base::dense_ptr_at(dst, insert_pos) = value; // insert the value
    for (auto index = insert_pos + 1; index < dst_size; ++index) {
      dst_array[index] = src_array[index - 1];
      dst_array[index]->add_ref();
    }

    return dst;
  }
};

// ----------------------------------------------------------------------------------------- NodeOps

template <typename KeyType,                           //
          typename ValueType,                         // Value Type for maps
          typename Hash = std::hash<KeyType>,         //
          typename KeyEqual = std::equal_to<KeyType>, //
          bool IsMap = false,                         // True if map
          bool IsThreadSafe = true>
struct NodeOps {
  using key_type = KeyType;
  using value_type = ValueType;
  using item_type = typename std::conditional_t<IsMap, std::pair<KeyType, ValueType>, ValueType>;
  using node_type = NodeData<IsThreadSafe>;
  using node_ptr_type = node_type*;
  using node_const_ptr_type = const node_type*;
  using size_type = std::size_t;
  using hash_type = std::size_t;
  using node_size_type = typename node_type::node_size_type;
  using ref_count_type = typename node_type::ref_count_type;
  using hasher = Hash;
  using key_equal = KeyEqual;

  using Branch = BranchNodeOps<IsThreadSafe>;
  using Leaf = LeafNodeOps<item_type, IsThreadSafe>;

  static constexpr bool is_thread_safe = IsThreadSafe;

  static constexpr void destroy(node_ptr_type node_ptr) {
    if (node_ptr == nullptr) {
      return;
    }

    if (node_ptr->type() == NodeType::Branch) {
      node_ptr_type* iterator = Branch::dense_ptr_at(node_ptr, 0); // i.e., node_type**
      node_ptr_type* end = iterator + Branch::size(node_ptr);
      while (iterator != end) {
        node_ptr_type node_ptr = *iterator++;
        dec_ref(node_ptr);
      }

    } else {
      // Destroy payload only if its of "class" type
      if constexpr (std::is_class<item_type>::value) {
        auto* iterator = Leaf::ptr_at(node_ptr, 0);
        auto* end = iterator + Leaf::size(node_ptr);
        while (iterator != end) {
          std::destroy_at(iterator++);
        }
      }
    }

    node_ptr->~node_type();
    std::free(node_ptr);
  }
  //@}

  //@{ Getters
  static constexpr NodeType type(node_const_ptr_type node) { return node->type(); }

  static constexpr size_type size(node_const_ptr_type node) {
    return (node->type() == NodeType::Branch) ? Branch::size(node) : Leaf::size(node);
  }

  static constexpr hash_type hash(node_const_ptr_type node) {
    assert(node != nullptr);
    assert(node->type() == NodeType::Leaf);
    assert(Leaf::size(node) > 0);
    Hash hasher;
    return hasher(*Leaf::ptr_at(node, 0));
  }
  //@}

  //@{ Reference counting
  static constexpr void add_ref(node_const_ptr_type node) {
    if (node != nullptr)
      node->add_ref();
  }
  static constexpr void dec_ref(node_const_ptr_type node) {
    if (node != nullptr && node->dec_ref() == 0)
      destroy(const_cast<node_ptr_type>(node));
  }
  static constexpr ref_count_type ref_count(node_const_ptr_type node) {
    return (node == nullptr) ? 0 : node->ref_count();
  }
  //@}

  //@{ Path
  struct TreePath {
    std::array<node_ptr_type, MaxTrieDepth> nodes; //!< the path is {Branch, Branch, Branch}
    node_ptr_type leaf_end = nullptr;              //!< set if the path ends in a leaf
    uint32_t size = 0;                             //!< number of branch elements in path
    void push(node_ptr_type node) {
      assert(size + 1 < nodes.size());
      nodes[size++] = node;
    }
  };

  static constexpr TreePath make_path(node_ptr_type root, hash_type hash) {
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
  static constexpr node_ptr_type rewrite_branch_path(const TreePath& path, hash_type hash,
                                                     node_ptr_type new_node,
                                                     node_ptr_type leaf_end) {
    // When editing the tree, we need to make a copy of all the branch nodes,
    // Returning the head+tail, i.e, the new root, and the new branch node at the tip
    if (path.size == 0)
      return new_node; // The trivial case

    // Working backwards, so start with the tail
    const uint32_t last_level = path.size - 1;
    auto last_index = hash_chunk(hash, last_level);
    node_ptr_type iterator = path.nodes[last_level];
    if (Branch::is_valid_index(iterator, last_index)) {
      const auto dense_index = to_dense_index(last_index, iterator->payload_);
      const auto old_leaf = *Branch::dense_ptr_at(iterator, dense_index);
      const auto skip_index =
          (old_leaf->payload_ == leaf_end->payload_)           // means overwriting old_leaf
              ? 0xffffffffu                                    // because it has been extended
              : dense_index;                                   // so skip this index
      iterator = Branch::duplicate(iterator, skip_index);      // duplicate the branch node and
      *Branch::dense_ptr_at(iterator, dense_index) = new_node; // overwrite

    } else {
      // insert into the branch node -- expanding it
      iterator = Branch::insert_into_branch_node(iterator, new_node, last_index);
    }

    return rewrite_and_attach(path, hash, last_level, iterator);
  }

  static constexpr node_ptr_type rewrite_and_attach(const TreePath& path, hash_type hash,
                                                    uint32_t level, node_ptr_type splice_node) {
    auto iterator = splice_node;
    for (auto i = level; i > 0; --i) {
      auto* node = path.nodes[i - 1];
      const auto sparse_index = hash_chunk(hash, i - 1); // The overwrite position
      const auto dense_index = to_dense_index(sparse_index, node->payload_);
      auto* new_branch_node = Branch::duplicate(node, dense_index);   // Private copy
      *Branch::dense_ptr_at(new_branch_node, dense_index) = iterator; // Do the overwite
      iterator = new_branch_node;                                     // Update the current tail
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
  template <typename ItemType>
  static constexpr std::pair<node_ptr_type, node_ptr_type>
  branch_to_leaves(const hash_type item_hash, uint32_t level, node_ptr_type existing_leaf,
                   ItemType&& item) {
    assert(type(existing_leaf) == NodeType::Leaf);
    assert(Leaf::size(existing_leaf) > 0);

    const auto existing_hash = calculate_hash(*Leaf::ptr_at(existing_leaf, 0));

    // Trivial cases: Hash collision => create new leaf with new value appended
    if (item_hash == existing_hash) { // Trivial case: hash collision
      auto* new_leaf = Leaf::copy_append(existing_leaf, std::forward<ItemType>(item));
      return {new_leaf, new_leaf};
    }

    assert(level < MaxTrieDepth); // otherwise the hashes would have to have been equal

    // Otherwise, make {branch, branch, branch, ...} until the indices diverge
    node_ptr_type branch = nullptr;
    node_ptr_type tail = nullptr;
    uint32_t last_index = 0u;

    auto append_to_tail = [&](node_ptr_type new_branch, uint32_t index) {
      if (branch == nullptr) {
        branch = new_branch;                            // nothing to connect
      } else {                                          //
        *Branch::ptr_at(tail, last_index) = new_branch; // connect [tail => new_branch]
      }                                                 //
      tail = new_branch;                                // update the tail
      last_index = index;                               // store the index for next insert into tail
    };

    node_ptr_type new_leaf = nullptr;
    for (auto i = level; new_leaf == nullptr; ++i) {
      assert(i < MaxTrieDepth); // otherwise there'd have to be a hash collission
      const auto index_lhs = hash_chunk(existing_hash, i);
      const auto index_rhs = hash_chunk(item_hash, i);
      if (index_lhs == index_rhs) {
        append_to_tail(Branch::make_uninitialized(1, 1u << index_lhs), index_lhs);
      } else { // divergence
        new_leaf = Leaf::make(std::forward<ItemType>(item));
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

  template <typename key_or_item_type>
  static constexpr size_t calculate_hash(const key_or_item_type& value) {
    hasher hash_func;
    constexpr bool is_key = std::is_same<key_type, key_or_item_type>::value;
    if constexpr (is_key) {
      return hash_func(value);
    } else {
      return hash_func(value.first);
    }
  }

  template <typename U, typename V>
  static constexpr bool calculate_equals(const U& lhs, const V& rhs) {
    key_equal equal_func;
    constexpr bool lhs_is_key = std::is_same<key_type, U>::value;
    constexpr bool rhs_is_key = std::is_same<key_type, V>::value;
    if constexpr (lhs_is_key && rhs_is_key) {
      return equal_func(lhs, rhs);
    } else if constexpr (lhs_is_key && !rhs_is_key) {
      return equal_func(lhs, rhs.first);
    } else if constexpr (!lhs_is_key && rhs_is_key) {
      return equal_func(lhs.first, rhs);
    } else {
      return equal_func(lhs.first, rhs.first);
    }
  }

  template <typename key_or_item_type>
  static constexpr uint32_t get_index_in_leaf(node_const_ptr_type leaf,
                                              const key_or_item_type& key) {
    if (leaf != nullptr) {
      assert(type(leaf) == NodeType::Leaf);
      auto* start = Leaf::ptr_at(leaf, 0);
      for (auto* iterator = start; iterator != start + Leaf::size(leaf); ++iterator) {
        assert(calculate_hash(*start) == calculate_hash(*iterator));
        if (calculate_equals(key, *iterator))
          return static_cast<uint32_t>(iterator - start);
      }
    }
    return detail::NotAnIndex;
  }

  static constexpr node_ptr_type erase(node_ptr_type root, const key_type& key) {
    // 1. Find the node (if it's not found, return zero)
    // 2. Delete the leaf, and "roll up"
    const hash_type hash = calculate_hash(key);
    const auto path = make_path(root, hash);
    const auto leaf_index = get_index_in_leaf(path.leaf_end, key);

    auto other_sibling = [&](node_ptr_type node, uint32_t level) {
      // If a Branch node as two siblings, then this method
      // will return the sibling that is *not* on the path
      // referenced by the `hash_chunk(hash, level)`... that is, the "other sibling"
      assert(type(node) == NodeType::Branch);
      assert(size(node) == 2);

      const auto child_sparse_index = hash_chunk(hash, level);
      assert(Branch::is_valid_index(node, child_sparse_index));
      const auto child_dense_index = to_dense_index(child_sparse_index, node->payload_);
      assert(child_dense_index < 2);
      const auto other_dense_index = 1 - child_dense_index;

      // Okay, is the other node a leaf node?
      return *Branch::dense_ptr_at(node, other_dense_index);
    };

    if (leaf_index == NotAnIndex) { // value not in tree
      return root;
    }

    if (size(path.leaf_end) > 1) { // Special case: deleting from "many value" leaf
      auto new_leaf = Leaf::duplicate_leaf(path.leaf_end, leaf_index);
      return rewrite_branch_path(path, hash, new_leaf, new_leaf);
    }

    // While rolling up the tree, we may find a single leaf node
    // that we can attach higher up. This is called the `leaf_in_hand`
    // and is held and attached if appropriate
    node_ptr_type leaf_in_hand = nullptr;
    node_ptr_type new_root = nullptr;
    auto level = path.size;

    while (level > 0) {
      --level;
      node_ptr_type iterator = path.nodes[level];
      const auto iterator_size = size(iterator);

      if (leaf_in_hand == nullptr && iterator_size == 2) {
        node_ptr_type sibling = other_sibling(iterator, level);
        if (type(sibling) == NodeType::Leaf) {
          leaf_in_hand = sibling;
          add_ref(leaf_in_hand); // this will be attached elsewhere
          continue;
        }
      }

      if (iterator_size > 1) {
        const auto last_index = hash_chunk(hash, level);
        const auto skip_index = to_dense_index(last_index, iterator->payload_);
        node_ptr_type new_node = nullptr;
        if (leaf_in_hand != nullptr) {
          new_node = Branch::duplicate(iterator, skip_index);         // Duplicate the branch node
          *Branch::dense_ptr_at(new_node, skip_index) = leaf_in_hand; // Overwrite with leaf
        } else {
          new_node = Branch::remove_from_branch_node(iterator, last_index);
        }
        return rewrite_and_attach(path, hash, level, new_node);
      }
    }

    // The tree is empty... so return the leaf in hand.
    // (If null, this would be deleting the last node.)
    return leaf_in_hand;
  }
  //@}

  template <typename ItemType>
  static constexpr node_ptr_type do_insert(node_ptr_type root, ItemType&& item) {
    if (root == nullptr) { // inserting into an empty tree: trivial case
      return Leaf::make(std::forward<ItemType>(item));
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
    const auto hash = calculate_hash(item);
    const auto path = make_path(root, hash);
    const auto leaf_index = get_index_in_leaf(path.leaf_end, item);

    if (leaf_index != NotAnIndex) {
      return root; // (2.a) Duplicate!
    }
    return finish_insert(hash, path, std::forward<ItemType>(item));
  }

  template <typename K, typename V>
  static constexpr node_ptr_type do_key_value_insert(node_ptr_type root, K&& key, V&& value) {
    assert(IsMap);
    if (root == nullptr) { // inserting into an empty tree: trivial case
      return Leaf::make(item_type{std::forward<K>(key), std::forward<V>(value)});
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
    const auto hash = calculate_hash(key);
    const auto path = make_path(root, hash);
    const auto leaf_index = get_index_in_leaf(path.leaf_end, key);

    if (leaf_index != NotAnIndex) { // Duplicate, so overwrite node
      auto new_leaf = Leaf::duplicate_leaf_with_overwrite(
          path.leaf_end, leaf_index, item_type{std::forward<K>(key), std::forward<V>(value)});
      auto ret = rewrite_branch_path(path, hash, new_leaf, new_leaf);
      if (path.size > 0)
        dec_ref(path.leaf_end); // rewrite_branch_path doesn't know to let this node go
      return ret;
    }
    return finish_insert(hash, path, item_type{std::forward<K>(key), std::forward<V>(value)});
  }

  template <typename ItemType>
  static constexpr node_ptr_type finish_insert(hash_type hash, const TreePath& path,
                                               ItemType&& item) {
    if (path.leaf_end == nullptr) {
      auto new_leaf = Leaf::make(std::forward<ItemType>(item));
      return rewrite_branch_path(path, hash, new_leaf, new_leaf);
    }
    auto [new_branch, leaf_end] =
        branch_to_leaves(hash,                          // Hash of the value
                         path.size,                     // The path starts here
                         path.leaf_end,                 // Includes this node
                         std::forward<ItemType>(item)); // Must include this new value
    return rewrite_branch_path(path, hash, new_branch, leaf_end);
  }

  static constexpr const item_type* find(node_const_ptr_type root, const key_type& key) {
    const auto hash = calculate_hash(key);
    auto node = root;
    auto level = 0u;
    if (node != nullptr) {
      while (type(node) == NodeType::Branch) {
        const auto sparse_index = hash_chunk(hash, level++);
        if (Branch::is_valid_index(node, sparse_index))
          node = *Branch::ptr_at(node, sparse_index);
        else
          break;
      }
      if (type(node) == NodeType::Leaf) {
        auto* start = Leaf::ptr_at(node, 0);
        auto* finish = start + Leaf::size(node);
        for (auto iterator = start; iterator != finish; ++iterator) {
          if (calculate_equals(*iterator, key))
            return iterator;
        }
      }
    }
    return nullptr;
  }
};

template <typename NodeOps, bool is_const_reference> class Iterator {
public:
  using iterator_category = std::bidirectional_iterator_tag;
  using item_type = typename NodeOps::item_type;
  using reference_type =
      typename std::conditional<is_const_reference, const item_type&, item_type&>::type;
  using pointer_type =
      typename std::conditional<is_const_reference, const item_type*, item_type*>::type;
  using node_type = typename NodeOps::node_type;
  using node_ptr_type = node_type*;
  using node_const_ptr_type = const node_type*;

private:
  static constexpr uint32_t NotADepth{static_cast<uint32_t>(-1)};
  std::array<node_ptr_type, MaxTrieDepth + 1> path_; // +1 for the leaf node
  std::array<uint32_t, MaxTrieDepth + 1> position_;  // of iteration in path_
  uint32_t depth_;

public:
  struct MakeBeginTag {};
  struct MakeEndTag {};

  constexpr Iterator(node_ptr_type root, MakeBeginTag tag) : depth_{0} {
    path_[0] = root;
    position_[0] = 0;
    if (root == nullptr)
      depth_ = NotADepth;          // At the end
    else                           //
      descend_to_leftmost_leaf_(); // Find first leaf
    assert_invariant_();
  }

  constexpr Iterator(node_ptr_type root, MakeEndTag tag) : depth_{NotADepth} {
    path_[0] = root;
    assert_invariant_();
  }

  constexpr bool operator==(const Iterator& other) const {
    return (is_end_() && other.is_end_())        // both at `end()`
           || (depth_ == other.depth_            // at same depth (so neither at `end()`)
               && current_() == other.current_() // pointer and cursor
               && cursor_() == other.cursor_()); // are same
  }

  constexpr bool operator!=(const Iterator& other) const { return !(*this == other); }

  constexpr Iterator& operator++() {
    increment_();
    return *this;
  }

  constexpr Iterator& operator--() {
    decrement_();
    return *this;
  }

  constexpr Iterator operator++(int) {
    Iterator tmp{*this};
    increment_();
    return tmp;
  }

  constexpr Iterator operator--(int) {
    Iterator tmp = *this;
    --(*this);
    return tmp;
  }

  constexpr reference_type operator*() const { return *operator->(); }

  constexpr pointer_type operator->() const {
    assert(current_() != nullptr);
    assert(!is_end_());
    assert(NodeOps::type(current_()) == NodeType::Leaf);
    assert(cursor_() < NodeOps::size(current_()));
    return NodeOps::Leaf::ptr_at(current_(), cursor_());
  }

private:
  constexpr void assert_invariant_() const {
    assert(is_end_() || NodeOps::type(current_()) == NodeType::Leaf);
  }

  constexpr bool is_branch_(node_ptr_type node) const {
    return NodeOps::type(node) == NodeType::Branch;
  }
  constexpr uint32_t is_end_() const { return depth_ == NotADepth; }
  constexpr node_ptr_type current_() const { return path_[depth_]; }
  constexpr uint32_t& cursor_() { return position_[depth_]; }
  constexpr const uint32_t& cursor_() const { return position_[depth_]; }

  constexpr void increment_() {
    if (is_end_()) {
      return; // Attempt to increment beyond the end of the collection
    }

    assert(NodeOps::type(current_()) == NodeType::Leaf); // precondition
    if (++cursor_() < NodeOps::size(current_()))
      return;

    while (true) {
      if (depth_ == 0) {
        depth_ = NotADepth; // We're beyond the end,
        break;              // and we're done
      }
      --depth_;                                      // iterate upwards
      assert(is_branch_(current_()));                // precondition
      if (++cursor_() < NodeOps::size(current_())) { // found a new node (branch or leaf)
        descend_to_leftmost_leaf_();                 // increments depth (if branch)
        break;                                       // and we're done
      }
    }

    assert_invariant_();
  }

  constexpr void decrement_() {
    if (is_end_()) {
      if (path_[0] != nullptr) {                   // if not an empty tree
        depth_ = 0;                                // Reset to root
        cursor_() = NodeOps::size(current_()) - 1; //
        descend_to_rightmost_leaf_();              // And find rightmost leaf
      }
      return;
    }

    assert(NodeOps::type(current_()) == NodeType::Leaf); // precondition
    if (cursor_() > 0) {
      --cursor_();
      return;
    }

    while (true) {
      if (depth_ == 0) {
        depth_ = NotADepth; // We're beyond the end,
        break;              // and we're done
      }
      --depth_;
      assert(is_branch_(current_()));
      if (cursor_() > 0) {
        --cursor_();
        descend_to_rightmost_leaf_();
        break;
      }
    }

    assert_invariant_();
  }

  constexpr void descend_to_leftmost_leaf_() {
    while (is_branch_(current_())) { // Descend to left-most leaf
      assert(depth_ + 1 < path_.size());
      path_[depth_ + 1] = *NodeOps::Branch::dense_ptr_at(current_(), cursor_());
      position_[depth_ + 1] = 0;
      ++depth_;
    }

    assert_invariant_();
  }

  constexpr void descend_to_rightmost_leaf_() {
    while (is_branch_(current_())) {
      assert(depth_ + 1 < path_.size());
      path_[depth_ + 1] = *NodeOps::Branch::dense_ptr_at(current_(), cursor_());
      position_[depth_ + 1] = NodeOps::size(path_[depth_ + 1]) - 1;
      assert(NodeOps::size(path_[depth_ + 1]) > 0);
      ++depth_;
    }
    assert_invariant_();
  }
};

// ---------------------------------------------------------------------------------------- base_set

template <typename KeyType,                           // The type used for Hash/KeyEqual
          typename ValueType,                         // Value Type for maps
          typename Hash = std::hash<KeyType>,         // Hash function for item
          typename KeyEqual = std::equal_to<KeyType>, // Equality comparision for Item
          bool IsMap = false,                         // True if map
          bool IsThreadSafe = true                    // True if Set is threadsafe
          >
class base_set {
private:
  using Ops = detail::NodeOps<KeyType, ValueType, Hash, KeyEqual, IsMap, IsThreadSafe>;
  using node_type = typename Ops::node_type;
  using node_ptr_type = typename Ops::node_ptr_type;
  using node_const_ptr_type = typename Ops::node_const_ptr_type;

public:
  //@{
  using key_type = KeyType;
  using value_type = ValueType;
  using item_type = typename Ops::item_type;
  using size_type = typename Ops::size_type;
  using hash_type = typename Ops::hash_type;
  using hasher = Hash;
  using key_equal = KeyEqual;
  using reference = item_type&;
  using const_reference = const item_type&;
  using iterator = typename detail::Iterator<Ops, false>;
  using const_iterator = typename detail::Iterator<Ops, true>;
  static constexpr bool is_map = IsMap;
  static constexpr bool is_thread_safe = IsThreadSafe;
  //@}

  static_assert(IsMap || std::is_same<KeyType, ValueType>::value);

private:
  node_ptr_type root_{nullptr}; //!< Root of the tree could be branch of leaf
  std::size_t size_{0};         //!< Current size of the Set

public:
  //@{ Construction/Destruction
  constexpr base_set() = default;
  constexpr base_set(const base_set& other) { *this = other; }
  constexpr base_set(base_set&& other) noexcept { swap(other); }
  constexpr ~base_set() { Ops::dec_ref(root_); }

  template <typename InputIt> constexpr base_set(InputIt first, InputIt last) {
    insert(first, last);
  }
  constexpr base_set(std::initializer_list<item_type> ilist) {
    insert(std::begin(ilist), std::end(ilist));
  }

  template <typename Predicate> constexpr base_set erase_if(Predicate predicate) {
    auto copy = *this;
    for (const auto& item : *this) {
      if constexpr (IsMap) {
        if (predicate(item.first))
          copy.erase(item.first);
      } else {
        if (predicate(item))
          copy.erase(item);
      }
    }
    return copy;
  }
  //@}

  //@{ Assignment
  constexpr base_set& operator=(const base_set& other) {
    Ops::dec_ref(root_);
    root_ = other.root_;
    size_ = other.size_;
    Ops::add_ref(root_);
    return *this;
  }

  constexpr base_set& operator=(base_set&& other) noexcept {
    swap(other);
    return *this;
  }
  //@}

  //@{ Iterators
  constexpr iterator begin() { return iterator{root_, typename iterator::MakeBeginTag{}}; }
  constexpr const_iterator begin() const { return cbegin(); }
  constexpr const_iterator cbegin() const {
    return const_iterator{root_, typename const_iterator::MakeBeginTag{}};
  }

  constexpr iterator end() { return iterator{root_, typename iterator::MakeEndTag{}}; }
  constexpr const_iterator end() const { return cend(); }
  constexpr const_iterator cend() const {
    return const_iterator{root_, typename const_iterator::MakeEndTag{}};
  }
  //@}

  //@{ Capacity
  constexpr bool empty() const { return size() == 0; }
  constexpr std::size_t size() const { return size_; }
  static constexpr std::size_t max_size() { return std::numeric_limits<std::size_t>::max(); }
  //@}

  //@{ Modifiers
  constexpr void clear() {
    Ops::dec_ref(root_);
    root_ = nullptr;
    size_ = 0;
  }

  constexpr bool insert(const item_type& value) { return insert_(value); }

  template <typename P> constexpr bool insert(P&& value) { return insert_(std::forward<P>(value)); }

  template <typename InputIt> constexpr void insert(InputIt first, InputIt last) {
    while (first != last) {
      insert_cref_(*first);
      ++first;
    }
  }
  constexpr void insert(std::initializer_list<item_type> ilist) {
    for (auto&& item : ilist)
      insert_(std::move(item));
  }

  template <typename V> constexpr bool insert_or_assign(const key_type& key, V&& value) {
    return insert_or_assign_(key, std::forward<V>(value));
  }

  template <typename K, typename V> constexpr bool insert_or_assign(K&& key, V&& value) {
    return insert_or_assign_(key, std::forward<V>(value));
  }

  template <typename... Args> constexpr bool emplace(Args&&... args) {
    return insert(item_type{std::forward<Args>(args)...});
  }

  constexpr size_type erase(const key_type& key) { return erase_(key); }

  constexpr void swap(base_set& other) noexcept { // Should be able to swap onto itself
    std::swap(root_, other.root_);
    std::swap(size_, other.size_);
  }

  constexpr std::optional<item_type> extract(const key_type& key) {
    auto* opt = find(key);
    if (opt != nullptr) {
      // move-construct
      auto result = std::optional<item_type>{std::move(const_cast<item_type&>(*opt))};
      erase(key);
      return result;
    }
    return {};
  }
  //@}

  //@{ Lookup
  constexpr std::size_t count(const key_type& key) const { return contains(key); }
  constexpr const item_type* find(const key_type& key) const { return Ops::find(root_, key); }
  constexpr bool contains(const key_type& key) const { return Ops::find(root_, key) != nullptr; }
  //@}

  //@{ Observers
  static constexpr hasher hash_function() { return hasher{}; }
  static constexpr key_equal key_eq() { return key_equal{}; }
  //@}

  //@{ Friends
  friend constexpr bool operator==(const base_set& lhs, const base_set& rhs) noexcept {
    if (lhs.size() != rhs.size())
      return false;

    if (lhs.root_ == rhs.root_)
      return true;

    if constexpr (IsMap) {
      for (const auto& [key, value] : lhs) {
        if (!rhs.contains(key))
          return false;
      }
    } else {
      for (const auto& item : lhs) {
        if (!rhs.contains(item))
          return false;
      }
    }

    return true;
  }

  friend constexpr bool operator!=(const base_set& lhs, const base_set& rhs) noexcept {
    return !(lhs == rhs);
  }
  //@}

private:
  constexpr node_ptr_type get_root_() { return root_; }

  template <typename Value> constexpr bool insert_cref_(const Value& value) {
    return update_root_after_insert_(Ops::do_insert(root_, value));
  }

  template <typename Value> constexpr bool insert_(Value&& value) {
    return update_root_after_insert_(Ops::do_insert(root_, std::forward<Value>(value)));
  }

  template <typename K, typename V> constexpr bool insert_or_assign_(K&& key, V&& value) {
    return update_root_after_insert_(
        Ops::do_key_value_insert(root_, std::forward<K>(key), std::forward<V>(value)));
  }

  constexpr bool update_root_after_insert_(node_ptr_type new_root) {
    const bool success = (new_root != root_);

    if (success) {
      if (root_ != nullptr && Ops::type(root_) == NodeType::Leaf &&
          Ops::type(new_root) == NodeType::Branch) {
        // Never `dec_ref` a root_ "leaf node", when new_root is a Branch
        // because that leaf will have become part of the tree.
        //
        // If new_root is a branch then there was a collision at the root
      } else {
        Ops::dec_ref(root_);
      }
      root_ = new_root;
      ++size_;
    }
    return success;
  }

  constexpr size_type erase_(const key_type& key) {
    node_ptr_type new_root = Ops::erase(root_, key);
    const bool success = (new_root != root_);
    if (success) {
      Ops::dec_ref(root_);
      root_ = new_root;
      --size_;
      return 1;
    }
    return 0;
  }
};

} // namespace niggly::trie::detail
