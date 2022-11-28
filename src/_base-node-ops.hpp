
#pragma once

#include "_node-data.hpp"

namespace niggly::trie::detail {

// ------------------------------------------------------------------------------------- BaseNodeOps

template <typename T, bool IsThreadSafe = true, bool IsBranchNode = false> struct BaseNodeOps {

  using node_type = NodeData<IsThreadSafe>;
  using node_ptr_type = node_type*;
  using node_const_ptr_type = const node_type*;
  using item_type = T;
  using hash_type = std::size_t;
  using node_size_type = typename node_type::node_size_type;
  using ref_count_type = typename node_type::ref_count_type;

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

  static NodeType type(node_const_ptr_type node) { return node->type(); }

  static std::size_t size(node_const_ptr_type node) {
    if constexpr (IsBranchNode) {
      return popcount(node->payload_);
    } else {
      return node->payload_;
    }
  }

  //@{ Member access
  static bool is_valid_index(node_const_ptr_type node, node_size_type index) {
    if constexpr (IsBranchNode) {
      return ::niggly::trie::detail::is_valid_index(index, node->payload_);
    } else {
      return index < node->payload_;
    }
  }

  static item_type* ptr_at(node_const_ptr_type node, node_size_type index) {
    if constexpr (IsBranchNode) {
      assert(index < 32);
      return dense_ptr_at(node, to_dense_index(index, node->payload_));
    } else {
      return dense_ptr_at(node, index);
    }
  }

  static item_type* dense_ptr_at(node_const_ptr_type node, node_size_type index) {
    auto ptr_idx = reinterpret_cast<uintptr_t>(node) + offset_at(index);
    assert(ptr_idx % alignof(item_type) == 0); // never unaligned access
    return reinterpret_cast<item_type*>(ptr_idx);
  }

  static item_type* begin(node_const_ptr_type node) { return ptr_at(node, 0); }

  static item_type* end(node_const_ptr_type node) { return ptr_at(node, 0) + size(node); }
  //@}

  //@{ Utility
  static node_ptr_type make_uninitialized(node_size_type size, node_size_type payload) {
    auto ptr = static_cast<node_ptr_type>(std::aligned_alloc(AlignOf, storage_size(size)));
    new (ptr) node_type{DefaultType, payload};
    return ptr;
  }

  //@}

  //@{ Factory/Destruct
  static node_ptr_type make_empty() { return make_uninitialized(0, 0); }
  //@}
};

// ------------------------------------------------------------------------------------- LeafNodeOps

template <typename T, bool IsThreadSafe = true>
struct LeafNodeOps : public BaseNodeOps<T, IsThreadSafe, false> {

  using Base = BaseNodeOps<T, IsThreadSafe, false>;

  using node_type = NodeData<IsThreadSafe>;
  using node_ptr_type = node_type*;
  using node_const_ptr_type = const node_type*;
  using item_type = T;
  using hash_type = std::size_t;
  using node_size_type = typename node_type::node_size_type;
  using ref_count_type = typename node_type::ref_count_type;

  static void initialize_one(const item_type& src, item_type* dst) { copy_one(src, dst); }

  static void copy_one(const item_type& src, item_type* dst) {
    if constexpr (std::is_trivial<item_type>::value) {
      std::memcpy(dst, &src, sizeof(item_type));
    } else {
      static_assert(std::is_copy_constructible<item_type>::value);
      new (dst) item_type{src};
    }
  }

  static void initialize_one(item_type&& src, item_type* dst) {
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

  static void copy_payload_to(node_const_ptr_type src, node_ptr_type dst) {
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

  static node_ptr_type make(const item_type& value) {
    auto* ptr = Base::make_uninitialized(1, 1);
    initialize_one(value, Base::ptr_at(ptr, 0));
    return ptr;
  }

  static node_ptr_type make(item_type&& value) {
    auto* ptr = Base::make_uninitialized(1, 1);
    initialize_one(std::move(value), Base::ptr_at(ptr, 0));
    return ptr;
  }

  /**
   * Duplicates a leaf node, optionally omitting the value at `index_to_skip`
   */
  static node_ptr_type duplicate_leaf(node_const_ptr_type node, uint32_t index_to_skip) {
    assert(node->type() == NodeType::Leaf); // and this too
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
   * Creates a new leaf node, with values copied, and `value` at the end
   */
  template <typename Value>
  static node_ptr_type copy_append(node_const_ptr_type src, Value&& value) {
    const auto sz = Base::size(src);
    auto new_node = Base::make_uninitialized(sz + 1, sz + 1);
    copy_payload_to(src, new_node);
    initialize_one(std::forward<Value>(value), Base::ptr_at(new_node, sz));
    return new_node;
  }
};

// ----------------------------------------------------------------------------------- BranchNodeOps

template <bool IsThreadSafe = true>
struct BranchNodeOps : public BaseNodeOps<NodeData<IsThreadSafe>*, IsThreadSafe, true> {

  using Base = BaseNodeOps<NodeData<IsThreadSafe>*, IsThreadSafe, true>;

  using node_type = NodeData<IsThreadSafe>;
  using node_ptr_type = node_type*;
  using node_const_ptr_type = const node_type*;
  using item_type = node_ptr_type;
  using hash_type = std::size_t;
  using node_size_type = typename node_type::node_size_type;
  using ref_count_type = typename node_type::ref_count_type;

  /**
   * Duplicate a Branch node, copying the values, perhaps skipping an index that is being
   * overwritten
   */
  static node_ptr_type duplicate(node_ptr_type node, uint32_t dense_index_to_skip) {
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
  static node_ptr_type remove_from_branch_node(node_ptr_type node,
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
  static node_ptr_type insert_into_branch_node(node_const_ptr_type src, item_type value,
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

} // namespace niggly::trie::detail
