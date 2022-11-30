
#pragma once

#include "_base-node-ops.hpp"

namespace niggly::trie::detail {

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

  static constexpr bool is_valid_index(node_const_ptr_type node) {
    return (node->type() == NodeType::Branch) ? Branch::is_valid_index(node)
                                              : Leaf::is_valid_index(node);
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

} // namespace niggly::trie::detail
