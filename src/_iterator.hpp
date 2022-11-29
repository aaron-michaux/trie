
#pragma once

#include "_node-ops.hpp"

namespace niggly::trie::detail {

// Forward? Bidirection?
template <typename NodeOps, bool is_const_reference> class Iterator {
public:
  using iterator_category = std::bidirectional_iterator_tag;
  using item_type = typename NodeOps::item_type;
  using reference_type = std::conditional<is_const_reference, const item_type&, item_type&>::type;
  using pointer_type = std::conditional<is_const_reference, const item_type*, item_type*>::type;
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
    Iterator tmp = *this;
    ++(*this);
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

} // namespace niggly::trie::detail
