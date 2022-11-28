
#pragma once

#include "_node-data.hpp"
#include "_base-node-ops.hpp"
#include "_node-ops.hpp"
#include "_iterator.hpp"

namespace niggly::trie::detail {

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
  using node_ptr_type = node_type*;
  using node_const_ptr_type = const node_type*;
  using NodeType = detail::NodeType;

public:
  //@{
  using key_type = KeyType;
  using value_type = ValueType;
  using item_type = typename Ops::item_type;
  using size_type = std::size_t;
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
  base_set() = default;
  base_set(const base_set& other) { *this = other; }
  base_set(base_set&& other) noexcept { swap(other); }
  ~base_set() { Ops::dec_ref(root_); }

  template <typename Predicate> base_set erase_if(Predicate predicate) {
    auto copy = *this;
    for (const auto& item : *this) {
      if (predicate(item)) {
        copy.erase(item);
      }
    }
    return copy;
  }
  //@}

  //@{ Assignment
  base_set& operator=(const base_set& other) {
    Ops::dec_ref(root_);
    root_ = other.root_;
    size_ = other.size_;
    Ops::add_ref(root_);
    return *this;
  }

  base_set& operator=(base_set&& other) noexcept {
    swap(other);
    return *this;
  }
  //@}

  //@{ Iterators
  iterator begin() { return iterator{root_, typename iterator::MakeBeginTag{}}; }
  const_iterator begin() const { return cbegin(); }
  const_iterator cbegin() const {
    return const_iterator{root_, typename const_iterator::MakeBeginTag{}};
  }

  iterator end() { return iterator{root_, typename iterator::MakeEndTag{}}; }
  const_iterator end() const { return cend(); }
  const_iterator cend() const {
    return const_iterator{root_, typename const_iterator::MakeEndTag{}};
  }
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

  bool insert(const item_type& value) { return insert_(value); }
  bool insert(item_type&& value) { return insert_(std::move(value)); }
  template <class InputIt> void insert(InputIt first, InputIt last) {
    while (first != last) {
      insert_(*first);
      ++first;
    }
  }
  void insert(std::initializer_list<item_type> ilist) {
    for (auto&& item : ilist)
      insert_(std::move(item));
  }

  template <class... Args> bool emplace(Args&&... args) {
    return insert(item_type{std::forward<Args>(args)...});
  }

  size_type erase(const item_type& key) { return erase_(key); }

  void swap(base_set& other) noexcept { // Should be able to swap onto itself
    std::swap(root_, other.root_);
    std::swap(size_, other.size_);
  }

  std::optional<item_type> extract(const item_type& key) {
    auto result = find(key);
    erase(key);
    return result;
  }
  //@}

  //@{ Lookup
  std::size_t count(const item_type& key) const { return contains(key); }
  std::optional<item_type> find(const item_type& key) const {
    auto* ptr = Ops::find(root_, key);
    if (ptr != nullptr)
      return {*ptr};
    return {};
  }
  bool contains(const item_type& key) const { return Ops::find(root_, key) != nullptr; }
  //@}

  //@{ Observers
  hasher hash_function() const { return hasher{}; }
  key_equal key_eq() const { return key_equal{}; }
  //@}

  //@{ Friends
  friend bool operator==(const base_set& lhs, const base_set& rhs) noexcept {
    return lhs.root_ == rhs.root_;
  }

  friend bool operator!=(const base_set& lhs, const base_set& rhs) noexcept {
    return !(lhs == rhs);
  }

  friend void swap(base_set& lhs, base_set& rhs) noexcept { lhs.swap(rhs); }

  template <typename Predicate> friend size_type erase_if(base_set& set, Predicate predicate) {
    const auto size_0 = set.size();
    set = set.erase_if(std::forward<Predicate>(predicate));
    return size_0 - set.size();
  }
  //@}

private:
  node_ptr_type get_root_() { return root_; }

  template <typename Value> bool insert_(Value&& value) {
    node_ptr_type new_root = Ops::do_insert(root_, std::forward<Value>(value));
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

  size_type erase_(const item_type& key) {
    hasher hash_func;
    return erase_(hash_func(key), [&key](const item_type& item) {
      key_equal f;
      return f(key, item);
    });
  }

  template <typename Predicate> size_type erase_(std::size_t hash, Predicate&& predicate) {
    node_ptr_type new_root = Ops::erase(root_, hash, std::forward<Predicate>(predicate));
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
