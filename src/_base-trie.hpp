
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
