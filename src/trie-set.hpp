
#pragma once

#include "_node-data.hpp"
#include "_base-node-ops.hpp"
#include "_node-ops.hpp"
#include "_iterator.hpp"
#include "_base-trie.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <type_traits>

#include <cstdint>
#include <cstring>
#include <cassert>
#include <cstdlib>

namespace niggly::trie {

// ---------------------------------------------------------------------------------- persistent_set

template <typename ItemType,                           // Type of item to store
          typename Hash = std::hash<ItemType>,         // Hash function for item
          typename KeyEqual = std::equal_to<ItemType>, // Equality comparision for Item
          bool IsThreadSafe = true                     // True if Set is threadsafe
          >
using persistent_set = detail::base_set<ItemType, ItemType, Hash, KeyEqual, false, IsThreadSafe>;

// template <typename ItemType,                           // Type of item to store
//           typename Hash = std::hash<ItemType>,         // Hash function for item
//           typename KeyEqual = std::equal_to<ItemType>, // Equality comparision for Item
//           bool IsThreadSafe = true                     // True if Set is threadsafe
//           >
// class persistent_set {
// private:
//   using Ops = detail::NodeOps<ItemType, Hash, KeyEqual, IsThreadSafe>;
//   using node_type = typename Ops::node_type;
//   using node_ptr_type = node_type*;
//   using node_const_ptr_type = const node_type*;
//   using NodeType = detail::NodeType;

// public:
//   //@{
//   using item_type = ItemType;
//   using size_type = std::size_t;
//   using difference_type = std::ptrdiff_t;
//   using hasher = Hash;
//   using key_equal = KeyEqual;
//   using reference = item_type&;
//   using const_reference = const item_type&;
//   using iterator = typename detail::Iterator<Ops, false>;
//   using const_iterator = typename detail::Iterator<Ops, true>;
//   static constexpr bool is_thread_safe = IsThreadSafe;
//   //@}

// private:
//   node_ptr_type root_{nullptr}; //!< Root of the tree could be branch of leaf
//   std::size_t size_{0};         //!< Current size of the Set

// public:
//   //@{ Construction/Destruction
//   persistent_set() = default;
//   persistent_set(const persistent_set& other) { *this = other; }
//   persistent_set(persistent_set&& other) noexcept { swap(other); }
//   ~persistent_set() { Ops::dec_ref(root_); }

//   template <typename Predicate> persistent_set erase_if(Predicate predicate) {
//     auto copy = *this;
//     for (const auto& item : *this) {
//       if (predicate(item)) {
//         copy.erase(item);
//       }
//     }
//     return copy;
//   }
//   //@}

//   //@{ Assignment
//   persistent_set& operator=(const persistent_set& other) {
//     Ops::dec_ref(root_);
//     root_ = other.root_;
//     size_ = other.size_;
//     Ops::add_ref(root_);
//     return *this;
//   }

//   persistent_set& operator=(persistent_set&& other) noexcept {
//     swap(other);
//     return *this;
//   }
//   //@}

//   //@{ Iterators
//   iterator begin() { return iterator{root_, typename iterator::MakeBeginTag{}}; }
//   const_iterator begin() const { return cbegin(); }
//   const_iterator cbegin() const {
//     return const_iterator{root_, typename const_iterator::MakeBeginTag{}};
//   }

//   iterator end() { return iterator{root_, typename iterator::MakeEndTag{}}; }
//   const_iterator end() const { return cend(); }
//   const_iterator cend() const {
//     return const_iterator{root_, typename const_iterator::MakeEndTag{}};
//   }
//   //@}

//   //@{ Capacity
//   bool empty() const { return size() == 0; }
//   std::size_t size() const { return size_; }
//   std::size_t max_size() const { return std::numeric_limits<std::size_t>::max(); }
//   //@}

//   //@{ Modifiers
//   void clear() {
//     Ops::dec_ref(root_);
//     root_ = nullptr;
//     size_ = 0;
//   }

//   bool insert(const item_type& value) { return insert_(value); }
//   bool insert(item_type&& value) { return insert_(std::move(value)); }
//   template <class InputIt> void insert(InputIt first, InputIt last) {
//     while (first != last) {
//       insert_(*first);
//       ++first;
//     }
//   }
//   void insert(std::initializer_list<item_type> ilist) {
//     for (auto&& item : ilist)
//       insert_(std::move(item));
//   }

//   template <class... Args> bool emplace(Args&&... args) {
//     return insert(item_type{std::forward<Args>(args)...});
//   }

//   size_type erase(const item_type& key) { return erase_(key); }

//   void swap(persistent_set& other) noexcept { // Should be able to swap onto itself
//     std::swap(root_, other.root_);
//     std::swap(size_, other.size_);
//   }

//   std::optional<item_type> extract(const item_type& key) {
//     auto result = find(key);
//     erase(key);
//     return result;
//   }
//   //@}

//   //@{ Lookup
//   std::size_t count(const item_type& key) const { return contains(key); }
//   std::optional<item_type> find(const item_type& key) const {
//     auto* ptr = Ops::find(root_, key);
//     if (ptr != nullptr)
//       return {*ptr};
//     return {};
//   }
//   bool contains(const item_type& key) const { return Ops::find(root_, key) != nullptr; }
//   //@}

//   //@{ Observers
//   hasher hash_function() const { return hasher{}; }
//   key_equal key_eq() const { return key_equal{}; }
//   //@}

//   //@{ Friends
//   friend bool operator==(const persistent_set& lhs, const persistent_set& rhs) noexcept {
//     return lhs.root_ == rhs.root_;
//   }

//   friend bool operator!=(const persistent_set& lhs, const persistent_set& rhs) noexcept {
//     return !(lhs == rhs);
//   }

//   friend void swap(persistent_set& lhs, persistent_set& rhs) noexcept { lhs.swap(rhs); }

//   template <typename Predicate>
//   friend size_type erase_if(persistent_set& set, Predicate predicate) {
//     const auto size_0 = set.size();
//     set = set.erase_if(std::forward<Predicate>(predicate));
//     return size_0 - set.size();
//   }
//   //@}

// private:
//   node_ptr_type get_root_() { return root_; }

//   template <typename Value> bool insert_(Value&& value) {
//     node_ptr_type new_root = Ops::do_insert(root_, std::forward<Value>(value));
//     const bool success = (new_root != root_);

//     if (success) {
//       if (root_ != nullptr && Ops::type(root_) == NodeType::Leaf &&
//           Ops::type(new_root) == NodeType::Branch) {
//         // Never `dec_ref` a root_ "leaf node", when new_root is a Branch
//         // because that leaf will have become part of the tree.
//         //
//         // If new_root is a branch then there was a collision at the root
//       } else {
//         Ops::dec_ref(root_);
//       }
//       root_ = new_root;
//       ++size_;
//     }
//     return success;
//   }

//   size_type erase_(const item_type& key) {
//     hasher hash_func;
//     return erase_(hash_func(key), [&key](const item_type& item) {
//       key_equal f;
//       return f(key, item);
//     });
//   }

//   template <typename Predicate> size_type erase_(std::size_t hash, Predicate&& predicate) {
//     node_ptr_type new_root = Ops::erase(root_, hash, std::forward<Predicate>(predicate));
//     const bool success = (new_root != root_);
//     if (success) {
//       Ops::dec_ref(root_);
//       root_ = new_root;
//       --size_;
//       return 1;
//     }
//     return 0;
//   }
// };

template <typename KeyType,                           // Key type to reference values
          typename ValueType,                         // Value type for item stored
          typename Hasher = std::hash<KeyType>,       // Hash function for item
          typename KeyEqual = std::equal_to<KeyType>, // Equality comparision for Item
          bool IsThreadSafe = true                    // True if Set is threadsafe
          >
class persistent_map {
private:
  using item_type_ = typename std::pair<KeyType, ValueType>;
  struct ItemHasher {
    std::size_t operator()(const item_type_& item) const {
      Hasher hasher;
      return hasher(item.first);
    }
  };

  struct ItemEquals {
    bool operator()(const item_type_& lhs, const item_type_& rhs) const {
      KeyEqual key_eq;
      return key_eq(lhs.first, rhs.first);
    }
  };

public:
  //@{
  using key_type = KeyType;
  using value_type = ValueType;
  using item_type = typename std::pair<KeyType, ValueType>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using hasher = Hasher;
  using key_equal = KeyEqual;
  using reference = item_type&;
  using const_reference = const item_type&;

private:
  using set_type = persistent_set<item_type, ItemHasher, ItemEquals, IsThreadSafe>;

  struct ItemPredicate {
    const key_type& key_;
    ItemPredicate(const key_type& key) : key_(key) {}
    bool operator()(const item_type_& item) const {
      KeyEqual key_eq;
      return key_eq(key_, item.first);
    }
  };

public:
  using iterator = typename set_type::iterator;
  using const_iterator = typename set_type::const_iterator;
  static constexpr bool is_thread_safe = IsThreadSafe;
  //@}

private:
  set_type set_;

public:
  //@{ Construction/Destruction
  persistent_map() = default;
  persistent_map(const persistent_map& other) = default;
  persistent_map(persistent_map&& other) noexcept = default;
  ~persistent_map() = default;
  //@}

  //@{ Assignment
  persistent_map& operator=(const persistent_map& other) = default;
  persistent_map& operator=(persistent_map&& other) noexcept = default;
  //@}

  //@{ Iterators
  iterator begin() { return set_.begin(); }
  const_iterator begin() const { return set_.begin(); }
  const_iterator cbegin() const { return set_.cbegin(); }

  iterator end() { return set_.end(); }
  const_iterator end() const { return set_.end(); }
  const_iterator cend() const { return set_.cend(); }
  //@}

  //@{ Capacity
  bool empty() const { return set_.empty(); }
  std::size_t size() const { return set_.size(); }
  std::size_t max_size() const { return set_.max_size(); }
  //@}

  //@{ Modifiers
  void clear() { set_.clear(); }
  bool insert(const item_type& value) { return set_.insert(value); }
  bool insert(item_type&& value) { return set_.insert(std::move(value)); }
  template <class InputIt> void insert(InputIt first, InputIt last) { set_.insert(first, last); }
  void insert(std::initializer_list<item_type> ilist) { set_.insert(std::move(ilist)); }
  template <class... Args> bool emplace(Args&&... args) {
    return set_.emplace(std::forward<Args>(args)...);
  }
  void swap(persistent_map& other) noexcept { set_.swap(other.set_); }

  size_type erase(const key_type& key) {
    hasher hash_func;
    return set_.erase_(hash_func(key), [&key](const item_type& item) {
      key_equal f;
      return f(key, item.first);
    });
  }
  // template<class...Args> bool try_emplace(const key_type& k, Args&&... args);
  // template <class M> bool insert_or_assign(const key_type& k, M&& obj);
  // template <class M> bool insert_or_assign(key_type&& k, M&& obj);
  std::optional<item_type> extract(const key_type& key) {
    auto result = find(key);
    if (result.has_value())
      erase(key);
    return result;
  }

  template <typename Predicate> persistent_map erase_if(Predicate predicate) {
    persistent_map new_map;
    new_map.set_ =
        set_.erase_if([&predicate](const item_type& item) { return predicate(item.first); });
    return new_map;
  }
  //@}

  //@{ Lookup
  // const value_type& at(const key_type& key) const;         // std::out_of_range
  // const value_type& operator[](const key_type& key) const; // std::out_of_range
  std::size_t count(const key_type& key) const { return contains(key); }
  std::optional<item_type> find(const key_type& key) const {
    auto* ptr = find_(key);
    if (ptr != nullptr)
      return {*ptr};
    return {};
  }
  bool contains(const key_type& key) const { return find_(key) != nullptr; }
  //@}

  //@{ Observers
  hasher hash_function() const { return hasher{}; }
  key_equal key_eq() const { return key_equal{}; }
  //@}

  //@{ Friends
  friend bool operator==(const persistent_map& lhs, const persistent_map& rhs) noexcept {
    return lhs.set_ == rhs.set_;
  }

  friend bool operator!=(const persistent_map& lhs, const persistent_map& rhs) noexcept {
    return !(lhs == rhs);
  }

  friend void swap(persistent_map& lhs, persistent_map& rhs) noexcept { lhs.swap(rhs); }

  template <typename Predicate>
  friend size_type erase_if(persistent_map& map, Predicate predicate) {
    return erase_if(map.set_,
                    [&predicate](const item_type& item) { return predicate(item.first); });
  }
  //@}

private:
  const item_type* find_(const key_type& key) const {
    hasher hash_func;
    return set_type::Ops::find_if(set_.root_, hash_func(key), [&key](const item_type& item) {
      key_equal f;
      return f(key, item.first);
    });
  }
};

} // namespace niggly::trie
