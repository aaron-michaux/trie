
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
#include <stdexcept>
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
class persistent_set {
private:
  using set_type = detail::base_set<ItemType, ItemType, Hash, KeyEqual, false, IsThreadSafe>;
  set_type set_;

public:
  using item_type = typename set_type::item_type;
  using size_type = typename set_type::size_type;
  using hash_type = typename set_type::hash_type;
  using hasher = typename set_type::hasher;
  using key_equal = typename set_type::key_equal;
  using reference = typename set_type::reference;
  using const_reference = typename set_type::const_reference;
  using iterator = typename set_type::iterator;
  using const_iterator = typename set_type::const_iterator;
  static constexpr bool is_thread_safe = IsThreadSafe;

  //@{ Construction/Destruction
  constexpr persistent_set() = default;
  constexpr persistent_set(const persistent_set& other) = default;
  constexpr persistent_set(persistent_set&& other) noexcept = default;
  constexpr ~persistent_set() = default;

  template <typename InputIt> constexpr persistent_set(InputIt first, InputIt last) {
    set_.insert(first, last);
  }
  constexpr persistent_set(std::initializer_list<item_type> ilist) {
    set_.insert(std::begin(ilist), std::end(ilist));
  }

  template <typename Predicate> persistent_set constexpr erase_if(Predicate&& predicate) {
    auto copy = *this;
    copy.set_ = set_.erase_if(std::forward<Predicate>(predicate));
    return copy;
  }
  //@}

  //@{ Assignment
  constexpr persistent_set& operator=(const persistent_set& other) = default;
  constexpr persistent_set& operator=(persistent_set&& other) noexcept = default;
  //@}

  //@{ Iterators
  constexpr iterator begin() { return set_.begin(); }
  constexpr const_iterator begin() const { return set_.begin(); }
  constexpr const_iterator cbegin() const { return set_.cbegin(); }

  constexpr iterator end() { return set_.end(); }
  constexpr const_iterator end() const { return set_.end(); }
  constexpr const_iterator cend() const { return set_.cend(); }
  //@}

  //@{ Capacity
  constexpr bool empty() const { return set_.empty(); }
  constexpr std::size_t size() const { return set_.size(); }
  static constexpr std::size_t max_size() { return set_type::max_size(); }
  //@}

  //@{ Modifiers
  constexpr void clear() { set_.clear(); }

  constexpr bool insert(const item_type& value) { return set_.insert(value); }
  constexpr bool insert(item_type&& value) { return set_.insert(std::move(value)); }
  template <class InputIt> constexpr void insert(InputIt first, InputIt last) {
    set_.insert(first, last);
  }
  constexpr void insert(std::initializer_list<item_type> ilist) { set_.insert(ilist); }

  template <class... Args> constexpr bool emplace(Args&&... args) {
    return set_.emplace(std::forward<Args>(args)...);
  }

  constexpr size_type erase(const item_type& key) { return set_.erase(key); }
  constexpr void swap(persistent_set& other) noexcept { set_.swap(other.set_); }

  constexpr std::optional<item_type> extract(const item_type& key) { return set_.extract(key); }
  //@}

  //@{ Lookup
  constexpr std::size_t count(const item_type& key) const { return set_.count(key); }
  constexpr const item_type* find(const item_type& key) const { return set_.find(key); }
  constexpr bool contains(const item_type& key) const { return set_.contains(key); }
  //@}

  //@{ Observers
  static constexpr hasher hash_function() { return set_type::hash_function(); }
  static constexpr key_equal key_eq() { return set_type::key_eq(); }
  //@}

  //@{ Friends
  friend constexpr bool operator==(const persistent_set& lhs, const persistent_set& rhs) noexcept {
    return lhs.set_ == rhs.set_;
  }

  friend constexpr bool operator!=(const persistent_set& lhs, const persistent_set& rhs) noexcept {
    return lhs.set_ != rhs.set_;
  }

  friend constexpr void swap(persistent_set& lhs, persistent_set& rhs) noexcept { lhs.swap(rhs); }

  template <typename Predicate>
  friend constexpr size_type erase_if(persistent_set& set, Predicate&& predicate) {
    const auto size_0 = set.size();
    set = set.erase_if(std::forward<Predicate>(predicate));
    return size_0 - set.size();
  }
  //@}
};

// ---------------------------------------------------------------------------------- persistent_map

template <typename KeyType,                           //
          typename ValueType,                         //
          typename Hash = std::hash<KeyType>,         // Hash function for item
          typename KeyEqual = std::equal_to<KeyType>, // Equality comparision for Item
          bool IsThreadSafe = true                    // True if Set is threadsafe
          >
class persistent_map {
private:
  using set_type = detail::base_set<KeyType, ValueType, Hash, KeyEqual, true, IsThreadSafe>;
  set_type set_;

public:
  using key_type = typename set_type::key_type;
  using value_type = typename set_type::value_type;
  using item_type = typename set_type::item_type;
  using size_type = typename set_type::size_type;
  using hash_type = typename set_type::hash_type;
  using hasher = typename set_type::hasher;
  using key_equal = typename set_type::key_equal;
  using reference = typename set_type::reference;
  using const_reference = typename set_type::const_reference;
  using iterator = typename set_type::iterator;
  using const_iterator = typename set_type::const_iterator;
  static constexpr bool is_thread_safe = IsThreadSafe;

  //@{ Construction/Destruction
  constexpr persistent_map() = default;
  constexpr persistent_map(const persistent_map& other) = default;
  constexpr persistent_map(persistent_map&& other) noexcept = default;
  constexpr ~persistent_map() = default;

  template <typename InputIt> constexpr persistent_map(InputIt first, InputIt last) {
    set_.insert(first, last);
  }
  constexpr persistent_map(std::initializer_list<item_type> ilist) {
    set_.insert(std::begin(ilist), std::end(ilist));
  }

  template <typename Predicate> persistent_map constexpr erase_if(Predicate&& predicate) {
    auto copy = *this;
    copy.set_ = set_.erase_if(std::forward<Predicate>(predicate));
    return copy;
  }
  //@}

  //@{ Assignment
  constexpr persistent_map& operator=(const persistent_map& other) = default;
  constexpr persistent_map& operator=(persistent_map&& other) noexcept = default;
  //@}

  //@{ Iterators
  constexpr iterator begin() { return set_.begin(); }
  constexpr const_iterator begin() const { return set_.begin(); }
  constexpr const_iterator cbegin() const { return set_.cbegin(); }

  constexpr iterator end() { return set_.end(); }
  constexpr const_iterator end() const { return set_.end(); }
  constexpr const_iterator cend() const { return set_.cend(); }
  //@}

  //@{ Capacity
  constexpr bool empty() const { return set_.empty(); }
  constexpr std::size_t size() const { return set_.size(); }
  static constexpr std::size_t max_size() { return set_type::max_size(); }
  //@}

  //@{ Modifiers
  constexpr void clear() { set_.clear(); }

  constexpr bool insert(const item_type& value) { return set_.insert(value); }
  constexpr bool insert(item_type&& value) { return set_.insert(std::move(value)); }
  template <class InputIt> constexpr void insert(InputIt first, InputIt last) {
    set_.insert(first, last);
  }
  constexpr void insert(std::initializer_list<item_type> ilist) { set_.insert(ilist); }

  template <typename K, typename V> constexpr bool insert_or_assign(K&& key, V&& value) {
    return set_.insert_or_assign(std::forward<K>(key), std::forward<V>(value));
  }

  template <class... Args> constexpr bool emplace(Args&&... args) {
    return set_.emplace(std::forward<Args>(args)...);
  }

  constexpr size_type erase(const key_type& key) { return set_.erase(key); }
  constexpr void swap(persistent_map& other) noexcept { set_.swap(other.set_); }

  constexpr std::optional<item_type> extract(const key_type& key) { return set_.extract(key); }
  //@}

  //@{ Lookup
  constexpr const value_type* at(const key_type& key) const { return find(key); }
  constexpr const value_type& operator[](const key_type& key) const {
    auto* value = at(key);
    if (value == nullptr)
      throw std::out_of_range{"index out of range"};
    return *value;
  }

  constexpr std::size_t count(const key_type& key) const { return set_.count(key); }
  constexpr const value_type* find(const key_type& key) const {
    auto* item = set_.find(key);
    return item != nullptr ? &item->second : nullptr;
  }
  constexpr bool contains(const key_type& key) const { return set_.contains(key); }
  //@}

  //@{ Observers
  static constexpr hasher hash_function() { return set_type::hash_function(); }
  static constexpr key_equal key_eq() { return set_type::key_eq(); }
  //@}

  //@{ Friends
  friend constexpr bool operator==(const persistent_map& lhs, const persistent_map& rhs) noexcept {
    return lhs.set_ == rhs.set_;
  }

  friend constexpr bool operator!=(const persistent_map& lhs, const persistent_map& rhs) noexcept {
    return lhs.set_ != rhs.set_;
  }

  friend constexpr void swap(persistent_map& lhs, persistent_map& rhs) noexcept { lhs.swap(rhs); }

  template <typename Predicate>
  friend constexpr size_type erase_if(persistent_map& set, Predicate&& predicate) {
    const auto size_0 = set.size();
    set = set.erase_if(std::forward<Predicate>(predicate));
    return size_0 - set.size();
  }
  //@}
};
} // namespace niggly::trie
