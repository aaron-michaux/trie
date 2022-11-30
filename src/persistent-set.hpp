
#pragma once

#include "bits/trie-base.hpp"

namespace niggly {

// ---------------------------------------------------------------------------------- persistent_set

template <typename ItemType,                           // Type of item to store
          typename Hash = std::hash<ItemType>,         // Hash function for item
          typename KeyEqual = std::equal_to<ItemType>, // Equality comparision for Item
          bool IsThreadSafe = true                     // True if Set is threadsafe
          >
class persistent_set {
private:
  using set_type = detail::trie::base_set<ItemType, ItemType, Hash, KeyEqual, false, IsThreadSafe>;
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

} // namespace niggly
