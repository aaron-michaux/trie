
#include <catch2/catch.hpp>

#include "trie-set.hpp"

#include <fmt/format.h>

#include <fstream>
#include <iostream>
#include <vector>

namespace niggly::trie::test {

using namespace std::string_literals;

class TracedItem {
private:
  uint32_t& counter_;
  std::size_t value_{0};

public:
  explicit TracedItem(uint32_t& counter) : TracedItem(counter, 0) {}
  TracedItem(uint32_t& counter, std::size_t value) : counter_{counter}, value_{value} {
    counter_++;
  }
  TracedItem(const TracedItem& o) : counter_{o.counter_}, value_{o.value_} { counter_++; }
  TracedItem(TracedItem&&) = delete;
  ~TracedItem() { counter_--; };
  TracedItem& operator=(const TracedItem&) = delete;
  TracedItem& operator=(TracedItem&&) = delete;
  std::size_t value() const { return value_; }
  bool operator==(const TracedItem& o) const { return o.value() == value(); }

  struct Hasher {
    std::size_t operator()(const TracedItem& item) const {
      return static_cast<std::size_t>(item.value() & static_cast<std::size_t>(0xffffffffu));
    }
  };
};

class MoveTracedItem {
private:
  uint32_t& counter_;
  std::size_t value_{0};

public:
  explicit MoveTracedItem(uint32_t& counter) : MoveTracedItem(counter, 0) {}
  MoveTracedItem(uint32_t& counter, std::size_t value) : counter_{counter}, value_{value} {
    counter_++;
  }
  MoveTracedItem(const MoveTracedItem& o) : counter_{o.counter_}, value_{o.value_} { counter_++; }
  MoveTracedItem(MoveTracedItem&& o) noexcept : counter_{o.counter_}, value_{o.value_} {
    counter_++;
  }
  ~MoveTracedItem() { counter_--; };
  MoveTracedItem& operator=(const MoveTracedItem&) = delete;
  MoveTracedItem& operator=(MoveTracedItem&&) = delete;
  std::size_t value() const { return value_; }
  bool operator==(const MoveTracedItem& o) const { return o.value() == value(); }

  struct Hasher {
    std::size_t operator()(const MoveTracedItem& item) const {
      return static_cast<std::size_t>(item.value() & static_cast<std::size_t>(0xffffffffu));
    }
  };
};

class TrivialTracedItem {
private:
  std::size_t value_{0};

public:
  TrivialTracedItem() = default;
  explicit TrivialTracedItem(uint32_t& counter) {}
  TrivialTracedItem(uint32_t& counter, std::size_t value) : value_{value} {}
  TrivialTracedItem(const TrivialTracedItem& o) = default;
  TrivialTracedItem(TrivialTracedItem&&) = default;
  ~TrivialTracedItem() = default;
  TrivialTracedItem& operator=(const TrivialTracedItem&) = default;
  TrivialTracedItem& operator=(TrivialTracedItem&&) = default;
  std::size_t value() const { return value_; }
  bool operator==(const TrivialTracedItem& o) const { return o.value() == value(); }

  struct Hasher {
    std::size_t operator()(const TrivialTracedItem& item) const {
      return static_cast<std::size_t>(item.value() & static_cast<std::size_t>(0xffffffffu));
    }
  };
};

// Set types used in testing need to be declared here to use the "private hack"
using TracedItemSetType = persistent_set<TracedItem, TracedItem::Hasher>;
using MoveTracedItemSetType = persistent_set<MoveTracedItem, MoveTracedItem::Hasher>;
using TrivialTracedItemSetType = persistent_set<TrivialTracedItem, TrivialTracedItem::Hasher>;

using ItemSetType =
    detail::base_set<typename TracedItemSetType::item_type, typename TracedItemSetType::item_type,
                     typename TracedItemSetType::hasher, typename TracedItemSetType::key_equal,
                     false, TracedItemSetType::is_thread_safe>;

namespace private_hack {
template <typename Tag> struct result {
  using type = typename Tag::type;
  static type ptr;
};
template <typename Tag> typename result<Tag>::type result<Tag>::ptr;

template <typename Tag, typename Tag::type p> struct rob : result<Tag> {
  struct filler {
    filler() { result<Tag>::ptr = p; }
  };
  static filler filler_obj;
};
template <typename Tag, typename Tag::type p> typename rob<Tag, p>::filler rob<Tag, p>::filler_obj;

template <typename T> struct Bf { using type = detail::NodeData<T::is_thread_safe>* (T::*)(); };

template struct rob<Bf<ItemSetType>, &ItemSetType::get_root_>;
template <typename T> auto get_root1(T& trie) { return (trie.*result<Bf<T>>::ptr)(); }

auto get_root(auto& trie) { return get_root1(*reinterpret_cast<ItemSetType*>(&trie)); }

} // namespace private_hack

template <typename NodeOps, typename Function>
void for_each_node(typename NodeOps::node_type* node, Function f) {
  using NodeType = detail::NodeType;
  if (node == nullptr)
    return; // empty
  f(node);
  if (NodeOps::type(node) == NodeType::Branch) {
    auto* start = NodeOps::Branch::dense_ptr_at(node, 0);
    for (auto* iterator = start; iterator != start + NodeOps::Branch::size(node); ++iterator) {
      for_each_node<NodeOps>(*iterator, f);
    }
  }
}

/**
 * Apply `f` to each leaf node descendent from `node`
 */
template <typename NodeOps, typename Function>
void for_each_leaf(typename NodeOps::node_type* root, Function f) {
  using NodeType = detail::NodeType;
  for_each_node<NodeOps>(root, [f](typename NodeOps::node_type* node) {
    if (NodeOps::type(node) == NodeType::Leaf)
      f(node);
  });
}

template <typename NodeOps>
void dot_graph(const char* filename, typename NodeOps::node_type* root) {
  if (root == nullptr)
    return;

  using NodeType = detail::NodeType;
  auto node_name = [](typename NodeOps::node_type* node) -> std::string {
    return fmt::format("{:c}0x{:08x}{}", (NodeOps::type(node) == NodeType::Branch ? 'B' : 'L'),
                       reinterpret_cast<uintptr_t>(node),
                       (NodeOps::type(node) == NodeType::Branch
                            ? std::string{""}
                            : fmt::format("sz_{}", NodeOps::size(node))));
  };

  std::ofstream out;
  out.open(filename);
  out << "digraph {\n";
  for_each_node<NodeOps>(root, [&](typename NodeOps::node_type* node) { //
    if (NodeOps::type(node) == NodeType::Branch) {
      for (auto i = 0u; i < 32; ++i) {
        if (NodeOps::Branch::is_valid_index(node, i)) {
          auto* other = *NodeOps::Branch::ptr_at(node, i);
          out << fmt::format("   {} -> {}[label=\"{}\"]\n", node_name(node), node_name(other), i);
        }
      }
    }
  });
  if (NodeOps::type(root) == NodeType::Leaf) {
    out << fmt::format("   {}\n", node_name(root));
  }

  out << "}\n";
  out.close();
}

template <typename NodeOps> void check_leaf_invariants(const typename NodeOps::node_type* leaf) {
  using NodeType = detail::NodeType;
  const auto size = NodeOps::size(leaf);
  const auto hasher = typename NodeOps::hasher{};
  const auto is_equal = typename NodeOps::key_equal{};
  CATCH_REQUIRE(NodeOps::type(leaf) == NodeType::Leaf); // Must be a leaf
  CATCH_REQUIRE(size > 0);                              // No empty leaves
  const auto* item0 = NodeOps::Leaf::begin(leaf);       // The start item
  const auto hash0 = hasher(*item0);                    // The zeroith hash
  CATCH_REQUIRE(std::all_of(item0 + 1, item0 + size,    // all items have the same hash
                            [&](auto& item) -> bool { return hasher(item) == hash0; }));
  // No two items are equal
  for (auto i = 0u; i < size; ++i)
    for (auto j = i + 1; j < size; ++j)
      CATCH_REQUIRE(!is_equal(*NodeOps::Leaf::ptr_at(leaf, i), *NodeOps::Leaf::ptr_at(leaf, j)));
}

template <typename NodeOps>
void check_trie_invariants(typename NodeOps::node_type* root, std::size_t tree_size) {
  using node_type_ptr = const typename NodeOps::node_type*;

  std::size_t leaf_count = 0;

  auto check_leaf = [root, &leaf_count](node_type_ptr leaf) {
    // 1. Check leaf invariants
    check_leaf_invariants<NodeOps>(leaf);
    leaf_count += NodeOps::Leaf::size(leaf);

    // 2. Check every path
    const auto hash = NodeOps::hash(leaf);
    const auto path = NodeOps::make_path(root, hash);
    CATCH_REQUIRE(path.leaf_end == leaf);
    for (auto i = 0u; i < path.size; ++i) {
      const auto node = path.nodes[i];
      const void* next_node = (i + 1 == path.size) ? path.leaf_end : path.nodes[i + 1];
      const auto sparse_index = detail::hash_chunk(hash, i);
      CATCH_REQUIRE(NodeOps::type(node) == detail::NodeType::Branch);
      CATCH_REQUIRE(NodeOps::Branch::is_valid_index(node, sparse_index));
      CATCH_REQUIRE(*NodeOps::Branch::ptr_at(node, sparse_index) == next_node);
    }
    CATCH_REQUIRE(path.leaf_end == leaf);
  };

  // Check each leaf nodes
  // for_each_leaf<NodeOps>(root, check_leaf);

  // If a branch node has a single child, it must be another branch node
  auto check_node = [&check_leaf](node_type_ptr node) {
    if (NodeOps::type(node) == detail::NodeType::Leaf) {
      check_leaf(node);
    } else {
      CATCH_REQUIRE(NodeOps::type(node) == detail::NodeType::Branch);
      CATCH_REQUIRE(NodeOps::size(node) > 0);
      // If the branch-node is size1, then the child must be a branch node
      if (NodeOps::size(node) == 1) {
        node_type_ptr child = *NodeOps::Branch::dense_ptr_at(node, 0);
        CATCH_REQUIRE(NodeOps::type(child) == detail::NodeType::Branch);
      }
    }
  };
  for_each_node<NodeOps>(root, check_node);

  CATCH_REQUIRE(leaf_count == tree_size);
}

template <typename Set, typename ForwardItr>
void check_trie_iterators(Set& set, ForwardItr start, ForwardItr finish) {
  std::vector<std::size_t> values{start, finish};
  auto is_value = [&values](std::size_t value) {
    return std::find(std::begin(values), std::end(values), value) != std::end(values);
  };

  {
    std::size_t counter = 0;
    for (auto ii = set.begin(); ii != set.end(); ++ii) {
      CATCH_REQUIRE(is_value((*ii).value()));
      ++counter;
    }
    CATCH_REQUIRE(counter == values.size());
  }

  {
    const Set& cset = set;
    std::size_t counter = 0;
    for (auto ii = cset.begin(); ii != cset.end(); ++ii) {
      CATCH_REQUIRE(is_value((*ii).value()));
      ++counter;
    }
    CATCH_REQUIRE(counter == values.size());
  }

  {
    std::size_t counter = 0;
    for (auto ii = set.cbegin(); ii != set.cend(); ++ii) {
      CATCH_REQUIRE(is_value((*ii).value()));
      ++counter;
    }
    CATCH_REQUIRE(counter == values.size());
  }

  {
    std::size_t counter = 0;
    for (auto ii = set.end(); ii != set.begin();) {
      --ii;
      CATCH_REQUIRE(is_value((*ii).value()));
      ++counter;
    }
    CATCH_REQUIRE(counter == values.size());
  }

  {
    for (auto ii = set.cbegin(); ii != set.cend(); ++ii) {
      CATCH_REQUIRE(set.count(*ii) == 1);
      CATCH_REQUIRE(*set.find(*ii) == *ii);
      CATCH_REQUIRE(set.contains(*ii));
    }
  }

  { // Should be able to move beyond the end, without effect
    std::size_t counter = 0;
    auto end = set.cend();
    ++end;
    for (auto ii = end; ii != set.cbegin();) {
      CATCH_REQUIRE(is_value((*--ii).value()));
      ++counter;
    }
    CATCH_REQUIRE(counter == values.size());
  }

  { // Should be able to move before the start, without effect
    std::size_t counter = 0;
    auto start = set.cbegin();
    --start; // start should now be end!
    CATCH_REQUIRE(start == set.cend());
  }

  { // Test postincrement
    std::size_t counter = 0;
    for (auto ii = set.cbegin(); ii != set.cend();) {
      CATCH_REQUIRE(is_value((*ii++).value()));
      ++counter;
    }
    CATCH_REQUIRE(counter == values.size());
  }

  { // Test postdecrement
    std::size_t counter = 0;
    for (auto ii = set.end(); ii != set.begin();) {
      ii--;
      CATCH_REQUIRE(is_value((*ii).value()));
      ++counter;
    }
    CATCH_REQUIRE(counter == values.size());
  }
}

CATCH_TEST_CASE("max_trie_depth", "[max_trie_depth]") {
  const auto hash_bits = sizeof(std::size_t) * 8;                      // 64 bits
  const auto chunk_bits = std::size_t{5};                              // 32 fits in 5 bits
  CATCH_REQUIRE(chunk_bits * (detail::MaxTrieDepth - 0) >= hash_bits); // 5 * 13 = 65
  CATCH_REQUIRE(chunk_bits * (detail::MaxTrieDepth - 1) < hash_bits);  // 5 * 12 = 60
}

CATCH_TEST_CASE("popcount", "[popcount]") {
  auto test_popcount = [](uint32_t x, int count) {
    CATCH_REQUIRE(detail::popcount(x) == count);
    CATCH_REQUIRE(detail::branch_free_popcount(x) == count);
  };
  test_popcount(0x00000000u, 0);
  test_popcount(0x01010101u, 4);
  test_popcount(0x11010101u, 5);
  test_popcount(0xf1010101u, 8);
  test_popcount(0xffffffffu, 32);
}

CATCH_TEST_CASE("sparse_index", "[sparse_index]") {
  // 4 == 100b
  const uint32_t bitmap = (1u << 0) | (1u << 7) | (1u << 16) | (1u << 22) | (1u << 31);
  CATCH_REQUIRE(detail::popcount(bitmap) == 5);
  CATCH_REQUIRE(detail::to_dense_index(0u, bitmap) == 0);
  CATCH_REQUIRE(detail::to_dense_index(7u, bitmap) == 1);
  CATCH_REQUIRE(detail::to_dense_index(16u, bitmap) == 2);
  CATCH_REQUIRE(detail::to_dense_index(22u, bitmap) == 3);
  CATCH_REQUIRE(detail::to_dense_index(31u, bitmap) == 4);

  // This indices are invalid
  CATCH_REQUIRE(detail::is_valid_index(1u, bitmap) == false);
  CATCH_REQUIRE(detail::is_valid_index(6u, bitmap) == false);
  CATCH_REQUIRE(detail::is_valid_index(23u, bitmap) == false);

  for (auto i = 0u; i < 32; ++i) {
    const bool is_valid = (i == 0) || (i == 7) || (i == 16) || (i == 22) || (i == 31);
    CATCH_REQUIRE(detail::is_valid_index(i, bitmap) == is_valid);
  }
}

CATCH_TEST_CASE("node_data_size_alignment", "[node_data_size_alignment]") {
  using NodeDataTheadSafe = detail::NodeData<true>;
  using NodeDataVanilla = detail::NodeData<false>;
  CATCH_REQUIRE(alignof(NodeDataTheadSafe) == alignof(NodeDataVanilla));
  CATCH_REQUIRE(sizeof(NodeDataTheadSafe) == sizeof(NodeDataVanilla));
  CATCH_REQUIRE(sizeof(NodeDataTheadSafe) == 8);
  CATCH_REQUIRE(alignof(NodeDataTheadSafe) == 4);
}

CATCH_TEST_CASE("node_data_add_dec_ref", "[node_data_add_dec_ref]") {
  auto test_node = [](auto& node) {
    CATCH_REQUIRE(node.ref_count() == 1);
    CATCH_REQUIRE(node.add_ref() == 2);
    CATCH_REQUIRE(node.dec_ref() == 1);
    CATCH_REQUIRE(node.dec_ref() == 0);
  };
  auto n1 = detail::NodeData<true>{detail::NodeType::Leaf, 0};
  auto n2 = detail::NodeData<false>{detail::NodeType::Leaf, 0};

  test_node(n1);
  test_node(n2);
}

CATCH_TEST_CASE("node_type", "[node_type]") {
  {
    detail::NodeData node{detail::NodeType::Branch, 0};
    CATCH_REQUIRE(node.type() == detail::NodeType::Branch);
  }
  {
    detail::NodeData node{detail::NodeType::Leaf, 0};
    CATCH_REQUIRE(node.type() == detail::NodeType::Leaf);
  }
}

template <typename T> void test_node_size() {
  using NodeType = detail::NodeType;
  {
    auto node = T::Branch::make_uninitialized(0, 0);
    CATCH_REQUIRE(T::size(node) == 0);
    CATCH_REQUIRE(T::type(node) == NodeType::Branch);
    CATCH_REQUIRE(T::Branch::offset() >= sizeof(typename T::node_type));
    auto address_0 = std::bit_cast<uintptr_t>(T::Branch::dense_ptr_at(node, 0));
    CATCH_REQUIRE(address_0 >= std::bit_cast<uintptr_t>(node) + sizeof(typename T::node_type));
    CATCH_REQUIRE(address_0 % T::Branch::AlignOf == 0);
    CATCH_REQUIRE(T::Branch::AlignOf == alignof(void*));
    CATCH_REQUIRE(T::ref_count(node) == 1);
    node->dec_ref();
    CATCH_REQUIRE(T::ref_count(node) == 0);
    std::free(node);
  }

  {
    auto node = T::Leaf::make_uninitialized(0, 0);
    CATCH_REQUIRE(T::size(node) == 0);
    CATCH_REQUIRE(T::type(node) == NodeType::Leaf);
    CATCH_REQUIRE(T::Leaf::offset() >= sizeof(typename T::node_type));
    auto address_0 = std::bit_cast<uintptr_t>(T::Leaf::ptr_at(node, 0));
    CATCH_REQUIRE(address_0 >= std::bit_cast<uintptr_t>(node) + sizeof(typename T::node_type));
    CATCH_REQUIRE(address_0 % alignof(typename T::Leaf::item_type) == 0);
    CATCH_REQUIRE(T::ref_count(node) == 1);
    node->dec_ref();
    CATCH_REQUIRE(T::ref_count(node) == 0);
    std::free(node);
  }
}

template <typename T> void test_node_configuration() {
  test_node_size<detail::NodeOps<T, T, std::hash<T>, std::equal_to<T>, false, true>>();
  test_node_size<detail::NodeOps<T, T, std::hash<T>, std::equal_to<T>, false, false>>();
}

CATCH_TEST_CASE("node_configuration", "[node_configuration]") {
  test_node_configuration<char>();
  test_node_configuration<int16_t>();
  test_node_configuration<int32_t>();
  test_node_configuration<int64_t>();
  test_node_configuration<void*>();
  test_node_configuration<std::array<char, 22>>();

  struct alignas(1) Weird0 {
    char value[5];
  };

  struct alignas(2) Weird1 {
    char value[17];
  };

  struct alignas(16) Weird2 {
    int64_t a;
    char b;
  };

  test_node_configuration<Weird0>();
  test_node_configuration<Weird1>();
  test_node_configuration<Weird2>();
}

CATCH_TEST_CASE("trie_construct_destruct", "[trie_construct_destruct]") {
  uint32_t counter{0}; // Tracks how many times the constructor/destructor is called
  using Ops = detail::NodeOps<TracedItem, TracedItem>;

  { // Constructor should be called 4 times, and same with destructor
    auto* node_ptr = Ops::Leaf::make_uninitialized(4, 4);
    CATCH_REQUIRE(Ops::size(node_ptr) == 4);
    for (auto i = 0u; i < Ops::size(node_ptr); ++i) {
      new (Ops::Leaf::ptr_at(node_ptr, i)) TracedItem{counter};
    }
    CATCH_REQUIRE(counter == Ops::size(node_ptr));
    Ops::dec_ref(node_ptr); // Calls destructor
    CATCH_REQUIRE(counter == 0);
  }
}

CATCH_TEST_CASE("trie_ops_safe_destroy", "[trie_ops_safe_destroy]") {
  using Ops = detail::NodeOps<TracedItemSetType::item_type, TracedItemSetType::item_type,
                              TracedItemSetType::hasher, TracedItemSetType::key_equal, false,
                              TracedItemSetType::is_thread_safe>;
  Ops::destroy(nullptr); // should not crash
}

CATCH_TEST_CASE("duplicate_leaf", "[dupcliate_leaf]") {
  uint32_t counter = 0;

  using Ops = detail::NodeOps<TracedItem, TracedItem>;
  using Leaf = Ops::Leaf;

  auto* leaf_0 = Leaf::make(TracedItem{counter, 0});
  auto* leaf_1 = Leaf::copy_append(leaf_0, TracedItem{counter, 1});
  auto* leaf_2 = Leaf::duplicate_leaf(leaf_1, detail::NotAnIndex);

  CATCH_REQUIRE(Leaf::size(leaf_0) == 1);
  CATCH_REQUIRE(Leaf::size(leaf_1) == 2);
  CATCH_REQUIRE(Leaf::size(leaf_2) == 2);

  for (auto index = 0u; index < 2; ++index) {
    CATCH_REQUIRE(Leaf::ptr_at(leaf_1, index)->value() == index);
    CATCH_REQUIRE(Leaf::ptr_at(leaf_2, index)->value() == index);
  }

  Ops::dec_ref(leaf_0);
  Ops::dec_ref(leaf_1);
  Ops::dec_ref(leaf_2);

  CATCH_REQUIRE(counter == 0);
}

template <typename SetType> void trie_ops_test() {
  constexpr bool skip_counter_test{std::is_same<SetType, TrivialTracedItemSetType>::value};
  uint32_t counter = 0;

  {
    SetType set;
    using ItemType = typename SetType::item_type;
    using Ops = detail::NodeOps<typename SetType::item_type, typename SetType::item_type,
                                typename SetType::hasher, typename SetType::key_equal, false,
                                SetType::is_thread_safe>;

    // Inserting the following sequence, to test code paths, hash is 32 bits
    // value =   1,         hash = 00|00-000|0 0000-|0000 0|000-00|00 000|0-0001
    // value =  55,         hash = 00|00-000|0 0000-|0000 0|000-00|00 001|1-0111
    // value = 119,         hash = 00|00-000|0 0000-|0000 0|000-00|00 011|1-0111
    // value =   3,         hash = 00|00-000|0 0000-|0000 0|000-00|00 000|0-0011
    // value =   0,         hash = 00|00-000|0 0000-|0000 0|000-00|00 000|0-0000
    // value =  31,         hash = 00|00-000|0 0000-|0000 0|000-00|00 000|1-1111
    // value = 0x100000000, hash = 00|00-000|0 0000-|0000 0|000-00|00 000|0-0000
    // value = 0x4000001F,  hash = 01|00-000|0 0000-|0000 0|000-00|00 000|1-1111
    // value = 0xC000001F,  hash = 11|00-000|0 0000-|0000 0|000-00|00 000|1-1111
    // value = 0x100001F,   hash = 00|00-000|1 0000-|0000 0|000-00|00 000|1-1111

    std::vector<std::size_t> values{
        {1, 55, 119, 3, 0, 31, 0x100000000, 0x4000001F, 0xC000001F, 0x100001F}};
    auto pos = 0u;

    uint32_t test_number = 0;
    auto run_standard_tests = [&set, &counter, &values, &test_number](uint32_t pos) {
      const auto value = values[pos];
      const auto new_size = pos + 1;
      bool was_inserted = set.insert(ItemType{counter, value});
      CATCH_REQUIRE(was_inserted == true);
      CATCH_REQUIRE(!set.empty());
      CATCH_REQUIRE(set.size() == new_size);
      CATCH_REQUIRE(set.size() < set.max_size());
      CATCH_REQUIRE((skip_counter_test || counter == new_size));
      auto root = private_hack::get_root(set);
      dot_graph<Ops>(fmt::format("/tmp/{}.dot", test_number++).c_str(), root);
      check_trie_invariants<Ops>(root, set.size());
      check_trie_iterators(set, std::begin(values), std::begin(values) + new_size);
    };

    { // Insert into empty
      // root -> L(1)
      CATCH_REQUIRE(set.empty());
      run_standard_tests(pos++);
    }

    { // root -> B ( 1)-> L(1)
      //           (23)-> L(55)
      run_standard_tests(pos++);
    }

    { // root -> B ( 1)-> L(1)
      //           (23)-> B ( 1) -> L(55)
      //                    ( 3) -> L(119)
      run_standard_tests(pos++);
    }

    { // root -> B ( 1)-> L(1)
      //           ( 3)-> L(3)
      //           (23)-> B ( 1) -> L(55)
      //                    ( 3) -> L(119)
      run_standard_tests(pos++);
    }

    { // root -> B ( 0)-> L(0)
      //           ( 1)-> L(1)
      //           ( 3)-> L(3)
      //           (23)-> B ( 1) -> L(55)
      //                    ( 3) -> L(119)
      run_standard_tests(pos++);
    }

    { // root -> B ( 0)-> L(0)
      //           ( 1)-> L(1)
      //           ( 3)-> L(3)
      //           (23)-> B ( 1) -> L(55)
      //                    ( 3) -> L(119)
      //           (31)-> L(31)
      run_standard_tests(pos++);
    }

    { // root -> B ( 0)-> L(0, 0x100000000)
      //           ( 1)-> L(1)
      //           ( 3)-> L(3)
      //           (23)-> B ( 1) -> L(55)
      //                    ( 3) -> L(119)
      //           (31)-> L(31)
      run_standard_tests(pos++);
    }

    { // root -> B ( 0)-> L(0, 0x100000000)
      //           ( 1)-> L(1)
      //           ( 3)-> L(3)
      //           (23)-> B ( 1) -> L (55)
      //                    ( 3) -> L(119)
      //           (31)-> B ( 0) -> B ( 0) -> B ( 0) -> B ( 0) -> B ( 0) -> B ( 0)-> L(31)
      //                                                                      ( 1)-> L(0x4000001F)
      run_standard_tests(pos++);
    }

    { // root -> B ( 0)-> L(0, 0x100000000)
      //           ( 1)-> L(1)
      //           ( 3)-> L(3)
      //           (23)-> B ( 1) -> L (55)
      //                    ( 3) -> L(119)
      //           (31)-> B ( 0) -> B ( 0) -> B ( 0) -> B ( 0) -> B ( 0) -> B ( 0)-> L(31)
      //                                                                      ( 1)-> L(0x4000001F)
      //                                                                      ( 3)-> L(0xC000001F)
      run_standard_tests(pos++);
    }

    { // root -> B ( 0)-> L(0, 0x100000000)
      //           ( 1)-> L(1)
      //           ( 3)-> L(3)
      //           (23)-> B ( 1) -> L (55)
      //                    ( 3) -> L(119)
      //           (31)-> B ( 0) -> B ( 0) -> B ( 0) -> B ( 0) -> B ( 0) -> B ( 0)-> L(31)
      //                                                                      ( 1)-> L(0x4000001F)
      //                                                                      ( 3)-> L(0xC000001F)
      //                                             -> B (16) -> L(0x100001F)
      run_standard_tests(pos++);
    }

    { // duplicate: graph unchanged
      for (auto value : values) {
        bool was_inserted = set.insert(ItemType{counter, value});
        CATCH_REQUIRE(was_inserted == false);
        CATCH_REQUIRE(!set.empty());
        CATCH_REQUIRE(set.size() == pos);
        CATCH_REQUIRE(set.size() < set.max_size());
        CATCH_REQUIRE((skip_counter_test || counter == pos));
        auto root = private_hack::get_root(set);
        check_trie_invariants<Ops>(root, set.size());
      }
    }

    { // Attempt to remove 'non-element'; graph unchanged
      bool was_erased = set.erase(ItemType{counter, 0xff112233u});
      CATCH_REQUIRE(was_erased == false);
      CATCH_REQUIRE(set.size() == pos);
      auto root = private_hack::get_root(set);
      check_trie_invariants<Ops>(root, set.size());
    }

    while (set.size() > 0) {
      { // Test removing any element
        for (const auto& item : set) {
          auto temp_set = set;
          bool was_erased = temp_set.erase(item);
          CATCH_REQUIRE(was_erased == true);
          CATCH_REQUIRE(temp_set.size() + 1 == set.size());
          auto root = private_hack::get_root(temp_set);
          check_trie_invariants<Ops>(root, temp_set.size());
        }
      }
      { // Erase if
        auto predicate = [](const ItemType& item) { return item.value() % 2 == 0; };
        auto temp_set = set;
        const auto delete_count = erase_if(temp_set, predicate);
        auto counter = 0u;
        for (const auto& item : set) {
          if (predicate(item)) {
            CATCH_REQUIRE(!temp_set.contains(item));
            counter++;
          } else {
            CATCH_REQUIRE(temp_set.contains(item));
          }
        }
        assert(counter == delete_count);
      }
      assert(pos > 0);
      const auto value = values[--pos];
      auto item = ItemType{counter, value};
      CATCH_REQUIRE(set.contains(item));
      bool was_erased = set.erase(item);
      CATCH_REQUIRE(was_erased == true);
      CATCH_REQUIRE(!set.contains(item));
      auto root = private_hack::get_root(set);
      dot_graph<Ops>(fmt::format("/tmp/{}.dot", test_number++).c_str(), root);
      check_trie_invariants<Ops>(root, set.size());
      check_trie_iterators(set, std::begin(values), std::begin(values) + set.size());
    }
  }

  CATCH_REQUIRE(counter == 0);
}

CATCH_TEST_CASE("trie_set_ops", "[trie_set_ops]") {
  trie_ops_test<TracedItemSetType>();
  trie_ops_test<MoveTracedItemSetType>();
  trie_ops_test<TrivialTracedItemSetType>();
}

CATCH_TEST_CASE("trie_set_default_construct", "[trie_set_default_construct]") {
  using set_type = persistent_set<TracedItem>;
  auto set = std::make_unique<set_type>();
  CATCH_REQUIRE(set->size() == 0);
}

CATCH_TEST_CASE("trie_set_insert", "[trie_set_insert]") {
  using set_type = persistent_set<int>;
  std::vector<int> values{{1, 2, 3}};
  set_type set{std::begin(values), std::end(values)};
  CATCH_REQUIRE(set.size() == values.size());
  for (auto value : values)
    CATCH_REQUIRE(set.contains(value));
  set.clear();
  CATCH_REQUIRE(set.size() == 0);

  set.insert(std::begin(values), std::end(values));
  CATCH_REQUIRE(set.size() == values.size());
  for (auto value : values)
    CATCH_REQUIRE(set.contains(value));

  CATCH_REQUIRE(set.insert(3) == false);
  CATCH_REQUIRE(set.insert(4) == true);
  CATCH_REQUIRE(set.contains(4));
  CATCH_REQUIRE(set.size() == 4);

  set.insert({4, 5, 6});
  CATCH_REQUIRE(set.size() == 6);

  CATCH_REQUIRE(set.emplace(7));
  CATCH_REQUIRE(set.size() == 7);
}

CATCH_TEST_CASE("trie_set_insert_const_ref", "[trie_set_insert_const_ref]") {
  using set_type = persistent_set<TracedItem, TracedItem::Hasher>;
  uint32_t counter = 0;
  {
    set_type set;
    TracedItem item{counter, 1};
    set.insert(item);
    CATCH_REQUIRE(set.size() == 1);
  }
  CATCH_REQUIRE(counter == 0);
}

CATCH_TEST_CASE("trie_set_extract", "[trie_set_extract]") {
  using set_type = persistent_set<int>;
  set_type set{{1, 2, 3}};
  auto x = set.extract(2);
  auto y = set.extract(2);

  CATCH_REQUIRE(x.has_value());
  CATCH_REQUIRE(!y.has_value());
  CATCH_REQUIRE(*x == 2);
}

CATCH_TEST_CASE("trie_set_ilist", "[trie_set_ilist]") {
  using set_type = persistent_set<int>;
  std::vector<int> values{{1, 2, 3}};
  set_type set{{1, 2, 3}};
  CATCH_REQUIRE(set.size() == values.size());
  for (auto value : values)
    CATCH_REQUIRE(set.contains(value));
}

CATCH_TEST_CASE("trie_set_op=", "[trie_set_op=]") {
  using set_type = persistent_set<int>;
  set_type set{{1, 2, 3}};
  set_type other;
  CATCH_REQUIRE(other.size() == 0);
  other = set;
  CATCH_REQUIRE(other.size() == set.size());
  for (auto value : set)
    CATCH_REQUIRE(other.contains(value));
  set.erase(2);
  CATCH_REQUIRE(!set.contains(2));
  CATCH_REQUIRE(other.contains(2));
}

CATCH_TEST_CASE("trie_set_op==", "[trie_set_op==]") {
  using set_type = persistent_set<int>;
  set_type set{{1, 2, 3}};
  set_type other;
  CATCH_REQUIRE(set == set);
  CATCH_REQUIRE(other == other);
  CATCH_REQUIRE(set != other);
  CATCH_REQUIRE(other != set);
  other = set;
  CATCH_REQUIRE(other == set);

  CATCH_REQUIRE(set.size() == 3);
  CATCH_REQUIRE(other.size() == 3);

  other.clear();
  CATCH_REQUIRE(set.size() == 3);
  CATCH_REQUIRE(other.size() == 0);
  CATCH_REQUIRE(other != set);
  other.insert(std::begin(set), std::end(set));
  CATCH_REQUIRE(other == set);

  other.erase(3);
  other.insert(4);
  CATCH_REQUIRE(other != set);
}

CATCH_TEST_CASE("trie_set_swap", "[trie_set_swap]") {
  using set_type = persistent_set<int>;
  set_type set{{1, 2, 3}};
  set_type other;

  CATCH_REQUIRE(other.size() == 0);
  other.swap(other);
  CATCH_REQUIRE(other.size() == 0);

  CATCH_REQUIRE(set.size() == 3);
  set.swap(set);
  CATCH_REQUIRE(set.size() == 3);

  set.swap(other);
  CATCH_REQUIRE(set.size() == 0);
  CATCH_REQUIRE(other.size() == 3);

  using std::swap;
  swap(set, other);
  CATCH_REQUIRE(set.size() == 3);
  CATCH_REQUIRE(other.size() == 0);
}

CATCH_TEST_CASE("trie_set_hash_func", "[trie_set_hash_func]") {
  using set_type = persistent_set<int>;
  auto func = set_type::hasher{};
  CATCH_REQUIRE(set_type::hash_function()(1) == func(1));
}

CATCH_TEST_CASE("trie_set_key_eq_func", "[trie_set_key_eq_func]") {
  using set_type = persistent_set<int>;
  CATCH_REQUIRE(set_type::key_eq()(1, 1));
  CATCH_REQUIRE(!set_type::key_eq()(1, 2));
}

CATCH_TEST_CASE("trie_map_default_constructor", "[trie_map_default_constructor]") {
  using map_type = persistent_map<std::string, TracedItem>;
  auto map = std::make_unique<map_type>();
  CATCH_REQUIRE(map->size() == 0);
  CATCH_REQUIRE(map->empty());
}

CATCH_TEST_CASE("trie_map_construct_insert", "[trie_map_construct_insert]") {
  using namespace std::string_literals;
  using map_type = persistent_map<std::string, int>;
  std::vector<map_type::item_type> values{{{"one"s, 1}, {"two"s, 2}, {"three"s, 3}}};
  map_type map{std::begin(values), std::end(values)};
  CATCH_REQUIRE(map.size() == 3);
  CATCH_REQUIRE(!map.empty());
  CATCH_REQUIRE(values[0].first == "one"s); // Make sure we didn't move anything
  CATCH_REQUIRE(values[1].first == "two"s);
  CATCH_REQUIRE(values[2].first == "three"s);

  for (const auto& [key, value] : values) {
    CATCH_REQUIRE(map.contains(key));
    CATCH_REQUIRE(map.count(key) == 1);
    CATCH_REQUIRE(map.find(key) != nullptr);
    CATCH_REQUIRE(*map.find(key) == value);
  }
  map.clear();
  CATCH_REQUIRE(map.size() == 0);
  for (const auto& [key, value] : values) {
    CATCH_REQUIRE(!map.contains(key));
    CATCH_REQUIRE(map.count(key) == 0);
    CATCH_REQUIRE(map.find(key) == nullptr);
  }
}

CATCH_TEST_CASE("trie_map_ilist", "[trie_map_ilist]") {
  using namespace std::string_literals;
  using map_type = persistent_map<std::string, int>;
  map_type map{{{"one"s, 1}, {"two"s, 2}, {"three"s, 3}}};
  CATCH_REQUIRE(map.size() == 3);
}

CATCH_TEST_CASE("trie_map_op=", "[trie_map_op=]") {
  using namespace std::string_literals;
  using map_type = persistent_map<std::string, int>;
  std::vector<map_type::item_type> values{{{"one"s, 1}, {"two"s, 2}, {"three"s, 3}}};

  map_type map{std::begin(values), std::end(values)};
  map_type other{map};
  map_type other2;

  CATCH_REQUIRE(map.size() == other.size());
  CATCH_REQUIRE(map.size() == 3);
  CATCH_REQUIRE(map.size() < map_type::max_size());
  CATCH_REQUIRE(map == other);
  CATCH_REQUIRE(map != other2);
  CATCH_REQUIRE(other2.size() == 0);

  for (const auto& [key, value] : values) {
    CATCH_REQUIRE(map.contains(key));
    CATCH_REQUIRE(other.contains(key));
    CATCH_REQUIRE(*map.find(key) == *other.find(key));
    CATCH_REQUIRE(*map.find(key) == value);
    CATCH_REQUIRE(!other2.contains(key));
  }

  other2 = map;
  CATCH_REQUIRE(map == other2);
  for (const auto& [key, value] : values) {
    CATCH_REQUIRE(map.contains(key));
    CATCH_REQUIRE(other2.contains(key));
    CATCH_REQUIRE(*map.find(key) == *other2.find(key));
  }
}

CATCH_TEST_CASE("trie_map_move=", "[trie_map_move=]") {
  using map_type = persistent_map<std::string, int>;
  std::vector<map_type::item_type> values{{{"one"s, 1}, {"two"s, 2}, {"three"s, 3}}};

  map_type map{std::begin(values), std::end(values)};
  map_type other{std::move(map)};

  CATCH_REQUIRE(map != other);
  CATCH_REQUIRE(map.size() != other.size());
  CATCH_REQUIRE(other.size() == 3);

  for (const auto& [key, value] : values) {
    CATCH_REQUIRE(!map.contains(key));
    CATCH_REQUIRE(other.contains(key));
    CATCH_REQUIRE(*other.find(key) == value);
  }

  map = std::move(other);
  CATCH_REQUIRE(map != other);
  CATCH_REQUIRE(map.size() != other.size());
  CATCH_REQUIRE(map.size() == 3);

  for (const auto& [key, value] : values) {
    CATCH_REQUIRE(map.contains(key));
    CATCH_REQUIRE(!other.contains(key));
    CATCH_REQUIRE(*map.find(key) == value);
  }
}

CATCH_TEST_CASE("trie_map_op==", "[trie_map_op==]") {
  using map_type = persistent_map<int, std::string>;
  map_type map{{{0, "a"s}, {1, "b"s}, {2, "c"s}}};
  map_type other;

  CATCH_REQUIRE(map == map);
  CATCH_REQUIRE(other == other);
  CATCH_REQUIRE(map != other);
  CATCH_REQUIRE(other != map);
  other = map;
  CATCH_REQUIRE(other == map);

  CATCH_REQUIRE(map.size() == 3);
  CATCH_REQUIRE(other.size() == 3);

  other.clear();
  CATCH_REQUIRE(map.size() == 3);
  CATCH_REQUIRE(other.size() == 0);
  CATCH_REQUIRE(other != map);
  other.insert(std::begin(map), std::end(map));
  CATCH_REQUIRE(other == map);

  other.erase(2);
  CATCH_REQUIRE(other.size() == 2);
  other.insert({3, "d"s});
  CATCH_REQUIRE(other.size() == 3);
  CATCH_REQUIRE(other != map);
}

CATCH_TEST_CASE("trie_map_swap", "[trie_map_swap]") {
  using map_type = persistent_map<int, std::string>;
  map_type map{{{0, "a"s}, {1, "b"s}, {2, "c"s}}};
  map_type other;

  CATCH_REQUIRE(other.size() == 0);
  other.swap(other);
  CATCH_REQUIRE(other.size() == 0);

  CATCH_REQUIRE(map.size() == 3);
  map.swap(map);
  CATCH_REQUIRE(map.size() == 3);

  map.swap(other);
  CATCH_REQUIRE(map.size() == 0);
  CATCH_REQUIRE(other.size() == 3);

  using std::swap;
  swap(map, other);
  CATCH_REQUIRE(map.size() == 3);
  CATCH_REQUIRE(other.size() == 0);
}

CATCH_TEST_CASE("trie_map_insert", "[trie_map_insert]") {
  using map_type = persistent_map<std::string, std::string>;
  std::vector<map_type::item_type> items{{{"A"s, "a"s}, {"B"s, "b"s}, {"C"s, "c"s}}};
  map_type map;

  CATCH_REQUIRE(map.size() == 0);
  for (const auto& item : items) {
    auto duplicate = item;
    auto key = item.first;
    CATCH_REQUIRE(!map.contains(key));
    CATCH_REQUIRE(map.insert(item));
    CATCH_REQUIRE(map.contains(key));
    CATCH_REQUIRE(!map.insert(item));
    CATCH_REQUIRE(map.erase(key) == 1);
    CATCH_REQUIRE(map.insert(std::move(duplicate)));
    CATCH_REQUIRE(map.contains(key));
    CATCH_REQUIRE(duplicate.first.size() == 0);
  }

  CATCH_REQUIRE(map.size() == items.size());
  map.clear();
  for (const auto& [key, value] : items) {
    CATCH_REQUIRE(!map.contains(key));
    CATCH_REQUIRE(map.at(key) == nullptr);
    CATCH_REQUIRE_THROWS_AS(map[key], std::out_of_range);
  }

  CATCH_REQUIRE(map.size() == 0);
  map.insert(std::begin(items), std::end(items));
  CATCH_REQUIRE(map.size() == 3);
  CATCH_REQUIRE(items[0].first == "A"s);
  CATCH_REQUIRE(items[0].second == "a"s);
  for (const auto& [key, value] : items) {
    CATCH_REQUIRE(map.contains(key));
    CATCH_REQUIRE(map.at(key) != nullptr);
    CATCH_REQUIRE(*map.at(key) == value);
    CATCH_REQUIRE(map[key] == value);
  }
}

CATCH_TEST_CASE("trie_map_insert_ilist", "[trie_map_insert_ilist]") {
  using map_type = persistent_map<std::string, std::string>;
  map_type map;
  map.insert({{"A"s, "a"s}, {"B"s, "b"s}, {"C"s, "c"s}});
  CATCH_REQUIRE(map.size() == 3);
}

CATCH_TEST_CASE("trie_map_erase_if", "[trie_map_erase_if]") {
  using map_type = persistent_map<int, std::string>;
  map_type map{{{0, "a"s}, {1, "b"s}, {2, "c"s}, {3, "d"s}}};
  CATCH_REQUIRE(map.size() == 4);

  auto predicate = [](int key) { return key % 2 == 0; };
  map_type other = map.erase_if(predicate);
  CATCH_REQUIRE(map.size() == 4);
  CATCH_REQUIRE(other.size() == 2);
  CATCH_REQUIRE(map != other);

  for (const auto& [key, value] : map)
    CATCH_REQUIRE(other.contains(key) == !predicate(key));

  erase_if(map, predicate);
  CATCH_REQUIRE(map.size() == 2);
  CATCH_REQUIRE(map == other);
}

CATCH_TEST_CASE("trie_map_emplace_extract", "[trie_map_emplace_extract]") {
  using map_type = persistent_map<int, TracedItem>;
  uint32_t counter = 0;
  {
    map_type map;
    CATCH_REQUIRE(map.emplace(0, TracedItem{counter, 1}));
    CATCH_REQUIRE(map.contains(0));

    auto opt = map.extract(0);
    CATCH_REQUIRE(opt.has_value());
    CATCH_REQUIRE(opt->second.value() == 1);
    CATCH_REQUIRE(!map.contains(0));

    auto other = map.extract(1);
    CATCH_REQUIRE(!other.has_value());
  }
  CATCH_REQUIRE(counter == 0);
}

CATCH_TEST_CASE("trie_map_insert_or_assign", "[trie_map_insert_or_assign]") {
  using map_type = persistent_map<int, TracedItem>;

  uint32_t counter = 0;
  {
    map_type map;
    CATCH_REQUIRE(map.insert_or_assign(10, TracedItem{counter, 1}));
    CATCH_REQUIRE(map.contains(10));
    CATCH_REQUIRE(map[10].value() == 1);

    auto b = TracedItem{counter, 2};
    auto c = TracedItem{counter, 3};
    auto d = TracedItem{counter, 4};

    CATCH_REQUIRE(map.insert_or_assign(1, b));
    CATCH_REQUIRE(map.contains(1));
    CATCH_REQUIRE(map[1].value() == 2);

    CATCH_REQUIRE(!map.insert({10, c}));
    CATCH_REQUIRE(map.contains(10));
    CATCH_REQUIRE(map[10].value() == 1);

    CATCH_REQUIRE(map.insert_or_assign(10, c));
    CATCH_REQUIRE(map.contains(10));
    CATCH_REQUIRE(map[10].value() == 3);
  }
  CATCH_REQUIRE(counter == 0);
}

CATCH_TEST_CASE("trie_map_insert_or_assign_traced_key", "[trie_map_insert_or_assign_traced_key]") {
  using map_type = persistent_map<TracedItem, TracedItem, TracedItem::Hasher>;

  const auto rollover = static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) + 1;

  uint32_t counter = 0;

  {
    auto a = TracedItem{counter, 1};
    auto b = TracedItem{counter, 2};
    auto c = TracedItem{counter, 3};
    auto d = TracedItem{counter, 4};
    auto e = TracedItem{counter, 5};
    auto a_plus = TracedItem{counter, a.value() + rollover};
    CATCH_REQUIRE(map_type::hash_function()(a) == map_type::hash_function()(a_plus));

    map_type map;
    CATCH_REQUIRE(map.insert_or_assign(TracedItem{counter, 10}, a));
    CATCH_REQUIRE(map.contains(TracedItem{counter, 10}));
    CATCH_REQUIRE(map[TracedItem{counter, 10}].value() == a.value());

    CATCH_REQUIRE(map.insert_or_assign(TracedItem{counter, 20}, a_plus));
    CATCH_REQUIRE(map.contains(TracedItem{counter, 20}));
    CATCH_REQUIRE(map[TracedItem{counter, 20}].value() == a_plus.value());

    CATCH_REQUIRE(map.insert_or_assign(TracedItem{counter, 10}, b));
    CATCH_REQUIRE(map.contains(TracedItem{counter, 10}));
    CATCH_REQUIRE(map[TracedItem{counter, 10}].value() == b.value());

    CATCH_REQUIRE(map.insert_or_assign(TracedItem{counter, 10 + rollover}, c));
    CATCH_REQUIRE(map.contains(TracedItem{counter, 10 + rollover}));
    CATCH_REQUIRE(map[TracedItem{counter, 10 + rollover}].value() == c.value());

    CATCH_REQUIRE(map.insert_or_assign(TracedItem{counter, 10 + 2 * rollover}, d));
    CATCH_REQUIRE(map.contains(TracedItem{counter, 10 + 2 * rollover}));

    CATCH_REQUIRE(map.insert_or_assign(TracedItem{counter, 10 + rollover}, e));
    CATCH_REQUIRE(map.contains(TracedItem{counter, 10 + rollover}));

    CATCH_REQUIRE(map[TracedItem{counter, 20}].value() == a_plus.value());
    CATCH_REQUIRE(map[TracedItem{counter, 10}].value() == b.value());
    CATCH_REQUIRE(map[TracedItem{counter, 10 + rollover}].value() == e.value());
    CATCH_REQUIRE(map[TracedItem{counter, 10 + 2 * rollover}].value() == d.value());
  }
  CATCH_REQUIRE(counter == 0);
}

CATCH_TEST_CASE("trie_map_insert_or_assign_with_move", "[trie_map_insert_or_assign_with_move]") {
  using map_type = persistent_map<int, std::string>;
  uint32_t counter = 0;
  {
    map_type map{{{0, "a"s}}};
    CATCH_REQUIRE(map.contains(0));
    CATCH_REQUIRE(map[0] == "a"s);

    auto b = "b"s;
    CATCH_REQUIRE(map.insert_or_assign(0, std::move(b)));
    CATCH_REQUIRE(b.size() == 0);
    CATCH_REQUIRE(map.contains(0));
    CATCH_REQUIRE(map[0] == "b"s);
  }
}

CATCH_TEST_CASE("trie_map_insert_or_assign_cref_key", "[trie_map_insert_or_assign_cref_key]") {
  using map_type = persistent_map<std::string, std::string>;
  map_type map{{{"A"s, "a"s}}};
  CATCH_REQUIRE(map.contains("A"s));
  CATCH_REQUIRE(map["A"s] == "a"s);

  auto key = "B"s;
  auto b = "b"s;
  CATCH_REQUIRE(map.insert_or_assign(key, std::move(b)));
  CATCH_REQUIRE(key.size() != 0);
  CATCH_REQUIRE(b.size() == 0);
  CATCH_REQUIRE(map.contains(key));
  CATCH_REQUIRE(map[key] == "b"s);
}

CATCH_TEST_CASE("trie_map_hash_func", "[trie_map_hash_func]") {
  using map_type = persistent_map<int, int>;
  auto func = map_type::hasher{};
  CATCH_REQUIRE(map_type::hash_function()(1) == func(1));
}

CATCH_TEST_CASE("trie_map_key_eq_func", "[trie_map_key_eq_func]") {
  using map_type = persistent_map<int, int>;
  CATCH_REQUIRE(map_type::key_eq()(1, 1));
  CATCH_REQUIRE(!map_type::key_eq()(1, 2));
}

CATCH_TEST_CASE("trie_map_const_iterators", "[trie_map_const_iterators]") {
  using map_type = persistent_map<int, std::string>;
  std::vector<map_type::item_type> items{{{0, "a"s}, {1, "b"s}, {2, "c"s}, {3, "d"s}}};
  map_type map{std::begin(items), std::end(items)};
  CATCH_REQUIRE(map.size() == 4);

  auto is_in = [&items](int key) {
    return std::find_if(std::begin(items), std::end(items),
                        [key](const auto& item) { return item.first == key; }) != std::end(items);
  };

  const auto& map_ref = map;
  for (auto [key, value] : map_ref) {
    CATCH_REQUIRE(is_in(key));
    CATCH_REQUIRE(map[key] == value);
  }

  for (auto ii = map.cbegin(); ii != map.cend(); ++ii) {
    CATCH_REQUIRE(is_in(ii->first));
    CATCH_REQUIRE(map[ii->first] == ii->second);
  }
}

} // namespace niggly::trie::test
