
#include "niggly/persistent-set.hpp"
#include "niggly/persistent-map.hpp"

#include <fmt/format.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <cassert>

using namespace std::string_literals;

using ticktock_type = std::chrono::time_point<std::chrono::steady_clock>;

static ticktock_type tick() { return std::chrono::steady_clock::now(); }
static std::chrono::microseconds tock(const ticktock_type& whence) {
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(now - whence);
}

struct Data {
  std::string label;
  std::size_t size;
  std::array<std::string, 6> columns;
  std::array<uint64_t, 6> insert_times;
  std::array<uint64_t, 6> iterate_times;
  std::array<uint64_t, 6> find_times;
  std::array<uint64_t, 6> delete_times;
  std::size_t volatile_data = 0; // to prevent optimizing away
};

template <typename T> T generate(std::size_t counter) {
  if constexpr (std::is_integral<T>::value) {
    return counter;
  } else if constexpr (std::is_same<T, std::string>::value) {
    return fmt::format("{0}:{1}:{0}:{2}", counter, 10 * counter, 437 * counter * 12345667);
  } else {
    assert(false);
  }
}

template <typename key_type, typename value_type>
std::vector<std::pair<key_type, value_type>> generate_items(std::size_t count) {
  using item_type = typename std::pair<key_type, value_type>;

  std::vector<item_type> items;
  items.reserve(count);
  for (auto i = 0u; i < count; ++i)
    items.push_back({generate<key_type>(i), generate<value_type>(i)});

  return items;
}

template <typename key_type, typename value_type,
          typename key_hasher = std::hash<key_type>,      // Hash function for item
          typename key_equal_to = std::equal_to<key_type> // Equality comparision for Item
          >
Data run_items(std::string label, const std::size_t size, const uint32_t sample_size) {
  using item_type = typename std::pair<key_type, value_type>;
  using vector_type = std::vector<item_type>;
  using persistent_map_type =
      niggly::persistent_map<key_type, value_type, key_hasher, key_equal_to, true>;
  using non_atomic_persistent_map_type =
      niggly::persistent_map<key_type, value_type, key_hasher, key_equal_to, false>;
  using std_map_type = std::unordered_map<key_type, value_type, key_hasher, key_equal_to>;

  const vector_type items = generate_items<key_type, value_type>(size);

  // 1. vector (with reserve)    // not find/delete
  // 2. vector (without reserve) // not find/delete
  // 3. unordered_map (with reserve)
  // 4. unordered_map (without reserve)
  // 5. persistent_map (with atomics)
  // 6. persistent_map (without atomics)

  Data data;
  data.label = label;
  data.size = size;
  data.columns = decltype(data.columns){
      {"vec-reserve"s, "vec"s, "std-map-reserve"s, "std-map"s, "atomic-trie"s, "na-trie"s}};

  vector_type vec_w_res;                          // 1.
  vector_type vec_wo_res;                         // 2.
  std_map_type std_map_w_res;                     // 3.
  std_map_type std_map_wo_res;                    // 4.
  persistent_map_type atomic_trie;                // 5.
  non_atomic_persistent_map_type non_atomic_trie; // 6.

  std_map_w_res.reserve(size);
  vec_w_res.reserve(size);

  const auto start = std::begin(items);
  const auto finish = std::end(items);

  auto profile = [sample_size](std::string_view label, auto thunk) {
    uint64_t total_us = 0;
    for (auto i = 0u; i < sample_size; ++i) {
      const auto reference = tick();
      thunk();
      total_us += tock(reference).count();
    }
    const auto average_us = uint64_t(total_us / double(sample_size));
    const auto seconds = average_us / 1000000;
    std::cout << fmt::format("             {:15s} = {}.{:06d}s\n", label, seconds,
                             average_us % 1000000);
    return average_us;
  };

  { // Insert
    std::cout << fmt::format("{}({}) -- INSERT\n", label, size);
    data.insert_times[0] =
        profile(data.columns[0], [&]() { vec_w_res.insert(vec_w_res.end(), start, finish); });
    data.insert_times[1] =
        profile(data.columns[1], [&]() { vec_wo_res.insert(vec_wo_res.end(), start, finish); });
    data.insert_times[2] = profile(data.columns[2], [&]() { std_map_w_res.insert(start, finish); });
    data.insert_times[3] =
        profile(data.columns[3], [&]() { std_map_wo_res.insert(start, finish); });
    data.insert_times[4] = profile(data.columns[4], [&]() { atomic_trie.insert(start, finish); });
    data.insert_times[5] =
        profile(data.columns[5], [&]() { non_atomic_trie.insert(start, finish); });
  }

  { // Iterate -- overhead for large iterator
    std::cout << fmt::format("{}({}) -- ITERATE\n", label, size);
    std::size_t counter = 0;
    data.iterate_times[0] = profile(data.columns[0], [&]() {
      for (const auto& item : vec_w_res)
        counter += item.second;
    });
    data.iterate_times[1] = profile(data.columns[1], [&]() {
      for (const auto& item : vec_wo_res)
        counter += item.second;
    });
    data.iterate_times[2] = profile(data.columns[2], [&]() {
      for (const auto& item : std_map_w_res)
        counter += item.second;
    });
    data.iterate_times[3] = profile(data.columns[3], [&]() {
      for (const auto& item : std_map_wo_res)
        counter += item.second;
    });
    data.iterate_times[4] = profile(data.columns[4], [&]() {
      for (const auto& item : atomic_trie)
        counter += item.second;
    });
    data.iterate_times[5] = profile(data.columns[5], [&]() {
      for (const auto& item : non_atomic_trie)
        counter += item.second;
    });
    data.volatile_data = counter;
  }

  { // Find
    std::cout << fmt::format("{}({}) -- ITERATE\n", label, size);
    std::size_t counter = 0;
    data.find_times[0] = profile(data.columns[0], [&]() {});
    data.find_times[1] = profile(data.columns[1], [&]() {});
    data.find_times[2] = profile(data.columns[2], [&]() {
      for (const auto& item : items)
        counter += std_map_w_res[item.first];
    });
    data.find_times[3] = profile(data.columns[3], [&]() {
      for (const auto& item : items)
        counter += std_map_wo_res[item.first];
    });
    data.find_times[4] = profile(data.columns[4], [&]() {
      for (const auto& item : items)
        counter += atomic_trie[item.first];
    });
    data.find_times[5] = profile(data.columns[5], [&]() {
      for (const auto& item : items)
        counter += non_atomic_trie[item.first];
    });
    data.volatile_data = counter;
  }

  { // Delete
    std::cout << fmt::format("{}({}) -- DELETE\n", label, size);
    data.delete_times[0] = profile(data.columns[0], [&]() {});
    data.delete_times[1] = profile(data.columns[1], [&]() {});
    data.delete_times[2] = profile(data.columns[2], [&]() {
      for (const auto& item : items)
        std_map_w_res.erase(item.first);
    });
    data.delete_times[3] = profile(data.columns[3], [&]() {
      for (const auto& item : items)
        std_map_wo_res.erase(item.first);
    });
    data.delete_times[4] = profile(data.columns[4], [&]() {
      for (const auto& item : items)
        atomic_trie.erase(item.first);
    });
    data.delete_times[5] = profile(data.columns[5], [&]() {
      for (const auto& item : items)
        non_atomic_trie.erase(item.first);
    });
  }

  std::cout << "\n";

  return data;
}

template <typename key_type, typename value_type, typename key_hasher = std::hash<key_type>>
void run_types(std::ostream& os, std::string label, std::size_t size0, std::size_t max_size,
               uint32_t sample_size) {
  // collect all the data
  std::vector<Data> data;
  for (std::size_t size = size0; size <= max_size; size *= 2)
    data.push_back(run_items<key_type, value_type, key_hasher>(label, size, sample_size));

  // const auto& headers = data[0].columns;

  auto output = [&](std::string op_type, auto fn) {
    os << fmt::format("{}_{}\t{}\n", label, op_type, fmt::join(data[0].columns, "\t"));
    for (const auto& datum : data)
      os << fmt::format("{}\t{}\n", datum.size, fmt::join(fn(datum), "\t"));
    os << "\n";
  };

  output("insert", std::mem_fn(&Data::insert_times));
  output("iterate", std::mem_fn(&Data::iterate_times));
  output("find", std::mem_fn(&Data::find_times));
  output("delete", std::mem_fn(&Data::delete_times));
}

void run_benchmark(std::string filename) {
  const std::size_t min_size = 1000;
  const std::size_t max_size = 2000000;
  const uint32_t sample_size = 20;
  std::fstream file(filename, file.out);
  if (!file.is_open()) {
    std::cerr << fmt::format("failed to open file '{}'\n", filename);
    std::exit(1);
  }

  run_types<int, int>(file, "integer", min_size, max_size, sample_size);
  run_types<std::string, int>(file, "string", min_size, max_size, sample_size);

  file.close();
  std::cout << fmt::format("Benchmark results collated in '{}'\n", filename);
}

int main(int argc, char* argv[]) { run_benchmark("/tmp/benchmark-data.csv"); }
