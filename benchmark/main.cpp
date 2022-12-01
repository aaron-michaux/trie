
#include "niggly/persistent-set.hpp"
#include "niggly/persistent-map.hpp"

#include <fmt/format.h>

#include <iostream>

int main(int argc, char* argv[]) {
  for (auto i = 0; i < argc; ++i)
    std::cout << fmt::format("[{}] = '{}'\n", i, argv[i]);
}
