
#define CATCH_CONFIG_RUNNER

#include "bits/trie-base.hpp"

#include <catch2/catch.hpp>

int main(int argc, char* argv[]) {
  int result = Catch::Session().run(argc, argv);
  return result;
}
