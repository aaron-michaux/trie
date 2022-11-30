
# HAMT Persistent Set and Map


A thread-safe persistent HAMT ([Hash Array Mapped Trie](https://idea.popcount.org/2012-07-25-introduction-to-hamt/)) for C++, inspired by [Phil Nash's hash trie](https://github.com/philsquared/hash_trie).

This is a tested rewrite that uses less memory, and is careful about object alignment.

and correctly handles the alignment of contained objects.

## Platform/Tested
 
 * Linux-x86_64 with gcc-12 and clang-14
 * Comprehensive testing
 * Asan, usan and tsan clean.
 
## Deficiencies

 * Containers are not Allocator aware.

## More information

 * [The Holy Grail - A Hash Array Mapped Trie for C++ - Phil Nash](https://www.youtube.com/watch?v=s9dwdo700eQ)
 * [CppConn 2017: Phil Nash "The Holy Grail..."](https://www.youtube.com/watch?v=imrSQ82dYns) (A longer version of the above talk.)
