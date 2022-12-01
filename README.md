
# HAMT Persistent Set and Map


A HAMT ([Hash Array Mapped Trie](https://idea.popcount.org/2012-07-25-introduction-to-hamt/)) for C++, with thread-safe implementations of `persistent_set` and `persistent_map`.
Inspired by [Phil Nash's hash trie](https://github.com/philsquared/hash_trie).

This is a tested rewrite that uses less memory, and is careful about object alignment.

## Platform/Tested
 
 * Linux-x86_64 with gcc-12 and clang-14
 * Comprehensive testing
 * Asan, ubsan and tsan clean.
 
```
------------------------------------------------------------------------------
                           GCC Code Coverage Report
Directory: .
------------------------------------------------------------------------------
File                                       Lines    Exec  Cover   Missing
------------------------------------------------------------------------------
include/niggly/bits/trie-base.hpp            591     591   100%   
include/niggly/persistent-map.hpp             62      62   100%   
include/niggly/persistent-set.hpp             52      52   100%   
------------------------------------------------------------------------------
TOTAL                                        705     705   100%
------------------------------------------------------------------------------
```

 
 
## Deficiencies

Containers are not Allocator aware. I used `std::aligned_alloc` and `std::free` to manage allocation.

## More information

 * [The Holy Grail - A Hash Array Mapped Trie for C++ - Phil Nash](https://www.youtube.com/watch?v=s9dwdo700eQ)
 * [CppConn 2017: Phil Nash "The Holy Grail..."](https://www.youtube.com/watch?v=imrSQ82dYns) (A longer version of the above talk.)
