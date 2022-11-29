
#include <iostream>

#include <chrono>
#include <type_traits>
#include <functional>
#include <thread>
#include <tuple>
#include <cmath>

std::chrono::steady_clock::time_point tick() { return std::chrono::steady_clock::now(); }

template <typename T> T tock(std::chrono::steady_clock::time_point reference) {
  return std::chrono::duration_cast<T>(tick() - reference);
}

/**
 * Profile the passed thunk
 */
template <typename T, typename ThunkType, typename... Args>
T profileThunk(ThunkType thunk, Args... args) {
  const auto reference = tick();
  thunk(std::forward<Args>(args)...);
  return tock<T>(reference);
}

/**
 * Profiles a function with the specified arguments; see examples.
 *
 * @note: overload sets can cause problems with function resolution, in
 *        which case, curry the profiled function so that overloaded
 *        function selection occurs normally.
 *
 * ~~~
 * // Thunk returns "void", so result is just over 444 milliseconds
 * std::chrono::milliseconds millis
 *     = profile<std::chrono::milliseconds>([] () {
 *          std::this_thread::sleep(std::chrono::milliseconds{444});
 *       });
 *
 * // Function return value in std::pair<Result, std::chrono::milliseconds>
 * auto profile_result = profile<std::chrono::milliseconds>(square, 4);
 * // profile_result is {16, std::chrono::milliseconds}
 * ~~~
 */
template <typename ChronoType, typename FunctionType, typename... Args>
auto profile(FunctionType function, Args... args) {
  using R = typename std::invoke_result<FunctionType, Args...>::type;
  const auto reference = tick();
  if constexpr (std::is_same<R, void>::value) {
    function(std::forward<Args>(args)...);
    return tock<ChronoType>(reference);
  } else {
    R result = function(std::forward<Args>(args)...);
    return std::pair<R, ChronoType>{std::move(result), tock<ChronoType>(reference)};
  }
}

double square(double x) { return x * x; }

void some_thing(int ticks) { std::this_thread::sleep_for(std::chrono::milliseconds{ticks}); }

#ifndef BUILD_EXAMPLES
int main(int, char**) {

  std::cout << "Hello World!" << std::endl;

  double ret1;
  const auto tock1 = profileThunk<std::chrono::microseconds>([&ret1]() { ret1 = square(4.0); });
  std::cout << "square(4.0) = " << ret1 << "; microseconds = " << tock1.count() << std::endl;

  const auto t2 = profile<std::chrono::microseconds>(square, 3.0);
  std::cout << "square(3.0) = " << t2.first << "; microseconds = " << t2.second.count()
            << std::endl;

  const auto t3 = profile<std::chrono::microseconds>(some_thing, 444);
  std::cout << "some_thing; microseconds = " << t3.count() << std::endl;

  const auto t4 = profile<std::chrono::microseconds>(square, 27.0);
  std::cout << "std::cbrt(9.0) = " << t4.first << "; microseconds = " << t4.second.count()
            << std::endl;

  return 0;
}
#endif
