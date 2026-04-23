#pragma once
#include <memory>
#include <mutex>
#include <utility>

template <typename T>
class ThreadSafe {
 private:
  T data;                  // The data to be protected
  mutable std::mutex mtx;  // The mutex for protecting the data

 public:
  // if arguments are correct, then forward constructor arguments to constructor
  // of T
  template <typename... Args,
            typename = typename std::enable_if<
                std::is_constructible<T, Args&&...>::value>::type>
  explicit ThreadSafe(Args&&... args) : data(std::forward<Args>(args)...) {}
  // other wise we have our ThreadSafe specific constructors:

  // Copy constructor
  ThreadSafe(const ThreadSafe& other) {
    std::lock_guard<std::mutex> lock(other.mtx);
    data = other.data;
  }

  // Copy assignment operator
  ThreadSafe& operator=(const ThreadSafe& other) {
    if (this != &other) {
      std::scoped_lock lock(mtx, other.mtx);
      data = other.data;
    }
    return *this;
  }
  // proxy object to provide scoped RAII-style mutable access to the data
  class AccessGuard {
   private:
    T& dataRef;
    std::lock_guard<std::mutex> lock;

   public:
    AccessGuard(T& dataRef, std::mutex& mtx) : dataRef(dataRef), lock(mtx) {}

    // access to data
    T& operator*() { return dataRef; }

    // access to data's members
    T* operator->() { return &dataRef; }
  };

  AccessGuard access() { return AccessGuard(data, mtx); }

  // const accessguard for immutable access
  class ConstAccessGuard {
   private:
    const T& dataRef;
    std::lock_guard<std::mutex> lock;

   public:
    ConstAccessGuard(const T& dataRef, std::mutex& mtx)
        : dataRef(dataRef), lock(mtx) {}
    const T& operator*() const { return dataRef; }
    const T* operator->() const { return &dataRef; }
  };

  ConstAccessGuard caccess() { return ConstAccessGuard(data, mtx); }
};
