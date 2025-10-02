#include <vector>
#include "CudaError.h"

// template<typename T>
// class cuda_iterator : public std::iterator<std::forward_iterator_tag, T>
// {

// }

template <typename T>
class cuda_vector {
 private:
  T* d_data;
  unsigned m_size;

 public:
  cuda_vector(int size) {
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(T) * size));
    m_size = size;
  }

  cuda_vector(const cuda_vector<T>& other) {
    m_size = other.m_size;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(T) * other.m_size));
    CUDA_CHECK(cudaMemcpy(d_data, other.d_data, sizeof(T) * m_size,
                          cudaMemcpyDeviceToDevice));
  }

  cuda_vector(const std::vector<T>& h_vec) {
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(T) * h_vec.size()));
    m_size = h_vec.size();
    CUDA_CHECK(cudaMemcpy(d_data, h_vec.data(), sizeof(T) * m_size,
                          cudaMemcpyHostToDevice));
  }
  ~cuda_vector() { CUDA_CHECK(cudaFree(d_data)); };

  int size() const { return m_size; }

  T operator[](int i) const {
    T element;
    CUDA_CHECK(
        cudaMemcpy(&element, &d_data[i], sizeof(T), cudaMemcpyDeviceToHost));
    return element;
  }

  cuda_vector& operator=(const cuda_vector<T>& other) {
    CUDA_CHECK(cudaFree(d_data));
    m_size = other.m_size;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(T) * other.m_size));
    CUDA_CHECK(cudaMemcpy(d_data, other.d_data, sizeof(T) * m_size,
                          cudaMemcpyDeviceToDevice));
    return *this;
  }

  T* data() { return d_data; }

  std::vector<T> to_host_vector() {
    std::vector<T> result(m_size);
    cudaMemcpy(result.data(), d_data, sizeof(T) * m_size,
               cudaMemcpyDeviceToHost);
    return result;
  }

  void print() {
    auto temp = to_host_vector();
    for (auto& x : temp) {
      std::cout << x << " ";
    }
    std::cout << "\n";
  }
};
