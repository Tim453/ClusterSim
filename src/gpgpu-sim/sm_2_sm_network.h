#ifndef SM_2_SM_NETWORK_H
#define SM_2_SM_NETWORK_H

#include "../abstract_hardware_model.h"
#include "../intersim2/interconnect_interface.hpp"
#include "array"
#include "local_interconnect.h"
#include "shader.h"

void sm2sm_network_options(class OptionParser* opp);

class cluster_shmem_request {
 public:
  cluster_shmem_request(warp_inst_t* warp, addr_t address, bool is_write,
                        bool is_atomic, unsigned origin_shader_id,
                        unsigned target_shader_id, unsigned tid,
                        unsigned latency, unsigned size);
  void send_response() { m_is_response = true; }
  // Called when data came from the target SM, then the request can be treated
  // as a normal request
  void atomic_sendback() {
    m_is_atomic = false;
    m_is_response = false;
  }
  warp_inst_t* get_warp() { return m_warp; }
  void send_request() { m_is_send = true; }

 private:
  // Warp the request belongs to
  warp_inst_t* m_warp;
  bool m_is_write;
  bool m_is_atomic;
  bool m_is_response;
  bool m_is_complete;
  addr_t m_address;
  unsigned m_tid;
  unsigned m_origin_shader_id;
  unsigned m_target_shader_id;
  unsigned m_size;
  // Time it takes to process the request for the target SM.
  unsigned m_latency;

 public:
  bool m_is_send = false;
  const bool& is_write = m_is_write;
  const bool& is_atomic = m_is_atomic;
  const bool& is_response = m_is_response;
  const bool& is_send = m_is_send;
  const unsigned& origin_shader_id = m_origin_shader_id;
  const unsigned& target_shader_id = m_target_shader_id;
  const unsigned& size = m_size;
  const unsigned& tid = m_tid;
  const addr_t address = m_address;
  const unsigned& latency = m_latency;
  bool complete = false;
};

class SM_2_SM_network {
 public:
  // Functions for local interconnect

  SM_2_SM_network(unsigned n_shader, const class shader_core_config* config,
                  const class gpgpu_sim* gpu);

  // void Init();
  virtual void Push(unsigned input_deviceID, unsigned output_deviceID,
                    std::shared_ptr<cluster_shmem_request> data,
                    unsigned int size, Interconnect_type network) = 0;
  virtual std::shared_ptr<cluster_shmem_request> Pop(
      unsigned ouput_deviceID, Interconnect_type network) = 0;
  virtual void Advance() = 0;
  virtual bool Busy() const = 0;
  virtual bool HasBuffer(unsigned deviceID, unsigned int size,
                         Interconnect_type network) const = 0;
  // virtual void DisplayStats() const;
  // virtual void DisplayOverallStats() const;
  // virtual unsigned GetFlitSize() const;

  // virtual void DisplayState(FILE* fp) const;

 protected:
  unsigned m_n_shader, m_n_mem;
  const class shader_core_config* m_config;
  const class gpgpu_sim* m_gpu;
  unsigned sid_to_gid(unsigned sid) const { return sid % m_n_shader; }
};

class Crossbar : public SM_2_SM_network {
 public:
  Crossbar(unsigned n_shader, const class shader_core_config* config,
           const class gpgpu_sim* gpu);
  ~Crossbar();
  void Init() {};
  void Push(unsigned input_deviceID, unsigned output_deviceID,
            std::shared_ptr<cluster_shmem_request>, unsigned int size,
            Interconnect_type network);
  std::shared_ptr<cluster_shmem_request> Pop(unsigned ouput_deviceID,
                                             Interconnect_type network);
  void Advance();
  bool Busy() const { return false; }
  bool HasBuffer(unsigned deviceID, unsigned int size,
                 Interconnect_type network) const {
    return true;
  };

  struct Message {
    int src;
    int dst;
    int size;  // in bits
    int sent_bits = 0;
    uint64_t time_injected;
    std::shared_ptr<cluster_shmem_request> data;
    Message(int s, int d, int sz, uint64_t t,
            std::shared_ptr<cluster_shmem_request> data)
        : src(s), dst(d), size(sz), time_injected(t), data(data) {}
  };

 private:
  std::vector<std::queue<Message>> input_queues;
  std::vector<std::queue<std::shared_ptr<cluster_shmem_request>>> output_queues;
  std::vector<std::pair<Message, uint64_t>> in_flight;
  std::vector<int> rr_pointers;
  uint64_t m_time;
  const int m_latency;
  const int m_bandwidth;
};

class IdealNetwork : public SM_2_SM_network {
 public:
  IdealNetwork(unsigned n_shader, const class shader_core_config* config,
               const class gpgpu_sim* gpu);
  ~IdealNetwork() {}
  void Init() {};
  void Push(unsigned input_deviceID, unsigned output_deviceID,
            std::shared_ptr<cluster_shmem_request> data, unsigned int size,
            Interconnect_type network);
  std::shared_ptr<cluster_shmem_request> Pop(unsigned ouput_deviceID,
                                             Interconnect_type network);
  void Advance();
  bool Busy() const { return false; }
  bool HasBuffer(unsigned deviceID, unsigned int size,
                 Interconnect_type network) const {
    return true;
  };

  struct Message {
    int src;
    int dst;
    int size;  // in bits
    int sent_bits = 0;
    uint64_t time_injected;
    std::shared_ptr<cluster_shmem_request> data;
    Message(int s, int d, int sz, uint64_t t,
            std::shared_ptr<cluster_shmem_request> data)
        : src(s), dst(d), size(sz), time_injected(t), data(data) {}
  };

 private:
  std::vector<std::queue<Message>> input_queues;
  std::vector<std::queue<std::shared_ptr<cluster_shmem_request>>> output_queues;
  uint64_t m_time;
  const int m_latency;  // Ideal network has no latency
};

#endif