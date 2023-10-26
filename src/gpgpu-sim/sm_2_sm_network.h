#ifndef SM_2_SM_NETWORK_H
#define SM_2_SM_NETWORK_H

#include "../abstract_hardware_model.h"
#include "local_interconnect.h"
#include "shader.h"

void sm2sm_network_options(class OptionParser* opp);

class cluster_shmem_request {
 public:
  cluster_shmem_request(warp_inst_t* warp, addr_t address, bool is_write,
                        bool is_atomic, unsigned origin_shader_id,
                        unsigned target_shader_id, unsigned tid);
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
  bool m_is_send;
  addr_t m_address;
  unsigned m_tid;
  unsigned m_origin_shader_id;
  unsigned m_target_shader_id;
  unsigned m_size;

 public:
  const bool& is_write = m_is_write;
  const bool& is_atomic = m_is_atomic;
  const bool& is_response = m_is_response;
  const bool& is_send = m_is_send;
  const unsigned& origin_shader_id = m_origin_shader_id;
  const unsigned& target_shader_id = m_target_shader_id;
  const unsigned& tid = m_tid;
  const addr_t address = m_address;
};

class sm_2_sm_network {
 public:
  enum Network_type { CROSSBAR, BOOKSIM };
  // Functions for local interconnect

  sm_2_sm_network(unsigned n_shader, const class shader_core_config* config,
                  const class gpgpu_sim* gpu);

  // void Init();
  virtual void Push(unsigned input_deviceID, unsigned output_deviceID,
                    void* data, unsigned int size,
                    Interconnect_type network) = 0;
  virtual void* Pop(unsigned ouput_deviceID, Interconnect_type network) = 0;
  virtual void Advance() = 0;
  virtual bool Busy() const = 0;
  virtual bool HasBuffer(unsigned deviceID, unsigned int size,
                         Interconnect_type network) const = 0;
  // virtual void DisplayStats() const;
  // virtual void DisplayOverallStats() const;
  // virtual unsigned GetFlitSize() const;

  // virtual void DisplayState(FILE* fp) const;

 protected:
  Network_type m_type;
  unsigned m_n_shader, m_n_mem;
  const class shader_core_config* m_config;
  const class gpgpu_sim* m_gpu;
};

class local_crossbar : public sm_2_sm_network {
 public:
  local_crossbar(unsigned n_shader, const class shader_core_config* config,
                 const class gpgpu_sim* gpu);
  ~local_crossbar();
  void Init();
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size, Interconnect_type network);
  void* Pop(unsigned ouput_deviceID, Interconnect_type network);
  void Advance();
  bool Busy() const;
  bool HasBuffer(unsigned deviceID, unsigned int size,
                 Interconnect_type network) const;
  void DisplayStats() const;
  void DisplayOverallStats() const;
  unsigned GetFlitSize() const;

  void DisplayState(FILE* fp) const;

 private:
  LocalInterconnect* m_localicnt_interface;

  std::ofstream m_request_net_in_log;
  std::ofstream m_request_net_out_log;
  std::ofstream m_reply_net_in_log;
  std::ofstream m_reply_net_out_log;
};

class ideal_network : public sm_2_sm_network {
 public:
  ideal_network(unsigned n_shader, const class shader_core_config* config,
                const class gpgpu_sim* gpu)
      : sm_2_sm_network(n_shader, config, gpu) {
    out_request.resize(n_shader);
    out_response.resize(n_shader);
  }
  ~ideal_network();
  void Init();
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size, Interconnect_type network);
  void* Pop(unsigned ouput_deviceID, Interconnect_type network);

  void Advance() { ; };
  bool Busy() const { return false; };
  bool HasBuffer(unsigned deviceID, unsigned int size,
                 Interconnect_type network) const {
    return true;
  }
  void DisplayStats() const { ; };
  void DisplayOverallStats() const { ; };
  unsigned GetFlitSize() const { return 1; };

  void DisplayState(FILE* fp) const { ; };

 private:
  std::vector<std::queue<void*>> out_request;
  std::vector<std::queue<void*>> out_response;
};

#endif  // SM_2_SM_NETWORK_H