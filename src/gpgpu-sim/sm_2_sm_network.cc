#include "sm_2_sm_network.h"

cluster_shmem_request::cluster_shmem_request(warp_inst_t* warp, addr_t address,
                                             bool is_write, bool is_atomic,
                                             unsigned origin_shader_id,
                                             unsigned target_shader_id,
                                             unsigned tid) {
  m_warp = warp;
  m_address = address;
  m_is_write = is_write;
  m_is_atomic = is_atomic;
  m_target_shader_id = target_shader_id;
  m_origin_shader_id = origin_shader_id;
  m_tid = tid;
  m_is_send = false;
  m_is_response = false;
  m_is_complete = false;
}

sm_2_sm_network::sm_2_sm_network(unsigned n_shader,
                                 const class shader_core_config* config) {
  m_n_shader = n_shader;
  m_config = config;
  m_type = CROSSBAR;
}

local_crossbar::local_crossbar(unsigned n_shader,
                               const class shader_core_config* config)
    : sm_2_sm_network(n_shader, config) {
  m_inct_config = inct_config{.in_buffer_limit = 10,
                              .out_buffer_limit = 10,
                              .subnets = 2,
                              .arbiter_algo = NAIVE_RR,
                              .verbose = 1,
                              .grant_cycles = 1};
  m_localicnt_interface = new LocalInterconnect(m_inct_config);
  m_localicnt_interface->CreateInterconnect(n_shader, 0);
}

local_crossbar::~local_crossbar() { delete m_localicnt_interface; }

void* local_crossbar::Pop(unsigned ouput_deviceID, Interconnect_type network) {
  ouput_deviceID = m_config->sid_to_cid(ouput_deviceID);
  return m_localicnt_interface->Pop(ouput_deviceID, network);
}

void local_crossbar::Push(unsigned input_deviceID, unsigned output_deviceID,
                          void* data, unsigned int size,
                          Interconnect_type network) {
  output_deviceID = m_config->sid_to_cid(output_deviceID);
  input_deviceID = m_config->sid_to_cid(input_deviceID);
  return m_localicnt_interface->Push(input_deviceID, output_deviceID, data,
                                     size, network);
}

void local_crossbar::Advance() { return m_localicnt_interface->Advance(); }

bool local_crossbar::Busy() const { return m_localicnt_interface->Busy(); }

bool local_crossbar::HasBuffer(unsigned deviceID, unsigned int size,
                               Interconnect_type network) const {
  return m_localicnt_interface->HasBuffer(deviceID, size, network);
}
