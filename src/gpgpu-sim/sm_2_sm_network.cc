#include "sm_2_sm_network.h"
#include "gpu-sim.h"
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
                                 const class shader_core_config* config,
                                 const class gpgpu_sim* gpu) {
  m_n_shader = n_shader;
  m_config = config;
  m_gpu = gpu;
  m_type = CROSSBAR;
}

local_crossbar::local_crossbar(unsigned n_shader,
                               const class shader_core_config* config,
                               const class gpgpu_sim* gpu)
    : sm_2_sm_network(n_shader, config, gpu) {
  m_inct_config = inct_config{.in_buffer_limit = 10,
                              .out_buffer_limit = 10,
                              .subnets = 2,
                              .arbiter_algo = NAIVE_RR,
                              .verbose = 0,
                              .grant_cycles = 1};
  m_localicnt_interface = new LocalInterconnect(m_inct_config);
  m_localicnt_interface->CreateInterconnect(n_shader, 0);

  if (m_config->sm_2_sm_network_log) {
    assert(m_config->n_simt_clusters == 1);
    m_request_net_in_log.open("req_in.csv");
    m_request_net_out_log.open("req_out.csv");
    m_reply_net_in_log.open("reply_in.csv");
    m_reply_net_out_log.open("reply_out.csv");

    m_request_net_in_log << "Cycle";
    m_reply_net_in_log << "Cycle";
    m_request_net_out_log << "Cycle";
    m_reply_net_out_log << "Cycle";
    for (int i = 0; i < n_shader; i++) {
      m_request_net_in_log << ",SM_" << i;
      m_reply_net_in_log << ",SM_" << i;
      m_request_net_out_log << ",SM_" << i;
      m_reply_net_out_log << ",SM_" << i;
    }
    m_request_net_in_log << std::endl;
    m_reply_net_in_log << std::endl;
    m_request_net_out_log << std::endl;
    m_reply_net_out_log << std::endl;
  }
}

local_crossbar::~local_crossbar() {
  delete m_localicnt_interface;

  m_reply_net_in_log.flush();
  m_reply_net_out_log.flush();
  m_request_net_in_log.flush();
  m_request_net_out_log.flush();
  m_request_net_in_log.close();
  m_request_net_out_log.close();
  m_reply_net_in_log.close();
  m_reply_net_out_log.close();
}

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

void local_crossbar::Advance() {
  m_localicnt_interface->Advance();

  if (m_config->sm_2_sm_network_log) {
    std::vector<int> req_in = m_localicnt_interface->get_req_in_size();
    std::vector<int> req_out = m_localicnt_interface->get_req_out_size();
    std::vector<int> reply_in = m_localicnt_interface->get_reply_in_size();
    std::vector<int> reply_out = m_localicnt_interface->get_reply_out_size();

    m_request_net_in_log << m_gpu->gpu_sim_cycle;
    m_request_net_out_log << m_gpu->gpu_sim_cycle;
    m_reply_net_in_log << m_gpu->gpu_sim_cycle;
    m_reply_net_out_log << m_gpu->gpu_sim_cycle;
    for (int i = 0; i < req_in.size(); i++) {
      m_request_net_in_log << "," << req_in.at(i);
      m_request_net_out_log << "," << req_out.at(i);
      m_reply_net_in_log << "," << reply_in.at(i);
      m_reply_net_out_log << "," << reply_out.at(i);
    }
    m_request_net_in_log << std::endl;
    m_request_net_out_log << std::endl;
    m_reply_net_in_log << std::endl;
    m_reply_net_out_log << std::endl;
  }
}

bool local_crossbar::Busy() const { return m_localicnt_interface->Busy(); }

bool local_crossbar::HasBuffer(unsigned deviceID, unsigned int size,
                               Interconnect_type network) const {
  return m_localicnt_interface->HasBuffer(deviceID, size, network);
}
