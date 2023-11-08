#include "sm_2_sm_network.h"
#include <algorithm>
#include "gpu-sim.h"

inct_config sm2sm_crossbar_config;

bool bi_directional_ringbus;

void sm2sm_network_options(class OptionParser* opp) {
  sm2sm_crossbar_config.subnets = 2;
  sm2sm_crossbar_config.verbose = 0;

  option_parser_register(opp, "-sm_2_sm_network_in_buffer_limit", OPT_INT32,
                         &sm2sm_crossbar_config.in_buffer_limit,
                         "Input Buffer Size of the SM 2 SM network", "32");
  option_parser_register(opp, "-sm_2_sm_network_out_buffer_limit", OPT_INT32,
                         &sm2sm_crossbar_config.out_buffer_limit,
                         "Output Buffer Size of the SM 2 SM network", "32");
  option_parser_register(opp, "-sm_2_sm_network_grant_cycles", OPT_INT32,
                         &sm2sm_crossbar_config.grant_cycles,
                         "Grant Cycles of the SM 2 SM network", "1");
  option_parser_register(
      opp, "-sm_2_sm_network_arbiter_algo", OPT_INT32,
      &sm2sm_crossbar_config.arbiter_algo,
      "Arbiter Algorithm of the SM 2 SM network NAIVE_RR=0, iSLIP=1", "0");
  option_parser_register(opp, "-bi_directional_ringbus", OPT_INT32,
                         &bi_directional_ringbus,
                         "Ringbus 0 = unidirectional, 1 = bidirectional", "0");
}

cluster_shmem_request::cluster_shmem_request(warp_inst_t* warp, addr_t address,
                                             bool is_write, bool is_atomic,
                                             unsigned origin_shader_id,
                                             unsigned target_shader_id,
                                             unsigned tid, unsigned latency) {
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
  m_latency = latency;
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
  m_localicnt_interface = new LocalInterconnect(sm2sm_crossbar_config);
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

void ideal_network::Push(unsigned input_deviceID, unsigned output_deviceID,
                         void* data, unsigned int size,
                         Interconnect_type network) {
  output_deviceID = m_config->sid_to_cid(output_deviceID);
  if (network == REQ_NET) {
    in_request[output_deviceID].push(data);
  } else if (network == REPLY_NET) {
    in_response[output_deviceID].push(data);
  }
}

void* ideal_network::Pop(unsigned ouput_deviceID, Interconnect_type network) {
  void* result = nullptr;
  ouput_deviceID = m_config->sid_to_cid(ouput_deviceID);
  if (network == REQ_NET && !out_request[ouput_deviceID].empty()) {
    result = out_request[ouput_deviceID].front();
    out_request[ouput_deviceID].pop();
  } else if (network == REPLY_NET && !out_response[ouput_deviceID].empty()) {
    result = out_response[ouput_deviceID].front();
    out_response[ouput_deviceID].pop();
  }
  return result;
}

void ideal_network::Advance() {
  for (int i = 0; i < m_n_shader; i++) {
    while (!in_request[i].empty()) {
      out_request[i].push(in_request[i].front());
      in_request[i].pop();
    }
    while (!in_response[i].empty()) {
      out_response[i].push(in_response[i].front());
      in_response[i].pop();
    }
  }
}

ringbus::ringbus(unsigned n_shader, const class shader_core_config* config,
                 const class gpgpu_sim* gpu)
    : sm_2_sm_network(n_shader, config, gpu) {
  m_bidirectional = bi_directional_ringbus;
  m_ring[REQ_NET].resize(n_shader);
  m_ring[REPLY_NET].resize(n_shader);

  m_out[REQ_NET].resize(n_shader);
  m_out[REPLY_NET].resize(n_shader);
  m_in[REQ_NET].resize(n_shader);
  m_in[REPLY_NET].resize(n_shader);
}

bool ringbus::HasBuffer(unsigned deviceID, unsigned int size,
                        Interconnect_type network) const {
  deviceID = m_config->sid_to_cid(deviceID);
  return m_in[network][deviceID].size() < m_in_out_buffer_size;
}

void ringbus::Push(unsigned input_deviceID, unsigned output_deviceID,
                   void* data, unsigned int size, Interconnect_type network) {
  output_deviceID = m_config->sid_to_cid(output_deviceID);
  input_deviceID = m_config->sid_to_cid(input_deviceID);
  m_in[network][input_deviceID].push(Packet(data, output_deviceID));
}

void* ringbus::Pop(unsigned ouput_deviceID, Interconnect_type network) {
  ouput_deviceID = m_config->sid_to_cid(ouput_deviceID);

  if (!m_out[network][ouput_deviceID].empty()) {
    auto packet = m_out[network][ouput_deviceID].front();
    m_out[network][ouput_deviceID].pop();
    assert(packet.output_deviceID == ouput_deviceID);
    return packet.data;
  }
  return nullptr;
}

void ringbus::Advance() {
  std::array<std::vector<std::queue<Packet>>, 2> next;
  next[REQ_NET].resize(m_n_shader);
  next[REPLY_NET].resize(m_n_shader);
  // Move messages to the next stage

  for (int subnet = 0; subnet < 2; subnet++) {
    for (int i = m_n_shader - 1; i >= 0; i--) {
      if (!m_ring[subnet][i].empty()) {
        int targetID = m_ring[subnet][i].front().output_deviceID;
        if (targetID == i && m_out[subnet][i].size() < m_in_out_buffer_size) {
          m_out[subnet][i].push(m_ring[subnet][i].front());
          m_ring[subnet][i].pop();
        } else if (targetID != i) {
          int next_node;

          if (m_bidirectional) {
            // Distance from current node to target
            int distance = std::min(abs(i + 1 - targetID),
                                    (int)m_n_shader - abs(i + 1 - targetID));
            // Distance if we move one step right
            int distance_right = std::min(
                abs(i + 1 - targetID), (int)m_n_shader - abs(i + 1 - targetID));
            if (distance_right < distance)
              next_node = (i + 1) % m_n_shader;
            else
              next_node = (i + m_n_shader - 1) % m_n_shader;
          } else {
            if (subnet == REQ_NET)
              next_node = (i + 1) % m_n_shader;
            else
              next_node = (i + m_n_shader - 1) % m_n_shader;
          }

          if (m_ring[subnet][next_node].size() < m_ring_buffer_size) {
            next[subnet][next_node].push(m_ring[subnet][i].front());
            m_ring[subnet][i].pop();
          }
        }
      }
    }

    for (unsigned i = 0; i < m_n_shader; i++) {
      while (!next[subnet][i].empty()) {
        m_ring[subnet][i].push(next[subnet][i].front());
        next[subnet][i].pop();
      }
    }

    // Move messages into the ringbus
    for (unsigned i = 0; i < m_n_shader; i++) {
      // Request net
      if (m_ring[subnet][i].size() < m_ring_buffer_size &&
          !m_in[subnet][i].empty()) {
        m_ring[subnet][i].push(m_in[subnet][i].front());
        m_in[subnet][i].pop();
      }
    }
  }
}