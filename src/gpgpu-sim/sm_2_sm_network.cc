#include "sm_2_sm_network.h"
#include <algorithm>
#include <unordered_set>
#include <utility>
#include "gpu-sim.h"

int sm2sm_latency = 171;  // Default latency for SM to SM network
int sm2sm_bandwidth =
    88;  // Default bandwidth for SM to SM network in bits/cycle

void sm2sm_network_options(class OptionParser* opp) {
  option_parser_register(opp, "-sm_2_sm_network_latency", OPT_INT32,
                         &sm2sm_latency, "Latency of the SM to SM network",
                         "171");
  option_parser_register(
      opp, "-sm_2_sm_network_bw", OPT_INT32, &sm2sm_bandwidth,
      "Bandwidth of the SM to SM network in bit/cycle", "88");
}

cluster_shmem_request::cluster_shmem_request(warp_inst_t* warp, addr_t address,
                                             bool is_write, bool is_atomic,
                                             unsigned origin_shader_id,
                                             unsigned target_shader_id,
                                             unsigned tid, unsigned latency,
                                             unsigned size)
    : m_warp(warp), m_address(address) {
  m_is_write = is_write;
  m_is_atomic = is_atomic;
  m_target_shader_id = target_shader_id;
  m_origin_shader_id = origin_shader_id;
  m_tid = tid;
  m_is_send = false;
  m_is_response = false;
  m_is_complete = false;
  m_latency = latency;
  m_size = size;
}

SM_2_SM_network::SM_2_SM_network(unsigned n_shader,
                                 const class shader_core_config* config,
                                 const class gpgpu_sim* gpu) {
  m_n_shader = n_shader;
  m_config = config;
  m_gpu = gpu;
}

Crossbar::Crossbar(unsigned n_shader, const class shader_core_config* config,
                   const class gpgpu_sim* gpu)
    : SM_2_SM_network(n_shader, config, gpu),
      rr_pointers(n_shader, 0),
      input_queues(n_shader),
      output_queues(n_shader),
      m_bandwidth(sm2sm_bandwidth),
      m_latency(sm2sm_latency) {}

void Crossbar::Push(unsigned input_deviceID, unsigned output_deviceID,
                    std::shared_ptr<cluster_shmem_request> data,
                    unsigned int size, Interconnect_type network) {
  output_deviceID = sid_to_gid(output_deviceID);
  input_deviceID = sid_to_gid(input_deviceID);
  if (data.get()->is_atomic) {
    size = data->size * 8 * 8;
  } else {
    size = data->size * 8;
  }

  assert(data.get() != nullptr);
  input_queues[input_deviceID].emplace(input_deviceID, output_deviceID, size,
                                       m_time, data);

  // output_queues[output_deviceID].emplace(data);
}

std::shared_ptr<cluster_shmem_request> Crossbar::Pop(
    unsigned ouput_deviceID, Interconnect_type network) {
  ouput_deviceID = sid_to_gid(ouput_deviceID);
  if (!output_queues[ouput_deviceID].empty()) {
    auto result = output_queues[ouput_deviceID].front();
    output_queues[ouput_deviceID].pop();
    return result;
  }
  return nullptr;
}

void Crossbar::Advance() {
  // Quit early if no messages
  bool empty = true;
  for (const auto& node : input_queues) {
    if (!node.empty()) {
      empty = false;
      break;
    }
  }
  if (in_flight.empty() && empty) return;

  const int num_ports = m_n_shader;
  // std::cout << "Time step: " << m_time << "\n";

  // Check in-flight messages
  for (size_t i = 0; i < in_flight.size(); ++i) {
    if (m_time - in_flight[i].second >= m_latency) {
      // std::cout << "Delivered message from " << in_flight[i].first.src << "
      // to "
      //           << in_flight[i].first.dst << "\n";

      auto& data = in_flight[i].first.data;
      const auto dst = in_flight[i].first.dst;
      assert(sid_to_gid(data->target_shader_id) == dst);
      assert(!data->complete);
      data->complete = true;
      in_flight.erase(in_flight.begin() + i);
      i = 0;
    }
  }

  // Build request lists for each output
  std::vector<std::vector<int>> requests(num_ports);
  for (int i = 0; i < num_ports; ++i) {
    if (!input_queues[i].empty()) {
      int dst = input_queues[i].front().dst;
      requests[dst].push_back(i);
    }
  }

  // 3. Perform round-robin arbitration per output port
  for (int out = 0; out < num_ports; ++out) {
    if (!requests[out].empty()) {
      int chosen = -1;
      int start = rr_pointers[out];

      for (int i = 0; i < num_ports; ++i) {
        int idx = (start + i) % num_ports;
        for (int requester : requests[out]) {
          if (requester == idx) {
            chosen = idx;
            rr_pointers[out] = (idx + 1) % num_ports;
            break;
          }
        }
        if (chosen != -1) break;
      }

      if (chosen != -1) {
        Message& msg = input_queues[chosen].front();
        int send_amount = std::min(m_bandwidth, msg.size - msg.sent_bits);
        msg.sent_bits += send_amount;

        if (msg.sent_bits >= msg.size) {
          in_flight.push_back({msg, m_time});
          input_queues[chosen].pop();
        }
      }
    }
  }
  ++m_time;
}

IdealNetwork::IdealNetwork(unsigned n_shader, const shader_core_config* config,
                           const gpgpu_sim* gpu)
    : SM_2_SM_network(n_shader, config, gpu),
      input_queues(n_shader),
      output_queues(n_shader),
      m_time(0),
      m_latency(sm2sm_latency) {}

void IdealNetwork::Push(unsigned input_deviceID, unsigned output_deviceID,
                        std::shared_ptr<cluster_shmem_request> data,
                        unsigned int size, Interconnect_type network) {
  output_deviceID = sid_to_gid(output_deviceID);
  input_deviceID = sid_to_gid(input_deviceID);
  size = data->size * 8;
  assert(data.get() != nullptr);
  input_queues[input_deviceID].emplace(input_deviceID, output_deviceID, size,
                                       m_time, data);
}

void IdealNetwork::Advance() {
  // Quit early if no messages
  bool empty = true;
  for (const auto& node : input_queues) {
    if (!node.empty()) {
      empty = false;
      break;
    }
  }
  if (empty) return;

  // Process all input queues

  for (int i = 0; i < m_n_shader; i++) {
    if (!input_queues[i].empty()) {
      auto& msg = input_queues[i].front();
      if (m_time - msg.time_injected > m_latency) {
        msg.data->complete = true;
        input_queues[i].pop();
      }
    }
  }
  ++m_time;
}

std::shared_ptr<cluster_shmem_request> IdealNetwork::Pop(
    unsigned ouput_deviceID, Interconnect_type network) {
  return nullptr;  // Ideal network does not have a pop operation
}

Ringbus::Ringbus(unsigned n_shader, const class shader_core_config* config,
                 const class gpgpu_sim* gpu)
    : SM_2_SM_network(n_shader, config, gpu),
      input_queues(n_shader),
      link_buffers(n_shader),
      link_ready_time(n_shader),
      num_nodes(n_shader),
      m_time(0),
      bandwidth(sm2sm_bandwidth),
      m_latency(sm2sm_latency) {}

void Ringbus::Push(unsigned input_deviceID, unsigned output_deviceID,
                   std::shared_ptr<cluster_shmem_request> data,
                   unsigned int size, Interconnect_type network) {
  output_deviceID = sid_to_gid(output_deviceID);
  input_deviceID = sid_to_gid(input_deviceID);
  size = data->size * 8;
  assert(data.get() != nullptr);
  input_queues[input_deviceID].emplace(input_deviceID, output_deviceID, size,
                                       m_time, data);
}

std::shared_ptr<cluster_shmem_request> Ringbus::Pop(unsigned ouput_deviceID,
                                                    Interconnect_type network) {
  return nullptr;  // ToDo
}

void Ringbus::Advance() {
  if (!Busy()) return;

  // 1. Move messages that finished hop
  for (int node = 0; node < num_nodes; ++node) {
    std::vector<Message> new_buffer;
    for (Message& msg : link_buffers[node]) {
      if (m_time - msg.time_injected >= m_latency) {
        msg.current_node = next_node(msg.current_node);
        if (msg.current_node == msg.dst) {
          // std::cout << "Delivered message from " << msg.src << " to " <<
          // msg.dst
          //           << " at node " << node << "\n";
          auto& data = msg.data;
          assert(!data->complete);
          data->complete = true;
        } else {
          input_queues[msg.current_node].push(msg);
        }
      } else {
        new_buffer.push_back(msg);  // still in flight
      }
    }
    link_buffers[node] = new_buffer;
  }

  // 2. Transmit messages onto links
  for (int node = 0; node < num_nodes; ++node) {
    if (!input_queues[node].empty()) {
      Message& msg = input_queues[node].front();

      // Check if we can send a chunk
      int send_amount = std::min(bandwidth, msg.size - msg.sent_bits);
      msg.sent_bits += send_amount;

      if (msg.sent_bits >= msg.size) {
        msg.time_injected = m_time;
        link_buffers[node].push_back(msg);
        input_queues[node].pop();
      }
    }
  }

  ++m_time;
}

bool Ringbus::Busy() const {
  for (const auto& node : input_queues) {
    if (!node.empty()) return true;
  }

  for (const auto& node : link_buffers) {
    if (!node.empty()) return true;
  }

  for (const auto& node : link_ready_time) {
    if (!node.empty()) return true;
  }

  return false;
}