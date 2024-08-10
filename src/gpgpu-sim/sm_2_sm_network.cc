#include "sm_2_sm_network.h"
#include <algorithm>
#include <unordered_set>
#include "gpu-sim.h"

inct_config sm2sm_crossbar_config;

bool bi_directional_ringbus;
char* sm2sm_intersim_config;

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
      "Arbiter Algorithm of the SM 2 SM network NAIVE_RR=0, iSLIP=1", "1");
  option_parser_register(opp, "-bi_directional_ringbus", OPT_INT32,
                         &bi_directional_ringbus,
                         "Ringbus 0 = unidirectional, 1 = bidirectional", "0");
  option_parser_register(opp, "-sm2sm_intersim_config_file", OPT_CSTR,
                         &sm2sm_intersim_config, "Config file for intersim",
                         "0");
}

cluster_shmem_request::cluster_shmem_request(warp_inst_t* warp, addr_t address,
                                             bool is_write, bool is_atomic,
                                             unsigned origin_shader_id,
                                             unsigned target_shader_id,
                                             unsigned tid, unsigned latency)
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
}

sm_2_sm_network::sm_2_sm_network(unsigned n_shader,
                                 const class shader_core_config* config,
                                 const class gpgpu_sim* gpu) {
  m_n_shader = n_shader;
  m_config = config;
  m_gpu = gpu;
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
  ouput_deviceID = sid_to_gid(ouput_deviceID);
  return m_localicnt_interface->Pop(ouput_deviceID, network);
}

void local_crossbar::Push(unsigned input_deviceID, unsigned output_deviceID,
                          void* data, unsigned int size,
                          Interconnect_type network) {
  output_deviceID = sid_to_gid(output_deviceID);
  input_deviceID = sid_to_gid(input_deviceID);
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
  deviceID = sid_to_gid(deviceID);
  return m_localicnt_interface->HasBuffer(deviceID, size, network);
}

void ideal_network::Push(unsigned input_deviceID, unsigned output_deviceID,
                         void* data, unsigned int size,
                         Interconnect_type network) {
  output_deviceID = sid_to_gid(output_deviceID);
  if (network == REQ_NET) {
    in_request[output_deviceID].push(data);
  } else if (network == REPLY_NET) {
    in_response[output_deviceID].push(data);
  }
}

booksim::booksim(unsigned n_shader, const class shader_core_config* config,
                 const class gpgpu_sim* gpu)
    : sm_2_sm_network(n_shader, config, gpu) {
  interface = InterconnectInterface::New(sm2sm_intersim_config);
  interface->CreateInterconnect(n_shader, 0);
}

void booksim::Push(unsigned input_deviceID, unsigned output_deviceID,
                   void* data, unsigned int size, Interconnect_type network) {
  output_deviceID = sid_to_gid(output_deviceID);
  input_deviceID = sid_to_gid(input_deviceID);
  mf_type type;
  cluster_shmem_request* request = (cluster_shmem_request*)(data);
  assert(request);
  if (network == REQ_NET) {
    if (request->is_write)
      type = WRITE_REQUEST;
    else
      type = READ_REQUEST;
  } else {
    if (request->is_write)
      type = WRITE_ACK;
    else
      type = READ_REPLY;
  }

  interface->Push(input_deviceID, output_deviceID, data, size, network, type);
}

void* booksim::Pop(unsigned ouput_deviceID, Interconnect_type network) {
  return interface->Pop(ouput_deviceID, network);
}

void booksim::Advance() { interface->Advance(); }
bool booksim::HasBuffer(unsigned deviceID, unsigned int size,
                        Interconnect_type network) const {
  return interface->HasBuffer(deviceID, size);
}

void* ideal_network::Pop(unsigned ouput_deviceID, Interconnect_type network) {
  void* result = nullptr;
  ouput_deviceID = sid_to_gid(ouput_deviceID);
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
  deviceID = sid_to_gid(deviceID);
  return m_in[network][deviceID].size() < m_in_out_buffer_size;
}

void ringbus::Push(unsigned input_deviceID, unsigned output_deviceID,
                   void* data, unsigned int size, Interconnect_type network) {
  output_deviceID = sid_to_gid(output_deviceID);
  input_deviceID = sid_to_gid(input_deviceID);
  m_in[network][input_deviceID].push(Packet(data, output_deviceID));
}

void* ringbus::Pop(unsigned ouput_deviceID, Interconnect_type network) {
  ouput_deviceID = sid_to_gid(ouput_deviceID);

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
            int distance = std::min(abs(i - targetID),
                                    (int)m_n_shader - abs(i - targetID));
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

H100Model::Node::Node(uint32_t packets_per_cycle)
    : packets_per_cycle(packets_per_cycle) {}

H100Model::Node::~Node() {}

H100Model::Processor::Processor(uint32_t packets_per_cycle,
                                           uint32_t block_rank)
    : Node(packets_per_cycle), block_rank(block_rank) {}

H100Model::Junction::Junction(uint32_t packets_per_cycle)
    : Node(packets_per_cycle) {}

H100Model::Pipe::Pipe(Node* in_node, Node* out_node,
                                 uint32_t packets_per_cycle,
                                 uint32_t buffer_capacity)
    : in_node(in_node),
      out_node(out_node),
      in_q(),
      out_q(),
      packets_per_cycle(packets_per_cycle),
      buffer_capacity(buffer_capacity),
      counterpart(nullptr),
      reachable_processors{false} {}

H100Model::H100Model(unsigned n_shader,
                       const class shader_core_config* config,
                       const class gpgpu_sim* gpu)
    : sm_2_sm_network(n_shader, config, gpu), node_list(), pipe_list() {
  // Default packets_per_cycle and buffer_capacity for all pipes and junctions.
  uint32_t ppc = 32;
  uint32_t bfs = 128;

  // Generate the 16 processors.
  for (uint32_t i = 0; i < 16; ++i) {
    node_list.push_back(new Processor(ppc, i));
  }

  // Generate the 8 junctions with the correct speeds.
  node_list.push_back(new Junction(ppc));      // JL0
  node_list.push_back(new Junction(ppc));      // JR1
  node_list.push_back(new Junction(ppc * 2));  // JM1
  node_list.push_back(new Junction(ppc));      // JL1
  node_list.push_back(new Junction(ppc));      // JR0
  node_list.push_back(new Junction(ppc));      // JM0
  node_list.push_back(new Junction(ppc));      // JL2
  node_list.push_back(new Junction(ppc));      // JR2

  // Hardcoded list of bidirectional edges for the interconnect.
  // Tuples follow the form (node_a, node_b, ppc, buf_size).
  // Will be expanded to the actual data structures down below.
  std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> edges;

  // SMs to Junctions
  edges.push_back({0, 16, ppc, bfs});  // JL0
  edges.push_back({1, 16, ppc, bfs});
  edges.push_back({2, 17, ppc, bfs});  // JR1
  edges.push_back({3, 17, ppc, bfs});
  edges.push_back({4, 18, ppc, bfs});  // JM1
  edges.push_back({5, 18, ppc, bfs});
  edges.push_back({6, 19, ppc, bfs});  // JL1
  edges.push_back({7, 19, ppc, bfs});
  edges.push_back({8, 20, ppc, bfs});  // JR0
  edges.push_back({9, 20, ppc, bfs});
  edges.push_back({10, 21, ppc, bfs});  // JM0
  edges.push_back({11, 21, ppc, bfs});
  edges.push_back({12, 22, ppc, bfs});  // JL2
  edges.push_back({13, 22, ppc, bfs});
  edges.push_back({14, 23, ppc, bfs});  // JR2
  edges.push_back({15, 23, ppc, bfs});

  // Junction to Junction
  edges.push_back({16, 19, ppc, bfs});  // JL0 <-> JL1
  edges.push_back({19, 22, ppc, bfs});  // JL1 <-> JL2

  edges.push_back({20, 17, ppc, bfs});  // JR0 <-> JR1
  edges.push_back({17, 23, ppc, bfs});  // JR1 <-> JR2

  edges.push_back({21, 18, ppc, bfs});  // JM0 <-> JM1

  edges.push_back({19, 18, ppc * 2, bfs * 2});  // JL1 <-> JM1
  edges.push_back({18, 17, ppc * 2, bfs * 2});  // JM1 <-> JR1

  // Translate the vector of tuples into actual edges.
  for (auto [node_idx_a, node_idx_b, packets_per_cycle, buffer_capacity] :
       edges) {
    // Create the two pipes.
    Pipe* ab_pipe =
        new Pipe(node_list[node_idx_a], node_list[node_idx_b],
                      packets_per_cycle, buffer_capacity);
    Pipe* ba_pipe =
        new Pipe(node_list[node_idx_b], node_list[node_idx_a],
                      packets_per_cycle, buffer_capacity);

    // Correctly set up counterparts.
    ab_pipe->counterpart = ba_pipe;
    ba_pipe->counterpart = ab_pipe;

    // Connect the pipes to their respective nodes.
    node_list[node_idx_a]->incoming_pipes.push_back(ba_pipe);
    node_list[node_idx_a]->outgoing_pipes.push_back(ab_pipe);
    node_list[node_idx_b]->incoming_pipes.push_back(ab_pipe);
    node_list[node_idx_b]->outgoing_pipes.push_back(ba_pipe);

    // Throw them onto the list.
    pipe_list.push_back(ab_pipe);
    pipe_list.push_back(ba_pipe);
  }

  // Perform BFS for each processor to calculate the routing hints.
  for (uint32_t proc = 0; proc < 16; ++proc) {
    // Prepare the node-queue and push the processor.
    std::deque<Node*> node_queue;
    node_queue.push_back(node_list[proc]);
    // Also keep around a set of nodes already visited.
    std::unordered_set<Node*> visited_nodes;

    // Continue as long as there are nodes in the queue.
    while (!node_queue.empty()) {
      // Grab the front node from the queue.
      Node* node = node_queue.front();
      node_queue.pop_front();

      // Mark it as visited.
      visited_nodes.insert(node);

      // Go through all incoming pipes of this node.
      for (auto in_pipe : node->incoming_pipes) {
        // What node is on the other side of this pipe?
        Node* other_node = in_pipe->in_node;

        // If the node on the other side of this pipe hasn't already
        // been visited, add a hint to this pipe that it can route packets
        // to the processor currently being evaluated and add the node to the
        // queue for further processing.
        if (visited_nodes.count(other_node) == 0) {
          // Mark the processor as reachable.
          in_pipe->reachable_processors[proc] = true;
          // And add the node to the queue.
          node_queue.push_back(other_node);
        }
      }
    }
  }

  // As a sanity check, print the whole network.
  // print_network();
}

void H100Model::print_network() {
  printf("Nodes: \n");
  for (uint32_t i = 0; i < node_list.size(); ++i) {
    Node* node = node_list[i];
    // Print differently depending on subtype.
    Processor* proc = dynamic_cast<Processor*>(node);
    Junction* jun = dynamic_cast<Junction*>(node);
    if (proc) {
      printf("  Node %2u is Processor %2u (ppc: %3u, ", i, proc->block_rank,
             proc->packets_per_cycle);
    } else if (jun) {
      printf("  Node %2u is Junction     (ppc: %3u, ", i,
             jun->packets_per_cycle);
    } else {
      printf("  Node %2u is INVALID                  ", i);
    }
    printf("incoming_pipes: (");
    for (auto pipe : node->incoming_pipes) {
      uint32_t idx = std::find(pipe_list.begin(), pipe_list.end(), pipe) -
                     pipe_list.begin();
      printf("%u,", idx);
    }
    printf("), outgoing_pipes: (");
    for (auto pipe : node->outgoing_pipes) {
      uint32_t idx = std::find(pipe_list.begin(), pipe_list.end(), pipe) -
                     pipe_list.begin();
      printf("%u,", idx);
    }
    printf("))\n");
  }

  printf("Pipes: \n");
  for (uint32_t i = 0; i < pipe_list.size(); ++i) {
    Pipe* pipe = pipe_list[i];
    // Determine the index of the incoming and outgoing node.
    uint32_t in_idx =
        std::find(node_list.begin(), node_list.end(), pipe->in_node) -
        node_list.begin();
    uint32_t out_idx =
        std::find(node_list.begin(), node_list.end(), pipe->out_node) -
        node_list.begin();

    printf("  Pipe %3u (in: %3u, out: %3u, ppc: %u, bufsize: %4u, reachable: (",
           i, in_idx, out_idx, pipe->packets_per_cycle, pipe->buffer_capacity);

    for (uint32_t j = 0; j < 16; ++j) {
      if (pipe->reachable_processors[j]) {
        printf("%u,", j);
      }
    }

    printf("))\n");
  }
}

H100Model::~H100Model() {
  // Free all of the pipes and nodes.
  for (auto pipe : pipe_list) delete pipe;
  for (auto node : node_list) delete node;
}

void H100Model::Push(unsigned input_deviceID, unsigned output_deviceID,
                      void* data, unsigned int size,
                      Interconnect_type network) {
  // Translate the SMIDs to the block rank of the corresponding processors.
  uint32_t input_rank = sid_to_gid(input_deviceID);
  uint32_t output_rank = sid_to_gid(output_deviceID);

  // Assemble the packet.
  Packet packet(input_rank, output_rank, network, data);

  // Grab a reference to the input-buffer of the input SMID.
  auto pipe = node_list[input_rank]->outgoing_pipes[0];

  // Push the packet onto the pipe.
  pipe->in_q.push_back(packet);

  // Check assumptions.
  if (pipe->in_q.size() > pipe->buffer_capacity) {
    printf("Packet count in input queue of pipe exceeds capacity.");
    exit(1);
  }
}

void* H100Model::Pop(unsigned output_deviceID, Interconnect_type network) {
  uint32_t output_block_rank = sid_to_gid(output_deviceID);
  auto pipe = node_list[output_block_rank]->incoming_pipes[0];

  // Check if there is a packet in the input-queue with the associated network.
  for (auto it = pipe->out_q.begin(); it != pipe->out_q.end(); it++) {
    if (it->associated_network == network) {
      // We found the packet we're looking for. Copy it.
      Packet packet = *it;
      // Remove it from the queue.
      pipe->out_q.erase(it);
      // And return the `cluster_shmem_request` contained within the packet.
      return packet.data;
    }
  }

  return nullptr;
}

// Nothing to advance within a processor.
void H100Model::Processor::Advance() {}

void H100Model::Junction::Advance() {
  // Iterate through all incoming pipes, in order.
  for (auto in_pipe : incoming_pipes) {
    // Process `ppc` packets.
    for (uint32_t i = 0; i < packets_per_cycle; ++i) {
      // We're done if there is nothing left to process.
      if (in_pipe->out_q.empty()) break;

      // Grab the next packet.
      Packet packet = in_pipe->out_q.front();

      // Find the destination pipe.
      Pipe* out_pipe = nullptr;
      for (auto op : outgoing_pipes) {
        if (op->reachable_processors[packet.destination_block_rank]) {
          out_pipe = op;
          break;
        }
      }

      if (out_pipe == nullptr) {
        printf(
            "Implementation error. Cannot determine route for given packet.");
        exit(1);
      }

      // Check if the pipe is full.
      if (out_pipe->in_q.size() >= out_pipe->buffer_capacity) {
        // We're stalling. Ignore the remaining packets at the input and try
        // again next cycle.
        break;
      }

      // Actually forward the packet by removing it from the incoming pipe and
      // putting it in the outgoing pipe.
      in_pipe->out_q.pop_front();
      out_pipe->in_q.push_back(packet);
    }
  }
}

void H100Model::Pipe::Advance() {
  // Forward as many packets as possible.
  for (uint32_t i = 0; i < packets_per_cycle; ++i) {
    // Quit if there are no more input packets *or* the output buffer is full.
    if (in_q.empty() || out_q.size() >= buffer_capacity) break;
    // Otherwise grab a packet from the in_q.
    Packet packet = in_q.front();
    in_q.pop_front();
    // And deposit it in the out_q.
    out_q.push_back(packet);
  }
}

void H100Model::Advance() {
  // First, advance all nodes.
  for (auto node : node_list) {
    node->Advance();
  }
  // Then, advance all pipes.
  for (auto pipe : pipe_list) {
    pipe->Advance();
  }
}

bool H100Model::Busy() const {
  // Check all pipes.
  for (auto pipe : pipe_list) {
    // If at least one pipe contains a packet somehwere, we're not done yet.
    if (!pipe->in_q.empty() || !pipe->out_q.empty()) {
      return true;
    }
  }
  return false;
}

bool H100Model::HasBuffer(unsigned deviceID, unsigned int size,
                           Interconnect_type network) const {
  // Translate the SMID to the processor's block rank.
  uint32_t block_rank = sid_to_gid(deviceID);
  // Grab a reference to the (only) outgoing pipe of the processor-node.
  auto pipe = node_list[block_rank]->outgoing_pipes[0];
  // And finally, check whether the input-queue has space for another packet.
  return pipe->in_q.size() < pipe->buffer_capacity;
}

H100Model::Packet::Packet(uint32_t source_block_rank,
                                     uint32_t destination_block_rank,
                                     Interconnect_type associated_network,
                                     void* data)
    : source_block_rank(source_block_rank),
      destination_block_rank(destination_block_rank),
      associated_network(associated_network),
      data(data) {}
